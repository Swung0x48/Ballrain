"""TAS dataset builder: replays each TAS record through agent mode with a replay policy (the record's
own frames, one frame per action) and stores the raw state of every frame.

Per record <out>/<stem>/:
  raw.npz     obs_bytes (u1, all OBS1 messages back to back), obs_offsets (n+1), frames_done (n),
              record_frames (the input), framelog (the mod's log), and convenience columns parsed
              from the raw state (ball_pos, ball_type, sr_double, n_mind, min_gap, n_fric, ...)
  level.msg   (+ level.msg.inc) the level messages written by the mod
  meta.json   record, level, sha256, counts, end reason, framelog check, SR, HS
The raw bytes are the source of truth: a parser fix never invalidates a dataset. The parsed columns
can be recomputed with load_observations().

To be executed by the gates phase (it launches the game); tests run it against ballrain.fakegame.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import instances, protocol as P, tas
from .client import AgentSession, SessionConfig
from .protocol import Action
from .recording import _json_default

DATASET_VERSION = 1


def level_of(name: str) -> int:
    m = re.match(r"(?:SR|HS)(\d\d)_", Path(name).stem)
    if not m:
        raise ValueError(f"no level in record name {name}")
    return int(m.group(1))


def _columns(obs_list) -> dict:
    n = len(obs_list)
    cols = {
        "ball_pos": np.full((n, 3), np.nan), "ball_type": np.full(n, 255, np.uint8),
        "ball_found": np.zeros(n, bool), "ball_vel": np.full((n, 3), np.nan),
        "sr_double": np.full(n, np.nan), "delta_ms": np.full(n, np.nan, np.float32),
        "n_mind": np.zeros(n, np.int16), "min_gap": np.full(n, np.nan, np.float32),
        "min_normal": np.full((n, 3), np.nan, np.float32), "min_partner": np.zeros(n, np.uint32),
        "n_fric": np.zeros(n, np.int16), "n_body": np.zeros(n, np.int16), "n_events": np.zeros(n, np.int16),
    }
    for i, o in enumerate(obs_list):
        if o.ball is not None:
            cols["ball_pos"][i] = o.ball["pos"]
            cols["ball_type"][i] = o.ball["type"]
            cols["ball_found"][i] = bool(o.ball["found"])
            cols["ball_vel"][i] = o.ball["speed"] + o.ball["speed_change"]
        if o.time is not None:
            cols["sr_double"][i] = o.time["sr_double"]
            cols["delta_ms"][i] = o.time["delta_ms"]
        cols["n_mind"][i] = len(o.mind)
        if len(o.mind):
            k = int(np.argmin(o.mind["gap"]))
            cols["min_gap"][i] = o.mind["gap"][k]
            cols["min_normal"][i] = o.mind["plane"][k][:3]
            cols["min_partner"][i] = o.mind["partner"][k]
        cols["n_fric"][i] = len(o.fric)
        cols["n_body"][i] = len(o.body)
        cols["n_events"][i] = len(o.events)
    return cols


def replay_record(record, out_dir, copy: str = instances.TRAINING_COPY, level: int | None = None, launcher=None,
                  frames_per_action: int = 1, obs_mask: int = 0, trace: bool = True, flags: int = 0,
                  save_raw: bool = True, name: str | None = None, prefix_frames: int = 0,
                  prefix_events: bool = False, pause=None) -> dict:
    """Replay policy: the record's own frames, frames_per_action per action (after an optional
    BALLRAIN_PREFIX of prefix_frames frames). flags are ACT1 flags sent with every action (ACT_WANT_TRAC);
    pause(step_index) is called before each action (stalls of the Python side, for fidelity tests).
    save_raw=False keeps no observations (only the checks); the run directory is <out>/<name>/run."""
    record = Path(record)
    frames = tas.load_any(record)
    level = level or level_of(record.name)
    out = Path(out_dir) / (name or record.stem)
    out.mkdir(parents=True, exist_ok=True)
    run = out / "run"
    pf = None
    if prefix_frames:
        pf = out / "prefix.bin"
        tas.save_raw(pf, frames[:prefix_frames])
    sess = AgentSession(SessionConfig(level=level, out_dir=run, copy=copy, trace=trace, prefix=pf,
                                      prefix_events=prefix_events), launcher)
    raws, fds = [], []
    obs_list = []
    error = None
    n_obs = 0
    gaps = 0  # observations whose frames_done is not the previous one + the block length
    last_fd = None
    steps = 0
    t0 = time.monotonic()
    t_start = None

    fol = {"checked": 0, "mismatch": 0, "first_mismatch": None, "max": 0}
    trafo = []  # [start, end) windows of tracker.transforming (frames_done)

    def keep(o, expect=None):
        nonlocal n_obs, gaps, last_fd
        n_obs += 1
        if expect is not None and o.frames_done != expect and not (o.final and o.frames_done < expect):
            gaps += 1
        last_fd = o.frames_done
        fds.append(o.frames_done)
        t = sess.tracker
        if o.scor_ext is not None and len(t.level.watch):
            # the mod's follower counter (SCOR, 0.3.1) against the Python rule on the WTCH history
            fol["checked"] += 1
            a, b = int(o.scor_ext["followers"]), t.follower_state()["total"]
            fol["max"] = max(fol["max"], a)
            if a != b:
                fol["mismatch"] += 1
                if fol["first_mismatch"] is None:
                    fol["first_mismatch"] = (o.frames_done, a, b)
        if t.transforming and (not trafo or trafo[-1][1] is not None):
            trafo.append([o.frames_done, None])
        elif not t.transforming and trafo and trafo[-1][1] is None:
            trafo[-1][1] = o.frames_done
        if save_raw:
            raws.append(o.raw)
            obs_list.append(o)

    if save_raw and prefix_events:
        sess.on_prefix_obs = keep
    try:
        o = sess.start()
        t_start = time.monotonic() - t0
        keep(o)
        while sess.frames_done < len(frames) and not sess.ended:
            i = sess.frames_done
            block = frames[i:i + frames_per_action]
            if pause is not None:
                pause(steps)
            o = sess.step(Action(block, 1, 0, flags, obs_mask))
            steps += 1
            if o is None:
                break
            keep(o, i + len(block))
        if not sess.ended:
            sess.stop()
    except Exception as e:  # recorded, the dataset entry is marked failed
        error = f"{type(e).__name__}: {e}"
    finally:
        sess.close()
    wall = time.monotonic() - t0
    framelog = tas.load_raw(sess.mod_framelog()) if sess.mod_framelog() else np.zeros(0, tas.FRAME_DTYPE)
    executed = sess.n_logged
    rec_prefix = frames[:len(framelog)]
    framelog_ok = len(framelog) == executed and framelog.tobytes() == rec_prefix.tobytes()
    if save_raw:
        offsets = np.cumsum([0] + [len(r) for r in raws]).astype(np.int64)
        blob = np.frombuffer(b"".join(raws), np.uint8)
        cols = _columns(obs_list)
        np.savez_compressed(out / "raw.npz", obs_bytes=blob, obs_offsets=offsets, frames_done=np.array(fds, np.int64),
                            record_frames=frames, framelog=framelog, **cols)
        for fname in ("level.msg", "level.msg.inc", "trace.txt", "events.txt"):
            src = run / fname
            if src.exists():
                shutil.copyfile(src, out / fname)
    t = sess.tracker
    final_sr = float(sess.final_time["sr_double"]) if sess.final_time is not None else None
    meta = {
        "dataset_version": DATASET_VERSION, "record": record.name,
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(), "level": level, "copy": copy,
        "record_frames": int(len(frames)), "executed_frames": int(executed), "observations": n_obs,
        "frames_per_action": frames_per_action, "end_reason": sess.end_reason_name, "error": error,
        "framelog_frames": int(len(framelog)), "framelog_equals_record_prefix": bool(framelog_ok),
        "levelfinish_frame": t.levelfinish_frame, "deaths": t.deaths,
        "sr_double_final": final_sr, "sr_ms": int(math.trunc(final_sr)) if final_sr is not None and t.levelfinish_frame else None,
        "sr_double_at_levelfinish": sess.sr_at_levelfinish,
        # official HS = end-screen score (docs/benchmarks.md "HS readout"): Points + 20 x followers + Lifes x
        # LifeBonus + LevelBonus at the end of the SR tail; hs_mod is the LevelFinish-hook snapshot without
        # the followers
        "hs": t.hs_at_finish, "hs_source": t.hs_source, "hs_followers": t.followers_at_finish,
        "hs_followers_at_hook": t.followers_at_hook, "hs_points_end": t.points_at_finish,
        "hs_lives_end": t.lives_at_finish, "followers_check": fol, "transforming_windows": trafo,
        "mod_version": (sess.hello or {}).get("mod_version"),
        "prefix_frames": int(prefix_frames), "prefix_events": bool(prefix_events), "prefix_observations": sess.n_prefix_obs,
        "observation_gaps": gaps, "steps": steps, "start_s": t_start, "wall_s": wall,
        "python_framelog_equals_mod": bool(len(framelog) == sess.n_logged
                                           and framelog.tobytes() == sess.frame_log().tobytes()),
        "sr_f32sum_final": float(sess.sren["sr_f32sum"]) if sess.sren is not None else None,
        "sr_bml_final": float(sess.sren["sr_bml"]) if sess.sren is not None else None,
        "sr_tail_frames": int(sess.sren["tail_frames"]) if sess.sren is not None else None,
        "sr_counter_inactive_seen": bool(sess.sren["counter_inactive_seen"]) if sess.sren is not None else None,
        "hs_mod": int(sess.sren["hs"]) if sess.sren is not None and sess.sren["level_finished"] else None,
        "hs_mod_valid": int(sess.sren["hs_valid"]) if sess.sren is not None else None,
        "hs_bml": int(sess.sren["bml_hs"]) if sess.sren is not None and sess.sren["level_finished"] else None,
        "hs_energy_at_finish": [int(x) for x in sess.sren["energy"]] if sess.sren is not None else None,
        "hs_level_bonus": int(sess.sren["level_bonus"]) if sess.sren is not None else None,
        "run_dir": str(run),
        "ok": error is None and framelog_ok and gaps == 0,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default), encoding="utf-8")
    return meta


def load_observations(npz_path):
    """Yield the parsed Observation of every stored message."""
    z = np.load(npz_path)
    blob, off = z["obs_bytes"].tobytes(), z["obs_offsets"]
    for i in range(len(off) - 1):
        yield P.parse_observation(blob[off[i]:off[i + 1]])


def build_dataset(records, out_dir, copies=(instances.TRAINING_COPY,), launcher_factory=None, **kw) -> list:
    """Replay records in parallel, one worker thread per copy (one game process per copy at a time)."""
    records = [Path(r) for r in records]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queue = list(records)
    lock = threading.Lock()
    results = []

    def worker(copy_name):
        launcher = launcher_factory() if launcher_factory else None
        while True:
            with lock:
                if not queue:
                    return
                rec = queue.pop(0)
            try:
                meta = replay_record(rec, out_dir, copy_name, launcher=launcher, **kw)
            except Exception as e:  # e.g. a record name without a level: reported, the batch goes on
                meta = {"dataset_version": DATASET_VERSION, "record": rec.name, "copy": copy_name,
                        "error": f"{type(e).__name__}: {e}", "ok": False}
            with lock:
                results.append(meta)

    with ThreadPoolExecutor(len(copies)) as ex:
        list(ex.map(worker, copies))
    results.sort(key=lambda m: m["record"])
    (out_dir / "index.json").write_text(json.dumps(results, indent=1, default=_json_default), encoding="utf-8")
    return results

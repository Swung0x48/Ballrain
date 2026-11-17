"""精彩回放: every hour, the best attempt of the last hour of training, replayed in the original game with rendering
on, recorded from the game's own picture, verified and published on the dashboard.

    uv run python tools/dashboard/highlight.py --every 3600          # the hourly job (low priority, hidden game)
    uv run python tools/dashboard/highlight.py --once [--force]      # one pass now (--force: even if unchanged)
    uv run python tools/dashboard/highlight.py gate                  # capture build vs. 0.3.2: traces byte-identical

Selection (current run, runs.csv): the runs of the last 60 minutes of learner time; of those that finished without a
death (LevelFinish, deaths 0, valid, frame log matched), the ones whose takeover frame k lies in the earliest
2000-frame start bin among them (the longest agent-controlled ones), fastest full-level SR first, then the other
finishes bin by bin (earlier takeover first), each by SR. Candidates are tried in that order until one replays exactly.

Replay input: the run's own <run_id>.tas, the complete frame log from level load (TAS prefix up to k, then the
agent's frames, exactly as injected). Retail runs are local; headless runs keep their finished run directories on
the Mac only, and the two small files are fetched over the key-only ssh setup of scripts/sim2real_m2.py pair
($BALLRAIN_MAC, $BALLRAIN_MAC_SSH, $BALLRAIN_SSH_EXE; shell environment only, never written anywhere).

Replay: BallrainIO >= 0.3.3 in agent mode on its own copy (runtime/video-00, 2002 physics), started through the
no-focus launcher (hidden window, never activated, muted): the first k frames are the BALLRAIN_PREFIX, the agent
part is sent in 2-frame blocks (the training's action repeat), then the SR tail and a few seconds of idle frames so
the airship ending is visible. BALLRAIN_CAPTURE streams the back buffer at fixed 1/60 s game-time steps to a pipe
(ballrain.capture); nothing on the screen is read.

Verification: the mod's frame log equals the .tas, the end is LevelFinish at the run's LevelFinish frame, no BallOff,
and the SR (trunc of the mod's double sum after the SR tail, BML+ 0.3.12 rule) equals the run's sr_ms; for a traced
run also the ball trace. A mismatch is retried once (retail's rare ball-pair ordering nondeterminism), then the next
candidate is taken. The published video and json are replaced only after a verified replay and a finished encode.

Output (runtime/dashboard/media/, untracked, live on the public page): only complete, verified sets, swapped in at once
(publish): highlight-<v>.mp4 (H.264, 1024x768, 60 fps, faststart, no audio), highlight-overlay-<v>.mp4 (1366x768, the
side panel) and highlight.json naming them (public-safe metadata only; the dashboard re-checks it). Work files stay in
runtime/video-work/; --media <dir> publishes into a draft folder instead (tests). Any failure keeps the previous set.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one core: the machine is busy training
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import data as D  # noqa: E402
from ballrain import capture, instances, tas  # noqa: E402

MEDIA = ROOT / "runtime" / "dashboard" / "media"
WORK = ROOT / "runtime" / "video-work"
COPY = "video-00"
VIDEO = "highlight.mp4"
OVERLAY = "highlight-overlay.mp4"
META = "highlight.json"
WINDOW_S = 3600.0
FPS = 60.0
POST_S = 6.0  # idle game seconds after the SR tail: the airship ending
STEP = 2  # agent frames per block (action repeat of the training)
MAX_CANDIDATES = 5
TAS_RECORD = "SR01_1_14_457_264"
TAS_FILE = ROOT / "runtime" / "ref" / "ModLoader" / "TASRecords" / f"{TAS_RECORD}.tas"
CAPTURE_BUILD = ROOT / "runtime" / "mod-builds" / "BallrainIO-0.3.3-capture.bmodp"
BASE_BUILD = ROOT / "runtime" / "mod-builds" / "BallrainIO-0.3.2.bmodp"

# Ballance terms shown in Chinese (wording of the Chinese Ballance community; renamed in one place)
TERMS = {"checkpoint": "盘点", "sector": "小节", "fan": "风扇", "airship": "飞船"}

REMOTE_DIR_RE = re.compile(r"^/[A-Za-z0-9_./-]+/episodes/hl-\d\d/L\d\d-[a-z]+-hl-\d\d-\d{8}-\d{6}-\d+-\d+$")
RUN_ID_RE = re.compile(r"^L\d\d-[a-z]+-(?:hl|train)-\d\d-\d{8}-\d{6}-\d+-\d+$")


def log(*a):
    print(dt.datetime.now().strftime("%m-%d %H:%M:%S"), *a, flush=True)


def set_priority(pid: int | None = None, cls: int = capture.BELOW_NORMAL_PRIORITY_CLASS) -> bool:
    """Priority class of this process (pid None; children started afterwards inherit it) or of `pid`."""
    if sys.platform != "win32":
        if pid is None:
            os.nice(10)
        return True
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    if pid is None:
        return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), cls))
    h = k32.OpenProcess(0x0200, False, int(pid))  # PROCESS_SET_INFORMATION
    if not h:
        return False
    try:
        return bool(k32.SetPriorityClass(h, cls))
    finally:
        k32.CloseHandle(h)


def lower_priority():
    set_priority()  # inherited by the game and the encoders started from here


# ---------------------------------------------------------------------------------------- selection

def backend_of(copy: str) -> str:
    return "headless" if str(copy).startswith("hl") else "retail"


def candidates(run: D.Run, window_s: float = WINDOW_S, bins=D.SR_BIN_EDGES) -> tuple[list[dict], dict]:
    """Ordered candidates of the run's last `window_s` learner seconds, and a summary of the choice."""
    cols = ["update", "t", "copy", "run_id", "end_reason", "sr_ms", "start_frame", "frames", "deaths",
            "framelog_match", "invalid", "dir"]
    d = D.read_many(run.files("runs*.csv"), usecols=cols, dedup=None)
    if d is None or not len(d):
        return [], {"reason": "no runs"}
    import pandas as pd
    for c in ("update", "t", "sr_ms", "start_frame", "deaths", "frames"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["t", "start_frame"])
    tmax = float(d["t"].max())
    h = d[d["t"] >= tmax - window_s]
    inv = h["invalid"].astype(str).str.strip().str.lower()
    valid = h["invalid"].isna() | inv.isin(["", "nan", "none", "false", "0", "0.0"])
    fm = h["framelog_match"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    ok = h[(h["end_reason"] == "LevelFinish") & (h["deaths"] == 0) & h["sr_ms"].notna() & valid & fm
           & h["run_id"].astype(str).str.match(RUN_ID_RE.pattern)]
    if not len(ok):
        return [], {"runs_in_window": int(len(h)), "finished_no_death": 0,
                    "t_window": [round(tmax - window_s, 1), round(tmax, 1)]}
    # the earliest start bin among the clean finishes (the longest agent-controlled ones); the rest follow bin by bin
    # (earlier takeover first), each by SR: the full-level SR alone would favour late takeovers (mostly TAS)
    edges = np.asarray(bins)
    sbin = edges[np.searchsorted(edges, ok["start_frame"].to_numpy(), side="right") - 1]
    ok = ok.assign(_bin=sbin)
    lo = int(ok["_bin"].min())
    hi = int(min(b for b in bins if b > lo))
    first = ok[ok["_bin"] == lo].sort_values(["sr_ms", "t"])
    rest = ok.drop(first.index).sort_values(["_bin", "sr_ms", "t"])
    rows = [{k: v for k, v in dict(r).items() if k != "_bin"} for _, r in pd.concat([first, rest]).iterrows()]
    for r in rows:
        r["in_first_bin"] = bool(lo <= r["start_frame"] < hi)
    return rows, {"runs_in_window": int(len(h)), "finished_no_death": int(len(ok)), "bin": [int(lo), int(hi)],
                  "in_bin": int(len(first)), "t_window": [round(tmax - window_s, 1), round(tmax, 1)]}


# ------------------------------------------------------------------------------------------- fetch

def ssh_cmd() -> list[str] | None:
    import shlex
    mac = os.environ.get("BALLRAIN_MAC")
    if not mac:
        return None
    exe = os.environ.get("BALLRAIN_SSH_EXE") or (r"C:\Program Files\Git\usr\bin\ssh.exe"
                                                 if Path(r"C:\Program Files\Git\usr\bin\ssh.exe").is_file() else "ssh")
    return [exe, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"] + shlex.split(os.environ.get("BALLRAIN_MAC_SSH", "")) + [mac]


def fetch(run: D.Run, row: dict, dst: Path) -> dict:
    """Puts <run_id>.tas / .json (and the agent trace when the run was traced) into dst; returns their paths."""
    rid = str(row["run_id"])
    dst.mkdir(parents=True, exist_ok=True)
    out = {"tas": dst / f"{rid}.tas", "json": dst / f"{rid}.json", "trace": None, "source": None}
    local = [run.path / "episodes-traced" / str(row["copy"]) / rid]
    if backend_of(row["copy"]) == "retail" and row.get("dir"):
        local.append(Path(str(row["dir"])))
    for p in local:
        if (p / f"{rid}.tas").is_file() and (p / f"{rid}.json").is_file():
            shutil.copyfile(p / f"{rid}.tas", out["tas"])
            shutil.copyfile(p / f"{rid}.json", out["json"])
            if (p / "trace.txt").is_file():
                shutil.copyfile(p / "trace.txt", dst / "agent_trace.txt")
                out["trace"] = dst / "agent_trace.txt"
            out["source"] = "traced" if "episodes-traced" in str(p) else "local"
            return out
    rdir = str(row.get("dir") or "")
    if not REMOTE_DIR_RE.match(rdir) or ".." in rdir or not rdir.endswith("/" + rid):
        raise RuntimeError("run directory not local and not a valid remote run directory")
    ssh = ssh_cmd()
    if ssh is None:
        raise RuntimeError("run files are on the training machine and BALLRAIN_MAC is not set")
    r = subprocess.run(ssh + [f"cd '{rdir}' && tar cf - '{rid}.json' '{rid}.tas'"], capture_output=True, timeout=120,
                       stdin=subprocess.DEVNULL, creationflags=capture.low_priority_flags(idle=False))
    if r.returncode != 0:
        raise RuntimeError(f"ssh fetch failed (exit {r.returncode}): {r.stderr[-300:]!r}")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        for name, key in ((f"{rid}.json", "json"), (f"{rid}.tas", "tas")):
            m = tf.getmember(name)
            if not m.isfile() or m.size > 64 << 20:
                raise RuntimeError(f"unexpected tar member {name}")
            out[key].write_bytes(tf.extractfile(m).read())
    out["source"] = "remote"
    return out


# ------------------------------------------------------------------------------------------- replay

def _recipe(copy, env):
    return instances.launch_nofocus(copy, env, visible=False, audio_on=False, keep_focus=False)


class _Launcher:
    """GameLauncher for the video copy with the capture variables added (or explicitly cleared)."""

    backend = "retail"

    def __init__(self, copy: str, capture_path: str | None, fps: float, extra: dict | None = None):
        from ballrain.client import GameLauncher
        if copy != COPY:
            raise ValueError(f"the highlight replay runs on {COPY} only")
        self.inner = GameLauncher(copy, recipe=_recipe)
        self.capture_path, self.fps = capture_path, fps
        self.extra = dict(extra or {})  # e.g. BALLRAIN_POST_FINISH / BALLRAIN_PREFIX_EVENTS for the env-driven replay
        self.starts = 0

    def start(self, env):
        self.starts += 1
        env = dict(env, BALLRAIN_CAPTURE=self.capture_path if self.starts == 1 else None,  # one pipe, one process
                   BALLRAIN_CAPTURE_FPS=f"{self.fps:g}" if self.capture_path and self.starts == 1 else None,
                   **self.extra)
        h = self.inner.start(env)
        set_priority(h.pid)  # our own game process only (the handle is bound to video-00's Player.exe)
        return h

    stop = staticmethod(lambda h, wait_s=10.0: instances.stop_handle(h, wait_s))


def replay(frames: np.ndarray, k: int, out: Path, *, level: int = 1, copy: str = COPY, record: bool = True,
           fps: float = FPS, post_s: float = POST_S, step: int = STEP, on_obs=None, prefix_events: bool = False) -> dict:
    """Plays `frames` (the complete frame log from level load) on the retail copy: frames[:k] as the prefix, the rest in
    `step`-frame blocks. record: capture the picture (rendering on) into out/capture.mp4 + out/capture.json; else
    rendering is skipped as in training. on_obs(obs, frames_done, session) after the hand-over and every block."""
    from ballrain.client import AgentSession, SessionConfig
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    frames = np.asarray(frames, tas.FRAME_DTYPE)
    if not 0 <= k <= len(frames):
        raise ValueError(f"k {k} outside 0..{len(frames)}")
    prefix = None
    if k > 0:
        prefix = out / "prefix.bin"
        tas.save_raw(prefix, frames[:k])
    last_delta = float(frames["delta"][-1]) if len(frames) else 1000 / 132
    post = int(math.ceil(post_s * 1000.0 / max(last_delta, 1.0))) if record else 0
    sink = srv = None
    if record:
        sink = capture.FfmpegSink(out / "capture.mp4", fps)
        srv = capture.CaptureServer(sink)
    cfg = SessionConfig(level=level, out_dir=out, copy=copy, prefix=prefix, trace=True, skip_render=not record,
                        launch="nofocus", timeout_ms=10 * 60 * 1000, prefix_events=prefix_events)
    # the ending: idle frames after the final observation (the run itself ends exactly as in training)
    launcher = _Launcher(copy, srv.path if srv else None, fps,
                         extra={"BALLRAIN_CAPTURE_TAIL": str(post)} if record else None)
    sess = AgentSession(cfg, launcher)
    t0 = time.time()
    cap = None
    try:
        obs = sess.start()
        if on_obs is not None:
            on_obs(obs, sess.frames_done, sess)
        i = sess.frames_done
        if i != k and not sess.ended:
            raise RuntimeError(f"hand-over at frame {i}, expected {k}")
        while not sess.ended and i < len(frames):
            blk = frames[i:i + step]
            obs = sess.act(blk)
            i += len(blk)
            if on_obs is not None and obs is not None:
                on_obs(obs, sess.frames_done, sess)
        if not sess.ended:
            sess.stop()
    finally:  # always release the game, the pipe and the encoder (a later replay reuses the work directory)
        try:
            sess.close()
        finally:
            if srv is not None:
                srv.join(60)
                srv.close()
                cap = sink.close()
    wall = time.time() - t0
    if cap is not None:
        cap["server_error"] = srv.error
        cap["launches"] = launcher.starts
        (out / "capture.json").write_text(json.dumps(cap), encoding="utf-8")
    evs = _events(out / "events.txt")
    t = sess.tracker
    sren = sess.sren or {}
    sr_double = float(sren["sr_double"]) if sren else None
    res = {"end": sess.end_reason_name, "levelfinish_frame": t.levelfinish_frame,
           "balloffs": sum(1 for _, n in evs if n == "BallOff"),
           "sr_double": sr_double, "sr_ms": int(math.trunc(sr_double)) if sr_double is not None and t.levelfinish_frame
           is not None else None, "wall_s": round(wall, 1), "post_frames": post,
           "framelog_equal": None, "capture": None if cap is None else {k_: v for k_, v in cap.items() if k_ != "rows"}}
    fl = sess.mod_framelog()
    if fl is not None:
        got = tas.load_raw(fl)
        res["framelog_equal"] = bool(len(got) == len(frames) and got.tobytes() == frames.tobytes())
        res["framelog_is_prefix"] = bool(len(got) <= len(frames) and got.tobytes() == frames[:len(got)].tobytes())
        res["framelog_frames"] = int(len(got))
    return res


def _events(p: Path) -> list[tuple[int, str]]:
    out = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.split()
            if len(s) >= 2 and s[0].isdigit():
                out.append((int(s[0]), s[1]))
    return out


def _trace_lines(p: Path) -> list[bytes]:
    from ballrain import trace
    return [trace._strip_eol(x) for x in trace.read_lines(p)]


def verify(meta: dict, frames: np.ndarray, res: dict, out: Path, agent_trace: Path | None) -> dict:
    v = {"framelog": res.get("framelog_equal") is True,
         "finish": res["end"] == "LevelFinish",
         "levelfinish_frame": res.get("levelfinish_frame") == meta.get("levelfinish_frame"),
         "no_death": res.get("balloffs") == 0,
         "sr_ms": res.get("sr_ms") is not None and res.get("sr_ms") == meta.get("sr_ms")}
    if agent_trace is not None and (out / "trace.txt").is_file():
        v["trace"] = _trace_lines(agent_trace) == _trace_lines(out / "trace.txt")
    v["matched"] = all(v.values())
    return v


# ------------------------------------------------------------------------------------------- publish

def tz_label() -> str:
    return D.tz_label()


def ensure_capture_build() -> str:
    """The capture build must be the BallrainIO of the video copy (a build without capture would render every replay
    for nothing). Installs it when another build is there and no game runs in the copy. Returns "" or why not."""
    if not CAPTURE_BUILD.is_file():
        return "capture build missing"
    want = instances.sha256_file(CAPTURE_BUILD)
    if instances.mod_hashes([COPY]).get(COPY) == want:
        return ""
    try:
        instances.sync_mod(CAPTURE_BUILD, [COPY])  # refuses while a Player.exe runs in the copy
    except instances.InstanceError as e:
        return str(e)[:200]
    return "" if instances.mod_hashes([COPY]).get(COPY) == want else "sync failed"


def capture_complete(work: Path, res: dict) -> tuple[bool, str]:
    """The video covers the whole run: one game process (a hand-over relaunch runs without capture), the encoder and
    the pipe ended cleanly, no frame was dropped, it starts at the level load and reaches the LevelFinish frame plus
    the capture tail."""
    try:
        cap = json.loads((work / "capture.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "no capture.json"
    rows = cap.get("rows") or []
    if cap.get("ffmpeg_exit") != 0 or cap.get("server_error"):
        return False, f"encoder / pipe: {cap.get('server_error')} {str(cap.get('ffmpeg_err'))[:200]}"
    if cap.get("launches", 1) != 1:
        return False, f"{cap.get('launches')} game launches"
    if list(cap.get("size") or ()) != [1024, 768]:  # what the panel layout and the page assume
        return False, f"capture size {cap.get('size')}, expected 1024x768"
    if not rows or sum(r[1] for r in rows) != cap.get("video_frames"):
        return False, "no frames / frame count mismatch"
    if rows[0][2] > 4:
        return False, f"first captured frame {rows[0][2]}"
    lf = res.get("levelfinish_frame")
    want = (lf or 0) + int(res.get("post_frames") or 0) - 4
    if lf is None or rows[-1][2] < want:
        return False, f"last captured frame {rows[-1][2]} < {want}"
    exp = int(rows[-1][3] * FPS / 1000.0) + 1  # video frames due by the last captured game time
    if abs(cap["video_frames"] - exp) > 2:
        return False, f"{cap['video_frames']} video frames for {rows[-1][3]:.0f} ms of game time"
    return True, ""


MEDIA_NAME_RE = re.compile(r"^highlight(?:-overlay)?-\d{9,12}\.mp4$")
LEGACY_NAMES = ("highlight.mp4", "highlight-overlay.mp4")  # the first version's fixed names


def publish(media: Path, meta_pub: dict, plain: Path, overlay_mp4: Path) -> dict:
    """Promotes one complete set (plain video, overlay video, json) at once: both videos go in under new versioned
    names that nothing references yet, then highlight.json is replaced in one step (the page and the server follow
    the json), then videos of older sets are removed (the set before stays for viewers still playing it)."""
    media.mkdir(parents=True, exist_ok=True)
    v = int(meta_pub["v"])
    names = {"video": f"highlight-{v}.mp4", "overlay": f"highlight-overlay-{v}.mp4"}
    for key, src in (("video", plain), ("overlay", overlay_mp4)):
        os.replace(src, media / names[key])
    try:
        prev = json.loads((media / META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev = {}
    pub = dict(meta_pub, **names)
    capture.write_json(media / META, pub)
    keep = set(names.values()) | {prev.get(k) for k in ("video", "overlay")}
    for f in media.iterdir():
        # older versioned sets, and the fixed names of the first version (no longer served once the json is versioned)
        if (MEDIA_NAME_RE.match(f.name) or f.name in LEGACY_NAMES) and f.name not in keep:
            try:
                f.unlink()
            except OSError:  # being sent right now: removed by a later pass
                pass
    return pub


def public_meta(run: D.Run, row: dict, meta: dict, res: dict, ver: dict, dur: float | None, sizes: dict) -> dict:
    started = float(meta.get("started_at") or 0)
    return {
        "run": run.name,
        "update": int(row["update"]),
        "episode_at": dt.datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M") if started else None,
        "recorded_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tz": tz_label(),
        "sr_ms": int(res["sr_ms"]),
        "k": int(row["start_frame"]),
        "tas_ms": int(D.benchmarks()["tas"]),
        # the run's own record of its backend (a copy name does not tell x87-soft from a headless 2002 engine)
        "backend": D.BACKEND_LABEL.get(str(meta.get("backend") or backend_of(row["copy"])), "其他"),
        "matched": bool(ver["matched"]),
        "duration_s": round(dur, 1) if dur else None,
        "fps": int(FPS),
        "size": [1024, 768],
        "files": {k: int(v) for k, v in sizes.items()},
        "v": int(time.time()),
    }


def run_once(force: bool = False, max_candidates: int = MAX_CANDIDATES, keep_work: bool = False,
             media: Path = MEDIA) -> dict:
    runs = D.discover_runs()
    name = D.default_run(runs)
    run = next((r for r in runs if r.name == name), None)
    if run is None:
        return {"status": "no run"}
    # a resumed learner writes runs.1.csv, runs.2.csv, ...: freshness comes from the newest of them
    logs = [p for p in run.path.glob("runs*.csv") if p.is_file()]
    if not logs or time.time() - max(p.stat().st_mtime for p in logs) > 2 * 3600:
        return {"status": "no recent training data"}
    rows, summ = candidates(run)
    log(f"{run.name}: {summ}")
    if not rows:
        return {"status": "no finished run in the window", **summ}
    why = ensure_capture_build()
    if why:
        log(f"capture build not installed: {why}")
        return {"status": "capture build not installed", "why": why, **summ}
    old = {}
    try:
        old = json.loads((media / META).read_text(encoding="utf-8"))
    except Exception:
        pass
    tried = []
    for row in rows[:max_candidates]:
        rid = str(row["run_id"])
        if not force and old.get("_id") == _short_id(rid):
            return {"status": "unchanged", **summ}
        work = WORK / "hl"
        try:
            src = fetch(run, row, WORK / "src")
            meta = json.loads(src["json"].read_text(encoding="utf-8"))
            frames = tas.load_any(src["tas"])
            k = int((meta.get("start_state") or {}).get("frame") or row["start_frame"])
            if k != int(row["start_frame"]) or len(frames) != int(meta.get("frames") or -1):
                raise RuntimeError(f"run files disagree with runs.csv (k {k}, frames {len(frames)})")
            if k > 0 and frames[:k].tobytes() != tas.load_any(TAS_FILE)[:k].tobytes():
                raise RuntimeError("prefix is not the TAS record's first k frames")
        except Exception as e:  # noqa: BLE001
            log(f"candidate {_short_id(rid)}: input unavailable: {type(e).__name__}: {e}")
            tried.append({"id": _short_id(rid), "error": type(e).__name__})
            continue
        panel = None
        ok = False
        for attempt in (1, 2):
            log(f"candidate {_short_id(rid)} u{int(row['update'])} k {k} sr {int(row['sr_ms'])} "
                f"({backend_of(row['copy'])}), replay {attempt}")
            try:  # the training env drives the replay: the panel's numbers come from this very replay
                import overlay
                res, panel = overlay.env_replay(run, row, meta, frames, k, work, post_s=POST_S, fps=FPS,
                                                launcher_cls=_Launcher, log=log)
            except Exception:  # noqa: BLE001  no complete set from this attempt
                log("  env replay failed:\n" + traceback.format_exc(limit=4))
                tried.append({"id": _short_id(rid), "attempt": attempt, "error": "env replay"})
                continue
            ver = verify(meta, frames, res, work, src["trace"])
            log(f"  replay: {json.dumps({x: res[x] for x in ('end', 'levelfinish_frame', 'balloffs', 'sr_ms', 'wall_s')})}"
                f" verify {ver}")
            cap_ok, why = capture_complete(work, res)
            if not cap_ok:
                log(f"  capture incomplete: {why}")
            if not panel.get("sr_check"):
                log("  panel SR clock does not match the mod's SR sum")
            tried.append({"id": _short_id(rid), "attempt": attempt, "matched": ver["matched"], "capture": cap_ok})
            ok = ver["matched"] and cap_ok and bool(panel.get("sr_check"))
            if ok:
                break
        if not ok:
            continue
        # both videos into the work folder first; the live folder only ever gets a complete set (publish)
        enc = WORK / "enc"
        shutil.rmtree(enc, ignore_errors=True)
        enc.mkdir(parents=True)
        t0 = time.time()
        try:
            plain = enc / VIDEO
            capture.encode_final(work / "capture.mp4", plain, fps=FPS)
            dur = capture.probe_duration(plain)
            pub = public_meta(run, row, meta, res, ver, dur, {})
            ov = overlay.render_overlay(work, frames, panel, pub, enc / OVERLAY, fps=FPS, log=log)
            dur_ov = capture.probe_duration(ov)
            if not dur or not dur_ov or abs(dur - dur_ov) > 0.1:
                raise RuntimeError(f"durations differ: plain {dur}, overlay {dur_ov}")
        except Exception:  # noqa: BLE001  keep the previous complete set live
            log("encode failed, the previous set stays published:\n" + traceback.format_exc(limit=6))
            tried.append({"id": _short_id(rid), "error": "encode"})
            continue
        (WORK / "last-agreement.json").write_text(json.dumps(panel["agree"]), encoding="utf-8")
        sizes = {VIDEO: plain.stat().st_size, OVERLAY: ov.stat().st_size}
        log(f"  encoded in {time.time() - t0:.0f}s: {sizes}, {dur}s")
        pub = public_meta(run, row, meta, res, ver, dur, sizes)
        pub["_id"] = _short_id(rid)
        _check_public(pub)
        pub = publish(media, pub, plain, ov)
        if not keep_work:
            shutil.rmtree(WORK / "src", ignore_errors=True)
        log(f"published {pub}")
        return {"status": "published", "meta": pub, "tried": tried, **summ}
    return {"status": "no candidate replayed exactly", "tried": tried, **summ}


def _short_id(rid: str) -> str:
    import zlib
    return f"{zlib.crc32(rid.encode()):08x}"


def _check_public(pub: dict) -> None:
    import report
    bad = report.check(json.dumps(pub, ensure_ascii=False))
    if bad:
        raise RuntimeError(f"highlight metadata not public-safe: {bad}")


# ---------------------------------------------------------------------------------------------- gate

def cmd_gate(a) -> int:
    """Byte-identity of the capture build: the same frames on video-00 with BallrainIO 0.3.2 (rendering skipped, no
    capture) and with the capture build (rendering on, capture on); traces, frame logs and SR must be identical."""
    from ballrain import trace  # noqa: F401
    records = a.records.split(",")
    out_root = WORK / "gate"
    res = {}
    try:
        for label, build, rec in (("base", BASE_BUILD, False), ("capture", CAPTURE_BUILD, True)):
            instances.sync_mod(build, [COPY])
            for name in records:
                frames = tas.load_any(ROOT / "runtime" / "ref" / "ModLoader" / "TASRecords" / f"{name}.tas")
                level = int(name[2:4])
                k = int(a.k)
                out = out_root / f"{name}-{label}"
                r = replay(frames, min(k, len(frames)), out, level=level, record=rec, prefix_events=rec)
                r["trace_sha256"] = instances.sha256_file(out / "trace.txt")
                r["trace_lines"] = len(_trace_lines(out / "trace.txt"))
                res[f"{name}/{label}"] = r
                keys = ("end", "levelfinish_frame", "sr_ms", "trace_sha256", "trace_lines", "framelog_is_prefix",
                        "wall_s")
                log(f"{name} {label}: {json.dumps({x: r.get(x) for x in keys})}")
    finally:  # whatever happened, the video copy gets its capture build back (the hourly job needs it)
        why = ensure_capture_build()
        if why:
            log(f"could not restore the capture build: {why}")
    ok = True
    for name in records:
        A, B = res[f"{name}/base"], res[f"{name}/capture"]
        same = (A["trace_sha256"] == B["trace_sha256"] and A["sr_double"] == B["sr_double"]
                and A["levelfinish_frame"] == B["levelfinish_frame"] and A["framelog_is_prefix"]
                and B["framelog_is_prefix"])  # a record runs on past LevelFinish; the frame log stops there
        fb_a = (out_root / f"{name}-base" / "framelog.bin").read_bytes()
        fb_b = (out_root / f"{name}-capture" / "framelog.bin").read_bytes()
        same = same and fb_a == fb_b
        ok &= same
        log(f"{name}: {'IDENTICAL' if same else 'DIFFERENT'} trace {A['trace_sha256'][:16]} vs {B['trace_sha256'][:16]}, "
            f"sr {A['sr_double']!r} vs {B['sr_double']!r}, video frames {(B.get('capture') or {}).get('video_frames')}")
    (out_root / "gate.json").write_text(json.dumps({"ok": ok, "k": a.k, "results": res}, indent=1), encoding="utf-8")
    return 0 if ok else 1


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--every", type=float, default=0, help="seconds between passes (0 = one pass)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--force", action="store_true", help="replay even if the best run is the one already published")
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--media", default=str(MEDIA),
                   help="where a complete set is published (default: the live folder; a scratch folder for tests)")
    p.add_argument("--test-env-fail", type=int, default=0, help=argparse.SUPPRESS)  # fallback test: fail after N steps
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gate")
    g.add_argument("--records", default="SR01_1_14_457_264,SR06_1_29_838_264")
    g.add_argument("--k", type=int, default=6000, help="prefix frames (the rest goes through 2-frame agent blocks)")
    a = p.parse_args(argv)
    lower_priority()
    if a.test_env_fail:
        import overlay
        overlay.FAIL_AFTER_STEPS = a.test_env_fail
    if a.cmd == "gate":
        return cmd_gate(a)
    while True:
        t0 = time.time()
        try:
            r = run_once(force=a.force, keep_work=a.keep_work, media=Path(a.media))
            log("pass:", json.dumps({k: v for k, v in r.items() if k != "meta"}, ensure_ascii=False))
        except Exception:  # noqa: BLE001  keep the hourly job alive; the published files stay as they were
            log("pass failed:\n" + traceback.format_exc())
        if not a.every:
            return 0
        time.sleep(max(60.0, a.every - (time.time() - t0)))


if __name__ == "__main__":
    raise SystemExit(main())

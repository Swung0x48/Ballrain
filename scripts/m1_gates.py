"""M1 gates: BallrainIO agent mode driven by the Python client (ballrain.client / ballrain.env), against M0
and against the real TASSupport.

Usage (from the repo root):
    uv run python scripts/m1_gates.py [--gates A,B,C,D,E] [--fresh] [--only NAME ...] [--m0 DIR]
                                      [--e-seconds 60]

  A  fidelity regression (BLOCKING). A Python replay policy feeds every frame (delta + keys) of each M0 G1
     record and each M0 G2 synthetic input through agent mode, one frame per action, with every
     observation section on, TRAC requested, the level message / geometry export written and parsed,
     and the client's tracker updated on every observation. trace.txt and framelog.bin must be
     byte-identical to the M0 BallrainIO outputs (<m0>/G1|G2/<case>/bio/). Variants on SR06: K=4 blocks,
     prefix hand-over with and without prefix observations, Python-side stalls, and 10 repeats. The agent
     runs are serial on train-00. Then replay mode (the new build) re-runs every case in parallel on
     train-00..07: trace, frame log and events (without the wall-clock Summary line) must equal M0.
  B  export gate. Random-policy episodes through ballrain.env.BallanceEnv on train-00..07 in parallel
     (level 1 with the scripted Q press, deaths, GameOver, frame limits, LevelFinish via a TAS prefix and a
     random agent tail). Every process's frame log is exported to a legacy .tas (ballrain.recording),
     replayed 3x by the real TASSupport in runtime/ref; PASS = at least 2 of 3 reference traces identical
     and the agent-mode trace byte-identical to that majority (ballrain.gates.majority_verdict). The
     planned coverage (level 1, a death, a GameOver, a LevelFinish, >= 8 processes) is checked.
  D  TAS dataset: ballrain.dataset.build_dataset on every TAS record (plus the patched records in
     <runtime>/tas-patched, e.g. HS04 with Esc pauses), 8 copies in parallel, into
     <runtime>/datasets/tas-v1/. Its traces are also compared with M0 (fidelity under 8-instance load), and
     the mod's per-frame follower counter (SCOR) with the Python rule on the WTCH history.
  C  SR/HS readout from the D metadata (SR: the mod's SREN block, BML+ 0.3.12 timer driven by the real
     CounterActive/Inactive hooks; HS: the official end-screen readout, Points + 20 x followers + Lifes x
     LifeBonus + LevelBonus at the end of the SR tail, docs/benchmarks.md), compared with
     benchmarks/records.json and the file names. Gated: every completing HS record must read its name
     score, except HS04_6925_2_27_295 (the original never presses Esc and loses small balls at checkpoints;
     its patched copy HS04_6925_2_27_295_esc is gated). --write-records updates sr_ms_bmlp0312 /
     sr_ms_float32 in benchmarks/records.json.
  E  throughput: null / random policy through BallanceEnv on 1, 4 and 8 instances (one OS process per
     instance), plus the raw client without features; decision steps/s, frames/s, lives per hour,
     process start overhead, CPU load.

Results: <runtime>/m1-results/results.json (rewritten after every case) and per-case directories.
Re-running resumes: a case already in results.json for the same installed mod hash is skipped unless
--fresh. All comparisons are byte-exact; nothing is ever reconciled.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from ctypes import wintypes
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

import m0_gates as m0  # noqa: E402  (reference replays: run_ref; sets the direct launch recipe)
from ballrain import dataset, gates, instances, protocol as P, tas  # noqa: E402

instances.LAUNCH_RECIPE = instances.launch_direct

RUNTIME = instances.runtime_dir()
OUT = RUNTIME / "m1-results"
RESULTS = OUT / "results.json"
M0_DIR = RUNTIME / "m0-results-r2"
RECORDS = m0.RECORDS  # reference copy TASRecords (read only, except installed m1_*.tas exports)
COPIES = instances.training_copy_names(8)
BUILD_MOD = REPO / "BallrainIO" / "build" / "BallrainIO.bmodp"
DATASET_OUT = RUNTIME / "datasets" / f"tas-v{dataset.DATASET_VERSION}"
PATCHED = RUNTIME / "tas-patched"  # extra records (e.g. HS04 with Esc pauses inserted), dataset + C only
RECORD_NAME = re.compile(r"^(SR|HS)\d\d_[\d_]+$")  # the TAS records of the reference copy
HS_NOT_GATED = {"HS04_6925_2_27_295": "name score not reachable without Esc pauses (small balls lost at "
                                      "checkpoints); see HS04_6925_2_27_295_esc"}
RECORDS_JSON = REPO / "benchmarks" / "records.json"
GATE_DEF = 1

SR06 = "SR06_1_36_163_132"
SR06_FINISH = 13205
SR11 = "SR11_2_00_382_132"


def sha256(path) -> str | None:
    p = Path(path)
    return instances.sha256_file(p) if p.is_file() else None


# ------------------------------------------------------------------------------------- results

class Results:
    def __init__(self, fresh: bool):
        self.lock = threading.Lock()
        self.data = {"cases": {}}
        if RESULTS.is_file() and not fresh:
            self.data = json.loads(RESULTS.read_text(encoding="utf-8"))
        mods = instances.mod_hashes(COPIES)
        self.mod = mods[COPIES[0]]
        self.data["meta"] = {
            "updated": None, "gate_def": GATE_DEF, "mod_sha256": self.mod, "copies_mod_sha256": mods,
            "build_sha256": sha256(BUILD_MOD), "physics_ref": sha256(m0.REF.physics_dll),
            "physics_train": {c: sha256(instances.get_copy(c).physics_dll) for c in COPIES},
            "tassupport_sha256": sha256(m0.REF.mods_dir / "TASSupport.bmodp"), "m0_dir": M0_DIR.name,
        }

    def done(self, key) -> bool:
        c = self.data["cases"].get(key)
        return bool(c) and c.get("mod_sha256") == self.mod and c.get("gate_def") == GATE_DEF and "error" not in c

    def get(self, key):
        return self.data["cases"].get(key)

    def put(self, key, value):
        value["mod_sha256"] = self.mod
        value["gate_def"] = GATE_DEF
        with self.lock:
            self.data["cases"][key] = value
            self.save()

    def save(self):
        self.data["meta"]["updated"] = dt.datetime.now().isoformat(timespec="seconds")
        current = {k: c for k, c in self.data["cases"].items()
                   if c.get("mod_sha256") == self.mod and c.get("gate_def") == GATE_DEF}
        self.data["summary"] = summarize(current)
        self.data["summary"]["stale_cases"] = sorted(set(self.data["cases"]) - set(current))
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = RESULTS.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=_jd), encoding="utf-8")
        os.replace(tmp, RESULTS)


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def summarize(cases: dict) -> dict:
    """Per gate / group: pass, fail, info (not gated), error, and the failed case names."""
    out: dict = {}
    for key, c in sorted(cases.items()):
        parts = key.split("/")
        group = "/".join(parts[:2]) if parts[0] in ("A", "B", "D") else parts[0]
        s = out.setdefault(group, {"pass": 0, "fail": 0, "info": 0, "error": 0, "failed": []})
        name = "/".join(parts[1:])
        if "error" in c:
            s["error"] += 1
            s["failed"].append(f"{name}: error {c['error']}")
        elif c.get("gated") is False:
            s["info"] += 1
            if not c.get("pass", True):
                s["failed"].append(f"{name} (info): {c.get('verdict')}")
        elif c.get("pass"):
            s["pass"] += 1
        else:
            s["fail"] += 1
            s["failed"].append(f"{name}: {c.get('verdict', 'FAIL')}")
    return out


def run_pool(jobs, copies, R: Results):
    """jobs: list of (key, fn(copy) -> case). One worker thread per copy (one game process per copy)."""
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(copy):
        while True:
            try:
                key, fn = q.get_nowait()
            except queue.Empty:
                return
            t0 = time.monotonic()
            try:
                case = fn(copy)
            except Exception as e:  # recorded, the batch goes on
                case = {"error": f"{type(e).__name__}: {e}", "pass": False, "verdict": f"error: {e}"}
            case["copy"] = copy
            case["case_wall_s"] = round(time.monotonic() - t0, 2)
            R.put(key, case)
            print(f"{key} [{copy}] {case.get('verdict')}", flush=True)

    ths = [threading.Thread(target=worker, args=(c,), daemon=True) for c in copies]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


# ------------------------------------------------------------------------------------ M0 cases

def m0_plan() -> dict:
    """{key: (frames source, level, m0 bio dir, strict)} for the G1 records and G2 cases of M0."""
    d = json.loads((M0_DIR / "results.json").read_text(encoding="utf-8"))
    plan = {}
    for k, c in d["cases"].items():
        gate, _, name = k.partition("/")
        if gate == "G1":
            bio = M0_DIR / "G1" / name / "bio"
            plan[f"G1/{name}"] = (RECORDS / f"{name}.tas", c["level"], bio, c.get("verdict") == "PASS")
        elif gate == "G2":
            bio = M0_DIR / "G2" / name / "bio"
            plan[f"G2/{name}"] = (M0_DIR / "G2" / name / "input.bin", c["level"], bio, c.get("verdict") == "PASS")
    return plan


def agent_replay_case(src, level, bio: Path, out_group: Path, name: str, copy: str, strict: bool = True,
                      **kw) -> dict:
    """Replay policy through agent mode (ballrain.dataset.replay_record) and byte comparison with M0 bio."""
    meta = dataset.replay_record(src, out_group, copy=copy, level=level, flags=P.ACT_WANT_TRAC, save_raw=False,
                                 name=name, trace=True, **kw)
    run = Path(meta["run_dir"])
    case = {k: meta.get(k) for k in ("record", "level", "end_reason", "executed_frames", "observations",
                                     "observation_gaps", "steps", "start_s", "wall_s", "prefix_frames",
                                     "prefix_events", "prefix_observations", "framelog_equals_record_prefix",
                                     "python_framelog_equals_mod", "levelfinish_frame", "deaths", "sr_ms",
                                     "sr_double_final", "sr_f32sum_final", "sr_tail_frames", "hs_mod")}
    case["run_error"] = meta.get("error")
    case["strict"] = strict
    case["trace"] = str(run / "trace.txt")
    case["trace_sha256"] = sha256(run / "trace.txt")
    case["m0_trace_sha256"] = sha256(bio / "trace.txt")
    case["trace_equal"] = gates.identical_files(run / "trace.txt", bio / "trace.txt")
    case["framelog_equal"] = gates.identical_files(run / "framelog.bin", bio / "framelog.bin")
    if not case["trace_equal"] and (bio / "trace.txt").is_file() and (run / "trace.txt").is_file():
        case["cmp"] = gates.compare(bio / "trace.txt", run / "trace.txt")
    checks = {"trace": case["trace_equal"], "framelog": case["framelog_equal"], "no_error": meta.get("error") is None,
              "python_log_equals_mod_log": bool(meta.get("python_framelog_equals_mod")),
              "framelog_is_input_prefix": bool(meta.get("framelog_equals_record_prefix")),
              "no_observation_gaps": meta.get("observation_gaps") == 0}
    case["checks"] = checks
    case["pass"] = all(checks.values())
    case["verdict"] = "PASS" if case["pass"] else "FAIL: " + ", ".join(k for k, v in checks.items() if not v)
    if not strict:
        case["gated"] = False
    return case


def replay_mode_case(src, level, bio: Path, out_dir: Path, copy: str, strict: bool) -> dict:
    r = instances.run_ballrain(src, level, out_dir, copy=copy, timeout=400)
    res = r["result"]
    paths = r["loads"][0]
    case = {"wall_s": round(res.wall_s, 2), "exit_code": res.exit_code, "timed_out": res.timed_out,
            "missing": res.missing_outputs, "strict": strict, "trace": str(paths["trace"])}
    case["trace_equal"] = gates.identical_files(paths["trace"], bio / "trace.txt")
    case["framelog_equal"] = gates.identical_files(paths["framelog"], bio / "framelog.bin")
    ev_new, ev_old = gates.events_without_summary(paths["events"]), gates.events_without_summary(bio / "events.txt")
    case["events_equal"] = bool(ev_new) and ev_new == ev_old
    if not case["trace_equal"] and Path(paths["trace"]).is_file():
        case["cmp"] = gates.compare(bio / "trace.txt", paths["trace"])
    case["pass"] = bool(case["trace_equal"] and case["framelog_equal"] and case["events_equal"] and res.ok)
    case["verdict"] = "PASS" if case["pass"] else "FAIL: " + ", ".join(
        k for k in ("trace_equal", "framelog_equal", "events_equal") if not case[k]) + ("" if res.ok else " (run not ok)")
    if not strict:
        case["gated"] = False
    return case


def stall_pause(seed: int):
    """Python-side stalls: random 0-3 ms sleeps on every 50th action, 2 s during loading, 5 s mid-level."""
    rng = random.Random(seed)

    def pause(step: int):
        if step == 60:
            time.sleep(2.0)
        elif step == 5000:
            time.sleep(5.0)
        elif step % 50 == 0:
            time.sleep(rng.uniform(0, 0.003))
    return pause


def gate_a(R: Results, only, repeats: int = 10):
    plan = m0_plan()
    agent_out = OUT / "A" / "agent"
    sr06 = plan[f"G1/{SR06}"]
    jobs = []
    for key, (src, level, bio, strict) in plan.items():
        name = key.split("/", 1)[1]
        k = f"A/agent/{key}"
        if (only and name not in only) or (not only and R.done(k)):
            continue
        jobs.append((k, lambda c, src=src, level=level, bio=bio, key=key, strict=strict:
                     agent_replay_case(src, level, bio, agent_out / key.split("/")[0], key.split("/")[1], c, strict)))
    variants = {
        "SR06_k4": dict(frames_per_action=4),
        "SR06_prefix6000": dict(prefix_frames=6000),
        "SR06_prefixev3000": dict(prefix_frames=3000, prefix_events=True),
        "SR06_stalls": dict(pause=stall_pause(7)),
    }
    variants.update({f"SR06_repeat{i:02d}": {} for i in range(1, repeats + 1)})
    for vname, kw in variants.items():
        k = f"A/agent/variants/{vname}"
        if (only and vname not in only and SR06 not in only) or (not only and R.done(k)):
            continue
        jobs.append((k, lambda c, vname=vname, kw=kw:
                     agent_replay_case(sr06[0], 6, sr06[2], agent_out / "variants", vname, c, True, **kw)))
    print(f"M1-A agent: {len(jobs)} runs, serial on {COPIES[0]}", flush=True)
    run_pool(jobs, COPIES[:1], R)
    # replay mode on the new build, all M0 cases, in parallel
    jobs = []
    for key, (src, level, bio, strict) in plan.items():
        name = key.split("/", 1)[1]
        k = f"A/replay/{key}"
        if (only and name not in only) or (not only and R.done(k)):
            continue
        jobs.append((k, lambda c, src=src, level=level, bio=bio, key=key, strict=strict:
                     replay_mode_case(src, level, bio, OUT / "A" / "replay" / key, c, strict)))
    print(f"M1-A replay: {len(jobs)} runs on {len(COPIES)} copies", flush=True)
    run_pool(jobs, COPIES, R)
    # SR06 x10: all repeats identical (and identical to M0)
    reps = [R.get(f"A/agent/variants/SR06_repeat{i:02d}") or {} for i in range(1, repeats + 1)]
    if not all(reps):
        return
    shas = [c.get("trace_sha256") for c in reps]
    c = Counter(shas)
    R.put("A/repeat/SR06_x10", {"trace_sha256": shas, "distinct": len(c), "m0_sha256": sha256(sr06[2] / "trace.txt"),
                                "pass": len(c) == 1 and shas[0] == sha256(sr06[2] / "trace.txt"),
                                "verdict": "PASS" if len(c) == 1 and shas[0] == sha256(sr06[2] / "trace.txt")
                                else f"FAIL: {len(c)} distinct traces"})


# ------------------------------------------------------------------------------------------ B

B_PLAN = [
    # name, copy, level, seed, prefix (record, cut before LevelFinish), max_process_frames, hold (steps)
    ("L01_random_a", "train-00", 1, 101, None, 60000, (2, 40)),
    ("L01_random_b", "train-01", 1, 102, None, 60000, (1, 10)),
    ("L02_random", "train-02", 2, 103, None, 60000, (2, 40)),
    ("L03_random", "train-03", 3, 104, None, 60000, (2, 40)),
    ("L06_prefix_finish", "train-04", 6, 105, (SR06, SR06_FINISH - 40), 60000, (1, 8)),
    ("L11_prefix_finish", "train-05", 11, 106, (SR11, None), 60000, (1, 8)),
    ("L06_random", "train-06", 6, 107, None, 60000, (2, 40)),
    ("L09_random_limit", "train-07", 9, 108, None, 12000, (1, 6)),
    # second wave (hold None = null policy: keys 0 until the frame limit ends the process)
    ("L02_null_limit", "train-00", 2, 109, None, 4000, None),
    ("L01_jitter_long", "train-01", 1, 110, None, 40000, (1, 1)),
    ("L04_jitter_limit", "train-02", 4, 111, None, 8000, (1, 2)),
    # HS mode: the Esc action is in the action space (random pauses / unpauses, the pause cap), and the
    # collect macro right after a TAS prefix that has just touched a score ball
    ("L01_hs_esc_random", "train-03", 1, 112, None, 30000, (1, 6), {"mode": "hs"}),
    ("L01_hs_collect", "train-04", 1, 113, ("HS01_4738_1_31_317", 6600), 12000, (1, 6),
     {"mode": "hs", "collect_action": True, "first_actions": ["collect"]}),
]


def finish_frame(record: str) -> int:
    """LevelFinish frame of a record from its M0 bio events (event number = frames done)."""
    for idx, name in gates.read_events(M0_DIR / "G1" / record / "bio" / "events.txt"):
        if name == "LevelFinish":
            return idx
    raise ValueError(f"no LevelFinish in the M0 events of {record}")


def random_episodes(name, copy, level, seed, prefix, max_frames, hold, opts=None, launcher=None, out=None) -> dict:
    """One game process: random sticky actions (action_repeat 4) over every life until the process ends.
    opts: mode ("sr" / "hs"), collect_action, first_actions (action names played first)."""
    from ballrain.env import BallanceEnv, EnvConfig
    opts = dict(opts or {})
    mode = opts.pop("mode", "sr")
    first = list(opts.pop("first_actions", []))
    out_root = Path(out or OUT / "B" / "episodes") / name
    if out_root.exists():
        shutil.rmtree(out_root)
    pf = None
    if prefix:
        rec, cut = prefix
        cut = cut if cut is not None else finish_frame(rec) - 30
        pf = OUT / "B" / "prefix" / f"{rec}_{cut}.bin"
        pf.parent.mkdir(parents=True, exist_ok=True)
        tas.save_raw(pf, tas.load(RECORDS / f"{rec}.tas")[:cut])
    env = BallanceEnv(EnvConfig(level=level, mode=mode, copy=copy, out_root=str(out_root), action_repeat=4,
                                max_process_frames=max_frames, trace=True, prefix=str(pf) if pf else None,
                                seed=seed, **opts), launcher=launcher)
    infos = []
    rng = np.random.default_rng(seed)
    lives = []
    steps = 0
    t0 = time.monotonic()
    try:
        obs, info = env.reset(seed=seed)
        a, left = 0, 0
        while True:
            if first:
                a, left = env.actions.index(first.pop(0)), 1
            elif hold is None:
                a, left = 0, 1
            elif left <= 0:
                a, left = int(rng.integers(env.action_space.n)), int(rng.integers(hold[0], hold[1] + 1))
            obs, r, term, trunc, info = env.step(a)
            for k in ("esc", "collect", "forced_unpause"):
                if k in info:
                    infos.append({"step": steps, k: info[k], "frames_done": info["frames_done"]})
            steps += 1
            left -= 1
            if term or trunc:
                lives.append(info.get("episode_end"))
                if env.session is None:
                    break
                obs, info = env.reset(options={"relaunch": False})
                if obs is None:
                    break
    finally:
        env.close()
    wall = time.monotonic() - t0
    metas = [json.loads(Path(p).read_text(encoding="utf-8")) for p in env.finished_runs]
    if len(metas) != 1:
        raise RuntimeError(f"{name}: expected one process run, got {len(metas)}")
    m = metas[0]
    run_dir = out_root / copy / m["run_id"]
    return {"name": name, "level": level, "seed": seed, "prefix": str(pf.name) if pf else None, "steps": steps,
            "wall_s": round(wall, 2), "life_ends": lives, "run_dir": str(run_dir), "meta": m,
            "mode": mode, "esc_infos": infos[:200], "esc_counts": dict(Counter(
                f"{k}:{v if isinstance(v, str) else v.get('result')}" for i in infos for k, v in i.items()
                if k in ("esc", "collect", "forced_unpause")))}


def gate_b(R: Results, only):
    plan = [p for p in B_PLAN if not only or p[0] in only]
    jobs = []
    for p in plan:
        k = f"B/run/{p[0]}"
        if not only and R.done(k):
            continue

        def fn(copy, p=p):
            if copy != p[1]:
                raise RuntimeError("copy mismatch")
            case = random_episodes(*p)
            case["pass"] = True  # judged in the reference phase below
            case["verdict"] = "ran"
            return case
        jobs.append((k, fn, p[1]))
    # one thread per planned copy (each job is bound to its copy)
    by_copy: dict = {}
    for k, fn, copy in jobs:
        by_copy.setdefault(copy, []).append((k, fn))
    ths = [threading.Thread(target=run_pool, args=(js, [c], R)) for c, js in by_copy.items()]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    # reference replays (serial: one reference copy)
    rec_dir = OUT / "B" / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    coverage = Counter()
    n_runs = 0
    for p in plan:
        run = R.get(f"B/run/{p[0]}")
        if not run or "error" in run:
            continue
        n_runs += 1
        m = run["meta"]
        names = [e[1] for e in m["events"]]
        coverage["level1"] += p[2] == 1
        coverage["BallOff"] += names.count("BallOff")
        coverage["GameOver"] += "GameOver" in names
        coverage["LevelFinish"] += "LevelFinish" in names
        coverage["frame_limit"] += m["end_reason"] == "frame_limit"
        coverage["lives"] += len(m["lives"])
        coverage["pauses"] += names.count("PauseLevel")
        k = f"B/export/{p[0]}"
        if not only and R.done(k):
            continue
        case = {"name": p[0], "level": p[2], "end_reason": m["end_reason"], "frames": m["frames"],
                "lives": len(m["lives"]), "deaths": names.count("BallOff"), "framelog_match": m["framelog_match"],
                "framelog_source": m["framelog_source"], "sr_ms": m.get("sr_ms"), "hs": m.get("hs"),
                "pauses": names.count("PauseLevel"), "unpauses": names.count("UnpauseLevel"),
                "esc_frames": int(np.count_nonzero(tas.load_raw(Path(run["run_dir"]) / "framelog.bin")["keys"]
                                                   & tas.KEY_ESC))}
        try:
            run_dir = Path(run["run_dir"])
            tas_path = run_dir / m["tas_path"]
            fl = tas.load_raw(run_dir / "framelog.bin")
            case["tas_roundtrip"] = tas.frames_to_bytes(tas.load(tas_path)) == tas.frames_to_bytes(fl)
            h = hashlib.sha256(tas_path.read_bytes()).hexdigest()[:16]
            rec = rec_dir / f"m1_{h}.tas"
            if not rec.is_file():
                shutil.copyfile(tas_path, rec)
            case["exported_tas"] = str(rec)
            agent_trace = run_dir / "trace.txt"
            case["agent_trace_sha256"] = sha256(agent_trace)
            d = OUT / "B" / "ref" / p[0]
            d.mkdir(parents=True, exist_ok=True)
            refs = []
            for i in range(1, 4):
                refs.append(m0.run_ref(rec, p[2], d / f"ref{i}.txt"))
            case["ref"] = refs
            shas = [r["sha256"] for r in refs]
            case["ref_sha256"] = shas
            ok, why = gates.majority_verdict(shas, case["agent_trace_sha256"])
            case["majority"] = why
            maj = Counter(s for s in shas if s).most_common(1)
            if maj:
                j = shas.index(maj[0][0])
                case["cmp_majority_vs_agent"] = gates.compare(d / f"ref{j + 1}.txt", agent_trace)
            case["pass"] = bool(ok and m["framelog_match"] is True and case["tas_roundtrip"])
            case["verdict"] = why if case["pass"] else f"FAIL: {why}, framelog_match={m['framelog_match']}, " \
                                                       f"roundtrip={case['tas_roundtrip']}"
        except Exception as e:
            case["error"] = f"{type(e).__name__}: {e}"
            case["pass"] = False
            case["verdict"] = f"error: {e}"
        R.put(k, case)
        print(k, case["verdict"], flush=True)
    need = {"level1": 1, "BallOff": 1, "GameOver": 1, "LevelFinish": 1}
    need["frame_limit"] = 1
    if any(len(p) > 7 and p[7].get("mode") == "hs" for p in plan):
        need["pauses"] = 1
    cov_ok = n_runs >= 8 and all(coverage[k] >= v for k, v in need.items())
    if not only:
        R.put("B/coverage", {"runs": n_runs, **dict(coverage), "pass": cov_ok,
                             "verdict": "PASS" if cov_ok else f"FAIL: coverage {dict(coverage)}, runs {n_runs}"})


def cleanup_m1_installs() -> list[str]:
    removed = []
    for dst in sorted(RECORDS.glob("m1_*.tas")):
        if gates.remove_installed(OUT / "B" / "records" / dst.name, dst):
            removed.append(dst.name)
    return removed


# ------------------------------------------------------------------------------------------ D

def gate_d(R: Results, only):
    recs = sorted(p for p in RECORDS.glob("*.tas") if RECORD_NAME.match(p.stem))
    recs += sorted(PATCHED.glob("*.tas")) if PATCHED.is_dir() else []
    if only:
        recs = [r for r in recs if r.stem in only]
    todo = [r for r in recs if only or not R.done(f"D/record/{r.stem}")]
    t0 = time.monotonic()
    metas = dataset.build_dataset(todo, DATASET_OUT, COPIES, frames_per_action=1) if todo else []
    wall = time.monotonic() - t0
    plan = m0_plan()
    for m in metas:
        stem = Path(m["record"]).stem
        case = dict(m)
        case["run_error"] = case.pop("error", None)  # "error" marks a driver error in results.json
        bio = plan.get(f"G1/{stem}")
        out = DATASET_OUT / stem
        case["dataset_dir"] = str(out)
        case["bytes"] = sum(f.stat().st_size for f in out.glob("*") if f.is_file()) if out.is_dir() else 0
        if bio:
            case["m0_trace_equal"] = gates.identical_files(out / "trace.txt", bio[2] / "trace.txt")
            case["m0_framelog_equal"] = gates.identical_files(out / "run" / "framelog.bin", bio[2] / "framelog.bin")
        completing = m.get("end_reason") == "LevelFinish"
        case["completing"] = completing
        fc = m.get("followers_check") or {}
        case["followers_counter_equal"] = fc.get("mismatch", 0) == 0 and fc.get("checked", 0) > 0
        case["pass"] = (bool(m.get("ok")) and case["followers_counter_equal"]
                        and (not bio or (case["m0_trace_equal"] and case["m0_framelog_equal"])))
        case["gated"] = completing  # non-completing records (GameOver, no level) are reported, not gated
        case["verdict"] = ("PASS" if case["pass"] else "FAIL") + ("" if completing else f" (not completing: "
                                                                  f"{m.get('end_reason') or m.get('error')})")
        if bio and not bio[3]:
            case["m0_strict"] = False  # M0 excluded / failed record: an M0 mismatch is reported, not gated
            case["gated"] = False
        R.put(f"D/record/{stem}", case)
    if metas:
        R.put("D/build", {"records": len(metas), "wall_s": round(wall, 1), "pass": True, "gated": False,
                          "verdict": "built", "out": str(DATASET_OUT)})
    for d in DATASET_OUT.glob("*/run"):  # the run directories duplicate the copied outputs
        for f in ("trace.txt", "events.txt", "level.msg", "level.msg.inc"):
            if (d / f).is_file() and (d.parent / f).is_file():
                (d / f).unlink()


# ------------------------------------------------------------------------------------------ C

def _name_time_ms(stem: str) -> int | None:
    m = re.search(r"_(\d)_(\d\d)_(\d\d\d)(?:_\d+)?$", stem)
    return (int(m.group(1)) * 60000 + int(m.group(2)) * 1000 + int(m.group(3))) if m else None


def _name_score(stem: str) -> int | None:
    m = re.match(r"HS\d\d_(\d+)_", stem)
    return int(m.group(1)) if m else None


def gate_c(R: Results, write_records: bool):
    rows = json.loads(RECORDS_JSON.read_text(encoding="utf-8"))
    by_src = {r["source"]: r for r in rows if r.get("kind") == "tas"}
    table = []
    for key, c in sorted(R.data["cases"].items()):
        if not key.startswith("D/record/"):
            continue
        stem = key.split("/")[-1]
        row = by_src.get(f"{stem}.tas")
        fin = c.get("levelfinish_frame") is not None and c.get("end_reason") == "LevelFinish"
        sr_d = c.get("sr_double_final")
        f32 = c.get("sr_f32sum_final")
        t = {"record": stem, "completing": fin, "end_reason": c.get("end_reason"),
             "sr_ms_measured": int(math.trunc(sr_d)) if fin and sr_d is not None else None,
             "sr_double": sr_d if fin else None,
             "sr_f32sum_ms": int(math.trunc(f32)) if fin and f32 is not None else None,
             "sr_bml_float": c.get("sr_bml_final") if fin else None,
             "tail_frames": c.get("sr_tail_frames"), "counter_inactive_seen": c.get("sr_counter_inactive_seen"),
             "name_time_ms": _name_time_ms(stem),
             "sr_ms_bmlp0312_estimate": row.get("sr_ms_bmlp0312") if row else None,
             "sr_ms_float32_estimate": row.get("sr_ms_float32") if row else None,
             "hs_measured": c.get("hs") if fin else None, "hs_source": c.get("hs_source"),
             "hs_followers": c.get("hs_followers"), "hs_followers_at_hook": c.get("hs_followers_at_hook"),
             "hs_levelfinish_hook": c.get("hs_mod") if fin else None, "hs_bml": c.get("hs_bml") if fin else None,
             "hs_points_end": c.get("hs_points_end"), "hs_lives_end": c.get("hs_lives_end"),
             "hs_level_bonus": c.get("hs_level_bonus"), "name_score": _name_score(stem),
             "stale": c.get("mod_sha256") != R.mod}
        if t["sr_ms_measured"] is not None:
            t["d_vs_estimate"] = None if t["sr_ms_bmlp0312_estimate"] is None else t["sr_ms_measured"] - t["sr_ms_bmlp0312_estimate"]
            t["d_f32_vs_name"] = None if t["name_time_ms"] is None else t["sr_f32sum_ms"] - t["name_time_ms"]
            t["d_vs_name"] = None if t["name_time_ms"] is None else t["sr_ms_measured"] - t["name_time_ms"]
        if t["hs_measured"] is not None and t["name_score"] is not None:
            t["hs_matches_name"] = t["hs_measured"] == t["name_score"]
            t["hs_gated"] = stem not in HS_NOT_GATED
            if stem in HS_NOT_GATED:
                t["hs_note"] = HS_NOT_GATED[stem]
        table.append(t)
    sr_rows = [t for t in table if t["record"].startswith("SR") and t["completing"]]
    hs_rows = [t for t in table if t["record"].startswith("HS") and t["completing"]]
    summary = {
        "records": len(table), "completing": sum(t["completing"] for t in table),
        "sr_estimate_equal": sum(t.get("d_vs_estimate") == 0 for t in sr_rows),
        "sr_estimate_differs": [(t["record"], t.get("d_vs_estimate")) for t in sr_rows if t.get("d_vs_estimate")],
        "sr_f32_equals_name": sum(t.get("d_f32_vs_name") == 0 for t in sr_rows),
        "sr_f32_vs_name_differs": [(t["record"], t.get("d_f32_vs_name")) for t in sr_rows if t.get("d_f32_vs_name")],
        "hs_equals_name": sum(bool(t.get("hs_matches_name")) for t in hs_rows),
        "hs_differs": [(t["record"], t["hs_measured"], t["name_score"]) for t in hs_rows if not t.get("hs_matches_name")],
        "hs_gated_records": [t["record"] for t in hs_rows if t.get("hs_gated")],
        "hs_gated_fail": [(t["record"], t["hs_measured"], t["name_score"]) for t in hs_rows
                          if t.get("hs_gated") and not t.get("hs_matches_name")],
        "stale_rows": [t["record"] for t in table if t["stale"]],  # D cases of an older build: reported only
        "hs_gated_stale": [t["record"] for t in hs_rows if t.get("hs_gated") and t["stale"]],
        "tail_frames": dict(Counter(t["tail_frames"] for t in table if t["completing"])),
        "all_counter_inactive_seen": all(t["counter_inactive_seen"] for t in table if t["completing"]),
    }
    written = []
    if write_records:
        for t in table:
            row = by_src.get(f"{t['record']}.tas")
            # the two fields are defined for SR-TAS rows only (docs/benchmarks.md); HS rows keep their
            # measured SR readings in results.json
            if row is None or row.get("category") != "SR" or not t["completing"] or t["sr_ms_measured"] is None:
                continue
            changed = {}
            if row.get("sr_ms_bmlp0312") != t["sr_ms_measured"]:
                changed["sr_ms_bmlp0312"] = (row.get("sr_ms_bmlp0312"), t["sr_ms_measured"])
            if row.get("sr_ms_float32") != t["sr_f32sum_ms"]:
                changed["sr_ms_float32"] = (row.get("sr_ms_float32"), t["sr_f32sum_ms"])
            row["sr_ms_bmlp0312"] = t["sr_ms_measured"]
            row["sr_ms_float32"] = t["sr_f32sum_ms"]
            written.append({"record": t["record"], "changed": changed})
        RECORDS_JSON.write_bytes((json.dumps(rows, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))  # LF kept
    hs_ok = bool(summary["hs_gated_records"]) and not summary["hs_gated_fail"] and not summary["hs_gated_stale"]
    R.put("C/readout", {"table": table, "summary": summary, "records_json_written": written, "pass": hs_ok,
                        "verdict": "PASS (HS = name score on every gated HS record)" if hs_ok else
                        f"FAIL: HS {summary['hs_gated_fail']}, stale {summary['hs_gated_stale']}"})
    print(json.dumps(summary, indent=1, ensure_ascii=False), flush=True)


# ------------------------------------------------------------------------------------------ E

class _FT(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


def _systimes():
    idle, kern, user = _FT(), _FT(), _FT()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    f = lambda t: (t.hi << 32) | t.lo  # noqa: E731
    return f(idle), f(kern), f(user)


def e_worker(spec: dict) -> dict:
    """One instance (runs in its own OS process): policy for `seconds`, then close and report."""
    from ballrain.client import AgentSession, SessionConfig
    from ballrain.env import ARROWS9, BallanceEnv, EnvConfig
    copy, level, seconds, policy, seed = spec["copy"], spec["level"], spec["seconds"], spec["policy"], spec["seed"]
    out_root = Path(spec["out"])
    rng = np.random.default_rng(seed)
    st = {"copy": copy, "policy": policy, "steps": 0, "frames": 0, "decision_frames": 0, "lives": 0,
          "processes": 0, "start_s": [], "start_to_control_s": [], "errors": []}
    t_end = time.monotonic() + seconds
    t0 = time.monotonic()
    if spec.get("raw"):
        # raw client: AgentSession, random sticky keys, action repeat 4, no features / rewards
        while time.monotonic() < t_end:
            ts = time.monotonic()
            s = AgentSession(SessionConfig(level=level, out_dir=out_root / f"raw{st['processes']}", copy=copy,
                                           max_frames=spec["max_frames"], events_txt=False))
            try:
                s.start()
                st["start_s"].append(time.monotonic() - ts)
                s.run_until(0, ["BallNavActive"])
                st["start_to_control_s"].append(time.monotonic() - ts)
                st["processes"] += 1
                f_ctrl = s.frames_done
                a, left = 0, 0
                while not s.ended and time.monotonic() < t_end:
                    if left <= 0:
                        a, left = int(rng.integers(9)), int(rng.integers(2, 40))
                    o = s.act(tas.make_frames([np.float32(1000 / 132)], [ARROWS9[a]]), repeat=4,
                              stop=["BallOff", "LevelFinish", "GameOver"])
                    st["steps"] += 1
                    left -= 1
                    if o is not None and o.has_event("BallOff"):
                        st["lives"] += 1
                        s.run_until(0, ["BallNavActive", "GameOver", "LevelFinish"])
                st["decision_frames"] += s.frames_done - f_ctrl
                st["frames"] += s.frames_done
                s.stop()
            except Exception as e:
                st["errors"].append(f"{type(e).__name__}: {e}")
            finally:
                s.close()
    else:
        env = BallanceEnv(EnvConfig(level=level, copy=copy, out_root=str(out_root), action_repeat=4,
                                    max_process_frames=spec["max_frames"]))
        prev_frames = 0
        try:
            while time.monotonic() < t_end:
                ts = time.monotonic()
                had = env.session
                obs, info = env.reset()
                if env.session is not had:
                    st["processes"] += 1
                    st["start_to_control_s"].append(time.monotonic() - ts)
                    prev_frames = 0
                f_ctrl = env.session.frames_done
                st["frames"] += f_ctrl - prev_frames
                a, left = 0, 0
                while time.monotonic() < t_end:
                    if policy == "random" and left <= 0:
                        a, left = int(rng.integers(env.action_space.n)), int(rng.integers(2, 40))
                    n0 = env.session.frames_done
                    obs, r, term, trunc, info = env.step(a if policy == "random" else 0)
                    st["steps"] += 1
                    left -= 1
                    sess = env.session
                    n1 = info["frames_done"]
                    st["frames"] += n1 - n0
                    st["decision_frames"] += n1 - n0
                    prev_frames = n1
                    if term or trunc:
                        st["lives" if term else "truncations"] = st.get("lives" if term else "truncations", 0) + 1
                        break
        except Exception as e:
            st["errors"].append(f"{type(e).__name__}: {e}")
        finally:
            env.close()
    st["wall_s"] = time.monotonic() - t0
    return st


def gate_e(R: Results, seconds: int):
    py = sys.executable
    configs = [("null", 1, False), ("random", 1, False), ("raw", 1, True), ("null", 4, False), ("random", 4, False),
               ("null", 8, False), ("random", 8, False), ("raw", 8, True)]
    for policy, n, raw in configs:
        key = f"E/{policy}_x{n}"
        if R.done(key):
            continue
        out = OUT / "E" / f"{policy}_x{n}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        procs = []
        for i, copy in enumerate(COPIES[:n]):
            spec = {"copy": copy, "level": 2, "seconds": seconds, "policy": "random" if raw else policy, "raw": raw,
                    "seed": 1000 + i, "max_frames": 30000, "out": str(out / copy)}
            f = out / f"{copy}.json"
            procs.append((subprocess.Popen([py, str(Path(__file__).resolve()), "--e-worker", json.dumps(spec), str(f)],
                                           cwd=str(REPO)), f))
        samples = []
        t_prev = _systimes()
        while any(p.poll() is None for p, _ in procs):
            time.sleep(2.0)
            t_now = _systimes()
            di, dk, du = (t_now[0] - t_prev[0]), (t_now[1] - t_prev[1]), (t_now[2] - t_prev[2])
            if dk + du:
                samples.append(1 - di / (dk + du))
            t_prev = t_now
        ws = [json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {"errors": ["no output"]} for _, f in procs]
        tot = lambda k: sum(w.get(k, 0) for w in ws)  # noqa: E731
        wall = max(w.get("wall_s", 0) for w in ws) or 1
        starts = [x for w in ws for x in w.get("start_to_control_s", [])]
        case = {"policy": policy, "instances": n, "seconds": seconds, "level": 2, "action_repeat": 4,
                "steps_per_s": round(tot("steps") / wall, 1), "frames_per_s": round(tot("frames") / wall, 1),
                "decision_frames_per_s": round(tot("decision_frames") / wall, 1),
                "game_s_per_wall_s": round(tot("frames") / wall * 1000 / 132 / 1000, 1),
                "lives_per_hour": round(tot("lives") / wall * 3600, 1), "terminated_lives": tot("lives"),
                "truncations": tot("truncations"), "processes": tot("processes"),
                "start_to_control_s_mean": round(float(np.mean(starts)), 2) if starts else None,
                "start_to_control_s_max": round(float(np.max(starts)), 2) if starts else None,
                "cpu_load_mean": round(float(np.mean(samples)), 3) if samples else None,
                "cpu_load_max": round(float(np.max(samples)), 3) if samples else None,
                "logical_cpus": os.cpu_count(), "workers": ws,
                "errors": [e for w in ws for e in w.get("errors", [])]}
        case["pass"] = not case["errors"]
        case["gated"] = False
        case["verdict"] = "measured" if case["pass"] else f"errors: {case['errors'][:3]}"
        R.put(key, case)
        print(key, {k: case[k] for k in ("steps_per_s", "frames_per_s", "lives_per_hour", "processes",
                                         "start_to_control_s_mean", "cpu_load_mean")}, flush=True)


# ---------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    global M0_DIR
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--e-worker":
        res = e_worker(json.loads(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(res, default=_jd), encoding="utf-8")
        return 0
    p = argparse.ArgumentParser()
    p.add_argument("--gates", default="A,B,D,C,E")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--only", nargs="*")
    p.add_argument("--m0", default=M0_DIR.name, help="M0 results directory name under the runtime directory")
    p.add_argument("--write-records", action="store_true", help="C: update benchmarks/records.json")
    p.add_argument("--e-seconds", type=int, default=60)
    a = p.parse_args(argv)
    M0_DIR = RUNTIME / a.m0
    mods = instances.mod_hashes(COPIES)
    if len(set(mods.values())) != 1 or mods[COPIES[0]] != sha256(BUILD_MOD):
        print(f"copies do not all hold the build {BUILD_MOD}: {mods}; run scripts/make_copies.py --sync-mod")
        return 1
    removed = cleanup_m1_installs() + m0.cleanup_stale_installs()
    if removed:
        print(f"removed stale installed records: {removed}")
    R = Results(a.fresh)
    R.save()
    only = set(a.only or [])
    sel = [g.strip().upper() for g in a.gates.split(",")]
    try:
        if "A" in sel:
            gate_a(R, only)
        if "B" in sel:
            gate_b(R, only)
        if "D" in sel:
            gate_d(R, only)
        if "C" in sel:
            gate_c(R, a.write_records)
        if "E" in sel:
            gate_e(R, a.e_seconds)
    finally:
        cleanup_m1_installs()
    print(json.dumps({g: {k: v for k, v in s.items() if k != "failed"} for g, s in R.data["summary"].items()
                      if isinstance(s, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

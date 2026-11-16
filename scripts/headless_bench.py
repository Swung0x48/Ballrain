"""Throughput of the headless backend (docs/headless.md), run where the engine runs (the Mac): N independent OS
processes, each stepping its own BallanceEnv(backend="headless") for a fixed wall time, L1, SR mode, f5 features,
action repeat 4 at 132 fps - the same worker loop and metrics as scripts/m2_start_states.py throughput (retail, PC).

    uv run python scripts/headless_bench.py --configs l1_null l1_random l1_curr_random --instances 1 6 10 12 \
        --seconds 60 --out runs/bench.json

Policies: null (always no key: lives do not end, up to the 60000-frame process cap), random (sticky random actions,
2-39 steps each), curr_random (random from curriculum TAS starts of SR01_1_14_457_264, last 3000 frames; needs the
tas-v1 record under $BALLRAIN_RUNTIME/datasets). Per instance it also samples the engine process's RSS.
Engine: BALLRAIN_HL_BIN / BALLRAIN_HL_ROOT (ballrain.headless.HeadlessConfig).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
RECORD = "SR01_1_14_457_264"
CONFIGS = {
    "l1_null": ("null", {}),
    "l1_random": ("random", {}),
    "l1_curr_random": ("random", {"start_record": RECORD, "curriculum": {"start_back": 3000}}),
    "l1_curr_null": ("null", {"start_record": RECORD, "curriculum": {"start_back": 3000}}),
}


def _rss_kb(pid: int) -> int | None:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return int(out.strip()) if out.strip() else None
    except Exception:  # noqa: BLE001
        return None


def worker(spec: dict) -> dict:
    import resource

    from ballrain.env import BallanceEnv, EnvConfig
    rng = np.random.default_rng(spec["seed"])
    cfg = EnvConfig(level=1, mode="sr", copy=spec["copy"], out_root=spec["out"], action_repeat=4, backend="headless",
                    headless=spec.get("headless") or {}, max_process_frames=60000, **spec.get("env", {}))
    env = BallanceEnv(cfg)
    fx_t = [0.0]
    orig = env.fx.compute

    def timed(tracker):
        t = time.perf_counter()
        try:
            return orig(tracker)
        finally:
            fx_t[0] += time.perf_counter() - t
    env.fx.compute = timed
    st = {"copy": spec["copy"], "steps": 0, "frames": 0, "prefix_frames": 0, "decision_frames": 0, "lives": 0,
          "finishes": 0, "processes": 0, "truncations": 0, "t_reset_new": 0.0, "t_reset_respawn": 0.0, "t_step": 0.0,
          "t_launch": 0.0, "t_handover": 0.0, "t_intro": 0.0, "errors": [], "ks": [], "engine_rss_kb": [],
          "launch_s": []}
    t_end = time.monotonic() + spec["seconds"]
    t0 = time.monotonic()
    next_rss = 0.0
    try:
        while time.monotonic() < t_end:
            had = env.session
            f_before = had.frames_done if had is not None else None
            ts = time.perf_counter()
            fx0 = fx_t[0]
            obs, info = env.reset()
            dt = time.perf_counter() - ts
            fx_t[0] = fx0
            s0 = info["start"]
            if s0.get("new_process"):
                st["processes"] += 1
                st["t_reset_new"] += dt
                tm = s0["timing"]
                st["ks"].append(tm["k"])
                st["t_launch"] += tm["launch_s"]
                st["launch_s"].append(round(tm["launch_s"], 3))
                st["t_handover"] += tm["handover_s"] + tm["handover_python_s"]
                st["t_intro"] += tm["intro_s"]
                st["frames"] += env.session.frames_done
                st["prefix_frames"] += tm.get("handover_frames", 0)
            else:
                st["t_reset_respawn"] += dt
                st["frames"] += env.session.frames_done - (f_before or 0)
            a, left = 0, 0
            while time.monotonic() < t_end:
                if spec["policy"] == "random" and left <= 0:
                    a, left = int(rng.integers(env.action_space.n)), int(rng.integers(2, 40))
                n0 = env.session.frames_done
                ts = time.perf_counter()
                obs, r, term, trunc, info = env.step(a if spec["policy"] == "random" else 0)
                st["t_step"] += time.perf_counter() - ts
                st["steps"] += 1
                left -= 1
                n1 = info["frames_done"]
                st["frames"] += n1 - n0
                st["decision_frames"] += n1 - n0
                if time.monotonic() > next_rss and env.session is not None and env.session.handle is not None:
                    rss = _rss_kb(env.session.handle.pid)
                    if rss:
                        st["engine_rss_kb"].append(rss)
                    next_rss = time.monotonic() + 5.0
                if term or trunc:
                    if term:
                        st["lives"] += 1
                        st["finishes"] += info.get("episode_end") == "LevelFinish"
                    else:
                        st["truncations"] += 1
                    break
    except Exception as e:  # noqa: BLE001
        st["errors"].append(f"{type(e).__name__}: {e}")
    finally:
        tc = time.perf_counter()
        env.close()
        st["t_close"] = time.perf_counter() - tc
    st["t_features"] = fx_t[0]
    st["wall_s"] = time.monotonic() - t0
    st["python_maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
    return st


def run(name: str, n: int, seconds: int, out: Path, headless: dict) -> dict:
    policy, opts = CONFIGS[name]
    d = out / f"{name}_x{n}"
    d.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(n):
        env = dict(opts)
        if "curriculum" in env:
            env["curriculum"] = dict(env["curriculum"], seed=1000 + i)
        spec = {"copy": f"hl-{i:02d}", "seconds": seconds, "policy": policy, "seed": 1000 + i, "out": str(d / "episodes"),
                "env": env, "headless": headless}
        f = d / f"w{i:02d}.json"
        procs.append((subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec), str(f)],
                                       cwd=str(REPO)), f))
    load = []
    while any(p.poll() is None for p, _ in procs):
        time.sleep(2.0)
        load.append(os.getloadavg()[0])
    ws = [json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {"errors": ["no output"]} for _, f in procs]
    tot = lambda k: sum(w.get(k, 0) for w in ws)  # noqa: E731
    wall = max(w.get("wall_s", 0) for w in ws) or 1
    busy = tot("wall_s") or 1
    rss = [x for w in ws for x in w.get("engine_rss_kb", [])]
    launches = [x for w in ws for x in w.get("launch_s", [])]
    case = {"config": name, "policy": policy, "instances": n, "seconds": seconds,
            "steps_per_s": round(tot("steps") / wall, 1), "frames_per_s": round(tot("frames") / wall, 1),
            "decision_frames_per_s": round(tot("decision_frames") / wall, 1),
            "feature_ms_per_step": round(1000 * tot("t_features") / max(tot("steps"), 1), 3),
            "step_ms": round(1000 * tot("t_step") / max(tot("steps"), 1), 3),
            "lives_per_hour": round(tot("lives") / wall * 3600, 1), "lives": tot("lives"), "finishes": tot("finishes"),
            "processes": tot("processes"),
            "time_split": {k: round(tot(k) / busy, 3) for k in ("t_launch", "t_handover", "t_intro", "t_reset_respawn",
                                                                 "t_step", "t_features", "t_close", "t_reset_new")},
            "engine_rss_mb_median": round(float(np.median(rss)) / 1024, 1) if rss else None,
            "engine_rss_mb_max": round(max(rss) / 1024, 1) if rss else None,
            "python_rss_mb_max": round(max(w.get("python_maxrss_mb", 0) for w in ws), 1),
            "launch_s_median": round(float(np.median(launches)), 3) if launches else None,
            "loadavg_mean": round(float(np.mean(load)), 2) if load else None,
            "errors": [e for w in ws for e in w.get("errors", [])][:10]}
    case["time_split"]["t_step_game_ipc"] = round((tot("t_step") - tot("t_features")) / busy, 3)
    return case


def main(argv=None) -> int:
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--worker":
        res = worker(json.loads(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(res), encoding="utf-8")
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["l1_null", "l1_random", "l1_curr_random"])
    ap.add_argument("--instances", nargs="+", type=int, default=[1, 6, 10, 12])
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("runs/headless_bench"))
    ap.add_argument("--binary", default=os.environ.get("BALLRAIN_HL_BIN", ""))
    ap.add_argument("--root", default=os.environ.get("BALLRAIN_HL_ROOT", ""))
    a = ap.parse_args(argv)
    headless = {"binary": a.binary, "root": a.root}
    res = {}
    out_json = a.out / "results.json"
    if out_json.exists():
        res = json.loads(out_json.read_text(encoding="utf-8"))
    for name in a.configs:
        for n in a.instances:
            case = run(name, n, a.seconds, a.out, headless)
            res[f"{name}_x{n}"] = case
            print(json.dumps(case), flush=True)
            out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

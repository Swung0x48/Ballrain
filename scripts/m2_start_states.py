"""M2 env preparation measurements (launches the game; results in <runtime>/m2-results/).

    uv run python scripts/m2_start_states.py measure  [--ks 0,5000,10000,15000,19000] [--reps 3]
    uv run python scripts/m2_start_states.py throughput [--seconds 60] [--configs ...]
    uv run python scripts/m2_start_states.py export   [--episodes 3]

measure     TAS start states on level 1 (SR01_1_14_457_264, ballrain.curriculum): for each k, `reps` fresh
            processes on train-00: process launch (Popen -> mod hello), hand-over (hello -> first observation:
            level load + the mod's prefix replay), Python's hand-over cost (parse + level message), intro (to
            control), and the hand-over state checked against the dataset's replay at frame k (ball position,
            sector, checkpoints, points, lives, SR double sum: all exact).
throughput  BallanceEnv in one OS process per instance for `seconds`: decision steps/s, frames/s, lives/hour and
            where the wall time goes (process start split into launch / hand-over / intro, respawn fast-forward,
            env.step split into game + IPC and feature extraction). Configurations: level 1 null / random from
            level load, level 1 random with curriculum starts (cold, and warm pool on two copies per env).
export      curriculum-start episodes with a random policy (action repeat 4, 132 fps after the hand-over),
            each process exported as .tas (prefix + agent frames), replayed 3x by the real TASSupport in
            runtime/ref, and once by BallrainIO in replay mode; PASS = the 3 reference traces identical and equal to
            the agent-mode trace; a failure is classified by gates.export_diagnosis (agent-run divergence vs a
            systematic difference).
Only runtime/<copy>/Bin/Player.exe is started (ballrain.instances); only the started PIDs are stopped.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from ctypes import wintypes
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from ballrain import gates, instances, tas  # noqa: E402

instances.LAUNCH_RECIPE = instances.launch_direct
RUNTIME = instances.runtime_dir()
OUT = RUNTIME / "m2-results"
RECORD = "SR01_1_14_457_264"
COPIES = instances.training_copy_names(8)


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save(name, data, merge: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{name}.json"
    if merge and f.is_file():
        data = {**json.loads(f.read_text(encoding="utf-8")), **data}
    f.write_text(json.dumps(data, indent=1, default=_jd), encoding="utf-8")


# ------------------------------------------------------------------------------------- measure

def measure(ks, reps: int, copy: str = "train-00") -> dict:
    from ballrain.curriculum import TasStarts
    from ballrain.env import BallanceEnv, EnvConfig
    st = TasStarts(RECORD)
    rows = []
    for k in ks:
        kk = st.nearest_valid(k)
        for r in range(reps):
            env = BallanceEnv(EnvConfig(level=1, mode="sr", copy=copy, out_root=str(OUT / "measure" / "episodes"),
                                        action_repeat=4, start_record=RECORD, start_frame=kk))
            try:
                t0 = time.perf_counter()
                obs, info = env.reset()
                t_reset = time.perf_counter() - t0
                s = env.session
                t = s.tracker
                check = None
                if kk > 0:
                    exp = st.expected(kk)
                    got = {"frame": int(s.frames_done), "sector": int(t.sector), "checkpoints": int(t.checkpoints),
                           "points": int(t.points), "lives": int(t.lives), "sr_double": float(t.sr_ms),
                           "pos": np.asarray(s.obs.ball["pos"], np.float64).tolist()}
                    check = {"equal": got == exp, "got": got, "expected": exp}
                n0 = s.frames_done
                for _ in range(50):
                    _, _, term, trunc, info = env.step(1)
                    if term or trunc:
                        break
                tim = dict(info["start"]["timing"])
                tim.update(k=kk, rep=r, reset_s=t_reset, check=check, prefix_fps=(kk / tim["handover_s"]) if kk else None,
                           steps_ok=s is env.session and env.session is not None)
                rows.append(tim)
                print(json.dumps({x: (round(v, 3) if isinstance(v, float) else v) for x, v in tim.items()
                                  if x != "check"}), "check", None if check is None else check["equal"], flush=True)
            finally:
                env.close()
    by = {}
    for k in sorted({r["k"] for r in rows}):
        rs = [r for r in rows if r["k"] == k]
        by[k] = {f: round(float(np.mean([r[f] for r in rs])), 3) for f in
                 ("launch_s", "handover_s", "handover_python_s", "intro_s", "total_s")}
        by[k]["checks_equal"] = all(r["check"] is None or r["check"]["equal"] for r in rs)
    # prefix replay speed: slope of the hand-over time over k
    ksn = np.array([r["k"] for r in rows], float)
    hs = np.array([r["handover_s"] for r in rows], float)
    slope = np.polyfit(ksn, hs, 1) if len(set(ksn)) > 1 else (np.nan, np.nan)
    res = {"record": RECORD, "copy": copy, "rows": rows, "by_k": by,
           "prefix_frames_per_s_fit": float(1 / slope[0]) if slope[0] > 0 else None,
           "handover_at_k0_fit_s": float(slope[1])}
    save("measure", res)
    return res


# ---------------------------------------------------------------------------------- throughput

class _FT(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


def _systimes():
    idle, kern, user = _FT(), _FT(), _FT()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    f = lambda t: (t.hi << 32) | t.lo  # noqa: E731
    return f(idle), f(kern), f(user)


def worker(spec: dict) -> dict:
    from ballrain.env import BallanceEnv, EnvConfig
    rng = np.random.default_rng(spec["seed"])
    cfg = EnvConfig(level=spec["level"], mode="sr", copy=spec["copy"], out_root=spec["out"], action_repeat=4,
                    max_process_frames=spec.get("max_frames"), **spec.get("env", {}))
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
    fin_t = [0.0]
    orig_fin = env._finish_process

    def timed_fin(*a, **k):
        t = time.perf_counter()
        try:
            return orig_fin(*a, **k)
        finally:
            fin_t[0] += time.perf_counter() - t
    env._finish_process = timed_fin
    st = {"copy": spec["copy"], "steps": 0, "frames": 0, "prefix_frames": 0, "decision_frames": 0, "lives": 0, "finishes": 0,
          "processes": 0, "truncations": 0, "t_reset_new": 0.0, "t_reset_respawn": 0.0, "t_step": 0.0,
          "t_launch": 0.0, "t_handover": 0.0, "t_intro": 0.0, "t_adopt_wait": 0.0, "errors": [], "ks": []}
    t_end = time.monotonic() + spec["seconds"]
    t0 = time.monotonic()
    try:
        while time.monotonic() < t_end:
            had = env.session
            f_before = had.frames_done if had is not None else None
            ts = time.perf_counter()
            fx0 = fx_t[0]
            obs, info = env.reset()
            dt = time.perf_counter() - ts
            fx_t[0] = fx0  # the reset observation's features are part of the reset
            s0 = info["start"]
            if s0.get("new_process"):
                st["processes"] += 1
                st["t_reset_new"] += dt
                tm = s0["timing"]
                st["ks"].append(tm["k"])
                if spec.get("env", {}).get("warm_pool"):
                    st["t_adopt_wait"] += s0.get("adopt_wait_s", 0.0)
                else:
                    st["t_launch"] += tm["launch_s"]
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
                if term or trunc:
                    if term:
                        st["lives"] += 1
                        st["finishes"] += info.get("episode_end") == "LevelFinish"
                    else:
                        st["truncations"] += 1
                    break
    except Exception as e:
        st["errors"].append(f"{type(e).__name__}: {e}")
    finally:
        tc = time.perf_counter()
        env.close()
        st["t_close"] = time.perf_counter() - tc
    st["t_features"] = fx_t[0]
    st["t_finish"] = fin_t[0] - st["t_close"]  # previous processes' teardown inside reset / step (STOP, exit, .tas)
    st["wall_s"] = time.monotonic() - t0
    return st


THROUGHPUT_CONFIGS = {
    # name: (policy, instances, env options, copies per env)
    "l1_null_x1": ("null", 1, {}, 1), "l1_random_x1": ("random", 1, {}, 1),
    "l1_null_x4": ("null", 4, {}, 1), "l1_random_x4": ("random", 4, {}, 1),
    "l1_null_x8": ("null", 8, {}, 1), "l1_random_x8": ("random", 8, {}, 1),
    "l1_curr_random_x1": ("random", 1, {"start_record": RECORD, "curriculum": {"start_back": 3000}}, 1),
    "l1_curr_random_x4": ("random", 4, {"start_record": RECORD, "curriculum": {"start_back": 3000}}, 1),
    "l1_curr_random_x8": ("random", 8, {"start_record": RECORD, "curriculum": {"start_back": 3000}}, 1),
    # null policy from curriculum starts: the ball stays where the TAS left it (near moving bodies at the end of L1),
    # so env.step's feature cost there is measured over the whole run
    "l1_curr_null_x8": ("null", 8, {"start_record": RECORD, "curriculum": {"start_back": 3000}}, 1),
    "l1_curr_random_warm_x1": ("random", 1, {"start_record": RECORD, "curriculum": {"start_back": 3000},
                                             "warm_pool": True}, 2),
    "l1_curr_random_warm_x4": ("random", 4, {"start_record": RECORD, "curriculum": {"start_back": 3000},
                                             "warm_pool": True}, 2),
}


def throughput(names, seconds: int) -> dict:
    res = {}
    py = sys.executable
    for name in names:
        policy, n, opts, per = THROUGHPUT_CONFIGS[name]
        out = OUT / "throughput" / name
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        procs = []
        for i in range(n):
            copy = COPIES[i * per]
            env = dict(opts)
            if per == 2:
                env["warm_copy"] = COPIES[i * per + 1]
            if "curriculum" in env:
                env["curriculum"] = dict(env["curriculum"], seed=1000 + i)
            spec = {"copy": copy, "level": 1, "seconds": seconds, "policy": policy, "seed": 1000 + i,
                    "max_frames": 60000, "out": str(out / copy), "env": env}
            f = out / f"{copy}.json"
            procs.append((subprocess.Popen([py, str(Path(__file__).resolve()), "--worker", json.dumps(spec), str(f)],
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
        busy = tot("wall_s") or 1
        split = {k: round(tot(k) / busy, 3) for k in ("t_launch", "t_handover", "t_intro", "t_adopt_wait",
                                                       "t_reset_respawn", "t_step", "t_features", "t_close",
                                                       "t_finish", "t_reset_new")}
        split["t_step_game_ipc"] = round((tot("t_step") - tot("t_features")) / busy, 3)
        case = {"policy": policy, "instances": n, "copies_per_env": per, "options": opts, "seconds": seconds,
                "steps_per_s": round(tot("steps") / wall, 1), "frames_per_s": round(tot("frames") / wall, 1),
                "frames_per_s_without_prefix": round((tot("frames") - tot("prefix_frames")) / wall, 1),
                "feature_ms_per_step": round(1000 * tot("t_features") / max(tot("steps"), 1), 2),
                "decision_frames_per_s": round(tot("decision_frames") / wall, 1),
                "lives_per_hour": round(tot("lives") / wall * 3600, 1), "lives": tot("lives"),
                "finishes": tot("finishes"), "processes": tot("processes"), "truncations": tot("truncations"),
                "time_split": split, "cpu_load_mean": round(float(np.mean(samples)), 3) if samples else None,
                "start_frames": [k for w in ws for k in w.get("ks", [])][:50],
                "errors": [e for w in ws for e in w.get("errors", [])], "workers": ws}
        res[name] = case
        print(name, json.dumps({k: case[k] for k in ("steps_per_s", "frames_per_s", "frames_per_s_without_prefix",
                                                     "feature_ms_per_step", "lives_per_hour", "processes",
                                                     "finishes", "time_split", "cpu_load_mean", "errors")}), flush=True)
        save("throughput", res, merge=True)
    return res


# -------------------------------------------------------------------------------------- export

def export(episodes: int, seed: int = 7) -> dict:
    import m0_gates as m0
    from ballrain.env import BallanceEnv, EnvConfig
    out = OUT / "export"
    if out.exists():
        shutil.rmtree(out)
    rec_dir = out / "records"
    rec_dir.mkdir(parents=True)
    cases = []
    starts = [None] * episodes
    for i in range(episodes):
        copy = COPIES[i % len(COPIES)]
        env = BallanceEnv(EnvConfig(level=1, mode="sr", copy=copy, out_root=str(out / "episodes"), action_repeat=4,
                                    trace=True, start_record=RECORD, max_process_frames=40000,
                                    curriculum={"start_back": 6000 + 3000 * i, "seed": seed + i, "recent": 0.0}))
        rng = np.random.default_rng(seed + i)
        lives, kinds = [], []
        try:
            obs, info = env.reset()
            starts[i] = info["start"]["frame"]
            a, left = 0, 0
            while True:
                kinds.append(info["start"]["kind"])
                while True:
                    if left <= 0:
                        a, left = int(rng.integers(env.action_space.n)), int(rng.integers(2, 40))
                    obs, r, term, trunc, info = env.step(a)
                    left -= 1
                    if term or trunc:
                        break
                lives.append(info.get("episode_end"))
                if env.session is None:
                    break
                obs, info = env.reset(options={"relaunch": False})
                if obs is None:
                    break
        finally:
            env.close()
        meta = json.loads(Path(env.finished_runs[0]).read_text(encoding="utf-8"))
        run_dir = Path(env.finished_runs[0]).parent
        tas_path = run_dir / meta["tas_path"]
        h = meta["tas_sha256"][:16]
        rec = rec_dir / f"m2_{h}.tas"
        shutil.copyfile(tas_path, rec)
        case = {"start_frame": starts[i], "copy": copy, "lives": lives, "episode_kinds": kinds,
                "end_reason": meta["end_reason"], "frames": meta["frames"], "framelog_match": meta["framelog_match"],
                "prefix_frames": meta["prefix"]["frames"] if meta["prefix"] else 0, "tas": rec.name,
                "agent_trace_sha256": instances.sha256_file(run_dir / "trace.txt")}
        refs = []
        for j in range(3):
            refs.append(m0.run_ref(rec, 1, out / "ref" / f"{rec.stem}_ref{j + 1}.txt"))
        shas = [r["sha256"] for r in refs]
        # the same .tas replayed by BallrainIO (replay mode, training copy): tells an agent-run divergence
        # (nondeterminism) from a systematic BallrainIO / TASSupport difference (gates.export_diagnosis)
        bio = m0.run_bio(rec, 1, out / "bio" / rec.stem)
        replay_sha = bio["loads"][0]["trace_sha256"] if bio["loads"] else None
        ok, why, kind = gates.export_diagnosis(shas, case["agent_trace_sha256"], replay_sha)
        case.update(ref_sha256=shas, all_three_equal=len(set(shas)) == 1, verdict=why, kind=kind,
                    replay_sha256=replay_sha, replay_equals_agent=replay_sha == case["agent_trace_sha256"],
                    pass_=bool(ok and len(set(shas)) == 1 and shas[0] == case["agent_trace_sha256"]
                               and meta["framelog_match"]))
        cases.append(case)
        print(json.dumps({k: v for k, v in case.items() if k not in ("ref_sha256",)}), flush=True)
        save("export", {"record": RECORD, "cases": cases})
    return {"record": RECORD, "cases": cases}


def main(argv=None) -> int:
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "--worker":
        res = worker(json.loads(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(res, default=_jd), encoding="utf-8")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["measure", "throughput", "export"])
    ap.add_argument("--ks", default="0,5000,10000,15000,19000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--configs", default=",".join(THROUGHPUT_CONFIGS))
    ap.add_argument("--episodes", type=int, default=3)
    a = ap.parse_args(argv)
    if a.what == "measure":
        r = measure([int(x) for x in a.ks.split(",")], a.reps)
        print(json.dumps(r["by_k"], indent=1), "prefix fps", r["prefix_frames_per_s_fit"])
    elif a.what == "throughput":
        throughput(a.configs.split(","), a.seconds)
    else:
        export(a.episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

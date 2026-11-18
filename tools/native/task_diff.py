"""Differential test of the native task runner (native/apps/nc_task, native/core/src/task_runner.cpp) against the
verifier's replay (ballrain/pool/rollout.py replay_task): the task ends and reward-shaping paths that honest policy
runs rarely reach -- stall (stall_terminal + stall_compensate), the frame cap, deaths / respawns / GameOver, gates,
max_steps, max_frames, a player's stop -- each produced by the native client (forced or policy actions) and replayed by
the Python verifier, which must find no mismatch. With forced actions the verifier's check that each action is the
task's seeded draw is bypassed (the only check skipped; everything else -- engine-state chain, observations,
log-prob, value, reward, flags, end, optional arrays -- is compared as for a real chunk).

    python tools/native/task_diff.py --ckpt runtime/nc/ckpt/r7_latest.pt --record <dataset record dir> \\
        --engine <x87-soft BallanceMMOSimTool.exe> --game <Ballance folder> --nc-task build/native-Release/nc_task.exe \\
        --out runtime/nc/task_diff
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ballrain.pool import protocol as P, rollout as R, task as T, weights as W  # noqa: E402


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def export(state, out: Path, run: str, **args):
    st = dict(state)
    st["args"] = dict(state.get("args") or {}, **args)
    return W.export_state(st, out, run)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--path-asset", default=str(REPO / "runtime" / "refpaths-alt" / "L01.npz"))
    ap.add_argument("--engine", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--nc-task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma-separated scenario names")
    a = ap.parse_args(argv)
    a.nc_task, a.engine, a.game = (str(Path(x).resolve()) for x in (a.nc_task, a.engine, a.game))
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    state = W.load_trusted_checkpoint(a.ckpt)
    m_r7 = export(state, out / "weights", "r7")
    # a variant: a short frame cap and two sub-goal gates (a box over the whole level, a ball near a point) -- the cap
    # and gate code paths of the shaping (r7 itself has no gates and a cap that a task rarely reaches)
    gates = {"everywhere": {"boxes": [[-1e5, 1e5, -1e5, 1e5, -1e5]], "bonus": 0.5},
             "spawn": {"center": [0.0, 0.0, 0.0], "radius": 1e5, "bonus": 0.25, "min_speed": 1.0, "heading": 90,
                       "heading_tol": 120}}
    m_cap = export(state, out / "weights", "r7cap", cap_mult=0.05, gates=json.dumps(gates))
    # the early-respawn rule with a draw (r7 itself: p_early_respawn 0, every early respawn skipped)
    m_er = export(state, out / "weights", "r7er", p_early_respawn=0.5, respawn_level_starts=False)
    ssha, spack = T.export_starts_pack(a.record, out / "assets")
    psha, passet = T.export_path_asset(a.path_asset, out / "assets")
    assert psha == m_r7["features"]["path_sha256"]
    sdir = T.install_starts_pack(spack, out / "cache", ssha)
    pdir = T.install_path_asset(passet, out / "cache", int(m_r7["env"]["level"]), psha)
    from ballrain.curriculum import TasStarts
    st = TasStarts(str(sdir))
    ex = m_r7["env"].get("exclude_starts") or []
    import numpy as np
    keep = np.ones(len(st.valid), bool)
    for lo, hi in ex:
        keep &= ~((st.valid >= lo) & (st.valid < hi))
    st.exclude(keep)  # excluded frames stay hand-over states (at-rest starts with a drive target)
    level = int(m_r7["env"]["level"])
    record = Path(a.record).name
    rec_sha = json.loads((Path(a.record) / "meta.json").read_text(encoding="utf-8"))["record_sha256"]
    if level != 1:  # another level (r8: L2): generic scenarios at the record's quartiles
        q = [st.nearest_valid(int(np.percentile(st.valid, p))) for p in (25, 50, 75)]
        pos = st.expected(q[1])["pos"]
        m_er = export(state, out / "weights", "er", p_early_respawn=0.5, respawn_level_starts=False)
        scenarios = {
            "sticky_spawn": (m_r7, 0, "sticky:1", 4000, 60000, 0, 31, "deaths, respawns from level load"),
            "policy_spawn": (m_r7, 0, "policy", 4000, 60000, 0, 32, "the policy from level load"),
            "sticky_q1": (m_r7, q[0], "sticky:2", 4000, 60000, 0, 33, "TAS start (first quartile): deaths, respawns"),
            "policy_q2": (m_r7, q[1], "policy", 4000, 60000, 0, 34, "TAS start (median)"),
            "policy_q3": (m_r7, q[2], "policy", 4000, 60000, 0, 35, "TAS start (third quartile)"),
            "policy_stop": (m_r7, q[1], "policy", 4000, 60000, 99, 36, "the player's stop"),
            "rest_brake": (m_r7, q[1], "sticky:3", 3000, 60000, 0, 37, "brake-only at-rest start", {"target": None}),
            "rest_drive": (m_r7, q[1], "policy", 3000, 60000, 0, 38, "at-rest start driven 1 unit aside",
                           {"target": [pos[0] + 0.8, pos[2] + 0.6], "max_frames": 300}),
            "early_respawn_frontier": (m_r7, q[0], "sticky:4", 3000, 60000, 0, 39,
                                       "the run's own early-respawn rule (p 0) with its frontier", None,
                                       int(np.percentile(st.valid, 60))),
            "early_respawn_draw": (m_er, 0, "sticky:5", 3000, 60000, 0, 40, "early-respawn draw (p 0.5) from level load",
                                   None, int(np.percentile(st.valid, 60))),
        }
        return run(a, out, scenarios, level, record, rec_sha, ssha, spack, psha, pdir, sdir, np)
    k_mid, k_late = st.nearest_valid(9900), st.nearest_valid(13500)
    k_ex = st.nearest_any(2000)  # inside r7's exclude_starts
    ex_pos = st.expected(k_ex)["pos"]
    # name: (manifest, k, actions, max_steps, max_frames, stop_after, ball seed, what it is for[, rest])
    # (r7's manifest has counter_rand: every k > 0 scenario carries per-life counter values from the task's life_seed)
    scenarios = {
        "rest_drive": (m_r7, st.nearest_valid(1860), "policy", 3000, 60000, 0, 21,
                       "at-rest start driven to r7's (3.5, -170) target", {"target": [3.4, -170.6]}),
        "rest_drive_excl": (m_r7, k_ex, "sticky:5", 3000, 60000, 0, 22,
                            "at-rest start handed over in an excluded range (no split reference), max_frames",
                            {"target": [ex_pos[0] + 0.7, ex_pos[2] - 0.6], "max_frames": 300}),
        "rest_brake": (m_r7, k_late, "sticky:6", 3000, 60000, 0, 23, "brake-only at-rest start, respawns",
                       {"target": None}),
        "rest_drive_late": (m_r7, st.nearest_any(14260), "policy", 3000, 60000, 0, 24,
                            "at-rest start driven to r7's (342, -831) target", {"target": [342.2, -830.7]}),
        "early_respawn": (m_er, k_mid, "sticky:7", 3000, 60000, 0, 25,
                          "early-respawn rule with a draw (p 0.5, frontier 12000): respawn_skipped or played",
                          None, 12000),
        "early_respawn_spawn": (m_er, 0, "sticky:8", 3000, 60000, 0, 26,
                                "early-respawn rule from level load (respawn_level_starts off)", None, 12000),
        "idle_spawn": (m_r7, 0, "idle", 4000, 60000, 0, 11, "stall_terminal + stall_compensate from level load"),
        "sticky_spawn": (m_r7, 0, "sticky:1", 4000, 60000, 0, 12, "deaths, respawns from level load"),
        "sticky_mid": (m_r7, k_mid, "sticky:2", 4000, 60000, 0, 13, "TAS start: deaths, respawns (kind respawn)"),
        "sticky_late": (m_r7, k_late, "sticky:3", 4000, 60000, 0, 14, "TAS start late in the level"),
        "policy_frames": (m_r7, k_mid, "policy", 4000, 900, 0, 15, "max_frames"),
        "policy_stop": (m_r7, k_late, "policy", 4000, 60000, 77, 16, "the player's stop (end stopped)"),
        "cap_gates": (m_cap, k_mid, "policy", 4000, 60000, 0, 17, "frame cap and gates (variant export)"),
        "sticky_cap_gates": (m_cap, 0, "sticky:4", 4000, 60000, 0, 18, "gates with forced actions, cap"),
    }
    return run(a, out, scenarios, level, record, rec_sha, ssha, spack, psha, pdir, sdir, np)


def run(a, out, scenarios, level, record, rec_sha, ssha, spack, psha, pdir, sdir, np) -> int:
    if a.only:
        scenarios = {k: v for k, v in scenarios.items() if k in a.only.split(",")}
    results, failed = {}, 0
    for name, (m, k, actions, max_steps, max_frames, stop_after, ball, what, *opt) in scenarios.items():
        rest = opt[0] if opt else None
        frontier = opt[1] if len(opt) > 1 else None
        d = out / name
        d.mkdir()
        task = T.Task(task_id=f"diff-{name}", run=m["source"]["run"], policy_sha=m["policy_sha"], level=level, record=record,
                      record_sha256=rec_sha, start_frame=int(k), ball_order_seed=ball, engine="x87soft",
                      assets={"starts": ssha, "path": psha}, chunk_steps=int(m["rollout"]["chunk"]),
                      max_steps=max_steps, max_frames=max_frames, deadline_s=3600, issued_at=0, sample_seed=1000 + ball,
                      rest=rest, life_seed=5000 + ball, respawn_frontier=frontier)
        (d / "task.json").write_text(json.dumps(task.to_dict()), encoding="utf-8")
        mp = out / "weights" / f"{m['policy_sha']}.json"
        t0 = time.time()
        p = subprocess.run([a.nc_task, "--task", str(d / "task.json"), "--manifest", str(mp), "--starts", str(spack),
                            "--path-dir", str(pdir), "--engine", a.engine, "--root", a.game, "--out", str(d / "native"),
                            "--actions", actions, "--stop-after", str(stop_after)], capture_output=True, text=True,
                           encoding="utf-8", timeout=3600)
        if p.returncode != 0:
            log(f"{name}: nc_task failed: {p.stderr[-500:]}")
            failed += 1
            continue
        res = json.loads(p.stdout.strip().splitlines()[-1])
        chunks = [P.decode_chunk(f.read_bytes()) for f in sorted((d / "native").glob("*.brpc"))]
        # the verifier's replay of those chunks
        pol = W.load_policy(mp.with_suffix(".npz"), m, expect_sha=m["policy_sha"])
        env = T.build_env(task, m, T.LocalGame(engine="x87soft", binary=a.engine, root=a.game), sdir, pdir, d / "replay")
        runner = R.TaskRunner(env, task, m)
        real = R.action_consistent
        if actions != "policy":
            R.action_consistent = lambda lp, u, act, eps=None: True  # forced actions: the draw check only
        try:
            mm = R.replay_task(runner, pol, chunks)
        finally:
            R.action_consistent = real
            runner.close()
            R.remove_run_dirs(env)
        rewards = np.concatenate([c.arrays["rewards"] for c in chunks]) if chunks else np.zeros(0)
        results[name] = dict(res, what=what, k=int(k), rest=rest, respawn_frontier=frontier, mismatch=None if mm is None else
                             {"reason": mm.reason.value, "step": mm.step, "detail": mm.detail},
                             min_reward=float(rewards.min()) if len(rewards) else None,
                             max_reward=float(rewards.max()) if len(rewards) else None)
        failed += mm is not None
        log(f"{name:17s} k={k:5d} {actions:9s} end={res['end']:10s} steps={res['steps']:5d} episodes={res['episodes']} "
            f"chunks={res['chunks']:3d} -> {'PASS' if mm is None else f'MISMATCH {mm.reason.value} step {mm.step} ({mm.detail})'}"
            f" [{time.time() - t0:.0f} s]")
    (out / "result.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    log("PASSED" if not failed else f"{failed} FAILED")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

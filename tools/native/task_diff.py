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
    ap.add_argument("--branch-dir", help="the stone branch files the learner sends with its policy (state "
                    "branch_assets): exported into the manifest (features.branch) and the branch pack as the gateway "
                    "does; adds the branch scenarios")
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
        if a.branch_dir and (Path(a.branch_dir) / "branches.json").is_file():
            # multi-branch routes (wire v2): each member's measured rest starts (rest-starts.txt), the legacy member's
            # unqualified rest, TAS starts upstream of every entrance, sticky actions through the phases
            from ballrain.branch_assets import read_bundle, validate
            bd = Path(a.branch_dir)
            ba = read_bundle(level, bd)
            pin = validate(ba, level, psha)
            ec = dict(state["env_config"], features=dict(state["env_config"]["features"], branch_sha256=pin,
                                                         branch_dir=None))
            m_v2 = export(dict(state, env_config=ec, branch_assets=ba), out / "weights", "r8multi")
            assert "multi_branch_v2" in m_v2["requires"] and m_v2["features"]["branch_sha256"] == pin
            bsha, bpack = T.export_branch_pack(ba, level, out / "assets", psha)
            branch = {pin: {"sha": bsha, "pack": bpack,
                            "dir": T.install_branch_pack(bpack, out / "cache", level, bsha, pin, psha)}}
            prefix = {}
            for e in json.loads(ba["manifest"])["branches"]:
                with np.load(bd / e["id"] / "drive.npz", allow_pickle=False) as dz:
                    prefix[e["id"]] = int(dz["prefix_frame"])
            rests = []
            txt = (bd / "rest-starts.txt").read_text(encoding="utf-8") if (bd / "rest-starts.txt").is_file() else ""
            for item in txt.replace("\n", ",").split(","):
                if "@branch:" in item:
                    rng_, tgt = item.strip().split("@branch:")
                    key, _, station = tgt.rpartition(":")
                    rests.append((int(rng_.split("-")[0]), key, float(station)))
            scenarios = {}
            for n, (k, key, station) in enumerate(rests):
                scenarios[f"multi_rest_{key}_{station:.0f}"] = (
                    m_v2, k, "policy" if n % 2 == 0 else f"sticky:{10 + n}", 2500, 60000, 0, 60 + n,
                    f"rest start on {key} at {station:.1f} m", {"target": None, "branch_s": station, "branch_id": key})
            scenarios["multi_rest_legacy"] = (m_v2, prefix["south-loop"], "policy", 2500, 60000, 0, 80,
                                              "the unqualified rest: the legacy south loop",
                                              {"target": None, "branch_s": 1700.0})
            scenarios["multi_rest_fallback"] = (m_v2, prefix["back-to-wood"], "policy", 2000, 60000, 0, 81,
                                                "a member rest outside its drive falls back to the TAS start",
                                                {"target": None, "branch_s": 99999.0, "branch_id": "back-to-wood"})
            for n, (key, k) in enumerate(sorted(prefix.items())):
                scenarios[f"multi_tas_{key}"] = (m_v2, st.nearest_valid(k - 300), "policy", 3000, 60000, 0, 90 + n,
                                                 f"TAS start upstream of {key} + the policy")
                scenarios[f"multi_tas_sticky_{key}"] = (m_v2, st.nearest_valid(k), f"sticky:{20 + n}", 3000, 60000, 0,
                                                        95 + n, f"TAS start at {key}'s drive prefix + sticky actions")
            return run(a, out, scenarios, level, record, rec_sha, ssha, spack, psha, pdir, sdir, np, branch)
        if a.branch_dir:  # the stone branch: its measured rest start, and TAS starts upstream of the trafo
            bd = Path(a.branch_dir)
            with np.load(bd / "drive.npz", allow_pickle=False) as dz:
                kb = int(dz["prefix_frame"])
            # as the learner: the run follows the branch and its state carries the files (the export pins them)
            from ballrain.branch_assets import read_bundle, validate
            ba = read_bundle(level, bd)
            pin = validate(ba, level, psha)
            ec = dict(state["env_config"], features=dict(state["env_config"]["features"], branch_sha256=pin,
                                                         branch_dir=None))
            st_b = dict(state, env_config=ec, branch_assets=ba)
            m_r7 = export(st_b, out / "weights", "r8b")
            # rest_speed 0: the drive replays in full, the brake runs to its frame limit, the at-rest check fails (a late
            # fallback after a full replay)
            m_slow = export(dict(st_b, env_config=dict(ec, rest_speed=0.0)), out / "weights", "r8slow")
            assert m_slow["env"]["rest_speed"] == 0.0 and m_r7["features"]["branch_sha256"] == pin
            # other rules (L02.json "rules", distributed with the files): the client must apply them as the env does
            desc = json.loads(ba["desc_json"])
            desc["rules"] = dict(desc.get("rules") or {}, stall_pause_speed=0.5, stall_odometer="s", drive_gain=0.6,
                                 drive_speed_cap=4.0, drive_deadband=0.2, rest_control_repeat=3,
                                 rest_max_brake_frames=400, rest_position_tolerance=3.0)
            import hashlib
            ba_r = dict(ba, desc_json=json.dumps(desc, indent=2).encode())
            ba_r["sha256"] = dict(ba["sha256"], desc_json=hashlib.sha256(ba_r["desc_json"]).hexdigest())
            pin_r = validate(ba_r, level, psha)
            ec_r = dict(ec, features=dict(ec["features"], branch_sha256=pin_r))
            m_rules = export(dict(st_b, env_config=ec_r, branch_assets=ba_r), out / "weights", "r8rules")
            assert m_rules["requires"][:2] == ["branch_v1", "branch_rules_v1"]
            branch = {}
            for p_, b_ in ((pin, ba), (pin_r, ba_r)):
                bsha, bpack = T.export_branch_pack(b_, level, out / "assets", psha)
                branch[p_] = {"sha": bsha, "pack": bpack,
                              "dir": T.install_branch_pack(bpack, out / "cache", level, bsha, p_, psha)}
            scenarios = {
                "branch_rest_policy": (m_r7, kb, "policy", 2500, 60000, 0, 41,
                                       "stone rest start (drive.npz replay, drive, brake) + the policy", {"target": None, "branch_s": 1700.0}),
                "branch_rest_sticky": (m_r7, kb, "sticky:6", 2500, 60000, 0, 42,
                                       "stone rest start + sticky actions (deaths, respawns: the follower resets)",
                                       {"target": None, "branch_s": 1750.0}),
                "branch_rest_fallback": (m_r7, kb, "policy", 2500, 60000, 0, 45,
                                         "a failed stone rest start (target outside the drive) falls back to the TAS start",
                                         {"target": None, "branch_s": 99999.0}),
                "branch_rest_late_fallback": (m_slow, kb, "policy", 2500, 60000, 0, 46,
                                              "the drive replays, the at-rest check fails, the TAS start follows",
                                              {"target": None, "branch_s": 1700.0}),
                "branch_rules_rest": (m_rules, kb, "policy", 2500, 60000, 0, 47,
                                      "other distributed rules: stone rest start (drive gain, cap, repeat, brake frames, "
                                      "tolerance) + the stall rules (pause speed, main-line odometer)",
                                      {"target": None, "branch_s": 1800.0}),
                "branch_rules_sticky": (m_rules, kb, "sticky:8", 2500, 60000, 0, 48,
                                        "other distributed rules + sticky actions", {"target": None, "branch_s": 1700.0}),
                "branch_tas_policy": (m_r7, st.nearest_valid(kb), "policy", 2500, 60000, 0, 43, "TAS start upstream of the trafo + the policy"),
                "branch_tas_sticky": (m_r7, st.nearest_valid(kb - 600), "sticky:7", 2500, 60000, 0, 44,
                                      "TAS start in 小节 3-4 + sticky actions"),
            }
        return run(a, out, scenarios, level, record, rec_sha, ssha, spack, psha, pdir, sdir, np,
                   branch if a.branch_dir else None)
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


def run(a, out, scenarios, level, record, rec_sha, ssha, spack, psha, pdir, sdir, np, branch=None) -> int:
    if a.only:
        scenarios = {k: v for k, v in scenarios.items() if k in a.only.split(",")}
    results, failed = {}, 0
    for name, (m, k, actions, max_steps, max_frames, stop_after, ball, what, *opt) in scenarios.items():
        rest = opt[0] if opt else None
        br = (branch or {}).get(m["features"].get("branch_sha256"))  # the policy's branch pack (None: no branch)
        frontier = opt[1] if len(opt) > 1 else None
        d = out / name
        d.mkdir()
        task = T.Task(task_id=f"diff-{name}", run=m["source"]["run"], policy_sha=m["policy_sha"], level=level, record=record,
                      record_sha256=rec_sha, start_frame=int(k), ball_order_seed=ball, engine="x87soft",
                      assets=dict({"starts": ssha, "path": psha},
                                  **({"branch": br["sha"]} if br else {})),
                      chunk_steps=int(m["rollout"]["chunk"]),
                      max_steps=max_steps, max_frames=max_frames, deadline_s=3600, issued_at=0, sample_seed=1000 + ball,
                      rest=rest, life_seed=5000 + ball, respawn_frontier=frontier)
        (d / "task.json").write_text(json.dumps(task.to_dict()), encoding="utf-8")
        mp = out / "weights" / f"{m['policy_sha']}.json"
        t0 = time.time()
        p = subprocess.run([a.nc_task, "--task", str(d / "task.json"), "--manifest", str(mp), "--starts", str(spack),
                            "--path-dir", str(pdir), "--engine", a.engine, "--root", a.game, "--out", str(d / "native"),
                            "--actions", actions, "--stop-after", str(stop_after)]
                           + (["--branch-pack", str(br["pack"])] if br else [])
                           # f7: the same fields files the Python side reads (<runtime>/fastreach)
                           + (["--reach-dir", str(REPO / "runtime" / "fastreach")] if m["features"].get("reach") else []),
                           capture_output=True, text=True, encoding="utf-8", timeout=3600)
        if p.returncode != 0:
            log(f"{name}: nc_task failed: {p.stderr[-500:]}")
            failed += 1
            continue
        res = json.loads(p.stdout.strip().splitlines()[-1])
        chunks = [P.decode_chunk(f.read_bytes()) for f in sorted((d / "native").glob("*.brpc"))]
        # the verifier's replay of those chunks
        pol = W.load_policy(mp.with_suffix(".npz"), m, expect_sha=m["policy_sha"])
        env = T.build_env(task, m, T.LocalGame(engine="x87soft", binary=a.engine, root=a.game), sdir, pdir, d / "replay",
                          branch_dir=br["dir"] if br else None)
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

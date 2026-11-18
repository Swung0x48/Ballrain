"""M2 evaluation and export check (launches the game).

    uv run python scripts/eval_m2.py --ckpt runtime/m2-train/<run>/latest.pt --runs 20 --temperature 0.3 \
        --copies train-12,train-13,train-14,train-15 --tag <name> [--greedy-runs 2] [--start-frame 0] [--export 1]

Each evaluation run is one game process from level load (k = 0: the scripted Q, then the policy) -- or from a TAS
start frame with --start-frame -- played by the checkpoint's policy closed-loop until LevelFinish, GameOver, a
stall (no 5 units of path progress for --stall-s game seconds) or --max-frames. Deaths respawn in the same
process (per-life episodes), so a run can finish after deaths. Actions: argmax (--greedy-runs runs) or sampled at
--temperature with a per-run seed (logged). Runs record a trace (trace.txt) for the export check.

Export check (--export n): the n fastest finishing runs' .tas (the mod's frame log) are replayed 3x by the real
TASSupport in runtime/ref and once by BallrainIO (replay mode) on --bio-copy; PASS = the 3 reference traces are
identical and equal to the agent-mode trace (gates.export_diagnosis classifies a failure). Output:
<run dir>/eval-<tag>/eval.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

RECORD = "SR01_1_14_457_264"


def build_env_config(ck: dict, out_root: str, start_frame: int, max_frames: int, backend: str | None = None,
                     headless: dict | None = None):
    from ballrain.env import EnvConfig
    from ballrain.features import FeatureConfig
    from ballrain.rewards import RewardConfig
    from ballrain.path import migrate_path_dir
    e = ck["env_config"]
    # checkpoints from before the fix (r5, r6) carry the learner machine's absolute path_dir: map it to this checkout
    fc = FeatureConfig(**dict(e["features"], path_dir=migrate_path_dir(e["features"].get("path_dir"))))
    rc = RewardConfig(**e["reward"])
    # the checkpoint's own level and start record (checkpoints from before r8 are level 1)
    level, record = int(e.get("level") or 1), e.get("start_record") or RECORD
    return EnvConfig(branch_assets=ck.get("branch_assets"), level=level, mode="sr", action_repeat=e["action_repeat"], out_root=out_root, features=fc, reward=rc,
                     start_record=record if start_frame else None, start_frame=start_frame or None,
                     max_process_frames=max_frames, trace=True, world_actions=e.get("world_actions", False),
                     repress_stale_keys=e.get("repress_stale_keys"), backend=backend or e.get("backend", "retail"),
                     headless=headless if headless is not None else e.get("headless"))


def play(spec: dict) -> list:
    import torch
    from ballrain import instances
    from ballrain.env import BallanceEnv
    from ballrain.train.ppo import ActorCritic, normalize
    instances.LAUNCH_RECIPE = instances.launch_direct
    torch.set_num_threads(1)
    ck = torch.load(spec["ckpt"], weights_only=False)
    a = ck["args"]
    model = ActorCritic(ck["n_obs"], ck["n_act"], tuple(int(x) for x in a["hidden"].split(",")), a["activation"],
                        a.get("actor_heads", "single"))
    model.load_state_dict(ck["model"])
    model.eval()
    rms = ck["rms"]
    out = []
    for job in spec["jobs"]:
        cfg = build_env_config(ck, spec["out_root"], job.get("start_frame", spec["start_frame"]), spec["max_frames"],
                               spec.get("backend"), spec.get("headless"))
        cfg.copy = spec["copy"]
        cfg.seed = job["seed"]
        env = BallanceEnv(cfg)
        gen = torch.Generator().manual_seed(job["seed"])
        stall_frames = int(1000 * spec["stall_s"] / cfg.delta_ms)
        lives, death_sectors, end, t0 = [], [], None, time.time()
        try:
            obs, info = env.reset()
            while True:
                mark_s, mark_f = env.reward.s, 0
                while True:
                    act, _, _ = model.act(normalize(obs, rms), temperature=job["temperature"], greedy=job["greedy"], gen=gen)
                    obs, r, term, trunc, info = env.step(int(act))
                    s = env.reward.s
                    if s is not None and (mark_s is None or s > mark_s + 5.0):
                        mark_s, mark_f = s, env._ep_frames
                    if term or trunc:
                        lives.append(info.get("episode_end"))
                        if info.get("episode_end") == "BallOff":
                            death_sectors.append(int(info.get("sector") or 0))
                        break
                    if env._ep_frames - mark_f >= stall_frames:
                        end = "stall"
                        break
                if end or env.session is None:
                    break
                obs, info = env.reset(options={"relaunch": False})
                if obs is None:
                    break
        except Exception as e:  # noqa: BLE001
            end = f"error: {type(e).__name__}: {e}"
        finally:
            env.close()
        meta_p = Path(env.finished_runs[0]) if env.finished_runs else None
        m = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p else {}
        out.append({**job, "copy": spec["copy"], "end_reason": m.get("end_reason"), "stopped": end, "sr_ms": m.get("sr_ms"),
                    "invalid": m.get("invalid"), "death_sectors": death_sectors, "backend": cfg.backend,
                    "start_frame": job.get("start_frame", spec["start_frame"]),
                    "deaths": sum(1 for x in m.get("lives") or [] if x.get("end_reason") == "BallOff"),
                    "lives": lives, "frames": m.get("frames"), "framelog_match": m.get("framelog_match"),
                    "meta": str(meta_p) if meta_p else None, "wall_s": round(time.time() - t0, 1)})
        print(json.dumps(out[-1]), flush=True)
    return out


def export_check(meta_path: Path, out: Path, bio_copy: str, n_ref: int = 5) -> dict:
    import m0_gates as m0
    from ballrain import gates, instances
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    run_dir = meta_path.parent
    rec = out / "records" / f"m2_{m['tas_sha256'][:16]}.tas"
    rec.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / m["tas_path"], rec)
    agent_sha = instances.sha256_file(run_dir / "trace.txt")
    # the real TASSupport replays the .tas n_ref times: the retail game itself occasionally diverges on the same
    # inputs (spot-checks: ~8% of long runs), so the verdict is the majority, and k / n_ref identical is reported
    refs = [m0.run_ref(rec, 1, out / "ref" / f"{rec.stem}_ref{j + 1}.txt") for j in range(n_ref)]
    shas = [r["sha256"] for r in refs]
    from collections import Counter
    if Counter(shas).most_common(1)[0][1] * 5 <= 3 * len(shas):  # split 3:2 or closer: two more replays (user rule)
        refs += [m0.run_ref(rec, 1, out / "ref" / f"{rec.stem}_ref{j + 1}.txt") for j in range(n_ref, n_ref + 2)]
        shas = [r["sha256"] for r in refs]
    r = instances.run_ballrain(rec, 1, out / "bio" / rec.stem, loads=1, timeout=m0.TIMEOUT + 60, copy=bio_copy)
    replay_sha = instances.sha256_file(r["loads"][0]["trace"]) if r["loads"] and Path(r["loads"][0]["trace"]).is_file() else None
    ok, why, kind = gates.export_diagnosis(shas[:3], agent_sha, replay_sha)
    k_equal = sum(x == agent_sha for x in shas)
    return {"run": str(run_dir), "tas": str(rec), "sr_ms": m.get("sr_ms"), "frames": m.get("frames"),
            "framelog_match": m.get("framelog_match"), "agent_trace_sha256": agent_sha, "ref_sha256": shas,
            "all_three_equal": len(set(shas[:3])) == 1, "replay_sha256": replay_sha, "verdict": why, "kind": kind,
            "ref_identical_to_agent": f"{k_equal}/{len(shas)}", "ref_distinct_traces": len(set(shas)),
            "pass_3of3": bool(ok and len(set(shas[:3])) == 1 and shas[0] == agent_sha and m.get("framelog_match")),
            "pass": bool(2 * k_equal > len(shas) and m.get("framelog_match"))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--greedy-runs", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--copies", default="train-12,train-13,train-14,train-15")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=40000)
    p.add_argument("--stall-s", type=float, default=15.0)
    p.add_argument("--export", type=int, default=0)
    p.add_argument("--export-only", help="eval.json of an earlier evaluation: only run the export check")
    p.add_argument("--bio-copy", default="train-15")
    p.add_argument("--backend", default="retail", help="evaluate on this backend (a headless-trained checkpoint on retail)")
    p.add_argument("--ref-replays", type=int, default=5, help="real-TASSupport replays per exported .tas (majority)")
    a = p.parse_args(argv)
    from ballrain import instances
    instances.LAUNCH_RECIPE = instances.launch_direct
    ck_path = Path(a.ckpt).resolve()
    out = ck_path.parent / f"eval-{a.tag}"
    if a.export_only:
        res = json.loads(Path(a.export_only).read_text(encoding="utf-8"))
    else:
        out.mkdir(parents=True, exist_ok=True)
        copies = a.copies.split(",")
        for c in copies:
            if instances.running_in_copy(instances.get_copy(c)):
                raise SystemExit(f"Player.exe already running in {c}")
        jobs = [{"i": i, "seed": a.seed + i, "greedy": i < a.greedy_runs, "temperature": 1.0 if i < a.greedy_runs else a.temperature}
                for i in range(a.runs + a.greedy_runs)]
        specs = [{"ckpt": str(ck_path), "copy": c, "out_root": str(out / "episodes"), "start_frame": a.start_frame,
                  "max_frames": a.max_frames, "stall_s": a.stall_s, "backend": a.backend, "jobs": jobs[j::len(copies)]}
                 for j, c in enumerate(copies)]
        with mp.get_context("spawn").Pool(len(copies)) as pool:
            rows = [r for part in pool.map(play, specs) for r in part]
        rows.sort(key=lambda r: r["i"])
        res = {"ckpt": str(ck_path), "start_frame": a.start_frame, "temperature": a.temperature, "seed": a.seed,
               "runs": rows}
        samp = [r for r in rows if not r["greedy"]]
        fin = [r for r in samp if r["end_reason"] == "LevelFinish" and not r.get("invalid")]
        res["summary"] = {
            "sampled_runs": len(samp), "finished": len(fin), "finished_no_death": sum(r["deaths"] == 0 for r in fin),
            "success_rate": len(fin) / max(len(samp), 1),
            "sr_ms_min": min((r["sr_ms"] for r in fin), default=None),
            "sr_ms_median": float(np.median([r["sr_ms"] for r in fin])) if fin else None,
            "sr_ms_max": max((r["sr_ms"] for r in fin), default=None),
            "greedy": [{"end_reason": r["end_reason"], "sr_ms": r["sr_ms"], "deaths": r["deaths"]} for r in rows if r["greedy"]]}
        print(json.dumps(res["summary"]), flush=True)
    if a.export:
        fin = sorted([r for r in res["runs"] if r["end_reason"] == "LevelFinish" and r["meta"] and not r.get("invalid")],
                     key=lambda r: r["sr_ms"])
        res["export"] = []
        for r in fin[:a.export]:
            e = export_check(Path(r["meta"]), out / "export", a.bio_copy, a.ref_replays)
            print(json.dumps({k: v for k, v in e.items() if k != "ref_sha256"}), flush=True)
            res["export"].append(e)
    (out / ("eval.json" if not a.export_only else "eval_export.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

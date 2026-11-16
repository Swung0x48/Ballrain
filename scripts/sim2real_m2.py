"""Sim-to-real checks for headless training (launches the game or the headless engine).

    # evaluate a checkpoint on one backend (run on the PC for retail, on the Mac for headless)
    uv run python scripts/sim2real_m2.py eval --ckpt <run>/latest.pt --backend retail --copies train-00,...,train-03 \
        [--runs 4] [--greedy] [--starts 0,frontier,frontier+400] [--tb <run>/tb-sim2real] [--watch 45]
    # export spot-check of headless training runs: replay their frame logs on retail BallrainIO (replay mode)
    uv run python scripts/sim2real_m2.py spotcheck --runs-dir <run>/episodes --copy train-11 [--max 6] [--watch 60]

eval: N runs per start (k = 0 and TAS starts at / above the checkpoint's curriculum frontier), same seeds on both
backends (seed = --seed + run index, start index * 1000), sampled at --temperature or greedy. Per start it logs the
success rate (LevelFinish), finishes without death, deaths per sector, SR (median / min of finishes) to
<out>/sim2real_<backend>.jsonl and to tensorboard as eval/<backend>_k<k>_<metric> at the checkpoint's step count.
--watch M: every M minutes take the newest latest.pt (copied first, so a checkpoint being written is not read) and
evaluate it when its update count changed. Runs on its own copies / instances, never in the learner's process.

spotcheck: the headless workers keep a sample of their runs with an agent-mode trace (scripts/train_m2.py
--trace-sample p: kept whatever the outcome, deaths / respawns / stalls included). Each such run not checked yet: its .tas
(the frame log) is replayed once by retail BallrainIO on --copy; the replay trace is compared with the headless run's
trace (sha256, first differing frame via ballrain.trace.compare). Results are appended to <runs-dir>/../spotcheck.jsonl
and the divergence rate goes to tensorboard (eval/spotcheck_divergence_rate); every divergence is saved (frame log,
both traces, meta) under <runs-dir>/../spotcheck-divergences/<run_id>/ for the x87-soft session.
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


def _starts(ck: dict, spec: str) -> list[int]:
    cu = ck.get("curriculum") or {}
    fr = int(cu.get("frontier", 0))
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x.startswith("frontier"):
            out.append(fr + int(x[len("frontier"):] or 0))
        else:
            out.append(int(x))
    return out


def evaluate(a, ck_path: Path) -> dict:
    import torch
    from eval_m2 import play
    ck = torch.load(ck_path, weights_only=False)
    starts = _starts(ck, a.starts)
    copies = a.copies.split(",")
    jobs = [{"i": j * 1000 + i, "seed": a.seed + i + 1000 * j, "greedy": a.greedy, "temperature": a.temperature,
             "start_frame": k} for j, k in enumerate(starts) for i in range(a.runs)]
    out_root = Path(a.out) / f"episodes-{a.backend}"
    hl = json.loads(a.headless) if a.headless else None
    specs = [{"ckpt": str(ck_path), "copy": c, "out_root": str(out_root), "start_frame": 0, "max_frames": a.max_frames,
              "stall_s": a.stall_s, "backend": a.backend, "headless": hl, "jobs": jobs[i::len(copies)]}
             for i, c in enumerate(copies)]
    with mp.get_context("spawn").Pool(len(copies)) as pool:
        rows = [r for part in pool.map(play, specs) for r in part]
    res = {"t": time.time(), "ckpt": str(ck_path), "update": ck["counters"]["update"], "steps": ck["counters"]["steps"],
           "backend": a.backend, "greedy": a.greedy, "temperature": a.temperature, "per_start": {}}
    for k in starts:
        rr = [r for r in rows if r["start_frame"] == k and not r.get("invalid")]
        fin = [r for r in rr if r["end_reason"] == "LevelFinish"]
        ds = [s for r in rr for s in r.get("death_sectors", [])]
        res["per_start"][str(k)] = {
            "runs": len(rr), "success": len(fin) / max(len(rr), 1), "finished_no_death": sum(r["deaths"] == 0 for r in fin),
            "deaths_per_run": len(ds) / max(len(rr), 1), "deaths_by_sector": {str(s): ds.count(s) for s in range(1, 5)},
            "sr_ms_median": float(np.median([r["sr_ms"] for r in fin])) if fin else None,
            "sr_ms_min": min((r["sr_ms"] for r in fin), default=None),
            "stopped": [r["stopped"] for r in rr if r["stopped"]]}
    res["rows"] = rows
    return res


def log_eval(a, res: dict):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"sim2real_{a.backend}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, default=str) + "\n")
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        w = SummaryWriter(a.tb)
        for k, v in res["per_start"].items():
            tag = f"eval/{a.backend}_k{k}"
            w.add_scalar(f"{tag}_success", v["success"], res["steps"])
            w.add_scalar(f"{tag}_deaths_per_run", v["deaths_per_run"], res["steps"])
            for s, n in v["deaths_by_sector"].items():
                w.add_scalar(f"{tag}_deaths_s{s}", n, res["steps"])
            if v["sr_ms_median"] is not None:
                w.add_scalar(f"{tag}_sr_ms_median", v["sr_ms_median"], res["steps"])
        w.close()
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}), flush=True)


def cmd_eval(a) -> int:
    src = Path(a.ckpt)
    last = None
    while True:
        if src.is_file():
            local = Path(a.out) / f"ckpt-{a.backend}.pt"
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, local)
            import torch
            u = torch.load(local, weights_only=False)["counters"]["update"]
            if u != last:
                log_eval(a, evaluate(a, local))
                last = u
        if not a.watch:
            return 0
        time.sleep(60 * a.watch)


def spot_one(run_dir: Path, copy: str, div_root: Path) -> dict:
    from ballrain import instances, trace
    meta_p = next(run_dir.glob("L*.json"))
    m = json.loads(meta_p.read_text(encoding="utf-8"))
    agent_trace = run_dir / "trace.txt"
    tas_p = run_dir / m["tas_path"]
    out = run_dir.parent.parent.parent / "spotcheck-replays" / run_dir.name
    r = instances.run_ballrain(tas_p, int(m.get("level", 1)), out, loads=1, timeout=300, copy=copy)
    rep = Path(r["loads"][0]["trace"]) if r["loads"] else None
    res = {"run": str(run_dir), "run_id": m.get("run_id"), "end_reason": m.get("end_reason"),
           "lives": len(m.get("lives") or []), "frames": m.get("frames"), "backend": m.get("backend"),
           "agent_sha": instances.sha256_file(agent_trace), "replay_sha": instances.sha256_file(rep) if rep and rep.is_file() else None}
    res["equal"] = res["agent_sha"] == res["replay_sha"]
    if not res["equal"] and res["replay_sha"]:
        d = trace.compare(agent_trace, rep)
        res["first_diff_frame"] = getattr(d, "first_diff_frame_a", None)
        res["diff"] = str(d)[:500]
        dst = div_root / run_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in (tas_p, agent_trace, meta_p):
            shutil.copyfile(f, dst / (("headless_" if f == agent_trace else "") + f.name))
        shutil.copyfile(rep, dst / "retail_replay_trace.txt")
    return res


def cmd_spotcheck(a) -> int:
    from ballrain import instances
    instances.LAUNCH_RECIPE = instances.launch_direct
    runs_dir = Path(a.runs_dir)
    base = runs_dir.parent
    log_p = base / "spotcheck.jsonl"
    div_root = base / "spotcheck-divergences"
    while True:
        done = set()
        if log_p.is_file():
            done = {json.loads(x)["run"] for x in log_p.read_text(encoding="utf-8").splitlines() if x.strip()}
        todo = sorted((p.parent for p in runs_dir.glob("*/*/trace.txt") if str(p.parent) not in done),
                      key=lambda p: p.stat().st_mtime)
        results = []
        for rd in todo[-a.max:]:
            if not any(rd.glob("L*.json")):
                continue  # still running
            try:
                res = spot_one(rd, a.copy, div_root)
            except Exception as e:  # noqa: BLE001
                res = {"run": str(rd), "error": f"{type(e).__name__}: {e}"}
            results.append(res)
            with open(log_p, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            print(json.dumps({k: v for k, v in res.items() if k != "diff"}), flush=True)
        if a.tb and results:
            from torch.utils.tensorboard import SummaryWriter
            allr = [json.loads(x) for x in log_p.read_text(encoding="utf-8").splitlines() if x.strip()]
            ok = [r for r in allr if "equal" in r]
            w = SummaryWriter(a.tb)
            w.add_scalar("eval/spotcheck_divergence_rate", 1 - float(np.mean([r["equal"] for r in ok])) if ok else 0.0,
                         len(ok))
            w.add_scalar("eval/spotcheck_checked", len(ok), len(ok))
            w.close()
        if not a.watch:
            return 0
        time.sleep(60 * a.watch)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--backend", choices=("retail", "headless"), required=True)
    e.add_argument("--headless", default=None, help="HeadlessConfig fields as JSON (headless backend)")
    e.add_argument("--copies", required=True, help="train-NN (retail) or hl-NN (headless) names")
    e.add_argument("--runs", type=int, default=4)
    e.add_argument("--starts", default="0,frontier,frontier+400")
    e.add_argument("--greedy", action="store_true")
    e.add_argument("--temperature", type=float, default=0.3)
    e.add_argument("--seed", type=int, default=777)
    e.add_argument("--max-frames", type=int, default=40000)
    e.add_argument("--stall-s", type=float, default=15.0)
    e.add_argument("--out", required=True, help="output directory (jsonl, episodes, local checkpoint copy)")
    e.add_argument("--tb", default=None)
    e.add_argument("--watch", type=float, default=0, help="minutes between checks (0 = once)")
    s = sub.add_parser("spotcheck")
    s.add_argument("--runs-dir", required=True)
    s.add_argument("--copy", default="train-11")
    s.add_argument("--max", type=int, default=6, help="runs per pass (newest first)")
    s.add_argument("--tb", default=None)
    s.add_argument("--watch", type=float, default=0)
    a = p.parse_args(argv)
    return cmd_eval(a) if a.cmd == "eval" else cmd_spotcheck(a)


if __name__ == "__main__":
    raise SystemExit(main())

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
    out_root = Path(a.out) / "episodes" / a.backend
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
    with open(out / f"sim2real_{res['backend']}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, default=str) + "\n")
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        w = SummaryWriter(a.tb)
        step = res["update"] if getattr(a, "tb_step", "steps") == "update" else res["steps"]
        for k, v in res["per_start"].items():
            tag = f"eval/{res['backend']}_k{k}"
            w.add_scalar(f"{tag}_success", v["success"], step)
            w.add_scalar(f"{tag}_deaths_per_run", v["deaths_per_run"], step)
            for s, n in v["deaths_by_sector"].items():
                w.add_scalar(f"{tag}_deaths_s{s}", n, step)
            if v["sr_ms_median"] is not None:
                w.add_scalar(f"{tag}_sr_ms_median", v["sr_ms_median"], step)
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
                try:
                    log_eval(a, evaluate(a, local))
                except Exception as e:  # noqa: BLE001  keep watching
                    import traceback
                    print(f"eval failed: {type(e).__name__}: {e}", traceback.format_exc(), sep="\n", flush=True)
                last = u
        if not a.watch:
            return 0
        time.sleep(60 * (a.watch if last is not None else min(a.watch, 2.0)))  # poll until the first checkpoint


def cmd_pair(a) -> int:
    """Exact pairs: take the newest local copy of the run's latest.pt, copy it to the Mac, evaluate the same file on
    both backends at once (retail here on --copies, headless on the Mac via ssh on --mac-copies), same starts, seeds
    and temperature; both results are logged under the checkpoint's update number (tensorboard step = update).
    ssh target from $BALLRAIN_MAC / $BALLRAIN_MAC_SSH (shell environment only)."""
    import os
    import shlex
    import subprocess
    import threading
    import torch
    mac = os.environ["BALLRAIN_MAC"]
    exe = os.environ.get("BALLRAIN_SSH_EXE", "ssh")  # on Windows Git's ssh.exe (the system OpenSSH rejects key ACLs)
    ssh = [exe, "-o", "BatchMode=yes"] + shlex.split(os.environ.get("BALLRAIN_MAC_SSH", "")) + [mac]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.tb_step = "update"
    last = None
    while True:
        src = Path(a.ckpt)
        if src.is_file():
            u = torch.load(src, weights_only=False)["counters"]["update"]
            if u != last:
                try:
                    local = out / f"ckpt_{u:06d}.pt"
                    shutil.copyfile(src, local)
                    rdir = f"{a.mac_dir}/repo/{a.mac_out}"
                    rfile = f"{rdir}/ckpt_{u:06d}.pt"
                    subprocess.run(ssh + [f"mkdir -p {rdir}"], check=True)
                    with open(local, "rb") as fh:
                        subprocess.run(ssh + [f"cat > {rfile}"], stdin=fh, check=True)
                    common = (f"--runs {a.runs} --starts {a.starts} --seed {a.seed} --temperature {a.temperature} "
                              f"--max-frames {a.max_frames} --stall-s {a.stall_s}" + (" --greedy" if a.greedy else ""))
                    rcmd = (f"cd {a.mac_dir}/repo && export BALLRAIN_RUNTIME=$HOME/{a.mac_dir}/repo/runtime "
                            f"BALLRAIN_HL_BIN={a.hl_bin} BALLRAIN_HL_ROOT=$HOME/{a.mac_dir}/game PYTHONPATH=$HOME/{a.mac_dir}/repo && "
                            f"nice -n 10 {a.mac_uv} run --offline python scripts/sim2real_m2.py eval --ckpt {a.mac_out}/ckpt_{u:06d}.pt "
                            f"--backend headless --copies {a.mac_copies} --out {a.mac_out} {common}")
                    box = {}

                    def remote():
                        r = subprocess.run(ssh + [rcmd], capture_output=True, text=True)
                        lines = [x for x in r.stdout.splitlines() if x.startswith('{"t":')]
                        box["res"] = json.loads(lines[-1]) if lines else None
                        box["err"] = r.stderr[-2000:]
                    th = threading.Thread(target=remote)
                    th.start()
                    ns = argparse.Namespace(**vars(a))
                    ns.backend = "retail"
                    try:
                        res_r = evaluate(ns, local)
                        log_eval(ns, res_r)
                    except Exception as e:  # noqa: BLE001
                        import traceback
                        print(f"retail eval failed: {type(e).__name__}: {e}", traceback.format_exc(), sep="\n", flush=True)
                    th.join()
                    if box.get("res"):
                        box["res"]["rows"] = []
                        log_eval(ns, box["res"])
                    else:
                        print(f"headless eval failed: {box.get('err')}", flush=True)
                    last = u
                except Exception as e:  # noqa: BLE001  (ssh / network): retry in 2 minutes
                    import traceback
                    print(f"pair failed: {type(e).__name__}: {e}", traceback.format_exc(), sep="\n", flush=True)
                    time.sleep(120)
                    continue
        if not a.watch:
            return 0
        time.sleep(60 * a.watch)


def replay_order(tas_path: Path, order: str, out: Path, hl: dict, copy: str = "hl-90") -> dict:
    """Replay a .tas on the headless backend with the ball-ball contact order mode `order` (BALLRAIN_HL_BALL_ORDER:
    ckid | reverse | perm:<seed> | table:<name=rank,...>): the whole .tas is the prefix, the session ends with it.
    Returns the trace sha256 (lines without line ends), end reason, LevelFinish frame, sectors reached, SR."""
    import hashlib
    from ballrain import tas, trace
    from ballrain.client import AgentSession, SessionConfig
    from ballrain.headless import HeadlessConfig, HeadlessLauncher
    out.mkdir(parents=True, exist_ok=True)
    frames = tas.load_any(tas_path)
    fb = out / "frames.bin"
    tas.save_raw(fb, frames)
    cfg = dict(hl or {})
    cfg["extra_env"] = dict(cfg.get("extra_env") or {}, BALLRAIN_HL_BALL_ORDER=order)
    sess = AgentSession(SessionConfig(level=1, out_dir=out, copy=copy, prefix=fb, trace=True),
                        HeadlessLauncher(HeadlessConfig(**cfg)))
    sectors = set()
    try:
        sess.start()
        sectors.add(int(sess.tracker.sector))
        if not sess.ended:
            sess.stop()
    finally:
        sess.close()
    t = sess.tracker
    names = {c for _, c, _ in t.events}
    from ballrain.protocol import Ev
    end = ("LevelFinish" if Ev.LevelFinish in names else "GameOver" if Ev.GameOver in names else
           "BallOff" if Ev.BallOff in names else "stop")
    lines = [trace._strip_eol(x) for x in trace.read_lines(out / "trace.txt")] if (out / "trace.txt").is_file() else []
    sr = t.sr_ms
    return {"order": order, "trace_sha": hashlib.sha256(b"\n".join(lines)).hexdigest(), "lines": lines, "end": end,
            "deaths": sum(1 for _, c, _ in t.events if c == Ev.BallOff), "levelfinish_frame": t.levelfinish_frame,
            "checkpoints": int(t.checkpoints), "sr_ms": None if sr is None else int(float(sr))}


def order_check(tas_path: Path, out: Path, hl: dict, orders: list[str]) -> dict:
    """The same .tas under several contact orders: order-sensitive = the traces differ; outcome-sensitive = the end
    reason, the checkpoints reached, the LevelFinish frame or the SR differ."""
    rows = [replay_order(tas_path, o, out / o.replace(":", "_"), hl) for o in orders]
    base = rows[0]["lines"]
    for r in rows:
        i = next((j for j in range(min(len(base), len(r["lines"]))) if base[j] != r["lines"][j]), None)
        if i is None and len(base) != len(r["lines"]):
            i = min(len(base), len(r["lines"]))
        r["first_diff_line"] = i
        r["first_diff_frame"] = None if i is None or i >= len(r["lines"]) else _frame(r["lines"][i])
        del r["lines"]
    outc = {(r["end"], r["checkpoints"], r["levelfinish_frame"], r["sr_ms"]) for r in rows}
    srs = [r["sr_ms"] for r in rows if r["end"] == "LevelFinish" and r["sr_ms"] is not None]
    return {"tas": str(tas_path), "orders": rows, "distinct_traces": len({r["trace_sha"] for r in rows}),
            "order_sensitive": len({r["trace_sha"] for r in rows}) > 1, "outcome_sensitive": len(outc) > 1,
            "finishes": f"{sum(r['end'] == 'LevelFinish' for r in rows)}/{len(rows)}",
            "sr_spread_ms": (max(srs) - min(srs)) if srs else None}


def _frame(line: bytes):
    from ballrain import trace
    return trace.frame_of(line)


def cmd_orders(a) -> int:
    """Order / outcome sensitivity of .tas files (run where the headless engine is: the Mac)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hl = json.loads(a.headless) if a.headless else {}
    orders = a.orders.split(",")
    files = [Path(x) for x in a.tas] + (sorted(Path(a.runs_dir).glob("*/*/*.tas"), key=lambda q: q.stat().st_mtime)[-a.max:]
                                        if a.runs_dir else [])
    log_p = out / "orders.jsonl"
    done = set()
    if log_p.is_file():
        done = {json.loads(x)["tas"] for x in log_p.read_text(encoding="utf-8").splitlines() if x.strip()}
    for f in files:
        if str(f) in done:
            continue
        try:
            res = order_check(f, out / f.stem, hl, orders)
        except Exception as e:  # noqa: BLE001
            res = {"tas": str(f), "error": f"{type(e).__name__}: {e}"}
        with open(log_p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "orders"}), flush=True)
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        allr = [json.loads(x) for x in log_p.read_text(encoding="utf-8").splitlines() if x.strip()]
        ok = [r for r in allr if "order_sensitive" in r]
        if ok:
            w = SummaryWriter(a.tb)
            w.add_scalar("eval/order_sensitive_rate", float(np.mean([r["order_sensitive"] for r in ok])), len(ok))
            w.add_scalar("eval/outcome_sensitive_rate", float(np.mean([r["outcome_sensitive"] for r in ok])), len(ok))
            w.close()
    return 0


def spot_one(run_dir: Path, copy: str, div_root: Path, label: str = "", after: float = 0.0) -> dict | None:
    from ballrain import instances, trace
    meta_p = next(run_dir.glob("L*.json"))
    m = json.loads(meta_p.read_text(encoding="utf-8"))
    if float(m.get("started_at") or 0) < after:
        return None  # a run of an older engine build (--after)
    agent_trace = run_dir / "trace.txt"
    tas_p = run_dir / m["tas_path"]
    out = run_dir.parent.parent.parent / "spotcheck-replays" / run_dir.name
    r = instances.run_ballrain(tas_p, int(m.get("level", 1)), out, loads=1, timeout=300, copy=copy)
    rep = Path(r["loads"][0]["trace"]) if r["loads"] else None
    res = {"run": str(run_dir), "run_id": m.get("run_id"), "end_reason": m.get("end_reason"),
           "lives": len(m.get("lives") or []), "frames": m.get("frames"), "backend": m.get("backend"),
           "build": m.get("mod_version"), "start_frame": (m.get("start_state") or {}).get("frame"),
           "started_at": m.get("started_at"),
           "agent_sha": instances.sha256_file(agent_trace), "replay_sha": instances.sha256_file(rep) if rep and rep.is_file() else None}
    res["equal"] = res["agent_sha"] == res["replay_sha"]
    if not res["equal"] and rep and rep.is_file():
        # the headless engine writes LF, the retail mod CRLF: compare the lines without their line ends
        la = [trace._strip_eol(x) for x in trace.read_lines(agent_trace)]
        lb = [trace._strip_eol(x) for x in trace.read_lines(rep)]
        res["equal"] = la == lb
        res["eol_only"] = res["equal"]
    res["label"] = label
    if not res["equal"] and res["replay_sha"]:
        la = [trace._strip_eol(x) for x in trace.read_lines(agent_trace)]
        lb = [trace._strip_eol(x) for x in trace.read_lines(rep)]
        i = next((j for j in range(min(len(la), len(lb))) if la[j] != lb[j]), min(len(la), len(lb)))
        res["first_diff_frame"] = trace.frame_of(la[i]) if i < len(la) else None
        res["first_diff_line"] = i
        res["diff"] = f"headless: {la[i][:160] if i < len(la) else b''!r} | retail: {lb[i][:160] if i < len(lb) else b''!r}"
        # classify (gates.export_diagnosis style): replay the same .tas on retail twice more; if the retail replays
        # disagree with each other the game itself diverged on these inputs (known, docs/observations.md 5.1)
        reps = [lb]
        for j in (2, 3):
            r2 = instances.run_ballrain(tas_p, int(m.get("level", 1)), out.with_name(out.name + f"_r{j}"), loads=1,
                                        timeout=300, copy=copy)
            t2 = Path(r2["loads"][0]["trace"]) if r2["loads"] else None
            reps.append([trace._strip_eol(x) for x in trace.read_lines(t2)] if t2 and t2.is_file() else None)
        res["retail_replays_equal"] = [x == lb for x in reps[1:]]
        res["retail_repeat_equals_headless"] = [x == la for x in reps[1:]]
        n_h = 1 + sum(res["retail_repeat_equals_headless"])
        n_r = sum(1 for x in reps if x == lb)
        res["kind"] = ("retail_outlier" if n_h >= 2 else "systematic" if n_r == 3 else "retail_split")
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
                res = spot_one(rd, a.copy, div_root, a.label, a.after)
            except Exception as e:  # noqa: BLE001
                res = {"run": str(rd), "error": f"{type(e).__name__}: {e}", "label": a.label}
            if res is None:
                continue
            results.append(res)
            with open(log_p, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            print(json.dumps({k: v for k, v in res.items() if k != "diff"}), flush=True)
        if a.tb and results:
            from torch.utils.tensorboard import SummaryWriter
            allr = [json.loads(x) for x in log_p.read_text(encoding="utf-8").splitlines() if x.strip()]
            ok = [r for r in allr if "equal" in r and r.get("label", "") == a.label]
            w = SummaryWriter(a.tb)
            tag = f"_{a.label}" if a.label else ""
            w.add_scalar(f"eval/spotcheck{tag}_divergence_rate", 1 - float(np.mean([r["equal"] for r in ok])) if ok else 0.0,
                         len(ok))
            w.add_scalar(f"eval/spotcheck{tag}_checked", len(ok), len(ok))
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
    q = sub.add_parser("pair", help="the same checkpoint on retail (here) and headless (the Mac, via ssh)")
    for n, kw in (("--ckpt", {"required": True}), ("--copies", {"required": True}), ("--runs", {"type": int, "default": 4}),
                  ("--starts", {"default": "0,frontier,frontier+400"}), ("--temperature", {"type": float, "default": 0.3}),
                  ("--seed", {"type": int, "default": 777}), ("--max-frames", {"type": int, "default": 40000}),
                  ("--stall-s", {"type": float, "default": 15.0}), ("--out", {"required": True}), ("--tb", {"default": None}),
                  ("--watch", {"type": float, "default": 0}), ("--mac-dir", {"default": "ballrain"}),
                  ("--mac-out", {"default": "runtime/m2-train/r3/sim2real-pairs"}), ("--mac-copies", {"default": "hl-50,hl-51"}),
                  ("--mac-uv", {"default": "$HOME/Library/Python/3.9/bin/uv"}), ("--hl-bin", {"required": True})):
        q.add_argument(n, **kw)
    q.add_argument("--greedy", action="store_true")
    q.add_argument("--headless", default=None)
    o = sub.add_parser("orders", help="ball-contact order / outcome sensitivity of .tas files (headless)")
    o.add_argument("--tas", nargs="*", default=[])
    o.add_argument("--runs-dir", default=None, help="also the newest --max kept runs under this directory")
    o.add_argument("--max", type=int, default=10)
    o.add_argument("--orders", default="ckid,reverse,perm:1,perm:2,perm:3")
    o.add_argument("--headless", default=None, help="HeadlessConfig fields as JSON")
    o.add_argument("--out", required=True)
    o.add_argument("--tb", default=None)
    s = sub.add_parser("spotcheck")
    s.add_argument("--runs-dir", required=True)
    s.add_argument("--copy", default="train-11")
    s.add_argument("--max", type=int, default=6, help="runs per pass (newest first)")
    s.add_argument("--tb", default=None)
    s.add_argument("--watch", type=float, default=0)
    s.add_argument("--label", default="", help="engine build label written to each record (rates are per label)")
    s.add_argument("--after", type=float, default=0.0, help="only runs started after this unix time (a build switch)")
    a = p.parse_args(argv)
    return {"eval": cmd_eval, "pair": cmd_pair, "spotcheck": cmd_spotcheck, "orders": cmd_orders}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

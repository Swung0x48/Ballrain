"""Off-route retail references for the headless backend (docs/headless.md): random-input game processes on the retail
game (BallanceEnv on runtime copies, the M1-B random_episodes driver of scripts/m1_gates.py), each exported as a legacy
.tas and replayed 3 times by the reference TASSupport (runtime/ref); the majority trace is the reference.

    uv run python scripts/offroute_refs.py --levels 5 7 8 10 12 13 --copies train-12 train-13 train-14 train-15

Per level: <L>_random (SR, sticky random 2-40 steps: deaths, usually GameOver), <L>_hs_esc_random (HS mode with the Esc
action: pauses, unpauses, deaths), <L>_jitter (1-2 steps, frame limit 8000). Output: runtime/offroute-refs/L<nn>/<case>/
{record.tas, record.raw, agent_trace.txt, ref1..3.txt, case.json}, and runtime/offroute-refs/results.json. Launches
Player.exe on the given copies (one process per copy at a time) and on ref (serial).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

import m0_gates as m0  # noqa: E402  (run_ref: the reference TASSupport replay on runtime/ref)
import m1_gates as m1  # noqa: E402  (random_episodes)
from ballrain import gates, instances, tas  # noqa: E402

OUT = instances.runtime_dir() / "offroute-refs"
KINDS = {  # suffix: (max_process_frames, hold, opts)
    "random": (60000, (2, 40), {}),
    "hs_esc_random": (30000, (1, 6), {"mode": "hs"}),
    "jitter": (8000, (1, 2), {}),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_case(name: str, level: int, kind: str, seed: int, copy: str) -> dict:
    max_frames, hold, opts = KINDS[kind]
    d = OUT / f"L{level:02d}" / name
    d.mkdir(parents=True, exist_ok=True)
    case = m1.random_episodes(name, copy, level, seed, None, max_frames, hold, opts, out=str(OUT / "_episodes"))
    m = case["meta"]
    run_dir = Path(case["run_dir"])
    shutil.copyfile(run_dir / m["tas_path"], d / "record.tas")
    tas.save_raw(d / "record.raw", tas.load(d / "record.tas"))
    shutil.copyfile(run_dir / "trace.txt", d / "agent_trace.txt")
    names = [e[1] for e in m["events"]]
    return {"name": name, "level": level, "kind": kind, "seed": seed, "copy": copy, "frames": m["frames"],
            "end_reason": m["end_reason"], "deaths": names.count("BallOff"), "pauses": names.count("PauseLevel"),
            "game_over": "GameOver" in names, "level_finish": "LevelFinish" in names,
            "framelog_match": m["framelog_match"], "lives": len(m["lives"]), "dir": str(d),
            "run_dir": str(run_dir), "steps": case["steps"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", type=int, default=[5, 7, 8, 10, 12, 13])
    ap.add_argument("--copies", nargs="+", default=["train-12", "train-13", "train-14", "train-15"])
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    a = ap.parse_args(argv)
    for c in a.copies:
        gc = instances.get_copy(c)
        if instances.running_in_copy(gc) or gc.has_tassupport() or not gc.has_ballrainio():
            raise SystemExit(f"copy {c} is busy or not an agent copy")
    OUT.mkdir(parents=True, exist_ok=True)
    plan = []
    for lv in a.levels:
        for j, kind in enumerate(a.kinds):
            plan.append((f"L{lv:02d}_{kind}", lv, kind, 5000 + 10 * lv + j))
    results: dict = {}
    lock = threading.Lock()
    todo = list(plan)

    def worker(copy):
        while True:
            with lock:
                if not todo:
                    return
                p = todo.pop(0)
            t0 = time.monotonic()
            try:
                r = run_case(*p, copy)
            except Exception as e:  # noqa: BLE001
                r = {"name": p[0], "level": p[1], "kind": p[2], "error": f"{type(e).__name__}: {e}"}
            r["wall_s"] = round(time.monotonic() - t0, 1)
            with lock:
                results[p[0]] = r
                print("ran", p[0], {k: r.get(k) for k in ("frames", "end_reason", "deaths", "pauses", "error")}, flush=True)

    ths = [threading.Thread(target=worker, args=(c,)) for c in a.copies]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    # reference replays: serial, the one reference copy
    for name, lv, kind, seed in plan:
        r = results.get(name)
        if not r or "error" in r:
            continue
        d = Path(r["dir"])
        rec = d.parent / f"offroute_{name}_{sha(d / 'record.tas')[:12]}.tas"  # unique name in TASRecords
        shutil.copyfile(d / "record.tas", rec)
        try:
            refs = [m0.run_ref(rec, lv, d / f"ref{i}.txt") for i in range(1, 4)]
        finally:
            rec.unlink(missing_ok=True)
        shas = [x["sha256"] for x in refs]
        ok, why = gates.majority_verdict(shas, sha(d / "agent_trace.txt"))
        r.update(ref_sha256=shas, majority=why, refs_identical=len(set(shas)) == 1 and bool(shas[0]),
                 agent_equals_ref=ok)
        (d / "case.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print("ref", name, why, flush=True)
    cov = Counter()
    for r in results.values():
        cov["deaths"] += r.get("deaths", 0)
        cov["pauses"] += r.get("pauses", 0)
        cov["game_over"] += bool(r.get("game_over"))
        cov["level_finish"] += bool(r.get("level_finish"))
    (OUT / "results.json").write_text(json.dumps({"cases": results, "coverage": dict(cov)}, indent=1), encoding="utf-8")
    print(json.dumps(dict(cov)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

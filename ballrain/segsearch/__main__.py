"""python -m ballrain.segsearch search --a 18653 --b 23300 --out runtime/segsearch/s4 [options]
python -m ballrain.segsearch plot runtime/segsearch/s4/result.json --title "..."   (top + side view)
python -m ballrain.segsearch rollouts runtime/segsearch/s4 --title "..."   (every rollout tried, by how it ended)
python -m ballrain.segsearch scan --a 18653 --b 23300 --out runtime/segsearch/s4-scan   (every single edit of the TAS keys)

Search runs the TAS reference A -> B on the headless engine, then the staged beam search (ballrain.segsearch.search),
then the cold verification of the best lines; everything lands in OUT (result.json, search.log, plots)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import EndCheck
from .search import SearchConfig, run_search

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "runtime/datasets/tas-v1/HS01_4740_1_30_344"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m ballrain.segsearch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--a", type=int, required=True, help="gate A: TAS frame of the hand-over")
    s.add_argument("--b", type=int, required=True, help="gate B: TAS frame of the end gate")
    s.add_argument("--dataset", default=str(DATASET), help="tas-v1 dataset folder (run/framelog.bin)")
    s.add_argument("--out", required=True)
    d = SearchConfig(0, 0)
    for k in ("gate_every", "k", "k_speed", "k_surv", "ahead", "m", "rounds", "block", "window", "workers", "seed", "verify_top", "verify_failed"):
        s.add_argument("--" + k.replace("_", "-"), type=int, default=getattr(d, k))
    for k in ("radius", "slack", "p_plain", "p_random", "mean_edits", "y_margin"):
        s.add_argument("--" + k.replace("_", "-"), type=float, default=getattr(d, k))
    e = EndCheck()
    for k in ("dv", "dheading", "dpos", "dy"):
        s.add_argument("--end-" + k, type=float, default=getattr(e, k))
    s.add_argument("--ball-order", default="ckid")
    sc = sub.add_parser("scan", help="run every single edit of the TAS keys between A and B (ballrain.segsearch.scan)")
    sc.add_argument("--a", type=int, required=True)
    sc.add_argument("--b", type=int, required=True)
    sc.add_argument("--dataset", default=str(DATASET))
    sc.add_argument("--out", required=True)
    sc.add_argument("--workers", type=int, default=7)
    sc.add_argument("--gate-every", type=int, default=264)
    sc.add_argument("--radius", type=float, default=6.0)
    sc.add_argument("--max-shift", type=int, default=24)
    sc.add_argument("--verify-top", type=int, default=3)
    mp_ = sub.add_parser("mpc", help="receding-horizon CEM search with rollback (ballrain.segsearch.mpc)")
    mp_.add_argument("--a", type=int, required=True)
    mp_.add_argument("--b", type=int, required=True)
    mp_.add_argument("--dataset", default=str(DATASET))
    mp_.add_argument("--out", required=True)
    from .mpc import MPCConfig
    dm = MPCConfig(0, 0)
    for k in ("window", "commit", "lookahead", "pop", "iters", "elites", "block", "workers", "seed", "max_backtrack"):
        mp_.add_argument("--" + k.replace("_", "-"), type=int, default=getattr(dm, k))
    for k in ("p_base", "alpha", "fall_cost", "off_route", "corridor_cost", "radius", "margin"):
        mp_.add_argument("--" + k.replace("_", "-"), type=float, default=getattr(dm, k))
    mp_.add_argument("--oracle", default="", help="dataset whose keys are committed (score diagnostics)")
    mp_.add_argument("--null", action="store_true", help="commit the TAS keys every window (must reproduce the TAS)")
    w = sub.add_parser("rollouts", help="plot every rollout a search tried (OUT/rollouts.npz) by how it ended")
    w.add_argument("dir")
    w.add_argument("--title", default="")
    q = sub.add_parser("plot")
    q.add_argument("result")
    q.add_argument("--title", default="")
    q.add_argument("--out", default="")
    a = p.parse_args(argv)
    if a.cmd == "search":
        if a.workers > 8:
            raise SystemExit("at most 8 workers (copies hl-40..hl-47)")
        cfg = SearchConfig(a.a, a.b, **{k: getattr(a, k) for k in (
            "gate_every", "k", "k_speed", "k_surv", "ahead", "m", "rounds", "block", "window", "workers", "seed", "verify_top", "verify_failed", "radius",
            "slack", "p_plain", "p_random", "mean_edits", "y_margin")},
            end=EndCheck(a.end_dv, a.end_dheading, a.end_dpos, a.end_dy))
        from .engine import default_engine
        eng = default_engine()
        eng.ball_order = a.ball_order
        run_search(cfg, Path(a.dataset), Path(a.out), eng)
    elif a.cmd == "scan":
        if a.workers > 8:
            raise SystemExit("at most 8 workers (copies hl-40..hl-47)")
        from .scan import run_scan
        run_scan(SearchConfig(a.a, a.b, gate_every=a.gate_every, radius=a.radius, workers=a.workers,
                              verify_top=a.verify_top), Path(a.dataset), Path(a.out), max_shift=a.max_shift)
    elif a.cmd == "mpc":
        if a.workers > 7:
            raise SystemExit("at most 7 workers (hl-40..hl-46; hl-47 is the reference / verify copy)")
        from .mpc import MPCConfig, run_mpc
        run_mpc(MPCConfig(a.a, a.b, **{k: getattr(a, k) for k in (
            "window", "commit", "lookahead", "pop", "iters", "elites", "block", "workers", "seed", "max_backtrack",
            "p_base", "alpha", "fall_cost", "off_route", "corridor_cost", "radius", "margin", "null", "oracle")}), Path(a.dataset), Path(a.out))
    elif a.cmd == "rollouts":
        from .plot import plot_rollouts
        print(plot_rollouts(Path(a.dir), a.title))
        if (Path(a.dir) / "result.json").is_file() and "scan" in json.loads((Path(a.dir) / "result.json").read_text(encoding="utf-8")):
            from .plot import plot_scan
            print(plot_scan(Path(a.dir), a.title))
    else:
        from .plot import plot_result
        print(plot_result(Path(a.result), a.title, Path(a.out) if a.out else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""python -m ballrain.segsearch search --a 18653 --b 23300 --out runtime/segsearch/s4 [options]
python -m ballrain.segsearch plot runtime/segsearch/s4/result.json --title "..."   (top + side view)
python -m ballrain.segsearch rollouts runtime/segsearch/s4 --title "..."   (every rollout tried, by how it ended)

Search runs the TAS reference A -> B on the headless engine, then the staged beam search (ballrain.segsearch.search),
then the cold verification of the best lines; everything lands in OUT (result.json, search.log, plots)."""
from __future__ import annotations

import argparse
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
    elif a.cmd == "rollouts":
        from .plot import plot_rollouts
        print(plot_rollouts(Path(a.dir), a.title))
    else:
        from .plot import plot_result
        print(plot_result(Path(a.result), a.title, Path(a.out) if a.out else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

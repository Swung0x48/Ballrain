"""Warm the f7 reach block on a host before its workers start: builds (or loads) the fastreach fields of a level from the
TAS dataset and runs the reach block on a few states per sector, so the field cache and the numba cache are written
once instead of stalling the first worker step (~16 s cold JIT).

    .venv/Scripts/python.exe tools/reach_warmup.py [--level 2] [--record SR02_1_28_992_264]

Prints the reach version, the field cache files and the timings. Needs numba (the reach block refuses without it).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--record", default="SR02_1_28_992_264")
    ap.add_argument("--dataset", default=None, help="TAS dataset dir (default <runtime>/datasets/tas-v1/<record>)")
    a = ap.parse_args()

    import numpy as np
    from ballrain import instances, protocol as P
    from ballrain.dataset import load_observations
    from ballrain.fastreach import FastReach
    from ballrain.state import GameTracker

    rec = Path(a.dataset) if a.dataset else instances.runtime_dir() / "datasets" / "tas-v1" / a.record
    tr = GameTracker(a.level)
    full, _ = P.parse_level_message((rec / "level.msg").read_bytes())
    tr.merge_level(full)
    inc = rec / "level.msg.inc"
    if inc.exists():
        for m in P.parse_level_stream(inc.read_bytes())[0]:
            tr.merge_level(m)
    tr.level_no = a.level
    t0 = time.time()
    fr = FastReach.fields_for_tracker(tr) if hasattr(FastReach, "fields_for_tracker") else FastReach.for_tracker(tr)
    t1 = time.time()
    states = []
    for o in load_observations(rec / "raw.npz"):
        b = o.ball
        if b is None:
            continue
        p = np.asarray(b["pos"], float)
        if np.abs(p).sum() < 1e-6:
            continue
        states.append((p, np.asarray(b["speed"], float) + np.asarray(b["speed_change"], float), int(b["type"])))
    pick = states[:: max(1, len(states) // 40)]
    t2 = time.time()
    for p, v, bt in pick[:1]:
        fr.perceive(p, v, bt, 0)
    t3 = time.time()
    for p, v, bt in pick:
        fr.perceive(p, v, bt, 0)
    t4 = time.time()
    try:
        from ballrain.reach_features import reach_version
        ver = reach_version()
    except Exception as e:  # noqa: BLE001
        ver = f"? ({type(e).__name__})"
    print(f"reach version {ver}; fields {t1 - t0:.1f} s; first perceive {t3 - t2:.1f} s; "
          f"{len(pick)} perceives {1000 * (t4 - t3) / max(1, len(pick)):.2f} ms each")
    cache = instances.runtime_dir() / "fastreach"
    print("field cache:", sorted(x.name for x in cache.glob(f"L{a.level:02d}-*.npz")))


if __name__ == "__main__":
    main()

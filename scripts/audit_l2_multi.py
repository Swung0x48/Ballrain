"""Summarize controller milestones and replay the distributed FSM against observations."""

from build_l2_multi import MAIN, DEST
from probe_l2_multi import OUT
from ballrain.branch import load_branch
from ballrain.path import ReferencePath, file_sha256
from ballrain.branch_assets import read_bundle, validate
import numpy as np, json, sys

for root in (DEST, DEST / "south-loop"):
    print(root.name, validate(read_bundle(2, root), 2, file_sha256(MAIN)))
for run in sys.argv[1:]:
    main = ReferencePath.load(MAIN)
    m = load_branch(main, 2, DEST, MAIN.parent)
    a = np.load(OUT / run / "trace.npz")["samples"]
    print(run, "SR", (a[-1, 7] - a[0, 7]) / 1000, "sector", a[-1, 6])
    prev = None
    pk = -1
    for frame, x, y, z, bt, stage, sec, t in a:
        path, pr = m.update([x, y, z], int(sec), int(bt))
        assert pr["s"] >= pk - 1e-9
        pk = pr["s"]
        state = (pr["branch_id"], pr["phase"])
        if state != prev:
            print(
                state,
                "frame",
                frame,
                "sr",
                (t - a[0, 7]) / 1000,
                "pos",
                np.round([x, y, z], 2),
                "credit",
                round(pk, 2),
            )
            prev = state
    print(
        "stage milestones",
        [
            (int(r[5]), round((r[7] - a[0, 7]) / 1000, 2), r[1:4].round(2).tolist())
            for r in a[np.r_[True, np.diff(a[:, 5]) != 0]]
        ],
    )

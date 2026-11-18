"""First-passage game times; discard absent ball poses during the transformation."""

import csv
import sys
from pathlib import Path
import numpy as np


def main(root, out):
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            "copy station mode envelope actual_s x y z horizontal_speed t5 t10 t_end_board t_paper t_cp4 last_t".split()
        )
        for file in sorted(root.glob("hlw-*-*.csv")):
            _, copy, st, mode, *v = file.stem.split("-")
            a = np.genfromtxt(file, delimiter=",", names=True)
            ok = a["x"] != 0
            stone = ok & (a["ball"] == 2)
            init = a[0]
            masks = (
                stone & (a["s"] > init["s"] + 5),
                stone & (a["s"] > init["s"] + 10),
                stone & (a["s"] > 2120) & (a["z"] >= -24) & (abs(a["x"] - 1040.55) < 2),
                ok & (a["ball"] == 1),
                a["sector"] == 5,
            )
            times = [
                round(a["frame"][np.flatnonzero(m)[0]] / 132, 3) if m.any() else ""
                for m in masks
            ]
            w.writerow(
                [
                    copy,
                    st,
                    mode,
                    "v1" if v else "v2",
                    *[round(init[k], 3) for k in "sxyz"],
                    round(np.hypot(init["vx"], init["vz"]), 3),
                    *times,
                    round(a["frame"][-1] / 132, 3),
                ]
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

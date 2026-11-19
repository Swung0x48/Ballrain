"""Join first-life intervals (not prefix playback) to r8 rest-start trace positions."""

import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain.path import ReferencePath


def main(r, out):
    path = ReferencePath.load(
        Path(__file__).resolve().parents[1]
        / "benchmarks/refpaths-alt/L02-stone/L02.npz"
    )
    eps = {
        e["run_id"]: e
        for e in csv.DictReader(open(r / "episodes.csv"))
        if int(e["update"]) >= 12541
        and e["kind"] == "tas_rest"
        and e["new_process"] == "True"
    }
    with out.open("w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow("run update station end t s dist x y z ball".split())
        for d in (r / "episodes-traced").glob("*/*"):
            if d.name not in eps:
                continue
            e = eps[d.name]
            j = json.loads((d / (d.name + ".json")).read_text())
            st = j["start_state"]["branch_s"]
            life = j["lives"][0]
            a, b = life["start_frame"], life["end_frame"]
            s = st
            for line in (d / "trace.txt").open():
                t = line.split()
                f = int(t[0])
                if (
                    f < a
                    or f > b
                    or (f - a) % 8
                    and f != b
                    or len(t) < 14
                    or not t[1].startswith("Ball_")
                ):
                    continue
                p = np.array(t[11:14], float)
                pr = path.project(p, 4, s)
                s = pr["s"]
                w.writerow(
                    [
                        d.name,
                        e["update"],
                        st,
                        e["end"],
                        round((f - a) / 132, 4),
                        round(s, 3),
                        round(pr["dist"], 3),
                        *p,
                        t[1],
                    ]
                )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

"""Sample the live corner and reconstruct one r8 first life using the same static geometry."""

import csv, json, sys
from pathlib import Path
import numpy as np
from probe_l2_chokes import probe, ReferencePath
from ballrain.features import level_geometry
from ballrain.reach_features import ReachBlock, REACH_NAMES


def main(runtime, build):
    out = runtime / "viz/l2chokes"
    rb = ReachBlock(cache_dir=out / "fields")
    path = ReferencePath.load(
        Path(__file__).resolve().parents[1]
        / "benchmarks/refpaths-alt/L02-stone/L02.npz"
    )
    offsets = (5, 10, 15, 20, 30, 45)
    seen = set()

    def sample(k, tr, s, p, v, frame):
        marker = next(
            (
                m
                for m in [
                    1740,
                    1750,
                    1760,
                    1770,
                    1775,
                    1780,
                    1783,
                    1785,
                    1790,
                    1795,
                    1800,
                    1805,
                    1810,
                    1820,
                ]
                if s >= m and (k, m) not in seen
            ),
            None,
        )
        if marker is None:
            return
        seen.add((k, marker))
        mesh = level_geometry(tr).tris
        near = mesh.closest(p, 12)
        line = mesh.closest(path.point_at(s), 12)

        def pair(n):
            return [tr.level.name(n["owner"]), n["d"]] if n else ["", None]

        w.writerow(
            [
                k,
                marker,
                frame,
                s,
                *p,
                *v,
                *pair(near),
                *pair(line),
                *np.round((path.points_at(s, offsets) - p).ravel(), 4),
                *[path.speed_at(s + i) for i in (0, 15, 30)],
                *rb.compute(tr, p, v, 0, 2)[1],
            ]
        )

    def obs(tr, s, p, v, frame):
        if not frame:
            mesh = level_geometry(tr).tris
            with (out / "corner-geometry.csv").open("w", newline="") as ff:
                ww = csv.writer(ff)
                ww.writerow("s x y z surface distance qx qy qz".split())
                for m in np.r_[
                    np.arange(1595, 1606, 0.25), np.arange(1760, 1822, 0.25)
                ]:
                    q = path.point_at(m)
                    n = mesh.closest(q, 8)
                    ww.writerow([m, *q, tr.level.name(n["owner"]), n["d"], *n["q"]])
            rid = "<run id of the traced life>"
            d = runtime / "m2-train-mac/r8/episodes-traced/hl-07" / rid
            life = json.loads((d / (rid + ".json")).read_text())["lives"][0]
            last = None
            t = 1700
            for row in (d / "trace.txt").open():
                a = row.split()
                f = int(a[0])
                if (
                    not life["start_frame"] <= f <= life["end_frame"]
                    or a[1] != "Ball_Stone"
                ):
                    continue
                pp = np.array(a[11:14], float)
                t = path.project(pp, 4, t)["s"]
                if last is not None:
                    sample(
                        "trace", tr, t, pp, (pp - last) * 66, f - life["start_frame"]
                    )
                last = pp
        sample("pass", tr, s, p, v, frame)

    with (out / "corner-observations.csv").open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "kind",
                "marker",
                "frame",
                "s",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "surface",
                "d",
                "line_surface",
                "line_d",
                *[f"wp{o}{a}" for o in offsets for a in "xyz"],
                "ref0",
                "ref15",
                "ref30",
                *REACH_NAMES,
            ]
        )
        probe(runtime, build, 1740, "runup", True, observer=obs)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])

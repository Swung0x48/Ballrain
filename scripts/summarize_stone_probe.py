"""Summarize individual attempts; use event IGT, not host time or time since prefix."""

import json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_l2_stone import geom, OUT


def summarize():
    tr, T = geom()
    for folder in sorted(OUT.glob("hlw-*-*")):
        rows = [
            json.loads(line)
            for line in (folder / "controller.jsonl").read_text().splitlines()
            if line.startswith("{")
        ]
        if not rows:
            continue
        print(
            folder.name,
            "handover",
            rows[0]["frame"],
            "end",
            rows[-1],
            "first stone",
            next((r["frame"] for r in rows if r["bt"] == 2), None),
        )
        if rows[-1]["sector"] < 5:
            continue
        ev = (folder / "events.txt").read_text()
        events = [
            (int(a), b)
            for a, b in re.findall(r"^(\d+) (\w+)", ev, re.M)
            if int(a) > rows[0]["frame"]
        ]
        entry = next(a for a, b in events if b == "BallNavInactive")
        cp = next(a for a, b in events if b.startswith("PreC"))
        active = next(a for a, b in events if b == "BallNavActive")
        dt = (rows[-1]["sr"] - rows[0]["sr"]) / (rows[-1]["frame"] - rows[0]["frame"])
        print(
            "entry, stone control, CP:",
            entry,
            active,
            cp,
            "seconds (entry/control to CP)",
            (cp - entry) * dt / 1000,
            (cp - active) * dt / 1000,
        )
        legs = {}
        for row in rows:
            if row["bt"] != 2:
                continue
            p = row["pos"]
            c = T.closest(p, 6)
            if c:
                legs.setdefault(tr.level.name(c["owner"]), []).append(
                    [row["speed"], c["d"] - 2]
                )
        print("nearest owner: samples, speed p10/p50/p90, max surface gap")
        for name, a in legs.items():
            a = np.array(a)
            print(
                name,
                len(a),
                np.percentile(a[:, 0], [10, 50, 90]).round(3),
                round(max(a[:, 1]), 4),
            )


if __name__ == "__main__":
    summarize()

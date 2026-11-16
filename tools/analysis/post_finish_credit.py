"""Analyse BALLRAIN_POST_LOG files (tools/probe/post_finish.py): when are the small score balls that still
follow the player ball at LevelFinish credited, and what score does the end-screen drain count?

    uv run python tools/analysis/post_finish_credit.py <probe out dir>/<record>/post.log [...]

Per log: Points / Lifes / followers at the LevelFinish hook (H) and at the end of the SR tail (last T
line), every positive Points step after that (arrivals), the frame where the drain starts (first
decrease), Points and Lifes at drain start, the reconstructed end-screen score Points_drain + Lifes x
LifeBonus + LevelBonus, and the readout Points_tail + 20 x followers_tail + Lifes x LifeBonus + LevelBonus.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"([HTP]) (\d+) tail=(\d+) post=(\d+) e=([-\d,]+) hs=(-?\d+) fol=(\d+)")


def analyse(path: Path, level_bonus: int | None = None) -> dict:
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        m = LINE.match(line)
        if m:
            e = [int(x) for x in m.group(5).split(",")]
            rows.append({"tag": m.group(1), "frame": int(m.group(2)), "points": e[0], "lifes": e[1],
                         "life_bonus": e[5], "hs": int(m.group(6)), "followers": int(m.group(7))})
    hook = next(r for r in rows if r["tag"] == "H")
    tail = [r for r in rows if r["tag"] == "T"][-1]
    lb = level_bonus if level_bonus is not None else hook["hs"] - hook["points"] - hook["lifes"] * hook["life_bonus"]
    post = [r for r in rows if r["tag"] == "P"]
    arrivals, drain = [], None
    prev = tail
    for r in post:
        d = r["points"] - prev["points"]
        if d > 0 and drain is None:
            arrivals.append((r["frame"] - hook["frame"], d, r["followers"]))
        elif d < 0 and drain is None:
            drain = prev
        prev = r
    out = {"log": str(path), "levelfinish_frame": hook["frame"],
           "hook": {k: hook[k] for k in ("points", "lifes", "followers")},
           "tail_end": {k: tail[k] for k in ("frame", "points", "lifes", "followers")},
           "arrivals": arrivals, "arrived_points": sum(d for _, d, _ in arrivals), "level_bonus": lb,
           "readout_hook": hook["points"] + 20 * hook["followers"] + hook["lifes"] * hook["life_bonus"] + lb,
           "readout_tail": tail["points"] + 20 * tail["followers"] + tail["lifes"] * tail["life_bonus"] + lb}
    if drain is not None:
        out["drain_start"] = {"frames_after_levelfinish": drain["frame"] - hook["frame"] + 1,
                              "points": drain["points"], "lifes": drain["lifes"], "followers": drain["followers"]}
        out["end_screen_score"] = drain["points"] + drain["lifes"] * drain["life_bonus"] + lb
        last = post[-1]
        out["window_end"] = {"frames_after_levelfinish": last["frame"] - hook["frame"], "points": last["points"],
                             "lifes": last["lifes"]}
    return out


def main(argv=None) -> int:
    for p in (argv or sys.argv[1:]):
        print(json.dumps(analyse(Path(p))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

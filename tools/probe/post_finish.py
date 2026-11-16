"""Post-finish probe: replay a TAS record through agent mode and keep the game running idle frames past
LevelFinish (BALLRAIN_POST_FINISH), logging Energy, events, follower visibility and every data-array
change per frame (BALLRAIN_POST_LOG, BallrainIO >= 0.3.1). Used to measure when the small score balls
still following at LevelFinish are credited (docs/benchmarks.md, HS readout).

    uv run python tools/probe/post_finish.py HS01_4738_1_31_317 --copy train-00 --frames 4000 --out <dir>

The record is looked up in the reference copy's TASRecords unless a path is given. Writes <out>/<stem>/
(run outputs, post.log) and prints a summary of the Points timeline after LevelFinish.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ballrain import dataset, instances, protocol as P  # noqa: E402
from ballrain.client import AgentSession, SessionConfig  # noqa: E402
from ballrain.protocol import Action  # noqa: E402
from ballrain import tas  # noqa: E402


def run(record: Path, copy: str, frames: int, out: Path) -> dict:
    level = dataset.level_of(record.name)
    fr = tas.load_any(record)
    d = out / record.stem
    d.mkdir(parents=True, exist_ok=True)
    log = d / "post.log"
    s = AgentSession(SessionConfig(level=level, out_dir=d / "run", copy=copy, post_finish=frames, post_log=log,
                                   obs_timeout_s=600))
    s.start()
    try:
        while s.frames_done < len(fr) and not s.ended:
            i = s.frames_done
            s.step(Action(fr[i:i + 4096], 1, P.event_mask("LevelFinish", "GameOver"), 0, 0))
        if not s.ended:
            s.stop()
    finally:
        s.close()
    return {"record": record.name, "end": s.end_reason_name, "levelfinish": s.tracker.levelfinish_frame,
            "sren": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in
                     zip(s.sren.dtype.names, s.sren.item())} if s.sren is not None else None,
            "post_log": str(log)}


def summarize(log: Path) -> dict:
    """Points / followers timeline: every frame where Energy row 0 or the follower count changes."""
    rows, last = [], None
    arrays = {}
    for line in log.read_text(encoding="latin-1").splitlines():
        m = re.match(r"([HTP]) (\d+) tail=(\d+) post=(\d+) e=([-\d,]+) hs=(-?\d+) fol=(\d+)", line)
        if m:
            key = (m.group(5), m.group(7))
            if key != last:
                rows.append(line.split(" sr=")[0])
                last = key
        elif line.startswith("A "):
            parts = line.split(" ", 4)
            arrays.setdefault(parts[2], 0)
            arrays[parts[2]] += 1
    return {"changes": rows, "array_cell_changes": arrays}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--copy", default=instances.TRAINING_COPY)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = instances.get_copy(instances.REFERENCE_COPY).records_dir
    for r in a.records:
        p = Path(r) if Path(r).suffix else recs / f"{r}.tas"
        res = run(p, a.copy, a.frames, Path(a.out))
        res["summary"] = summarize(Path(res["post_log"]))
        (Path(a.out) / p.stem / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "summary"}))
        print("\n".join(res["summary"]["changes"][:200]))
        print(res["summary"]["array_cell_changes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

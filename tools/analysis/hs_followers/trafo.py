import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P, dataset
from ballrain.state import GameTracker
for name in sys.argv[1:]:
    d = Path(_REPO + r"\runtime\datasets\tas-v1") / name
    t = GameTracker(dataset.level_of(name))
    lm, _ = P.parse_level_message((d / "level.msg").read_bytes()); t.merge_level(lm)
    msgs, _ = P.parse_level_stream((d / "level.msg.inc").read_bytes())
    for m in msgs: t.merge_level(m)
    win, start, fwin, fstart = [], None, [], None
    for o in dataset.load_observations(d / "raw.npz"):
        t.update(o)
        tr = t.transforming
        if tr and start is None: start = o.frames_done
        if not tr and start is not None: win.append((start, o.frames_done)); start = None
        nf = o.ball is not None and not o.ball["found"] and t.flags.started
        if nf and fstart is None: fstart = o.frames_done
        if not nf and fstart is not None: fwin.append((fstart, o.frames_done)); fstart = None
    print(name, "transforming", win, start, "| ball not found", fwin, fstart, "| hs", t.hs_at_finish, t.hs_source, t.followers_at_finish)

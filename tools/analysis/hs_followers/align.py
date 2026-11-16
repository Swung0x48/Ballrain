import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, re, json, numpy as np
from collections import Counter
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P, dataset
from pathlib import Path
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
rx = re.compile(r"^P_Extra_Point_Ball([0-6])(\d{3})?$")
tot = Counter()
for name in sys.argv[1:]:
    d = ROOT / name
    lm, _ = P.parse_level_message((d / "level.msg").read_bytes())
    names = {e.ckid: e.name for e in lm.entities}
    vis = {}; prevn = 0; prevpts = None
    rows = []
    for o in dataset.load_observations(d / "raw.npz"):
        for w in o.wtch:
            m = rx.match(names.get(int(w["ckid"]), "") or "")
            if m: vis[(m.group(2) or "", int(m.group(1)))] = int(w["visible"])
        n = sum(1 for (s, k), v in vis.items() if k > 0 and v == 1 and vis.get((s, 0)) == 0)
        mains_hidden = sum(1 for (s,k),v in vis.items() if k==0 and v==0)
        pts = int(np.int32(o.scor["energy"][0])) if o.scor is not None else None
        ep = sum(1 for e in o.events if int(e["code"]) == 18)
        dp = (pts - prevpts) if (pts is not None and prevpts is not None) else 0
        if n != prevn or (dp > 0) or ep:
            rows.append((o.frames_done, n - prevn, dp, ep, [int(e["frame"]) for e in o.events if int(e["code"])==18]))
        prevn, prevpts = n, pts
    print(name)
    for r in rows[:60]: print("  ", r)

import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, pickle, collections
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P
res = pickle.load(open("tl.pkl","rb"))
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
sel = sys.argv[1] if len(sys.argv) > 1 else "HS0"
for name, r in res.items():
    if sel not in name: continue
    lm, _ = P.parse_level_message((ROOT/name/"level.msg").read_bytes())
    names = {e.ckid: e.name for e in lm.entities}
    watch = dict(lm.watch)
    groups = collections.Counter(watch.values())
    if name == sel or sel == "HS0X": print(groups); print([ (names.get(k), g) for k,g in list(watch.items())[:40]])
    # small-ball entities
    small = {k for k in watch if names.get(k, "") and "Extra_Point_Ball" in names.get(k, "")}
    vis = {}
    hist = []
    for f, w in zip(r["fd"], r["wvis"]):
        for ck, v in w:
            if ck in small:
                if vis.get(ck) != v: hist.append((f, names[ck], v))
                vis[ck] = v
    nvis = sum(vis.values())
    print(name, "small-ball entities watched", len(small), "visible at end:", nvis, "of", len(vis))
    if len(sys.argv) > 2:
        for h in hist: print("  ", h)

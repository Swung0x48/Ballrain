import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, pickle, numpy as np, collections
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P, dataset
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
res = pickle.load(open("tl.pkl","rb"))
sel = sys.argv[1] if len(sys.argv) > 1 else "HS"
out = {}
for name, r in res.items():
    if sel not in name: continue
    d = ROOT/name
    lm, _ = P.parse_level_message((d/"level.msg").read_bytes())
    names = {e.ckid: e.name for e in lm.entities}
    import glob
    # include .inc entity names too
    inc = d/"level.msg.inc"
    if inc.exists():
        msgs, _ = P.parse_level_stream(inc.read_bytes())
        for mm in msgs:
            for e in mm.entities: names.setdefault(e.ckid, e.name)
    fd = r["fd"]
    # points jumps per frame
    pj = {}
    for i in range(1, len(fd)):
        a, b = r["pts"][i-1], r["pts"][i]
        if a is not None and b is not None and b > a and fd[i] > 200:
            dp = b - a; x = next(dp + t for t in (0,1,2) if (dp + t) % 20 == 0); pj[fd[i]] = x
    st = {}; moving_since = {}; ev = []
    for f, w in zip(fd, r["wvis"]):
        for ck, v in w:
            nm = names.get(ck, "?")
            if not ("Extra_Point_Ball" in nm and "HitFrame" not in nm and nm[-3:].isdigit() and nm[-3:] != "000"): continue
            pv = st.get(ck)
            if pv == 1 and v == 0:
                ev.append((f, nm))
            st[ck] = v
    # classify vanish events by whether a small-ball points jump happened within +-3 frames
    small_frames = {f: x for f, x in pj.items() if x != 100}
    used = collections.Counter()
    van_np = []
    byf = collections.defaultdict(list)
    for f, nm in ev: byf[f].append(nm)
    tot_v = 0; tot_pts_v = 0
    for f in sorted(byf):
        n = len(byf[f]); tot_v += n
        near = sum(small_frames.get(g, 0)//20 for g in range(f-3, f+4))
        if near == 0: van_np.append((f, n, sorted(set(x[-3:] for x in byf[f]))))
    print(name, "vanish events", tot_v, "small units collected", sum(x//20 for x in small_frames.values()), "vanish WITHOUT points:", sum(v[1] for v in van_np))
    for v in van_np: print("   ", v)

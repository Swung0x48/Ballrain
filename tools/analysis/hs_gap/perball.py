import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, pickle, numpy as np, json
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P, dataset
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
res = pickle.load(open("tl.pkl","rb"))
PRE = "P_Extra_Point_Ball"
allout = {}
for name, r in res.items():
    d = ROOT/name
    lm, _ = P.parse_level_message((d/"level.msg").read_bytes())
    names = {e.ckid: e.name for e in lm.entities}
    z = np.load(d/"raw.npz"); bp = z["ball_pos"]
    cps = sorted(f for f, n, p in r["evs"] if n == "PreCheckpoint")
    vis, hit, balls = {}, {}, []   # balls: dict(hit, suf, got=[frames], lost_cp=None, pending_end=0)
    cur = {}
    for i, o in enumerate(dataset.load_observations(d/"raw.npz")):
        f = o.frames_done
        if f in cps:
            for s, b in cur.items():
                n = sum(1 for k in range(1, 7) if vis.get((s, k)) == 1)
                if n: b["lost_at_cp"] = (f, n)
            cur = {}
        for wv in o.wtch:
            nm = names.get(int(wv["ckid"]), "")
            if not (nm.startswith(PRE) and ((len(nm) == len(PRE)+4 and nm[-3:].isdigit()) or (len(nm) == len(PRE)+1 and nm[-1].isdigit()))): continue
            k, s, v = int(nm[len(PRE)]), nm[len(PRE)+1:], int(wv["visible"])
            pv = vis.get((s, k)); vis[(s, k)] = v
            dist = float(np.linalg.norm(wv["pos"] - bp[i])) if np.all(np.isfinite(bp[i])) else 99
            if k == 0:
                if pv == 1 and v == 0 and dist < 4:
                    b = dict(hit=f, inst=s or "tmpl", got=[], lost_at_cp=None); balls.append(b); cur[s] = b
                continue
            if pv == 1 and v == 0 and s in cur and f not in [c + x for c in cps for x in range(0, 15)]:
                cur[s]["got"].append(f)
    for s, b in cur.items():
        b["pending_at_end"] = sum(1 for k in range(1, 7) if vis.get((s, k)) == 1)
    out = []
    for b in balls:
        g = b["got"]
        out.append(f"{b['hit']}[{b['inst']}]: {len(g)}/6" + (f" @{g[0]}-{g[-1]}" if g else "") +
                   (f" LOST@CP{b['lost_at_cp'][0]}x{b['lost_at_cp'][1]}" if b.get("lost_at_cp") else "") +
                   (f" PENDING@END x{b['pending_at_end']}" if b.get("pending_at_end") else ""))
    allout[name] = out
    print(name, "balls", len(balls)); print("   " + "\n   ".join(out))
json.dump(allout, open("perball.json", "w"), indent=1)

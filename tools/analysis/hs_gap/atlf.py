import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, pickle, numpy as np
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P, dataset
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
res = pickle.load(open("tl.pkl","rb"))
for name, r in res.items():
    if name.startswith(("HS09","HS13")): continue
    d = ROOT/name
    lm, _ = P.parse_level_message((d/"level.msg").read_bytes())
    names = {e.ckid: e.name for e in lm.entities}
    z = np.load(d/"raw.npz"); bp = z["ball_pos"]; fd = z["frames_done"]
    state = {}; lastmove = {}
    for i, o in enumerate(dataset.load_observations(d/"raw.npz")):
        for w in o.wtch:
            ck = int(w["ckid"]); nm = names.get(ck, "")
            if "Extra_Point_Ball" in nm and "HitFrame" not in nm:
                prev = state.get(ck)
                if prev is not None and prev[0] == 1 and int(w["visible"]) == 1 and np.any(prev[1] != w["pos"]):
                    lastmove[ck] = int(o.frames_done)
                state[ck] = (int(w["visible"]), np.array(w["pos"]))
    lf = r["meta"]["levelfinish_frame"]
    ball = bp[-1]
    vis = [(names[ck], st[1], lastmove.get(ck)) for ck, st in state.items() if st[0] == 1 and names[ck][-3:].isdigit()]
    chasing = [v for v in vis if v[2] is not None and v[2] >= lf - 5]
    ballX = [v for v in vis if v[0][len("P_Extra_Point_Ball")] == "0"]
    gap = int(name.split("_")[1]) - r["meta"]["hs"]
    print(f"{name}: gap/20={gap//20} visible instance balls at end={len(vis)} (Ball0 kind={len(ballX)}) moving within 5f of LF={len(chasing)}")
    for v in sorted(vis, key=lambda v: v[0]):
        print(f"    {v[0]:26s} dist={np.linalg.norm(v[1]-ball):7.2f} lastmove={v[2]}")

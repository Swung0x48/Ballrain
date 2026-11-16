import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, pickle, numpy as np
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P, dataset
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
name, suf, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = ROOT/name
lm, _ = P.parse_level_message((d/"level.msg").read_bytes())
names = {e.ckid: e.name for e in lm.entities}
z = np.load(d/"raw.npz"); bp = z["ball_pos"]
last = {}
for i, o in enumerate(dataset.load_observations(d/"raw.npz")):
    f = o.frames_done
    if f < a: 
        for w in o.wtch: last[int(w["ckid"])] = (int(w["visible"]), w["pos"].copy())
        continue
    if f > b: break
    for w in o.wtch:
        nm = names.get(int(w["ckid"]), "?")
        if nm.endswith(suf) or suf == "*":
            pv = last.get(int(w["ckid"]))
            if pv is None or pv[0] != int(w["visible"]) or (f % 50 == 0):
                print(f, nm, int(w["visible"]), np.round(w["pos"], 2), "ball", np.round(bp[i], 2), "dist", round(float(np.linalg.norm(w["pos"] - bp[i])), 2))
        last[int(w["ckid"])] = (int(w["visible"]), w["pos"].copy())

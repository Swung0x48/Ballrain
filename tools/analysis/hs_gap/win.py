import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, numpy as np
sys.path.insert(0, _REPO + r"")
from ballrain import tas
fr = tas.load(_REPO + r"\runtime\ref\ModLoader\TASRecords\HS04_6925_2_27_295.tas")
print(len(fr), sorted(set(fr["delta"].tolist()))[:5])
z = np.load(_REPO + r"\runtime\datasets\tas-v1\HS04_6925_2_27_295\raw.npz")
fd = z["frames_done"]; nm = z["n_mind"]; vel = z["ball_vel"]; bt=z["ball_type"]
k = fr["keys"]
# run-length of keys
for lo, hi in [(10360,10831),(15980,17474),(24140,24967),(30085,33170),(36660,39793)]:
    segs=[]; s=lo
    for i in range(lo+1, hi+1):
        if k[i]!=k[i-1] or i==hi:
            segs.append((s,i-1,hex(int(k[s])))); s=i
    print(lo,hi,[x for x in segs if x[1]-x[0]>=30])

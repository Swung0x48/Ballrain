import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import pickle, numpy as np, sys
sys.path.insert(0, _REPO + r"")
from ballrain import tas
res = pickle.load(open("tl.pkl","rb"))
for name in sys.argv[1:]:
    r = res[name]
    fr = tas.load(_REPO + rf"\runtime\ref\ModLoader\TASRecords\{name}.tas")
    fd, pts = r["fd"], r["pts"]
    P = [f for f,n,p in r["evs"] if n=="PauseLevel"]; U=[f for f,n,p in r["evs"] if n=="UnpauseLevel"]
    for p,u in zip(P,U):
        inc = [(fd[i], pts[i]-pts[i-1]) for i in range(1,len(fd)) if p-3<=fd[i]<=u+3 and pts[i] is not None and pts[i-1] is not None and pts[i]!=pts[i-1]]
        k = fr["keys"]
        print(name, "pause",p,"unpause",u,"len",u-p,"incs",inc[:3],"...",inc[-2:] if inc else None, "n",len(inc))
        print("   keys around pause esc", [hex(int(x)) for x in k[p-4:p+2]], "during", sorted(set(hex(int(x)) for x in k[p+2:u-3])), "around unpause", [hex(int(x)) for x in k[u-4:u+2]])
        print("   delta", sorted(set(float(x) for x in fr[fr.dtype.names[0]][p-4:u+3])))

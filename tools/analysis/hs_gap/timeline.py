import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, json, re, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, _REPO + r"")
from ballrain import dataset, protocol as P, tas
OUT = Path(__file__).parent
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
INTEREST = {"ExtraPoint","PauseLevel","UnpauseLevel","PreCheckpoint","PostCheckpoint","LevelFinish","PreLifeUp","PostLifeUp","BallOff","PreEndLevel","PostEndLevel","GameOver","CounterInactive","CounterActive","RunEnd"}
res = {}
for d in sorted(ROOT.glob("HS*")):
    name = d.name
    m = json.loads((d/"meta.json").read_text())
    z = np.load(d/"raw.npz")
    rf = z["record_frames"]
    keys = rf["keys"].astype(np.uint32)
    esc = (keys >> 7) & 1
    esc_down = [int(i) for i in np.nonzero((esc[1:] == 1) & (esc[:-1] == 0))[0] + 1]
    if esc[0]: esc_down = [0] + esc_down
    fd, pts, lif, bo, nev = [], [], [], [], []
    evs = []
    wvis = []
    for o in dataset.load_observations(d/"raw.npz"):
        fd.append(o.frames_done)
        if o.scor is not None and o.scor["hs_valid"]:
            pts.append(int(o.scor["energy"][0])); lif.append(int(o.scor["energy"][1]))
        else:
            pts.append(None); lif.append(None)
        for e in o.events:
            evs.append((int(e["frame"]), P.EVENT_NAMES.get(int(e["code"]), str(e["code"])), int(e["param"])))
        wvis.append([(int(w["ckid"]), int(w["visible"])) for w in o.wtch])
    # text events from events.txt
    tev = []
    for line in (d/"events.txt").read_text().splitlines():
        p = line.split(" ", 1)
        if len(p) == 2 and p[0].isdigit():
            nm = p[1].split()[0]
            if nm in INTEREST or nm in ("SR","HS","SRFinal","Summary"):
                tev.append((int(p[0]), p[1]))
        else:
            tev.append((None, line))
    res[name] = dict(meta=m, fd=fd, pts=pts, lif=lif, esc_down=esc_down, nframes=len(rf), evs=evs, tev=tev, wvis=wvis)
    print(name, len(fd), "esc_down", len(esc_down))
pickle.dump(res, open(OUT/"tl.pkl","wb"))

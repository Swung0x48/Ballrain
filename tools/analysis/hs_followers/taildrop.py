import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
for d in sorted(ROOT.glob("*_*")):
    z = np.load(d / "raw.npz")
    blob, off = z["obs_bytes"].tobytes(), z["obs_offsets"]
    o = P.parse_observation(blob[off[-2]:off[-1]])
    if o.sren is None or not o.sren["level_finished"]: continue
    a, b = int(np.int32(o.sren["energy"][0])), int(np.int32(o.scor["energy"][0]))
    dl = 1000 / float(o.time["delta_ms"])
    print(d.name, "LF pts", a, "after tail", b, "diff", b - a, "fps %.0f" % dl, "sr_at_finish %.3f final %.3f" % (o.sren["sr_double_at_finish"], o.sren["sr_double"]))

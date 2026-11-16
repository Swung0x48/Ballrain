import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, re, json, numpy as np
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P, dataset
from pathlib import Path
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
rx = re.compile(r"^P_Extra_Point_Ball([0-6])(\d{3})?$")
for d in sorted(ROOT.glob("*_*")):
    if not d.is_dir(): continue
    lm, _ = P.parse_level_message((d / "level.msg").read_bytes())
    names = {e.ckid: e.name for e in lm.entities}
    meta = json.loads((d / "meta.json").read_text())
    vis = {}; prev = 0; odd = []
    evs = []
    for o in dataset.load_observations(d / "raw.npz"):
        for e in o.events:
            evs.append((int(e["frame"]), P.EVENT_NAMES.get(int(e["code"]))))
        for w in o.wtch:
            m = rx.match(names.get(int(w["ckid"]), "") or "")
            if m: vis[(m.group(2) or "", int(m.group(1)))] = int(w["visible"])
        n = sum(1 for (s, k), v in vis.items() if k > 0 and v == 1 and vis.get((s, 0)) == 0)
        if n > prev and (n - prev) % 6: odd.append((o.frames_done, prev, n))
        if n < prev:
            near = [x for x in evs if o.frames_done - 20 <= x[0] <= o.frames_done and x[1] in ("PreCheckpoint","PostCheckpoint","BallOff","PostSubLife","PreSubLife")]
            if near and n < prev: odd.append(("drop", o.frames_done, prev, n, near[-1]))
        prev = n
    gap = None
    if meta.get("hs") is not None and d.name.startswith("HS"):
        gap = (int(d.name.split("_")[1]) - meta["hs"]) // 20
    print(d.name, "end", meta.get("end_reason"), "followers_at_end", prev, "gap_units", gap, "odd", odd[:8])

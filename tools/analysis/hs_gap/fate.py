import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, pickle, numpy as np, collections, json
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
res = pickle.load(open("tl.pkl","rb"))
summary = {}
for name, r in res.items():
    d = ROOT/name
    lm, _ = P.parse_level_message((d/"level.msg").read_bytes())
    names = {e.ckid: e.name for e in lm.entities}
    fd = r["fd"]; m = r["meta"]
    pj = {}
    for i in range(1, len(fd)):
        a, b = r["pts"][i-1], r["pts"][i]
        if a is not None and b is not None and b > a and fd[i] > 200:
            dp = b - a; x = next(dp + t for t in (0,1,2) if (dp + t) % 20 == 0); pj[fd[i]] = x
    cps = [f for f, n, p in r["evs"] if n == "PreCheckpoint"]
    pause = [f for f, n, p in r["evs"] if n == "PauseLevel"]
    vis = {}  # (suffix, idx) -> v
    hit = {}  # suffix -> frame main hit
    fates = []  # (frame, suffix, idx, fate)
    avail = dict(pj)  # remaining small units per frame to attribute
    for f, w in zip(fd, r["wvis"]):
        for ck, v in w:
            nm = names.get(ck, "")
            if not (nm.startswith("P_Extra_Point_Ball") and nm[-3:].isdigit() and len(nm) == len("P_Extra_Point_Ball") + 4): continue
            idx = int(nm[len("P_Extra_Point_Ball")]); suf = nm[-3:]
            if suf == "000": continue
            pv = vis.get((suf, idx))
            vis[(suf, idx)] = v
            if idx == 0:
                if pv == 1 and v == 0: hit[suf] = f
                if v == 1: hit.pop(suf, None)
                continue
            if pv == 1 and v == 0 and suf in hit:
                # collected? find a points jump at f-3..f+3 with remaining units
                got = None
                for g in range(f - 3, f + 4):
                    if avail.get(g, 0) >= 20 and avail[g] != 100:
                        avail[g] -= 20; got = g; break
                if got is not None: fate = "collected"
                elif any(0 <= f - c <= 15 for c in cps): fate = "lost_at_checkpoint"
                else: fate = "vanished_other"
                fates.append((f, suf, idx, hit[suf], fate))
    inflight = [(s, i, hit[s]) for (s, i), v in vis.items() if v == 1 and i > 0 and s in hit]
    c = collections.Counter(x[4] for x in fates)
    left = sum(v for v in avail.values() if v != 100)
    gap = None if m["hs"] is None else int(name.split("_")[1]) - m["hs"]
    lostcp = collections.defaultdict(int)
    for x in fates:
        if x[4] == "lost_at_checkpoint": lostcp[x[0]] += 1
    summary[name] = dict(gap=gap, collected=c["collected"], lost_at_cp=c["lost_at_checkpoint"], other=c["vanished_other"],
                         inflight_at_end=len(inflight), unattributed_points_units=left//20, lost_by_frame=dict(lostcp),
                         inflight_hit_frames=sorted(set(x[2] for x in inflight)))
    print(name, json.dumps(summary[name]))
json.dump(summary, open("fate.json", "w"), indent=1)

import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import sys, pickle, numpy as np, json
from pathlib import Path
sys.path.insert(0, _REPO + r"")
from ballrain import protocol as P
ROOT = Path(_REPO + r"\runtime\datasets\tas-v1")
res = pickle.load(open("tl.pkl","rb"))
PRE = "P_Extra_Point_Ball"
out = {}
for name, r in res.items():
    d = ROOT/name
    lm, _ = P.parse_level_message((d/"level.msg").read_bytes())
    names = {e.ckid: e.name for e in lm.entities}
    z = np.load(d/"raw.npz"); bp = z["ball_pos"]
    fd = r["fd"]; m = r["meta"]; lf = m["levelfinish_frame"]
    cps = sorted(f for f, n, p in r["evs"] if n == "PreCheckpoint")
    vis, pos, hit = {}, {}, {}
    collected_d, vanish_far = 0, []
    snaps = {}
    want = set(cps) | ({lf} if lf else {fd[-1]})
    for i, (f, w) in enumerate(zip(fd, r["wvis"])):
        pass
    # need positions: reload from obs (wvis lacks pos) -> reparse quickly
    from ballrain import dataset
    for i, o in enumerate(dataset.load_observations(d/"raw.npz")):
        f = o.frames_done
        for wv in o.wtch:
            nm = names.get(int(wv["ckid"]), "")
            if not (nm.startswith(PRE) and ((len(nm) == len(PRE) + 4 and nm[-3:].isdigit()) or (len(nm) == len(PRE) + 1 and nm[-1].isdigit()))): continue
            k, suf, v = int(nm[len(PRE)]), nm[len(PRE)+1:], int(wv["visible"])
            pv = vis.get((suf, k)); vis[(suf, k)] = v; pos[(suf, k)] = np.array(wv["pos"])
            dist = float(np.linalg.norm(wv["pos"] - bp[i])) if np.all(np.isfinite(bp[i])) else 99
            if k == 0:
                if pv == 1 and v == 0 and dist < 4: hit[suf] = f
                elif v == 1: hit.pop(suf, None)
            elif pv == 1 and v == 0 and suf in hit:
                if dist < 2.6: collected_d += 1
                else: vanish_far.append((f, suf, k, round(dist, 1)))
        if f in want:
            snaps[f] = sum(1 for (s, k), v in vis.items() if k > 0 and v == 1 and s in hit)
    gap = None if m["hs"] is None else int(name.split("_")[1]) - m["hs"]
    inflight_end = sum(1 for (s, k), v in vis.items() if k > 0 and v == 1 and s in hit)
    o = dict(gap_units=None if gap is None else gap // 20, collected_by_touch=collected_d,
             inflight_at_PreCheckpoint={c: snaps.get(c) for c in cps}, inflight_at_end=inflight_end,
             end_frame=fd[-1], LF=lf, vanished_far=len(vanish_far))
    o["sum_lost_at_cp_plus_inflight_end"] = sum(v or 0 for v in o["inflight_at_PreCheckpoint"].values()) + inflight_end
    out[name] = o
    print(name, json.dumps(o))
json.dump(out, open("fate2.json", "w"), indent=1)

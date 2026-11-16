import sys, struct, math
import numpy as np
sys.path.insert(0, '.')
import pf
frames, ev = pf.parse(sys.argv[1], want={"K","D","DT","P"})
P = pf.parse_partners(frames)
def core_at(obj, idx):
    c = P[obj]["cores"]
    k = max((i for i in c if i <= idx), default=min(c))
    m = c[k]
    R = np.array([m[0:3], m[4:7], m[8:11]]); t = np.array(m[12:15])
    return R, t
def pt_tri(p, a, b, c):
    # closest point on triangle (Ericson)
    ab, ac, ap = b-a, c-a, p-a
    d1, d2 = ab@ap, ac@ap
    if d1 <= 0 and d2 <= 0: return a
    bp = p-b; d3, d4 = ab@bp, ac@bp
    if d3 >= 0 and d4 <= d3: return b
    vc = d1*d4-d3*d2
    if vc <= 0 and d1 >= 0 and d3 <= 0: return a + d1/(d1-d3)*ab
    cp = p-c; d5, d6 = ab@cp, ac@cp
    if d6 >= 0 and d5 <= d6: return c
    vb = d5*d2-d1*d6
    if vb <= 0 and d2 >= 0 and d6 <= 0: return a + d2/(d2-d6)*ac
    va = d3*d6-d5*d4
    if va <= 0 and (d4-d3) >= 0 and (d5-d6) >= 0: return b + (d4-d3)/((d4-d3)+(d5-d6))*(c-b)
    den = 1/(va+vb+vc); return a + ab*vb*den + ac*vc*den
rows = []
bad_unit = 0; n = 0
for fr in frames:
    K = fr.get("K")
    if not K: continue
    pos = np.array(K["pos"])
    for d in fr["D"]:
        raw = d["raw"]
        if len(raw) < 0x90 or "tri" not in d or d["pobj"] not in P: continue
        n += 1
        L = pf.f32(raw, 0x54); sxr = pf.f32(raw, 0x50)
        pl = np.array(struct.unpack_from("<3f", raw, 0x68)); hes = pf.f32(raw, 0x74)
        if abs(np.linalg.norm(pl)-1) > 1e-3: bad_unit += 1
        R, t = core_at(d["pobj"], fr["idx"])
        v = [R @ np.array(x) + t for x in d["tri"]["v"]]
        q = pt_tri(pos, *v)
        dist = np.linalg.norm(pos - q)
        dirv = (pos - q) / max(dist, 1e-9)
        rows.append((fr["idx"], L, dist - sxr, abs(pl @ dirv), int(d["pst"]), d["partner"]))
print("mindists with tri", n, "non-unit planes", bad_unit)
a = np.array([(r[1], r[2], r[3]) for r in rows])
err = a[:,0]-a[:,1]
print("len - (geomdist - sum_extra_radius): median %.3g p95 %.3g max %.3g" % (np.median(abs(err)), np.percentile(abs(err),95), abs(err).max()))
print("|plane . dir| median %.6f p5 %.6f min %.6f" % (np.median(a[:,2]), np.percentile(a[:,2],5), a[:,2].min()))
for r in rows[:5]: print(r)
big = [r for r in rows if abs(r[1]-r[2]) > 0.05]
print("big errors", len(big)); print(big[:10])

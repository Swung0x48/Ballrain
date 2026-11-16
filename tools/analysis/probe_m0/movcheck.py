import sys, struct, math
import numpy as np
sys.path.insert(0, '.')
import pf, geom
out = sys.argv[1]
frames, ev = pf.parse(out + "/frames.txt", want={"K","BW","D","DT","P","B"})
meshes = geom.load_meshes(out + "/meshes.txt")
def qmat(x,y,z,w):
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],[2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],[2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
res = {"tri_psi": [], "ckmesh_psi": [], "ckmesh_now": [], "ckmesh_now_ckball": []}
names = {}
for fr in frames:
    K = fr.get("K")
    if not K or fr["ball"] == "Ball_Paper": continue
    p = np.array(K["pos"]); pnow = np.array(fr["BW"][9:12])
    Pm = {}
    for s in fr["P"]:
        t = s.split(" ")
        if " movable=1 " not in " " + s + " ": continue
        obj = t[0].split("=")[1]; eid = int(t[2].split("=")[1])
        i = t.index("core="); m = [float(x) for x in t[i+1:i+17]]
        q = [float(x) for x in t[i+17].split("=")[1].split(",")]
        sh = [float(x) for x in t[i+18].split("=")[1].split(",")]
        Pm[obj] = (eid, np.array([m[0:3], m[4:7], m[8:11]]), np.array(m[12:15]), q, np.array(sh))
    Bw = {}
    for bl in fr["B"]:
        t = bl.split(" "); Wi = t.index("W"); Bw[int(t[1])] = geom.W12_to_44([float(x) for x in t[Wi+1:Wi+13]])
    for d in fr["D"]:
        if d["pobj"] not in Pm or "tri" not in d: continue
        L = pf.f32(d["raw"], 0x54)
        if L > 1.0: continue
        eid, R, vv, q, sh = Pm[d["pobj"]]
        Rq = qmat(*q)
        # (a) IVP ledge triangle in object coords -> world at PSI
        best = None
        for conv in ("q", "qT"):
            Rc = Rq if conv == "q" else Rq.T
            V = np.array([R @ (Rc @ np.array(v) + sh) + vv for v in d["tri"]["v"]])
            Q, _ = geom.closest_pt_tris(p, V[0:1], V[1:2], V[2:3])
            e = abs(np.linalg.norm(p - Q[0]) - 2.0 - L)
            best = e if best is None or e < best else best
        res["tri_psi"].append(best)
        # (b) CK mesh local -> world with the IVP PSI pose, and (c) with the CK entity matrix now
        if eid in meshes:
            if True:
                Vl = np.concatenate([m["V"] for m in meshes[eid]]); offs = np.cumsum([0]+[len(m["V"]) for m in meshes[eid]])
                T = np.concatenate([m["T"] + o for m, o in zip(meshes[eid], offs)])
                m = {"V": Vl}
                Vp = (R @ (Rq @ m["V"].T + sh[:, None])).T + vv
                Q, _ = geom.closest_pt_tris(p, Vp[T[:,0]], Vp[T[:,1]], Vp[T[:,2]])
                e_ = abs(np.linalg.norm(p - Q, axis=1).min() - 2.0 - L); res["ckmesh_psi"].append(e_); names.setdefault(d["partner"], []).append(e_)
                if eid in Bw:
                    Vn = geom.to_world(m["V"], Bw[eid])
                    Q, _ = geom.closest_pt_tris(p, Vn[T[:,0]], Vn[T[:,1]], Vn[T[:,2]])
                    res["ckmesh_now"].append(abs(np.linalg.norm(p - Q, axis=1).min() - 2.0 - L))
                    Q, _ = geom.closest_pt_tris(pnow, Vn[T[:,0]], Vn[T[:,1]], Vn[T[:,2]])
                    res["ckmesh_now_ckball"].append(abs(np.linalg.norm(pnow - Q, axis=1).min() - 2.0 - L))
for k, v in res.items():
    if v:
        v = np.array(v); print(k, "n", len(v), "median %.3g p95 %.3g max %.3g" % (np.median(v), np.percentile(v, 95), v.max()))
    else: print(k, "none")

for n, v in sorted(names.items()):
    print("  %-30s n %5d median %.3g p95 %.3g meshes %s" % (n, len(v), np.median(v), np.percentile(v, 95), [ (m["mesh"], m["kind"], len(m["V"])) for m in meshes.get(next((int(d) for d in []), 0), [])]))

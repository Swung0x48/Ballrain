import sys, math, json
import numpy as np
sys.path.insert(0, '.')
import pf, geom
out = sys.argv[1]; step = int(sys.argv[2])
frames, ev = pf.parse(out + "/frames.txt", want={"K","C","D","DT"})
meshes = geom.load_meshes(out + "/meshes.txt")
world = geom.StaticWorld(meshes, lambda m: True)
floors = set(); grp = None
for l in open(out + "/static_start.txt", encoding="latin-1"):
    if l.startswith("GROUP "): grp = l.split(" ")[2]
    elif l.startswith("  - ") and grp == "Phys_Floors": floors.add(int(l.split(" ")[3]))
    elif not l.startswith("  "): grp = None
def ray_down(P, A, B, C):
    # P (R,3) rays straight down; returns highest hit y per ray (nan if none). Moller-Trumbore with dir (0,-1,0).
    d = np.array([0, -1.0, 0])
    e1 = B - A; e2 = C - A
    h = np.cross(d, e2); a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-9
    f = np.where(ok, 1 / np.where(ok, a, 1), 0)
    s = P[:, None, :] - A[None]
    u = f[None] * (s * h[None]).sum(2)
    q = np.cross(s, e1[None])
    v = f[None] * (q @ d)
    t = f[None] * (q * e2[None]).sum(2)
    hit = ok[None] & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    y = np.where(hit, P[:, 1:2] - t, -np.inf)
    return y.max(1)
dirs = np.array([[math.cos(a), 0, math.sin(a)] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)])
steps = np.arange(0.25, 8.01, 0.25)
rows = []
n = 0
for fr in frames:
    K = fr.get("K")
    if not K or fr["ball"] == "Ball_Paper" or not fr["C"] or not all(int(c["pid"]) in floors for c in fr["C"]): continue
    n += 1
    if n % step: continue
    p = np.array(K["pos"])
    g = world.nearest(p, 6)
    if not g or g["d"] - 2 > 0.05 or abs(g["n_face"][1]) < 0.5 or g["ent"] not in floors: continue
    h = g["q"][1]
    P = (p[None, None, :] + steps[None, :, None] * dirs[:, None, :]).reshape(-1, 3)
    P[:, 1] = p[1] + 1.0
    idx = world.grid.query(p, 10)
    y = ray_down(P, world.A[idx], world.B[idx], world.C[idx]).reshape(16, len(steps))
    drop = (y < h - 1.0)
    first = np.where(drop.any(1), steps[np.argmax(drop, 1)], np.inf)
    edge_mesh = world.edge_distance(g["q"])
    rows.append((fr["idx"], float(first.min()), edge_mesh, int(fr["D"][0]["pst"]) if fr["D"] else -1))
a = np.array([(r[1], r[2]) for r in rows])
fan = a[:, 0]; fin = np.isfinite(fan)
print("frames", len(rows), "fan-drop min distance: finite share %.3f median %.2f p10 %.2f p90 %.2f" % (fin.mean(), np.median(fan[fin]), np.percentile(fan[fin], 10), np.percentile(fan[fin], 90)))
print("mesh-boundary edge distance median %.3f p10 %.3f" % (np.median(a[:, 1]), np.percentile(a[:, 1], 10)))
print("corr(fan, meshedge) finite:", np.corrcoef(fan[fin], np.minimum(a[fin, 1], 8))[0, 1])
print("share meshedge<0.3 while fan>=2:", float(np.mean((a[:, 1] < 0.3) & (fan >= 2))))

hist = np.histogram(np.minimum(fan, 8.5), bins=[0, 0.5, 1, 2, 3, 4, 6, 8, 9])[0]
print("fan histogram [0,.5,1,2,3,4,6,8,inf):", hist.tolist())

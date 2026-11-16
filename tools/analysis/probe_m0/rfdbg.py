import sys, math
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], "5"]
exec(open('rayfan.py').read().split("a = np.array")[0])
bad = [r for r in rows if r[1] <= 0.25][:3]
print(bad)
for fr in frames:
    if fr["idx"] != bad[0][0]: continue
    K = fr["K"]; p = np.array(K["pos"]); g = world.nearest(p, 6); h = g["q"][1]
    print("p", p, "q", g["q"], "nface", g["n_face"], "ent", world.names.get(g["ent"]), "reg", g["reg"])
    P = (p[None, None, :] + steps[None, :4, None] * dirs[:, None, :]).reshape(-1, 3); P[:, 1] = p[1] + 1
    idx = world.grid.query(p, 10)
    y = ray_down(P, world.A[idx], world.B[idx], world.C[idx]).reshape(16, 4)
    print(np.round(y - h, 3))

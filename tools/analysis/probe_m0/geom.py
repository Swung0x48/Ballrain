"""Geometry from the probe's meshes.txt: world triangles, grid lookup, closest points, walkable boundary edges."""
import numpy as np


def load_meshes(path):
    """-> {ent_id: [ {ent, entname, mesh, kind, fixed, W(4x4), V(n,3), T(m,3)} ]} (first export per entity/mesh)."""
    out = {}
    with open(path, "r", encoding="latin-1") as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines):
        l = lines[i]
        if not l.startswith("MESH "):
            i += 1
            continue
        p = l.split(" ")
        d = dict(t.split("=", 1) for t in p if "=" in t)
        ent = int(d["ent"])
        entname = p[3]
        nv, nf = int(d["nv"]), int(d["nf"])
        W = np.array([float(x) for x in lines[i + 1].split(" ")[1:17]]).reshape(4, 4)
        V = np.array([[float(x) for x in lines[i + 2 + k].split(" ")[1:4]] for k in range(nv)]).reshape(-1, 3)
        T = np.array([[int(x) for x in lines[i + 2 + nv + k].split(" ")[1:4]] for k in range(nf)], dtype=np.int64).reshape(-1, 3)
        out.setdefault(ent, []).append({"ent": ent, "entname": entname, "mesh": p[5] if len(p) > 5 else "-", "kind": d["kind"],
                                        "fixed": d["fixed"] == "1", "W": W, "V": V, "T": T})
        i += 2 + nv + nf
    return out


def to_world(V, W):
    """Virtools row-vector convention: [x y z 1] * W."""
    return V @ W[:3, :3] + W[3, :3]


def W12_to_44(w12):
    W = np.zeros((4, 4))
    W[:4, :3] = np.array(w12).reshape(4, 3)
    W[3, 3] = 1
    return W


def closest_pt_tris(p, A, B, C):
    """Vectorised closest point on N triangles to p (Ericson). Returns (Q (N,3), region code)."""
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - B
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - C
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    v = vb / den
    w = vc / den
    Q = A + ab * v[:, None] + ac * w[:, None]
    reg = np.full(len(A), 2)  # 2 face, 1 edge, 0 vertex
    # edges
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where(((d4 - d3) + (d5 - d6)) == 0, 1, (d4 - d3) + (d5 - d6))
    Q[m] = (B + (C - B) * t[:, None])[m]; reg[m] = 1
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where((d2 - d6) == 0, 1, d2 - d6)
    Q[m] = (A + ac * t[:, None])[m]; reg[m] = 1
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where((d1 - d3) == 0, 1, d1 - d3)
    Q[m] = (A + ab * t[:, None])[m]; reg[m] = 1
    # vertices
    m = (d6 >= 0) & (d5 <= d6); Q[m] = C[m]; reg[m] = 0
    m = (d3 >= 0) & (d4 <= d3); Q[m] = B[m]; reg[m] = 0
    m = (d1 <= 0) & (d2 <= 0); Q[m] = A[m]; reg[m] = 0
    return Q, reg


def seg_dist(p, S0, S1):
    d = S1 - S0
    L2 = (d * d).sum(1)
    t = np.clip(((p - S0) * d).sum(1) / np.where(L2 == 0, 1, L2), 0, 1)
    Q = S0 + d * t[:, None]
    return np.linalg.norm(p - Q, axis=1)


class Grid:
    def __init__(self, lo, hi, cell=8.0):
        self.cell = cell
        self.map = {}
        cl = np.floor(lo / cell).astype(int)
        ch = np.floor(hi / cell).astype(int)
        for i in range(len(lo)):
            for x in range(cl[i, 0], ch[i, 0] + 1):
                for y in range(cl[i, 1], ch[i, 1] + 1):
                    for z in range(cl[i, 2], ch[i, 2] + 1):
                        self.map.setdefault((x, y, z), []).append(i)

    def query(self, p, r):
        lo = np.floor((p - r) / self.cell).astype(int)
        hi = np.floor((p + r) / self.cell).astype(int)
        s = set()
        for x in range(lo[0], hi[0] + 1):
            for y in range(lo[1], hi[1] + 1):
                for z in range(lo[2], hi[2] + 1):
                    s.update(self.map.get((x, y, z), ()))
        return np.fromiter(s, dtype=np.int64, count=len(s))


class StaticWorld:
    """All triangles of fixed physicalized entities, in world coordinates."""

    def __init__(self, meshes, include):
        A, B, C, E = [], [], [], []
        self.names = {}
        for ent, lst in meshes.items():
            for m in lst:
                if not m["fixed"] or not include(m):
                    continue
                Vw = to_world(m["V"], m["W"])
                T = m["T"]
                if len(T) == 0:
                    continue
                A.append(Vw[T[:, 0]]); B.append(Vw[T[:, 1]]); C.append(Vw[T[:, 2]])
                E.append(np.full(len(T), ent))
                self.names[ent] = m["entname"]
        self.A, self.B, self.C = np.concatenate(A), np.concatenate(B), np.concatenate(C)
        self.E = np.concatenate(E)
        n = np.cross(self.B - self.A, self.C - self.A)
        ln = np.linalg.norm(n, axis=1)
        self.N = n / np.where(ln == 0, 1, ln)[:, None]
        lo = np.minimum(np.minimum(self.A, self.B), self.C)
        hi = np.maximum(np.maximum(self.A, self.B), self.C)
        self.grid = Grid(lo, hi)
        self._edges()

    def _edges(self, up=0.5):
        """Boundary edges of the walkable set (faces with |n.y| > up), vertices welded at 1e-3."""
        walk = np.abs(self.N[:, 1]) > up
        idx = np.where(walk)[0]
        key = lambda P: [tuple(x) for x in np.round(P / 1e-3).astype(np.int64)]
        ka, kb, kc = key(self.A[idx]), key(self.B[idx]), key(self.C[idx])
        cnt = {}
        for i in range(len(idx)):
            for u, v in ((ka[i], kb[i]), (kb[i], kc[i]), (kc[i], ka[i])):
                e = (u, v) if u < v else (v, u)
                cnt[e] = cnt.get(e, 0) + 1
        bnd = [e for e, c in cnt.items() if c == 1]
        if bnd:
            self.S0 = np.array([e[0] for e in bnd], dtype=float) * 1e-3
            self.S1 = np.array([e[1] for e in bnd], dtype=float) * 1e-3
        else:
            self.S0 = self.S1 = np.zeros((0, 3))
        self.egrid = Grid(np.minimum(self.S0, self.S1), np.maximum(self.S0, self.S1)) if len(self.S0) else None

    def nearest(self, p, r=8.0):
        idx = self.grid.query(p, r)
        if len(idx) == 0:
            return None
        Q, reg = closest_pt_tris(p, self.A[idx], self.B[idx], self.C[idx])
        d = np.linalg.norm(p - Q, axis=1)
        k = int(np.argmin(d))
        return {"d": float(d[k]), "q": Q[k], "reg": int(reg[k]), "tri": int(idx[k]), "ent": int(self.E[idx[k]]),
                "n_face": self.N[idx[k]], "all_d": d, "all_idx": idx}

    def edge_distance(self, q, r=12.0):
        if self.egrid is None:
            return float("inf")
        idx = self.egrid.query(q, r)
        if len(idx) == 0:
            return float("inf")
        return float(seg_dist(q, self.S0[idx], self.S1[idx]).min())

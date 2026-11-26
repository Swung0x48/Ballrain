"""Collision geometry in Python (state-spec.md 3.4): static triangle soup with a 2-D (x, z) grid,
posed moving bodies, closest points, vertical rays, walkable boundary edges.

Conventions (verified on probe data, see docs/observations.md):
- CK matrices are row-vector: world = v @ M[:3, :3] + M[3, :3].
- IVP quaternions (x, y, z, w) give column-vector rotations: world = R(q) @ v + pos. On SR06
  R(q_last_psi) matched the transposed CK ball matrix to 0.04 (the CK matrix is half a PSI ahead).
- Moving bodies (state-spec.md 2.5): world = R(q_core) @ (R(q_cfo) @ v + shift) + pos, with v the CK
  mesh's local vertices. The q_cfo convention is unverified (identity in every probed sample).
All functions use float64.
"""
from __future__ import annotations

import os

import numpy as np

try:  # the per-step vertical rays (features: height grid, edge fan) run through a compiled kernel when numba is there
    if os.environ.get("BALLRAIN_NO_NUMBA") or os.environ.get("BALLRAIN_RAY_PY"):
        raise ImportError("disabled")
    import numba
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    numba = None
    HAVE_NUMBA = False

UP_WALKABLE = 0.5  # |n_y| above this: walkable face
WELD = 1e-3


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def ck_to_world(v, m) -> np.ndarray:
    """Row-vector CK transform; m is (4,4) or (4,3)."""
    m = np.asarray(m, np.float64)
    return np.asarray(v, np.float64) @ m[:3, :3] + m[3, :3]


def body_to_world(v, pos, q, q_cfo, shift) -> np.ndarray:
    R = quat_matrix(q)
    Rc = quat_matrix(q_cfo)
    v = np.asarray(v, np.float64)
    return (v @ Rc.T + np.asarray(shift, np.float64)) @ R.T + np.asarray(pos, np.float64)


def closest_point_triangles(p, A, B, C):
    """Closest point on each of N triangles to p (Ericson 5.1.5). Returns (Q (N,3), region) with
    region 2 = face, 1 = edge, 0 = vertex."""
    p = np.asarray(p, np.float64)
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - B
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - C
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    v, w = vb / den, vc / den
    Q = A + ab * v[:, None] + ac * w[:, None]
    reg = np.full(len(A), 2)

    def put(mask, val, r):
        Q[mask] = val[mask]
        reg[mask] = r

    def sdiv(a, b):
        return a / np.where(b == 0, 1, b)

    t = sdiv(d4 - d3, (d4 - d3) + (d5 - d6))
    put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), B + (C - B) * t[:, None], 1)
    t = sdiv(d2, d2 - d6)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + ac * t[:, None], 1)
    t = sdiv(d1, d1 - d3)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + ab * t[:, None], 1)
    put((d6 >= 0) & (d5 <= d6), C, 0)
    put((d3 >= 0) & (d4 <= d3), B, 0)
    put((d1 <= 0) & (d2 <= 0), A, 0)
    return Q, reg


def ray_down_pairs(P, A, B, C, eps: float = 1e-9):
    """Vertical ray from each P[i] downwards against triangle i (pairwise). Returns hit y (nan = miss)."""
    x, z = P[:, 0], P[:, 2]
    x0, z0 = A[:, 0], A[:, 2]
    e1x, e1z = B[:, 0] - x0, B[:, 2] - z0
    e2x, e2z = C[:, 0] - x0, C[:, 2] - z0
    det = e1x * e2z - e2x * e1z
    ok = np.abs(det) > eps
    det = np.where(ok, det, 1.0)
    px, pz = x - x0, z - z0
    u = (px * e2z - e2x * pz) / det
    v = (e1x * pz - px * e1z) / det
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    y = A[:, 1] + u * (B[:, 1] - A[:, 1]) + v * (C[:, 1] - A[:, 1])
    hit &= y <= P[:, 1] + 1e-9
    return np.where(hit, y, np.nan)


def _seg_dist(p, S0, S1):
    d = S1 - S0
    L2 = (d * d).sum(1)
    t = np.clip(((p - S0) * d).sum(1) / np.where(L2 == 0, 1, L2), 0, 1)
    return np.linalg.norm(p - (S0 + d * t[:, None]), axis=1)


class Grid2D:
    """Uniform grid over (x, z) of axis-aligned boxes; CSR cell -> item lists."""

    def __init__(self, lo, hi, cell: float = 4.0):
        self.cell = float(cell)
        n = len(lo)
        if n == 0:
            self.origin = np.zeros(2)
            self.shape = (1, 1)
            self.start = np.zeros(2, np.int64)
            self.items = np.zeros(0, np.int64)
            return
        lo2, hi2 = lo[:, [0, 2]], hi[:, [0, 2]]
        self.origin = lo2.min(0) - 1e-6
        c0 = np.floor((lo2 - self.origin) / self.cell).astype(np.int64)
        c1 = np.floor((hi2 - self.origin) / self.cell).astype(np.int64)
        nx, nz = int(c1[:, 0].max()) + 1, int(c1[:, 1].max()) + 1
        self.shape = (nx, nz)
        sx, sz = c1[:, 0] - c0[:, 0] + 1, c1[:, 1] - c0[:, 1] + 1
        counts = sx * sz
        item = np.repeat(np.arange(n), counts)
        k = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        ix = c0[item, 0] + k // sz[item]
        iz = c0[item, 1] + k % sz[item]
        cid = ix * nz + iz
        order = np.argsort(cid, kind="stable")
        self.items = item[order]
        self.start = np.searchsorted(cid[order], np.arange(nx * nz + 1))

    def cells_of(self, xz_lo, xz_hi):
        c0 = np.floor((np.asarray(xz_lo) - self.origin) / self.cell).astype(np.int64)
        c1 = np.floor((np.asarray(xz_hi) - self.origin) / self.cell).astype(np.int64)
        nx, nz = self.shape
        c0 = np.maximum(c0, 0)
        c1 = np.minimum(c1, [nx - 1, nz - 1])
        if (c1 < c0).any():
            return np.zeros(0, np.int64)
        ix, iz = np.meshgrid(np.arange(c0[0], c1[0] + 1), np.arange(c0[1], c1[1] + 1), indexing="ij")
        return (ix * nz + iz).ravel()

    def query_box(self, xz_lo, xz_hi) -> np.ndarray:
        if HAVE_NUMBA:
            lo = np.asarray(xz_lo, np.float64)
            hi = np.asarray(xz_hi, np.float64)
            return _query_box_grid(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]), float(self.origin[0]),
                                   float(self.origin[1]), self.cell, int(self.shape[0]), int(self.shape[1]),
                                   self._start64(), self._items64())
        cells = self.cells_of(xz_lo, xz_hi)
        if len(cells) == 0:
            return np.zeros(0, np.int64)
        parts = [self.items[self.start[c]:self.start[c + 1]] for c in cells]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)

    def _start64(self):
        if getattr(self, "_s64", None) is None:
            self._s64 = np.ascontiguousarray(self.start, np.int64)
        return self._s64

    def _items64(self):
        if getattr(self, "_i64", None) is None:
            self._i64 = np.ascontiguousarray(self.items, np.int64)
        return self._i64

    def point_cells(self, xz) -> np.ndarray:
        """Cell id per point, -1 outside the grid."""
        c = np.floor((np.asarray(xz) - self.origin) / self.cell).astype(np.int64)
        nx, nz = self.shape
        ok = (c[:, 0] >= 0) & (c[:, 0] < nx) & (c[:, 1] >= 0) & (c[:, 1] < nz)
        return np.where(ok, c[:, 0] * nz + c[:, 1], -1)


def _query_box_grid(lx, lz, hx, hz, ox, oz, cell, nx, nz, start, items):
    """Grid2D.query_box: the sorted distinct items of the cells overlapping [lo, hi] (same cell arithmetic)."""
    c0x = max(int(np.floor((lx - ox) / cell)), 0)
    c0z = max(int(np.floor((lz - oz) / cell)), 0)
    c1x = min(int(np.floor((hx - ox) / cell)), nx - 1)
    c1z = min(int(np.floor((hz - oz) / cell)), nz - 1)
    if c1x < c0x or c1z < c0z:
        return np.zeros(0, np.int64)
    n = 0
    for ix in range(c0x, c1x + 1):
        for iz in range(c0z, c1z + 1):
            c = ix * nz + iz
            n += start[c + 1] - start[c]
    if n == 0:
        return np.zeros(0, np.int64)
    out = np.empty(n, np.int64)
    k = 0
    for ix in range(c0x, c1x + 1):
        for iz in range(c0z, c1z + 1):
            c = ix * nz + iz
            for j in range(start[c], start[c + 1]):
                out[k] = items[j]
                k += 1
    out.sort()
    m = 1
    for j in range(1, n):
        if out[j] != out[m - 1]:
            out[m] = out[j]
            m += 1
    return out[:m].copy()


def _ray_down_grid(P, ox, oz, cell, nx, nz, start, items, A, B, C, eps):
    """TriangleSet.ray_down over the grid, one point at a time: the same arithmetic as ray_down_pairs (same operation
    order, float64), and the same choice: the highest hit, among equal heights the last in cell order."""
    M = P.shape[0]
    y = np.full(M, np.nan)
    tri = np.full(M, -1, np.int64)
    for i in range(M):
        x = P[i, 0]
        z = P[i, 2]
        cx = int(np.floor((x - ox) / cell))
        cz = int(np.floor((z - oz) / cell))
        if cx < 0 or cx >= nx or cz < 0 or cz >= nz:
            continue
        c = cx * nz + cz
        best = 0.0
        bt = -1
        for j in range(start[c], start[c + 1]):
            t = items[j]
            x0 = A[t, 0]
            z0 = A[t, 2]
            e1x = B[t, 0] - x0
            e1z = B[t, 2] - z0
            e2x = C[t, 0] - x0
            e2z = C[t, 2] - z0
            det = e1x * e2z - e2x * e1z
            if not abs(det) > eps:
                continue
            px = x - x0
            pz = z - z0
            u = (px * e2z - e2x * pz) / det
            v = (e1x * pz - px * e1z) / det
            if not (u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9):
                continue
            hy = A[t, 1] + u * (B[t, 1] - A[t, 1]) + v * (C[t, 1] - A[t, 1])
            if not (hy <= P[i, 1] + 1e-9):
                continue
            if bt < 0 or hy >= best:
                best = hy
                bt = t
        if bt >= 0:
            y[i] = best
            tri[i] = bt
    return y, tri


if HAVE_NUMBA:
    _ray_down_grid = numba.njit(cache=True, nogil=True)(_ray_down_grid)
    _query_box_grid = numba.njit(cache=True, nogil=True)(_query_box_grid)


class TriangleSet:
    """World-space triangles with owner entity ids and optional grid."""

    def __init__(self, A, B, C, owner=None, grid_cell: float | None = 4.0):
        self.A = np.asarray(A, np.float64).reshape(-1, 3)
        self.B = np.asarray(B, np.float64).reshape(-1, 3)
        self.C = np.asarray(C, np.float64).reshape(-1, 3)
        self.owner = np.zeros(len(self.A), np.int64) if owner is None else np.asarray(owner, np.int64)
        n = np.cross(self.B - self.A, self.C - self.A)
        ln = np.linalg.norm(n, axis=1)
        self.N = n / np.where(ln == 0, 1, ln)[:, None]
        self.lo = np.minimum(np.minimum(self.A, self.B), self.C)
        self.hi = np.maximum(np.maximum(self.A, self.B), self.C)
        self.grid = Grid2D(self.lo, self.hi, grid_cell) if grid_cell and len(self.A) else None

    def __len__(self):
        return len(self.A)

    @classmethod
    def from_meshes(cls, items, grid_cell: float | None = 4.0):
        """items: iterable of (owner, world vertices (n,3), faces (m,3))."""
        A, B, C, O = [], [], [], []
        for owner, V, F in items:
            F = np.asarray(F, np.int64).reshape(-1, 3)
            if len(F) == 0:
                continue
            V = np.asarray(V, np.float64)
            A.append(V[F[:, 0]]); B.append(V[F[:, 1]]); C.append(V[F[:, 2]])
            O.append(np.full(len(F), owner))
        if not A:
            z = np.zeros((0, 3))
            return cls(z, z, z, np.zeros(0, np.int64), grid_cell)
        return cls(np.concatenate(A), np.concatenate(B), np.concatenate(C), np.concatenate(O), grid_cell)

    def candidates(self, p, r) -> np.ndarray:
        p = np.asarray(p, np.float64)
        if self.grid is None:
            m = np.all((self.lo <= p + r) & (self.hi >= p - r), axis=1)
            return np.flatnonzero(m)
        idx = self.grid.query_box(p[[0, 2]] - r, p[[0, 2]] + r)
        if len(idx):
            m = (self.lo[idx, 1] <= p[1] + r) & (self.hi[idx, 1] >= p[1] - r)
            idx = idx[m]
        return idx

    def closest(self, p, r: float = 6.0):
        """Nearest triangle within r of p: dict(d, q, region, tri, owner, normal) or None."""
        idx = self.candidates(p, r)
        if len(idx) == 0:
            return None
        Q, reg = closest_point_triangles(p, self.A[idx], self.B[idx], self.C[idx])
        d = np.linalg.norm(np.asarray(p) - Q, axis=1)
        k = int(np.argmin(d))
        if d[k] > r:
            return None
        return {"d": float(d[k]), "q": Q[k], "region": int(reg[k]), "tri": int(idx[k]),
                "owner": int(self.owner[idx[k]]), "face_normal": self.N[idx[k]]}

    def ray_down(self, P, owner_mask=None):
        """Highest hit at or below each P (vertical rays). Returns (y (M,), tri (M,)); nan / -1 = miss."""
        P = np.asarray(P, np.float64).reshape(-1, 3)
        M = len(P)
        y = np.full(M, np.nan)
        tri = np.full(M, -1, np.int64)
        if len(self.A) == 0 or M == 0:
            return y, tri
        if HAVE_NUMBA and owner_mask is None and self.grid is not None:
            g = self.grid
            return _ray_down_grid(P, float(g.origin[0]), float(g.origin[1]), g.cell, int(g.shape[0]), int(g.shape[1]),
                                  g.start.astype(np.int64), g.items.astype(np.int64), self.A, self.B, self.C, 1e-9)
        if self.grid is None:
            pi = np.repeat(np.arange(M), len(self.A))
            ti = np.tile(np.arange(len(self.A)), M)
        else:
            cells = self.grid.point_cells(P[:, [0, 2]])
            ok = cells >= 0
            s = np.where(ok, self.grid.start[np.maximum(cells, 0)], 0)
            e = np.where(ok, self.grid.start[np.maximum(cells, 0) + 1], 0)
            cnt = e - s
            pi = np.repeat(np.arange(M), cnt)
            k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            ti = self.grid.items[s[pi] + k]
        if owner_mask is not None and len(ti):
            keep = owner_mask[ti]
            pi, ti = pi[keep], ti[keep]
        if len(pi) == 0:
            return y, tri
        hy = ray_down_pairs(P[pi], self.A[ti], self.B[ti], self.C[ti])
        good = ~np.isnan(hy)
        pi, ti, hy = pi[good], ti[good], hy[good]
        order = np.lexsort((hy, pi))  # by point, then height: the last per point is the highest
        pi, ti, hy = pi[order], ti[order], hy[order]
        last = np.r_[pi[1:] != pi[:-1], True] if len(pi) else np.zeros(0, bool)
        y[pi[last]] = hy[last]
        tri[pi[last]] = ti[last]
        return y, tri


class BoundaryEdges:
    """Boundary edges of the walkable faces (|n_y| > UP_WALKABLE), vertices welded at WELD
    (state-spec.md 3.4, mesh-boundary method)."""

    def __init__(self, tris: TriangleSet, owner_filter=None, up: float = UP_WALKABLE):
        walk = np.abs(tris.N[:, 1]) > up
        if owner_filter is not None:
            walk &= owner_filter(tris.owner)
        idx = np.flatnonzero(walk)
        if len(idx) == 0:
            self.S0 = self.S1 = np.zeros((0, 3))
            self.grid = None
            return
        P = np.concatenate([tris.A[idx], tris.B[idx], tris.C[idx]])  # (3n,3)
        key = np.round(P / WELD).astype(np.int64)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(3, -1)
        a, b, c = inv[0], inv[1], inv[2]
        e = np.concatenate([np.stack([a, b], 1), np.stack([b, c], 1), np.stack([c, a], 1)])
        e.sort(1)
        e = e[e[:, 0] != e[:, 1]]
        ue, cnt = np.unique(e, axis=0, return_counts=True)
        bnd = ue[cnt == 1]
        V = uniq.astype(np.float64) * WELD
        self.S0, self.S1 = V[bnd[:, 0]], V[bnd[:, 1]]
        self.grid = Grid2D(np.minimum(self.S0, self.S1), np.maximum(self.S0, self.S1), 4.0) if len(bnd) else None

    def distance(self, q, r: float = 12.0) -> float:
        if self.grid is None:
            return float("inf")
        q = np.asarray(q, np.float64)
        idx = self.grid.query_box(q[[0, 2]] - r, q[[0, 2]] + r)
        if len(idx) == 0:
            return float("inf")
        return float(_seg_dist(q, self.S0[idx], self.S1[idx]).min())


def merge_ray(y1, t1, y2, t2):
    """Combine two ray_down results (static + dynamic): the higher hit wins. Returns (y, src, tri)
    with src 0 = first, 1 = second, -1 = miss."""
    a = np.where(np.isnan(y1), -np.inf, y1)
    b = np.where(np.isnan(y2), -np.inf, y2)
    second = b > a
    y = np.where(second, y2, y1)
    src = np.where(np.isnan(y), -1, second.astype(np.int64))
    tri = np.where(second, t2, t1)
    return y, src, tri

"""Multilayer FastReach support graph over the sector's placement footprint (not just the start/goal rectangle). Coplanar opposing banks near flap mechanisms get hypothetical, directional-lane crossings. Never count a guide node as an engine pass."""
from pathlib import Path
import json, sys, heapq, math, numpy as np
from ballrain import protocol as P
from ballrain.state import GameTracker
from ballrain.features import level_geometry
from ballrain.fastreach import FastReach

def build(tr, m, sector, cache):
    t = level_geometry(tr).tris
    props = [o for o in m['placements'] if o['sector'] == sector]
    src, dst = np.array(m['anchors'][sector - 1:sector + 1])
    scope = np.array([src, dst, *[o['pos'] for o in props]])
    lo = np.floor(scope[:, [0, 2]].min(0) - 25)
    hi = np.ceil(scope[:, [0, 2]].max(0) + 25)
    x, z = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing='ij')
    nx, nz = x.shape
    f = FastReach.fields_for_tracker(tr, cache_dir=cache)
    R = np.full((nx, nz, 5), np.nan)
    N = np.zeros_like(R)
    G = np.zeros_like(R, bool)
    walls = np.isin(t.owner, [k for k in np.unique(t.owner) if 'obstacle' in (tr.level.name(k) or '').lower()])
    ps = np.c_[x.ravel(), np.full(x.size, scope[:, 1].max() + 50), z.ravel()]
    H = np.concatenate([t.ray_down(c, owner_mask=~walls)[0] for c in np.array_split(ps, 128)]).reshape(nx, nz)
    W = np.isfinite(np.concatenate([t.ray_down(c, owner_mask=walls)[0] for c in np.array_split(ps, 128)])).reshape(nx, nz)
    W = W | np.roll(W, 1, 0) | np.roll(W, -1, 0) | np.roll(W, 1, 1) | np.roll(W, -1, 1)
    for i, j in np.ndindex(nx, nz):
        for k, (y, low, fl, ny) in enumerate(f.rest_layers(x[i, j], z[i, j])):
            if ny >= 0.62 and (not W[i, j]) and (scope[:, 1].min() - 10 < y < scope[:, 1].max() + 16):
                R[i, j, k] = y
                N[i, j, k] = ny
    proposals = []
    for obj in props:
        if obj['name'] != 'P_Modul_19':
            continue
        cx, cy, cz = obj['pos']
        box = (abs(x - cx) <= 18) & (abs(z - cz) <= 18)
        banks = np.isfinite(H) & (abs(H - cy) < 1.7) & ~W
        def bank(i,j):
            return 0<=i<nx and 0<=j<nz and bool(banks[i,j])
        for di, dj in [(1, 0), (0, 1)]:
            for i, j in zip(*np.where(box & banks)):
                if bank(i + di, j + dj) or not all((bank(i - n * di, j - n * dj) for n in range(1, 6))):
                    continue
                for n in range(3, 13):
                    if bank(i + n * di, j + n * dj):
                        if all((bank(i + (n + e) * di, j + (n + e) * dj) for e in range(1, 6))) and abs(H[i, j] - H[i + n * di, j + n * dj]) < 0.75:
                            for e in range(1, n):
                                u, v = (i + e * di, j + e * dj)
                                if not W[u, v]:
                                    R[u, v, 4] = cy + 2
                                    N[u, v, 4] = 1.0
                                    G[u, v, 4] = True
                            proposals.append([float(cx), float(cz), int(x[i, j]), int(z[i, j]), n, di, dj])
                        break
    cost = np.zeros_like(R)
    for dx, dz in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
        a = np.roll(R, (dx, dz), (0, 1))
        support = np.any(abs(R[:, :, :, None] - a[:, :, None, :]) < 1.6, axis=-1)
        cost += 0.55 * ~support
    cost += 1.5 * (1 - N) + 0.35 * G
    pts = np.argwhere(np.isfinite(R))

    def nearest(p):
        return tuple(pts[np.argmin((x[pts[:, 0], pts[:, 1]] - p[0]) ** 2 + (z[pts[:, 0], pts[:, 1]] - p[2]) ** 2 + (R[tuple(pts.T)] - p[1]) ** 2)])
    a = nearest(src)
    b = nearest(dst)
    D = {a: 0.0}
    par = {}
    heap = [(0.0, 0.0, a)]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    while heap:
        _, dist, k = heapq.heappop(heap)
        if k == b:
            break
        if D[k] != dist:
            continue
        i, j, l = k
        y = R[k]
        for di, dj in dirs:
            u, v = (i + di, j + dj)
            if not (0 <= u < nx and 0 <= v < nz):
                continue
            h = math.hypot(di, dj)
            for n in np.flatnonzero(np.isfinite(R[u, v])):
                m0 = (u, v, n)
                dy = R[m0] - y
                if dy > h * 0.9 + 0.12 or dy < -16:
                    continue
                if di and dj and (not (any(abs(R[u, j] - y) < 1.6) or any(abs(R[i, v] - y) < 1.6))):
                    continue
                nd = dist + h * (1 + cost[m0]) + abs(dy) * 0.8 + max(0, -dy - 1) * 0.8
                if nd < D.get(m0, 1e+99):
                    D[m0] = nd
                    par[m0] = k
                    heapq.heappush(heap, (nd + math.hypot(u - b[0], v - b[1]), nd, m0))
    end = b if b in D else min(D, key=lambda k: math.hypot(k[0] - b[0], k[1] - b[1]) + abs(R[k] - R[b]))
    ids = [end]
    while ids[-1] != a:
        ids.append(par[ids[-1]])
    ids = ids[::-1]
    guide = [[float(x[i, j]), R[i, j, k], float(z[i, j])] for i, j, k in ids]
    out = dict(guide=guide, connected=b in D, crossings=[n for n, k in enumerate(ids) if G[k]], gate_proposals=proposals, reachable=len(D), last=guide[-1])
    return out

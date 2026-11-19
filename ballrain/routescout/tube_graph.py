"""A floor graph with isolated bank gates and tangential tube edges. Pair and single-rail hypotheses have different costs. Never project through overlapping turns merely because their XZ cells coincide."""
from pathlib import Path
import sys, json, math, heapq, numpy as np, collections
from ballrain import protocol as P
from ballrain.state import GameTracker
from ballrain.features import level_geometry
from ballrain.fastreach import extract_rails

def build_tubes(tr, atlas, sector, flaps):
    geo = level_geometry(tr)
    t = geo.tris
    rt = geo.rails
    src, dst = np.array(atlas['anchors'][sector - 1:sector + 1])
    props = [o for o in atlas['placements'] if o['sector'] == sector]
    area = np.array([src, dst, *[o['pos'] for o in props]])[:, [0, 2]]
    lb = np.floor(area.min(0) - 25)
    ub = np.ceil(area.max(0) + 25)
    x, z = np.meshgrid(np.arange(lb[0], ub[0] + 1), np.arange(lb[1], ub[1] + 1), indexing='ij')
    X, Z = x.shape
    qq = np.c_[x.ravel(), np.full(x.size, max((p[1] for p in [src, dst, *[o['pos'] for o in props]])) + 60.0), z.ravel()]
    roof = np.isin(t.owner, np.unique(rt.owner))
    wall = np.isin(t.owner, [k for k in np.unique(t.owner) if 'obstacle' in (tr.level.name(k) or '').lower()])
    floor = np.concatenate([t.ray_down(q, owner_mask=~(wall | roof))[0] for q in np.array_split(qq, 128)]).reshape(x.shape) + 2
    w = np.isfinite(np.concatenate([t.ray_down(q, owner_mask=wall)[0] for q in np.array_split(qq, 128)])).reshape(x.shape)
    W = w.copy()
    for i, j in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        W |= np.roll(w, (i, j), (0, 1))
    floor[W] = np.nan
    gated = np.zeros_like(W)
    proposals = flaps['gate_proposals']
    for cx, cz, xx, zz, n, di, dj in proposals:
        i, j = map(int, [xx - lb[0], zz - lb[1]])
        y = floor[i, j]
        if np.isfinite(y):
            for e in range(1, n):
                floor[i + di * e, j + dj * e] = y
                gated[i + di * e, j + dj * e] = True
    pos = []
    tangent = []
    kind = []
    buckets = collections.defaultdict(list)

    def add(p, tan, typ):
        ix = len(pos)
        pos.append(np.array(p))
        tangent.append(tan)
        kind.append(typ)
        buckets[tuple(np.floor(np.array(p) / 2).astype(int))].append(ix)
        return ix
    floorids = {}
    for i, j in zip(*np.where(np.isfinite(floor))):
        floorids[i, j] = add([x[i, j], floor[i, j], z[i, j]], None, 'gate' if gated[i, j] else 'floor')
    base = len(pos)
    nr = 0
    for ck in np.unique(rt.owner):
        sel = rt.owner == ck
        if np.any(rt.lo[sel].min(0)[[0, 2]] > ub) or np.any(rt.hi[sel].max(0)[[0, 2]] < lb):
            continue
        seg, _, _ = extract_rails(rt.A[sel], rt.B[sel], rt.C[sel], rt.owner[sel])
        a = seg[:, :3]
        h = seg[:, 3:6] - a
        ll = np.linalg.norm(h, axis=1)
        u = h / np.maximum(1e-06, ll[:, None])
        mids = list()
        dedup = set()
        for i, seg0 in enumerate(seg):
            for v in np.linspace(0, 1, max(2, int(ll[i] / 0.5) + 1)):
                q = a[i] + v * h[i]
                up = np.array([0.0, 1.0, 0.0]) - u[i] * u[i, 1]
                up /= max(0.001, np.linalg.norm(up))
                single = q + (seg0[6] + 2) * up
                sid = tuple(np.round(single / 0.32).astype(int))
                if sid not in dedup:
                    dedup.add(sid)
                    add(single, u[i], 'single')
                factor = np.clip(((q - a) * u).sum(1) / np.maximum(ll, 1e-06), 0, 1)
                s = a + h * factor[:, None]
                side = s - q
                gap = np.linalg.norm(side, axis=1)
                cross = np.cross(u[i], side)
                N = cross / np.maximum(1e-06, np.linalg.norm(cross, axis=1)[:, None])
                N *= np.where(N[:, 1] < 0, -1, 1)[:, None]
                ok = (abs(u @ u[i]) > 0.91) & (gap > 3.2) & (gap < 4.25) & (abs(side[:, 1]) < 1.15) & (abs(side @ u[i]) < 0.9) & (N[:, 1] > 0.65)
                if not ok.any():
                    continue
                j = np.argmin(np.where(ok, abs(gap - 3.8) + 0.5 * abs(side @ u[i]), 1000000.0))
                p = 0.5 * (q + s[j]) + N[j] * math.sqrt(max(0, (2 + 0.5 * (seg0[6] + seg[j, 6])) ** 2 - (gap[j] / 2) ** 2))
                key = tuple(np.round(p / 0.35).astype(int))
                if key in dedup:
                    continue
                dedup.add(key)
                add(p, u[i], 'pair')
                nr += 1
    pos = np.array(pos)
    edges = [[] for _ in pos]

    def edge(i, j, c, both=True):
        edges[i].append((j, c))
        if both:
            edges[j].append((i, c))
    for (i, j), a0 in floorids.items():
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            b = floorids.get((i + di, j + dj))
            if b is None:
                for jmp in range(2, 7):
                    b = floorids.get((i + di * jmp, j + dj * jmp))
                    if b is not None:
                        dh = math.hypot(di, dj) * jmp
                        dy = pos[b, 1] - pos[a0, 1]
                        if -18 < dy < -1 and dh < 2 + 9 * math.sqrt(-dy / 10):
                            edge(a0, b, dh + 8 - dy, False)
                        break
                continue
            dy = pos[b, 1] - pos[a0, 1]
            dh = math.hypot(di, dj)
            if dy > 0.7 * dh + 0.1 or dy < -17:
                continue
            cost = dh + 0.6 * abs(dy) + 0.4 * sum(((i + di + u, j + dj + v) not in floorids for u, v in ((2, 0), (-2, 0), (0, 2), (0, -2))))
            edge(a0, b, cost, False)
    for i in range(base, len(pos)):
        p = pos[i]
        cell = np.floor(p / 2).astype(int)
        ux = tangent[i]
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for j in buckets.get(tuple(cell + [di, dj, dk]), []):
                        if j >= i:
                            continue
                        diff = pos[j] - p
                        dist = np.linalg.norm(diff)
                        cos = abs(diff @ ux) / max(0.0001, dist)
                        if j < base:
                            if dist <= 2.8 and abs(diff[1]) <= 1.8:
                                edge(i, j, dist + 0.4)
                        elif kind[j] != kind[i] and dist < 2.6 and (abs(ux @ tangent[j]) > 0.84) and (abs(diff[1]) < 1.8):
                            edge(i, j, dist + 7)
                        elif dist <= 2.4 and abs(ux @ tangent[j]) > 0.75 and (dist < 0.64 or (cos > 0.74 and abs(diff @ tangent[j]) > 0.69 * dist)):
                            edge(i, j, dist * (1.7 if kind[i] == 'single' else 1) + 1.6 * (1 - cos) * dist)
    a = np.linalg.norm(pos - src, axis=1).argmin()
    b = np.linalg.norm(pos - dst, axis=1).argmin()
    D = {a: 0.0}
    par = {}
    heap = [(0.0, a)]
    while heap:
        c, i = heapq.heappop(heap)
        if c != D[i]:
            continue
        if i == b:
            break
        for j, w in edges[i]:
            n = c + w
            if n < D.get(j, 1e+90):
                D[j] = n
                par[j] = i
                heapq.heappush(heap, (n, j))
    end = b if b in D else min(D, key=lambda i: np.linalg.norm(pos[i] - dst))
    ids = [end]
    while ids[-1] != a:
        ids.append(par[ids[-1]])
    ids = ids[::-1]
    GG = pos[ids]
    doc = dict(guide=GG.tolist(), kinds=[kind[i] for i in ids], connected=b in D, reachable=len(D), gate_proposals=proposals, rail_nodes=nr)
    doc.update(level=tr.level_no, sector=sector, static_key=tr.level.static_key())
    return doc

def groove(tr, doc):
    """Two vertically offset tubes can guide a sphere from the inside of a coil.
    Make an inner tangent-centre proposal, not a feasibility assertion. Outward
    pressure and tangential speed have to be checked with the engine.
    """
    import itertools
    g = dict(doc)
    P = np.array(g['guide'])
    K = np.array(g['kinds'])
    rt = level_geometry(tr).rails
    seg = extract_rails(rt.A, rt.B, rt.C, rt.owner)[0]
    a = seg[:, :3]
    h = seg[:, 3:6] - a
    ll = np.linalg.norm(h, axis=1)
    u = h / np.maximum(ll, 1e-08)[:, None]
    S = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1).cumsum()]
    new = P.copy()
    rad = np.zeros((len(P), 2))
    for kind, group in itertools.groupby(range(len(P)), key=lambda k: K[k]):
        ix = np.array(list(group))
        if kind != 'single' or S[ix[-1]] - S[ix[0]] < 12:
            continue
        xy = P[ix][:, [0, 2]]
        cx, cz, _ = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(ix))], (xy ** 2).sum(1), rcond=None)[0]
        center = np.array([cx, cz])
        radial = xy-center
        radii=np.linalg.norm(radial,axis=1)
        angles=np.unwrap(np.arctan2(radial[:,1],radial[:,0]))
        if np.ptp(angles)<np.pi or np.std(radii)>.25*max(.01,np.mean(radii)):continue
        for j in ix:
            p = P[j]
            q = a + np.clip(((p - a) * u).sum(1) / np.maximum(ll, 1e-08), 0, 1)[:, None] * h
            up = np.array([0.0, 1.0, 0.0]) - u * u[:, 1:2]
            up /= np.maximum(0.01, np.linalg.norm(up, axis=1)[:, None])
            lower = np.argmin(np.linalg.norm(q + up * (2 + seg[:, 6, None]) - p, axis=1))
            axis = u[lower]
            q0 = q[lower]
            q2 = a + np.clip(((q0 - a) * u).sum(1) / np.maximum(ll, 1e-08), 0, 1)[:, None] * h
            diff = q2 - q0
            gap = np.linalg.norm(diff, axis=1)
            ok = (diff[:, 1] > 2.6) & (gap < 4.5) & (abs(u @ axis) > 0.87) & (abs(diff @ axis) < 1)
            outward = q0[[0, 2]] - center
            outward /= max(0.001, np.linalg.norm(outward))
            rad[j] = outward
            if not ok.any():
                new[j, [0, 2]] -= 1.35 * outward
                new[j, 1] -= 0.48
                continue
            k = np.argmin(np.where(ok, np.linalg.norm(diff[:, [0, 2]], axis=1) + 0.3 * abs(gap - 3.65), 10000000.0))
            normal = np.cross(axis, diff[k])
            normal /= max(0.001, np.linalg.norm(normal))
            normal *= -1 if normal[[0, 2]] @ outward > 0 else 1
            new[j] = 0.5 * (q0 + q2[k]) + normal * np.sqrt(max(0, (2 + 0.5 * (seg[lower, 6] + seg[k, 6])) ** 2 - gap[k] ** 2 / 4))
    g.update(guide=new.tolist(), radial=rad.tolist())
    return g

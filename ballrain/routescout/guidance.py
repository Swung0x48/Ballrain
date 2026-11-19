"""Static floor proposals from the route-scout pickup planner; not traversability evidence."""

import math
import heapq
import numpy as np


def ribbon(pts):
    pts = np.array(pts, float)
    pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.005]]
    if len(pts) < 2:
        pts = np.vstack([pts, pts[-1] + [0.01, 0, 0]])
    seg = np.diff(pts, axis=0)
    length = np.linalg.norm(seg, axis=1)
    return pts, seg, length, np.r_[0.0, length.cumsum()]


def project(line, p, prior=None):
    x, v, l, s = line
    idx = (
        np.arange(len(l))
        if prior is None
        else np.flatnonzero((s[1:] >= prior - 7) & (s[:-1] <= prior + 14))
    )
    idx = idx if len(idx) else np.arange(len(l))
    f = np.clip(((p - x[idx]) * v[idx]).sum(1) / (l[idx] ** 2), 0, 1)
    d = np.linalg.norm(x[idx] + f[:, None] * v[idx] - p, axis=1)
    j = d.argmin()
    i = idx[j]
    return s[i] + f[j] * l[i], d[j]


def point(line, st):
    x, v, l, s = line
    i = min(len(l) - 1, max(0, np.searchsorted(s, st, side="right") - 1))
    return x[i] + v[i] * np.clip((st - s[i]) / l[i], 0, 1)


def local_line(t, p, q):
    lo = np.floor(np.minimum(p, q)[[0, 2]] - 11)
    hi = np.ceil(np.maximum(p, q)[[0, 2]] + 11)
    X, Z = np.meshgrid(
        np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij"
    )
    pts = np.c_[X.ravel(), np.full(X.size, max(p[1], q[1]) + 5.0), Z.ravel()]
    y = (
        np.concatenate(
            [
                t.ray_down(c, owner_mask=abs(t.N[:, 1]) > 0.5)[0]
                for c in np.array_split(pts, 1 + len(pts) // 512)
            ]
        ).reshape(X.shape)
        + 2
    )
    ok = np.isfinite(y)
    ids = np.argwhere(ok)
    if not len(ids):
        return None

    def near(a):
        return tuple(
            ids[
                np.argmin(
                    (X[ok] - a[0]) ** 2 + (Z[ok] - a[2]) ** 2 + (y[ok] - a[1]) ** 2
                )
            ]
        )

    a = near(p)
    b = near(q)
    cost = np.zeros(X.shape)
    for dx, dz in ((2, 0), (-2, 0), (0, 2), (0, -2)):
        cost += np.where(
            abs(np.where(ok, y, 1e6) - np.roll(np.where(ok, y, -1e6), (dx, dz), (0, 1)))
            < 1.7,
            0.0,
            0.85,
        )
    dist = {a: 0.0}
    par = {}
    queue = [(0.0, 0.0, a)]
    while queue:
        _, w, k = heapq.heappop(queue)
        if k == b:
            break
        if w != dist[k]:
            continue
        for di, dj in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            for jump in (1, 2, 3, 4, 5, 6):
                m = k[0] + di * jump, k[1] + dj * jump
                if not (0 <= m[0] < X.shape[0] and 0 <= m[1] < X.shape[1]):
                    break
                if not ok[m]:
                    continue
                dy = y[m] - y[k]
                ds = math.hypot(di, dj) * jump
                if -12 < dy < 0.85 * ds + 0.3:
                    nw = w + ds * (1 + cost[m]) + max(0.0, dy) * 2 + 4 * (jump - 1)
                    if nw < dist.get(m, 1e99):
                        dist[m] = nw
                        par[m] = k
                        heapq.heappush(queue, (nw + math.dist(m, b), nw, m))
                break
    if b not in dist:
        return None
    path = [b]
    while path[-1] != a:
        path.append(par[path[-1]])
    return ribbon([p, *[[X[k], y[k], Z[k]] for k in reversed(path)], q])

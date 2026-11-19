"""Sparse ball-centre graph sampled from fastreach rest layers (not a route witness)."""

from heapq import heappop, heappush
import math
import numpy as np
from ballrain.fastreach import RES, TL
from .guidance import ribbon, point, project


class FloorGuide:
    def __init__(self, field):
        self.field = field
        self.step = 4 * RES
        tiles = np.argwhere(field.tiles >= 0)
        ids = field.tiles[tuple(tiles.T)]
        self.columns = cols = {}
        self.penalty = {}
        for i in range(0, TL, 4):
            for j in range(0, TL, 4):
                ys = field.Rr[ids, i, j]
                flags = field.Rf[ids, i, j]
                for t, k in zip(*np.where(np.isfinite(ys))):
                    x, z = (tiles[t] * TL + [i, j]) // 4
                    node = int(x), int(z), float(ys[t, k])
                    normal = field.Rn[ids[t], i, j, k].astype(float) / 127.0
                    self.penalty[node] = float(3 * (normal @ normal))
                    cols.setdefault(node[:2], []).append((node, int(flags[t, k])))
        for (x, z), layers in cols.items():
            for n, _ in layers:
                # A ray on the lip of a catwalk is not a steerable centre line.
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    for radius, weight in ((1, 3.0), (2, 1.5), (3, 0.6)):
                        nearby = cols.get((x + dx * radius, z + dz * radius), ())
                        if not any(
                            (fl & 3) and abs(m[2] - n[2]) < 1 + radius * 1.5
                            for m, fl in nearby
                        ):
                            self.penalty[n] += weight
        self.nodes = [n for v in cols.values() for n, _ in v]
        self.xyz = np.array([self.world(n) for n in self.nodes])

    def world(self, n):
        x, z, y = n
        return [self.field.ox + x * self.step, y, self.field.oz + z * self.step]

    def path(self, start, goal, hazards=()):
        if goal is None or not self.nodes:
            return None

        def snap(p):
            return self.nodes[np.argmin(((self.xyz - p) ** 2).sum(1))]

        # A failing return point poisons only this proposal, never the base graph.
        danger = {}
        for p in hazards:
            delta = self.xyz - p
            for i in np.flatnonzero(np.einsum("ij,ij->i", delta, delta) < 64):
                danger[self.nodes[i]] = 12.0
        a, b = snap(start), snap(goal)

        def h(n):
            return math.hypot(n[0] - b[0], n[1] - b[1]) * self.step + 0.2 * abs(
                n[2] - b[2]
            )

        costs, parents, queue, closest = {a: 0.0}, {}, [(h(a), 0.0, a)], a
        while queue:
            _, cost, n = heappop(queue)
            if cost != costs[n]:
                continue
            if h(n) < h(closest):
                closest = n
            if n == b:
                break
            for dx, dz in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                for gap in range(1, 10):
                    ds = self.step * math.hypot(dx, dz) * gap
                    linked = False
                    for m, flags in self.columns.get(
                        (n[0] + dx * gap, n[1] + dz * gap), ()
                    ):
                        dy = m[2] - n[2]
                        if not -22 <= dy <= 1.1 + 0.75 * ds / gap:
                            continue
                        linked = True
                        w = (
                            cost
                            + ds * (self.penalty[m] + danger.get(m, 0.0))
                            + math.hypot(ds, dy)
                            + 8 * max(0, abs(dy) - ds * 0.6)
                            + 24 * (gap - 1)
                            + (5 if flags & 4 or flags & 3 == 0 else 0)
                        )
                        # Rest samples on consecutive columns can belong to opposite
                        # sides of a step. Ask the ground marcher whether the ascent
                        # actually stays on this layer before preferring a short cut.
                        if dy > 0.4 and gap == 1 and w < costs.get(m, math.inf):
                            march = self.field.march(
                                self.world(n), [dx, dz], y=n[2], lmax=ds
                            )
                            if (
                                min(march["edge_s"], march["obst_s"]) < ds - 0.25
                                or abs(march["y_last"] - m[2]) > 1
                            ):
                                w += 35.0
                        if w < costs.get(m, math.inf):
                            costs[m], parents[m] = w, n
                            heappush(queue, (w + h(m), w, m))
                    if linked:
                        break
        route = [closest]
        while route[-1] != a:
            route.append(parents[route[-1]])
        pts = [start, *(self.world(n) for n in reversed(route))]
        return ribbon(pts + ([goal] if closest == b else []))


def waypoint(field, rules, p, path, ball, look, rng, explore, commit=False):
    """Score short local marches. Permit uncertain drops if the guide descends."""
    target = point(path, project(path, p)[0] + look) if path is not None else p
    y, _, flag = field.support(p, tol=2.0)
    if flag < 0:
        return target
    desired = (target - p)[[0, 2]]
    desired /= max(1.0, np.linalg.norm(desired))
    choices = []
    directions = [[math.cos(a), math.sin(a)] for a in np.arange(8) * math.pi / 4]
    if np.linalg.norm(desired) > 0.01:
        directions.insert(0, desired / np.linalg.norm(desired))
    for d in map(np.asarray, directions):
        m = field.march(p, d, y=y, lmax=look + 1, ball_type=ball)
        # A rest-layer discontinuity can be a mountable lip or a moving module.
        # Some branches deliberately carry speed across it; don't veto it on every replay.
        obstacle = look if commit and m["obst_h"] < 3 else m["obst_s"]
        safe = min(
            look, obstacle, m["edge_s"] if target[1] >= y - 2 and not commit else look
        )
        end = p.copy()
        end[[0, 2]] += d * max(0.5, safe)
        score = (
            safe * (d @ desired + 0.45) - 2 * (look - safe) + rng.uniform(0, explore)
        )
        if rules.gate(p, end, ball):
            score -= 50
        choices.append((score, end))
    return max(choices, key=lambda c: c[0])[1]

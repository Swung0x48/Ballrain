"""Reference path for progress features and SR rewards.

For now the path is the ordered polyline start -> checkpoints -> airship (the "sector line"):
  start      ResetPoints[0] position (fallback: PS_FourFlames_MF)
  checkpoint Checkpoints[k] matrix position, k = 0..n-1 (sector k+1 ends there)
  end        PE_Balloon_Platform position (fallback: PE_Balloon_MF)
Node s-1 -> node s is the segment of sector s. TAS-trajectory reference lines can replace the nodes
later (ReferencePath(points, sector_of_node)).
"""
from __future__ import annotations

import numpy as np

END_NAMES = ("PE_Balloon_Platform", "PE_Balloon_MF", "PE_Levelende")
START_NAMES = ("PS_FourFlames_MF",)


def _matrix_pos(v):
    if isinstance(v, np.ndarray) and v.shape == (4, 4):
        return v[3, :3].astype(np.float64)
    return None


class ReferencePath:
    def __init__(self, points, node_sector=None):
        self.P = np.asarray(points, np.float64).reshape(-1, 3)
        if len(self.P) < 2:
            raise ValueError("a reference path needs at least two points")
        seg = self.P[1:] - self.P[:-1]
        self.seg = seg
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.r_[0.0, np.cumsum(self.seg_len)]
        self.length = float(self.cum[-1])
        # sector of each segment: segment i belongs to the sector that ends at node i+1
        self.seg_sector = np.arange(1, len(self.P)) if node_sector is None else np.asarray(node_sector[1:])

    @classmethod
    def from_tracker(cls, tracker):
        arr, lvl = tracker.arrays, tracker.level
        pts = []
        start = _matrix_pos(arr.get("ResetPoints", 0, 0))
        if start is None:
            for n in START_NAMES:
                start = lvl.entity_position(n)
                if start is not None:
                    break
        if start is None:
            return None
        pts.append(start)
        for r in range(arr.rows("Checkpoints")):
            p = None
            for c in range(2):
                p = _matrix_pos(arr.get("Checkpoints", r, c))
                if p is not None:
                    break
            if p is not None:
                pts.append(p)
        end = None
        for n in END_NAMES:
            end = lvl.entity_position(n)
            if end is not None:
                break
        if end is not None:
            pts.append(end)
        if len(pts) < 2:
            return None
        return cls(pts)

    def project(self, p, sector: int | None = None, window: int = 1):
        """Project p onto the path. With a sector, only segments of sectors sector-window ..
        sector+window are considered (so the projection cannot jump to a far part of the level).
        Returns dict(s = arc length, seg, t, point, tangent (horizontal unit), lateral distance)."""
        p = np.asarray(p, np.float64)
        n = len(self.seg)
        if sector is None:
            cand = np.arange(n)
        else:
            cand = np.flatnonzero(np.abs(self.seg_sector - sector) <= window)
            if len(cand) == 0:
                cand = np.array([int(np.clip(sector - 1, 0, n - 1))])
        A = self.P[cand]
        d = self.seg[cand]
        L2 = (d * d).sum(1)
        t = np.clip(((p - A) * d).sum(1) / np.where(L2 == 0, 1, L2), 0, 1)
        Q = A + d * t[:, None]
        dist = np.linalg.norm(p - Q, axis=1)
        k = int(np.argmin(dist))
        i = int(cand[k])
        tan = self.seg[i].copy()
        tan[1] = 0
        ln = np.linalg.norm(tan)
        tan = tan / ln if ln > 0 else np.array([0.0, 0.0, 1.0])
        return {"s": float(self.cum[i] + t[k] * self.seg_len[i]), "seg": i, "t": float(t[k]),
                "point": Q[k], "tangent": tan, "dist": float(dist[k])}

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0, self.length))
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg) - 1))
        t = (s - self.cum[i]) / self.seg_len[i] if self.seg_len[i] > 0 else 0.0
        return self.P[i] + self.seg[i] * t

    def next_nodes(self, seg: int, k: int) -> np.ndarray:
        """The next k path nodes after segment seg (the end node repeated past the end)."""
        idx = np.minimum(np.arange(seg + 1, seg + 1 + k), len(self.P) - 1)
        return self.P[idx]

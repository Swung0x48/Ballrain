"""Reference paths for progress features and SR rewards (docs/observations.md, "参考路径").

Two sources, one class (ReferencePath: points P[i] with a sector label per point, arc length cum[i]):
  tas         a TAS trajectory of the level (the fastest completing record of the dataset), resampled by arc
              length; a small committed asset benchmarks/refpaths/L<nn>.npz built by
              scripts/build_refpaths.py (build_tas_path). The default (FeatureConfig.path_source="tas").
  checkpoints the polyline start -> checkpoints -> airship (the "sector line"), from the level message:
              start      ResetPoints[0] position (fallback: PS_FourFlames_MF)
              checkpoint Checkpoints[k] matrix position, k = 0..n-1 (sector k+1 ends there)
              end        the airship platform (LevelData.physical_position("PE_Balloon_Platform"): world bbox
                         centre of its collision meshes; the ENTS matrix has it parked at the origin)
A TAS asset is used only when its stored start and checkpoint positions match the running level (so the
toy level of ballrain.fakegame, or a changed level, falls back to the polyline); levels without a
completing TAS record (9, 13 in tas-v1) have no asset and also use the polyline.

Projection (PathFollower) keeps continuity: it searches a window [s_prev - back, s_prev + ahead] of arc
length around the previous projection, restricted to the arc range of the current sector (plus a margin).
Fallback ("reacquire"): when the best point of the whole sector range is closer than the windowed best by
more than reacquire units (the ball left the line, e.g. fell to a lower part of the route), or when there is
no previous projection (episode start, respawn, TAS hand-over) or the window is empty, the sector-range best
is used and the result is flagged jump=True (the SR progress reward pays nothing for that step).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

START_NAMES = ("PS_FourFlames_MF",)
AIRSHIP_NAMES = ("PE_Balloon_Platform", "PE_Balloon_MF")
REFPATH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "refpaths"
REFPATH_VERSION = 1
MATCH_TOL = 1.0  # stored start / checkpoint positions must match the level within this (units)


def _matrix_pos(v):
    if isinstance(v, np.ndarray) and v.shape == (4, 4):
        return v[3, :3].astype(np.float64)
    return None


def level_anchors(tracker) -> dict:
    """Start, checkpoint and airship positions of the running level (None where unknown)."""
    arr, lvl = tracker.arrays, tracker.level
    start = _matrix_pos(arr.get("ResetPoints", 0, 0))
    if start is None:
        for n in START_NAMES:
            start = lvl.entity_position(n)
            if start is not None:
                break
    cps = []
    for r in range(arr.rows("Checkpoints")):
        p = None
        for c in range(2):
            p = _matrix_pos(arr.get("Checkpoints", r, c))
            if p is not None:
                break
        if p is not None:
            cps.append(p)
    end = None
    for n in AIRSHIP_NAMES:
        end = lvl.physical_position(n)
        if end is not None:
            break
    return {"start": start, "checkpoints": cps, "airship": end}


class ReferencePath:
    """Polyline P (n,3) with a sector label per point. Segment i (P[i] -> P[i+1]) belongs to sector
    seg_sector[i] = point_sector[i+1]."""

    def __init__(self, points, point_sector=None, source: str = "custom", meta: dict | None = None):
        self.P = np.asarray(points, np.float64).reshape(-1, 3)
        if len(self.P) < 2:
            raise ValueError("a reference path needs at least two points")
        self.point_sector = (np.r_[1, np.arange(1, len(self.P))] if point_sector is None
                             else np.asarray(point_sector, np.int64))
        self.seg = self.P[1:] - self.P[:-1]
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.L2 = np.where(self.seg_len == 0, 1.0, self.seg_len ** 2)
        self.cum = np.r_[0.0, np.cumsum(self.seg_len)]
        self.length = float(self.cum[-1])
        self.seg_sector = self.point_sector[1:]
        self.n_sectors = int(self.point_sector.max())
        self.source = source
        self.meta = dict(meta or {})
        # arc range of every sector: from the start of its first segment to the end of its last segment
        self.sector_range = {}
        for s in np.unique(self.seg_sector):
            idx = np.flatnonzero(self.seg_sector == s)
            self.sector_range[int(s)] = (float(self.cum[idx[0]]), float(self.cum[idx[-1] + 1]))
        # horizontal tangent per segment, smoothed over +-tangent_span of arc length
        self._tan = self._tangents(self.meta.get("tangent_span", 0.0))

    # construction ---------------------------------------------------------------------------
    @classmethod
    def from_tracker(cls, tracker):
        """The checkpoint polyline (start -> checkpoints -> airship)."""
        a = level_anchors(tracker)
        if a["start"] is None:
            return None
        pts = [a["start"]] + list(a["checkpoints"])
        if a["airship"] is not None:
            pts.append(a["airship"])
        if len(pts) < 2:
            return None
        return cls(pts, np.r_[1, np.arange(1, len(pts))], source="checkpoints")

    @classmethod
    def load(cls, path) -> "ReferencePath":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(z["points"].astype(np.float64), z["sector"].astype(np.int64), source="tas", meta=meta)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, points=self.P.astype(np.float32), sector=self.point_sector.astype(np.uint8),
                            meta=np.array(json.dumps(self.meta, sort_keys=True)))

    def _tangents(self, span: float) -> np.ndarray:
        n = len(self.seg)
        mid = 0.5 * (self.cum[:-1] + self.cum[1:])
        if span > 0:
            a = np.array([self.point_at(s) for s in np.clip(mid - span, 0, self.length)])
            b = np.array([self.point_at(s) for s in np.clip(mid + span, 0, self.length)])
            d = b - a
        else:
            d = self.seg.copy()
        d[:, 1] = 0
        ln = np.linalg.norm(d, axis=1)
        out = np.zeros((n, 3))
        last = np.array([0.0, 0.0, 1.0])
        for i in range(n):  # carry the last good direction over vertical / zero-length parts
            if ln[i] > 1e-6:
                last = d[i] / ln[i]
            out[i] = last
        return out

    # queries ----------------------------------------------------------------------------------
    def arc_window(self, sector: int | None, margin: float) -> tuple[float, float]:
        if sector is None or not self.sector_range:
            return 0.0, self.length
        s = int(np.clip(sector, min(self.sector_range), max(self.sector_range)))
        if s not in self.sector_range:
            return 0.0, self.length
        lo, hi = self.sector_range[s]
        return max(0.0, lo - margin), min(self.length, hi + margin)

    def _segments(self, s_lo: float, s_hi: float) -> np.ndarray:
        i0 = int(np.clip(np.searchsorted(self.cum, s_lo, side="right") - 1, 0, len(self.seg) - 1))
        i1 = int(np.clip(np.searchsorted(self.cum, s_hi, side="left"), i0 + 1, len(self.seg)))
        return np.arange(i0, i1)

    def nearest(self, p, cand: np.ndarray) -> dict:
        p = np.asarray(p, np.float64)
        A = self.P[cand]
        d = self.seg[cand]
        t = np.clip(((p - A) * d).sum(1) / self.L2[cand], 0, 1)
        Q = A + d * t[:, None]
        dist = np.linalg.norm(p - Q, axis=1)
        k = int(np.argmin(dist))
        i = int(cand[k])
        return {"s": float(self.cum[i] + t[k] * self.seg_len[i]), "seg": i, "t": float(t[k]), "point": Q[k],
                "tangent": self._tan[i], "dist": float(dist[k]), "sector": int(self.seg_sector[i])}

    def project(self, p, sector: int | None = None, s_prev: float | None = None, back: float = 10.0,
                ahead: float = 30.0, margin: float = 20.0, reacquire: float = 5.0) -> dict:
        """Projection of p with continuity (see the module docstring). Returns dict(s, seg, t, point,
        tangent (horizontal unit), dist, sector, jump, mode)."""
        lo, hi = self.arc_window(sector, margin)
        glob = self.nearest(p, self._segments(lo, hi))
        if s_prev is None:
            glob.update(jump=True, mode="init")
            return glob
        wlo, whi = max(lo, s_prev - back), min(hi, s_prev + ahead)
        if whi <= wlo:
            glob.update(jump=True, mode="window_empty")
            return glob
        win = self.nearest(p, self._segments(wlo, whi))
        if win["dist"] <= glob["dist"] + reacquire:
            win.update(jump=False, mode="window")
            return win
        glob.update(jump=True, mode="reacquire")
        return glob

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0, self.length))
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg) - 1))
        t = (s - self.cum[i]) / self.seg_len[i] if self.seg_len[i] > 0 else 0.0
        return self.P[i] + self.seg[i] * t

    def points_ahead(self, s: float, k: int, spacing: float) -> np.ndarray:
        """k points at arc length s + spacing, s + 2 spacing, ... (the end point repeated past the end)."""
        return np.array([self.point_at(s + spacing * (j + 1)) for j in range(k)]).reshape(k, 3)

    def next_nodes(self, seg: int, k: int) -> np.ndarray:
        """The next k path points after segment seg (the end point repeated past the end)."""
        idx = np.minimum(np.arange(seg + 1, seg + 1 + k), len(self.P) - 1)
        return self.P[idx]


class PathFollower:
    """Stateful projection along one ReferencePath (one per episode; reset() at every episode start,
    respawn and hand-over)."""

    def __init__(self, path: ReferencePath, **kw):
        self.path = path
        self.kw = kw
        self.s_prev: float | None = None

    def reset(self) -> None:
        self.s_prev = None

    def update(self, p, sector: int | None) -> dict:
        r = self.path.project(p, sector, self.s_prev, **self.kw)
        self.s_prev = r["s"]
        return r


# ------------------------------------------------------------------------------- TAS reference lines

def asset_path(level: int, root=None) -> Path:
    return Path(root or REFPATH_DIR) / f"L{int(level):02d}.npz"


def anchors_match(meta: dict, anchors: dict, tol: float = MATCH_TOL) -> bool:
    if anchors["start"] is None or meta.get("start") is None:
        return False
    if np.linalg.norm(np.asarray(meta["start"]) - anchors["start"]) > tol:
        return False
    cps = meta.get("checkpoints", [])
    if len(cps) != len(anchors["checkpoints"]):
        return False
    return all(np.linalg.norm(np.asarray(a) - b) <= tol for a, b in zip(cps, anchors["checkpoints"]))


_ASSET_CACHE: dict = {}


def load_tas_path(level: int, tracker=None, root=None) -> ReferencePath | None:
    """The committed TAS reference line of a level, or None (no asset, or the tracker's level does not match
    the stored start / checkpoints)."""
    p = asset_path(level, root)
    if not p.is_file():
        return None
    key = (str(p), p.stat().st_mtime_ns)
    path = _ASSET_CACHE.get(key)
    if path is None:
        path = ReferencePath.load(p)
        _ASSET_CACHE[key] = path
    if tracker is not None and not anchors_match(path.meta, level_anchors(tracker)):
        return None
    return path


def build_tas_path(pos, sector, delta_ms, spacing: float = 0.5, time_sigma_ms: float = 40.0,
                   tangent_span: float = 2.0, meta: dict | None = None) -> ReferencePath:
    """Reference line from a trajectory: per-frame ball positions (frames with control only; nan rows are
    bridged), sector per frame and frame deltas. The positions are smoothed in time (Gaussian, sigma
    time_sigma_ms; removes bounce jitter so that a bouncing ball does not add arc length), then resampled at
    `spacing` units of arc length. The sector of a resampled point is the sector of the frame it came from."""
    pos = np.asarray(pos, np.float64)
    sector = np.asarray(sector, np.int64)
    ok = ~np.isnan(pos).any(1)
    pos, sector = pos[ok], sector[ok]
    t = np.cumsum(np.asarray(delta_ms, np.float64)[ok])
    if time_sigma_ms > 0:
        # Gaussian in time (frames may have different deltas): weights on the time axis
        sm = np.empty_like(pos)
        w = 4 * time_sigma_ms
        j0 = np.searchsorted(t, t - w)
        j1 = np.searchsorted(t, t + w, side="right")
        for i in range(len(t)):
            a, b = j0[i], j1[i]
            g = np.exp(-0.5 * ((t[a:b] - t[i]) / time_sigma_ms) ** 2)
            sm[i] = (g[:, None] * pos[a:b]).sum(0) / g.sum()
        sm[0], sm[-1] = pos[0], pos[-1]
        pos = sm
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(d)]
    keep = np.r_[True, d > 1e-9]
    cum, pos, sector = cum[keep], pos[keep], sector[keep]
    s_new = np.arange(0.0, cum[-1], spacing)
    if cum[-1] - s_new[-1] > 1e-6:
        s_new = np.r_[s_new, cum[-1]]
    P = np.stack([np.interp(s_new, cum, pos[:, k]) for k in range(3)], 1)
    idx = np.clip(np.searchsorted(cum, s_new, side="right") - 1, 0, len(cum) - 1)
    sec = np.maximum.accumulate(sector[idx])  # sectors never go back along the line
    m = dict(meta or {})
    m.update(version=REFPATH_VERSION, spacing=spacing, time_sigma_ms=time_sigma_ms, tangent_span=tangent_span)
    return ReferencePath(P, sec, source="tas", meta=m)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

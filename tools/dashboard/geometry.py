"""Level-1 top-view geometry for the dashboard maps, built once and cached on disk.

Rebuilding the level needs the TAS record replayed through a GameTracker, so the result is saved to
runtime/dashboard/cache/ and reused by every later regeneration and restart. The triangles are merged
into outlines (boundary-edge loops per object, x-z plane), so the browser draws a few hundred polygons
instead of ~18k triangles. Adapted from runtime/viz/viz_features.py (OfflineSession, replay_to).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECORD = "SR01_1_14_457_264"
REC_DIR = ROOT / "runtime" / "datasets" / "tas-v1" / RECORD
TAS_CACHE = ROOT / "runtime" / "viz" / f"cache_{RECORD}_f3.npz"
GEOM_VERSION = 2


class OfflineSession:
    """Feeds stored observations to a GameTracker the way AgentSession._ingest does."""

    def __init__(self, rec: Path, level: int):
        from ballrain import protocol as P
        from ballrain.state import GameTracker
        self.P = P
        self.tracker = GameTracker(level)
        self.full, _ = P.parse_level_message((rec / "level.msg").read_bytes())
        inc = rec / "level.msg.inc"
        self.inc = P.parse_level_stream(inc.read_bytes())[0] if inc.exists() else []
        self.inc_i = 0
        self.loaded = False
        z = np.load(rec / "raw.npz")
        self.blob, self.off = z["obs_bytes"].tobytes(), z["obs_offsets"]
        self.n = len(self.off) - 1

    def obs(self, i):
        return self.P.parse_observation(self.blob[self.off[i]:self.off[i + 1]])

    def ingest(self, o):
        from ballrain.protocol import Ev
        start = o.has_event(Ev.StartLevel)
        if start or (not self.loaded and o.frames_done >= self.full.frames_done):
            self.tracker.merge_level(self.full)
            self.loaded = True
        if self.loaded and (len(o.phys) or start):
            while self.inc_i < len(self.inc) and self.inc[self.inc_i].frames_done <= o.frames_done:
                self.tracker.merge_level(self.inc[self.inc_i])
                self.inc_i += 1
        self.tracker.update(o)


def _outlines(A, B, C, owner) -> list[list[float]]:
    """Boundary-edge loops (x-z) of each owner's triangle mesh; flat [x0, z0, x1, z1, ...] rings."""
    rings = []
    for o in np.unique(owner):
        m = owner == o
        V = np.concatenate([A[m], B[m], C[m]])
        key = np.round(V, 3)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        xz = np.zeros((inv.max() + 1, 2))
        xz[inv] = V[:, [0, 2]]
        inv = inv.reshape(3, -1).T  # triangle -> vertex ids
        cnt = defaultdict(int)
        for a, b, c in inv:
            for u, v in ((a, b), (b, c), (c, a)):
                cnt[(min(u, v), max(u, v))] += 1
        nb = defaultdict(list)
        for (u, v), k in cnt.items():
            if k == 1:
                nb[u].append(v)
                nb[v].append(u)
        used = set()
        for s in list(nb):
            for t in nb[s]:
                e = (min(s, t), max(s, t))
                if e in used:
                    continue
                loop = [s]
                prev, cur = s, t
                used.add(e)
                while cur != s and len(loop) < 100000:
                    loop.append(cur)
                    nxt = None
                    for w in nb[cur]:
                        e2 = (min(cur, w), max(cur, w))
                        if w != prev and e2 not in used:
                            nxt = w
                            break
                    if nxt is None:
                        break
                    used.add((min(cur, nxt), max(cur, nxt)))
                    prev, cur = cur, nxt
                if len(loop) >= 3:
                    P = xz[loop]
                    # drop collinear points
                    keep = [0]
                    for i in range(1, len(P) - 1):
                        a, b, c = P[keep[-1]], P[i], P[i + 1]
                        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-4:
                            keep.append(i)
                    keep.append(len(P) - 1)
                    P = P[keep]
                    if len(P) >= 3:
                        rings.append([round(float(v), 2) for v in P.ravel()])
    return rings


def _build(n_frames: int) -> dict:
    from ballrain.features import level_geometry
    s = OfflineSession(REC_DIR, 1)
    for i in range(min(n_frames, s.n)):
        s.ingest(s.obs(i))
    geo = level_geometry(s.tracker)
    T = geo.tris
    rail = np.isin(T.owner, np.unique(geo.rails.owner)) if len(geo.rails) else np.zeros(len(T), bool)
    ny = T.N[:, 1]
    out = {}
    for name, m in (("walk", (ny > 0.5) & ~rail), ("steep", (ny <= 0.5) & (ny > 0.05) & ~rail),
                    ("rail", rail & (ny > 0.05))):
        out[name] = _outlines(T.A[m], T.B[m], T.C[m], T.owner[m])
    return out


class Level:
    """TAS line (pos / sector / ball_nav per frame) + level outlines, held in memory."""

    def __init__(self, cache_dir: Path, log=print):
        c = np.load(TAS_CACHE, allow_pickle=True)
        self.pos = np.asarray(c["pos"], dtype=np.float64)
        self.sector = np.asarray(c["sector"])
        self.nav = np.asarray(c["ball_nav"])
        self.n = len(self.pos)
        gp = cache_dir / f"geom_{RECORD}_v{GEOM_VERSION}.json"
        if gp.exists():
            self.outlines = json.loads(gp.read_text(encoding="utf-8"))
        else:
            log("geometry: replaying the TAS record once ...")
            self.outlines = _build(self.n - 5)
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = gp.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.outlines), encoding="utf-8")
            tmp.replace(gp)
            log(f"geometry: cached {gp.name} ({sum(len(v) for v in self.outlines.values())} outlines)")
        ch = np.flatnonzero(np.diff(self.sector) > 0) + 1
        self.checkpoints = [int(i) for i in ch[ch > 936]]
        self.ok = np.flatnonzero(self.nav > 0)

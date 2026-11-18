"""Top-view geometry and TAS line of a level for the dashboard maps, built once per TAS record and cached on disk.

Rebuilding a level needs its TAS record (runtime/datasets/tas-v1/<record>) replayed through a GameTracker, so the
result is saved to runtime/dashboard/cache/ and reused by every later regeneration and restart. One pass gives both
the outlines (triangles merged into boundary-edge loops per object, x-z plane, so the browser draws a few hundred
polygons instead of ~18k triangles) and the TAS line (ball position, sector and ball_nav per frame). Level 1 can
still read the older runtime/viz feature cache for its line (same values). Adapted from runtime/viz/viz_features.py.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# default start record per level: mirrors scripts/train_m2.py RECORDS (a run's --record overrides; test_dashboard_levels)
LEVEL_RECORDS = {1: "SR01_1_14_457_264", 2: "SR02_1_28_992_264"}
RECORD = LEVEL_RECORDS[1]
DATASET = ROOT / "runtime" / "datasets" / "tas-v1"
RECORD_RE = re.compile(r"^SR(\d\d)_\d+_\d\d_\d{3}_\d{3}$")
GEOM_VERSION = 2
LINE_VERSION = 1
START_BACK = 1500  # ballrain.curriculum.CurriculumConfig.start_back: the first frontier, before any log line


def record_ok(record, level: int | None = None) -> bool:
    """A TAS record name as the dataset spells it (SR<level>_...), of `level` when given, present on this machine."""
    m = RECORD_RE.match(str(record or ""))
    if not m or (level is not None and int(m.group(1)) != int(level)):
        return False
    return (DATASET / str(record) / "raw.npz").is_file()


def viz_cache(record: str) -> Path:
    return ROOT / "runtime" / "viz" / f"cache_{record}_f3.npz"


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


def _split(geo) -> dict:
    T = geo.tris
    rail = np.isin(T.owner, np.unique(geo.rails.owner)) if len(geo.rails) else np.zeros(len(T), bool)
    ny = T.N[:, 1]
    out = {}
    for name, m in (("walk", (ny > 0.5) & ~rail), ("steep", (ny <= 0.5) & (ny > 0.05) & ~rail),
                    ("rail", rail & (ny > 0.05))):
        out[name] = _outlines(T.A[m], T.B[m], T.C[m], T.owner[m])
    return out


def replay_record(record: str, outlines: bool = True) -> tuple[dict, dict | None]:
    """One pass over the record's observations: the TAS line per observation (index = frames_done; pos is NaN where
    the ball is not found, sector 0 before the level is loaded, ball_nav as the tracker flags it) and, when asked,
    the level outlines as the tracker knows them 5 observations before the end (as built for level 1 before)."""
    from ballrain.features import level_geometry
    level = int(record[2:4])
    s = OfflineSession(DATASET / record, level)
    pos, sec, nav = np.full((s.n, 3), np.nan), np.zeros(s.n), np.zeros(s.n)
    geo = None
    for i in range(s.n):
        o = s.obs(i)
        s.ingest(o)
        t, b = s.tracker, o.ball
        if s.loaded and b is not None and bool(b["found"]) and int(b["type"]) in (0, 1, 2):
            pos[i] = np.asarray(b["pos"], np.float64)
        sec[i] = t.sector if s.loaded else 0
        nav[i] = float(t.flags.ball_nav)
        if outlines and i == s.n - 6:
            geo = _split(level_geometry(t))
    if outlines and geo is None:
        geo = _split(level_geometry(s.tracker))
    return {"pos": pos, "sector": sec, "ball_nav": nav}, geo


def _write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    tmp.replace(p)


class Level:
    """One level as its TAS record shows it: the TAS line (pos / sector / ball_nav per frame), the outlines, the
    checkpoint frames, the record's own SR clock and the benchmark thresholds; held in memory, cached on disk."""

    def __init__(self, cache_dir: Path, log=print, record: str = RECORD):
        import data as D
        if not record_ok(record):
            raise ValueError("unknown TAS record")
        self.record = record
        self.level = int(record[2:4])
        cache_dir.mkdir(parents=True, exist_ok=True)
        gp = cache_dir / f"geom_{record}_v{GEOM_VERSION}.json"
        lp = cache_dir / f"line_{record}_v{LINE_VERSION}.npz"
        line = geo = None
        if viz_cache(record).is_file():
            c = np.load(viz_cache(record), allow_pickle=False)
            line = {k: c[k] for k in ("pos", "sector", "ball_nav")}
        elif lp.is_file():
            c = np.load(lp, allow_pickle=False)
            line = {k: c[k] for k in ("pos", "sector", "ball_nav")}
        if gp.exists():
            geo = json.loads(gp.read_text(encoding="utf-8"))
        if line is None or geo is None:
            log(f"geometry: replaying the level {self.level} TAS record once ...")
            line2, geo2 = replay_record(record, outlines=geo is None)
            if line is None:
                line = line2
                tmp = lp.with_name(lp.stem + ".tmp.npz")
                np.savez_compressed(tmp, **line)
                tmp.replace(lp)
            if geo is None:
                geo = geo2
                _write_json(gp, geo)
                log(f"geometry: cached level {self.level} ({sum(len(v) for v in geo.values())} outlines)")
        self.outlines = geo
        self.pos = np.asarray(line["pos"], dtype=np.float64)
        self.sector = np.asarray(line["sector"])
        self.nav = np.asarray(line["ball_nav"])
        self.n = len(self.pos)
        self.ok = np.flatnonzero(self.nav > 0)
        self.first = int(self.ok[0])  # spawn: the first frame the ball can be steered (L1 936)
        ch = np.flatnonzero(np.diff(self.sector) > 0) + 1
        self.checkpoints = [int(i) for i in ch[ch > self.first]]
        self.clock = D.tas_clock(record)
        self.default_frontier = int(self.ok[-1]) - START_BACK  # the curriculum's first frontier (L1 19086)
        self.trace = DATASET / record / "trace.txt"

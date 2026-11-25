"""Score balls and life balls of a level: which are still there, during an episode (HS training).

The placements come from benchmarks/placements/L<nn>.json (the level file's P_Extra_Point / P_Extra_Life, with sector
and position). A score ball, once collected, never comes back; a life ball comes back after a death (docs/benchmarks.md,
计分规则), so a respawn restores the life balls of its sector.

What the agent collects is read from the counters after every step: Points rising by at least 95 in one step is a score
ball's core (+100; a small ball is +20, a time decrement -1; the harness's collect macro adds the six small balls in
the step after), Lifes rising is a life ball; the collected one is the nearest still-present placement of that kind
(within PICK_SEARCH of the ball). What a TAS prefix collected before the hand-over cannot be seen from Python (the mod
replays the prefix): it comes from the start record's dataset, where the same rule is applied frame by frame
(record_pickups, cached next to the record as pickups.json).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PLACEMENTS = REPO / "benchmarks" / "placements"
CORE_MIN = 95  # Points rise of a core (+100) net of a time decrement and with some small balls in the same step
PICK_SEARCH = 8.0  # the collected placement is the nearest one within this of the ball (radius 3 / 4.5 + a step)


def placements_file(level: int) -> Path:
    return PLACEMENTS / f"L{int(level):02d}.json"


@lru_cache(maxsize=16)
def load(level: int) -> tuple:
    """((name, kind, sector, position), ...) of the level's score ("point") and life ("life") balls; () without file."""
    p = placements_file(level)
    if not p.is_file():
        return ()
    doc = json.loads(p.read_text(encoding="utf-8"))
    return tuple((it["name"], it["kind"], int(it.get("sector") or 0), tuple(float(v) for v in it["position"]))
                 for it in doc.get("items", []) if it.get("kind") in ("point", "life"))


def _nearest(items, collected, kind, pos, limit=PICK_SEARCH):
    best, bd = None, limit
    for name, k, _, p in items:
        if k != kind or name in collected:
            continue
        d = float(np.linalg.norm(np.asarray(p) - pos))
        if d < bd:
            best, bd = name, d
    return best


def record_pickups(record_dir, level: int) -> list:
    """[(frame, name)] of the score / life balls the record collects (frames_done of the collecting frame), in order."""
    record_dir = Path(record_dir)
    items = load(level)
    if not items:
        return []
    sig = hashlib.sha256(placements_file(level).read_bytes()).hexdigest()[:16]
    cache = record_dir / "pickups.json"
    if cache.is_file():
        try:
            c = json.loads(cache.read_text(encoding="utf-8"))
            if c.get("placements") == sig:
                return [tuple(x) for x in c["pickups"]]
        except (OSError, ValueError, KeyError):
            pass
    from . import dataset
    tr = dataset.record_trajectory(record_dir, level)
    pts = np.asarray(tr["points"], np.int64)
    lives = np.asarray(tr["lives"], np.int64)
    pos = np.asarray(tr["pos"], np.float64)
    ctl = np.asarray(tr["control"], bool)
    fd = np.asarray(tr["frames_done"], np.int64)
    got, out = set(), []
    started = False
    for i in range(1, len(fd)):
        started = started or bool(ctl[i])
        if not started or not np.isfinite(pos[i]).all():
            continue
        for kind, hit in (("point", pts[i] - pts[i - 1] >= CORE_MIN), ("life", lives[i] > lives[i - 1])):
            if hit:
                name = _nearest(items, got, kind, pos[i])
                if name is not None:
                    got.add(name)
                    out.append((int(fd[i]), name))
    import os
    tmp = cache.with_name(f"pickups.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"placements": sig, "pickups": out}), encoding="utf-8")
    tmp.replace(cache)
    return out


class PickupState:
    """The score / life balls still present in one game process."""

    def __init__(self, level: int):
        self.level = int(level)
        self.items = load(self.level)
        self.sector_of = {n: s for n, _, s, _ in self.items}
        self.kind_of = {n: k for n, k, _, _ in self.items}
        self.collected: set = set()
        self.points = self.lives = None

    def __bool__(self):
        return bool(self.items)

    def start(self, tracker, collected=()):
        """A new process: `collected` = what a TAS prefix took before the hand-over."""
        self.collected = set(collected)
        self.points, self.lives = tracker.points, tracker.lives

    def respawn(self, tracker):
        """After a death: the life balls of the respawn sector come back."""
        s = tracker.sector
        self.collected = {n for n in self.collected if not (self.kind_of[n] == "life" and self.sector_of[n] == s)}
        self.points, self.lives = tracker.points, tracker.lives

    def update(self, tracker, pos) -> list:
        """After a step: mark what the counters say was collected; returns the names."""
        new = []
        if self.points is not None and pos is not None:
            p = np.asarray(pos, np.float64)
            if tracker.points - self.points >= CORE_MIN:
                n = _nearest(self.items, self.collected, "point", p)
                if n:
                    new.append(n)
            if tracker.lives > self.lives:
                n = _nearest(self.items, self.collected, "life", p)
                if n:
                    new.append(n)
        self.collected.update(new)
        self.points, self.lives = tracker.points, tracker.lives
        return new

    def present(self, sector: int | None = None) -> list:
        """(name, kind, sector, position) still present (in `sector` when given)."""
        return [it for it in self.items if it[0] not in self.collected and (sector is None or it[2] == sector)]

    def nearest(self, pos, sector: int, k: int = 2) -> list:
        """[(distance, kind, position)] of the k nearest still present in `sector`."""
        p = np.asarray(pos, np.float64)
        out = sorted((float(np.linalg.norm(np.asarray(it[3]) - p)), it[1], it[3]) for it in self.present(sector))
        return out[:k]

    def missed(self, sectors) -> int:
        """How many are still present in the given sectors (an agent that passed them left these behind)."""
        sectors = set(sectors)
        return sum(1 for it in self.present() if it[2] in sectors)

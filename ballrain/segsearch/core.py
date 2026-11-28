"""Pure parts of the segment search (no engine): gates, crossings, key runs and their edits, node selection, end-state
checks. docs/segsearch.md."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# arrow-key bits (ballrain.tas): up 1, down 2, left 4, right 8; the 8 directions as a ring (45 degrees apart)
RING = (1, 9, 8, 10, 2, 6, 4, 5)  # up, up+right, right, down+right, down, down+left, left, up+left
COMBOS = (0,) + RING  # the 9 actions: no key + 8 directions
ARROWS = 15


@dataclass
class Gate:
    """A plane through the TAS position at a TAS frame, normal along the TAS velocity there. A trajectory crosses it
    when the signed distance goes from below 0 to 0 or above while the crossing point lies within `radius`."""
    index: int
    frame: int  # TAS frame (absolute)
    pos: np.ndarray
    normal: np.ndarray
    radius: float

    def signed(self, p) -> np.ndarray:
        return (np.asarray(p, float) - self.pos) @ self.normal


def make_gates(frames, pos, vel, a: int, b: int, every: int, radius: float) -> list[Gate]:
    """Gates at TAS frames a+every, a+2*every, ... and at b (the last gate), from the reference arrays (absolute TAS
    frames `frames`, positions, velocities). A frame where the TAS is (almost) at rest takes the direction of travel
    over the next 64 frames."""
    frames = np.asarray(frames)
    marks = list(range(a + every, b - every // 2, every)) + [b]
    out = []
    for i, f in enumerate(marks):
        j = int(np.searchsorted(frames, f))
        v = np.asarray(vel[j], float)
        if np.linalg.norm(v) < 0.5:
            v = np.asarray(pos[min(j + 64, len(pos) - 1)], float) - np.asarray(pos[j], float)
        n = v / max(np.linalg.norm(v), 1e-9)
        out.append(Gate(i, int(frames[j]), np.asarray(pos[j], float), n, float(radius)))
    return out


def crossing(gate: Gate, t, p):
    """First crossing of `gate` by the samples (times t, positions p): (fractional time, index of the first sample at or
    past the plane) or None. The time is interpolated linearly between the two samples."""
    p = np.asarray(p, float)
    if len(p) < 2:
        return None
    d = gate.signed(p)
    idx = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0))
    for i in idx:
        w = -d[i] / (d[i + 1] - d[i])
        q = p[i] + w * (p[i + 1] - p[i])
        if np.linalg.norm(q - gate.pos) <= gate.radius:
            return float(t[i] + w * (t[i + 1] - t[i])), int(i + 1)
    return None


def crossings(gates, start: int, t, p) -> dict:
    """{gate index: (time, sample index)} for gates[start:], each searched after the previous one's crossing (in
    order); stops at the first gate not crossed."""
    out = {}
    i0 = 0
    for g in gates[start:]:
        c = crossing(g, t[i0:], p[i0:]) if len(t) - i0 >= 2 else None
        if c is None:
            break
        c_abs = c[1] + i0
        out[g.index] = (c[0], c_abs)
        i0 = max(i0, c_abs - 1)
    return out


# ------------------------------------------------------------------------------------------------- key runs / edits
def to_runs(keys) -> list[list[int]]:
    """Per-frame keys -> [[key, length], ...]."""
    keys = np.asarray(keys)
    if not len(keys):
        return []
    ch = np.flatnonzero(np.diff(keys.astype(np.int64))) + 1
    starts = np.r_[0, ch]
    lens = np.diff(np.r_[starts, len(keys)])
    return [[int(keys[s]), int(n)] for s, n in zip(starts, lens)]


def from_runs(runs) -> np.ndarray:
    if not runs:
        return np.zeros(0, np.uint32)
    return np.concatenate([np.full(int(n), int(k), np.uint32) for k, n in runs if n > 0]) if any(n > 0 for _, n in runs) \
        else np.zeros(0, np.uint32)


def rotate(key: int, steps: int) -> int:
    """The arrow combo `steps` x 45 degrees around the ring (other bits kept); no-arrow stays no-arrow."""
    arrows = key & ARROWS
    if arrows not in RING:
        return key
    return (key & ~ARROWS) | RING[(RING.index(arrows) + steps) % 8]


def mutate(keys, rng: np.random.Generator, window: int, n_edits: int, grain: int = 2) -> np.ndarray:
    """`n_edits` random local edits of the runs that start in the first `window` frames: shift a run boundary, rotate a
    run's direction by 45 / 90 degrees, release (no key), overwrite part of a run with a short run of any combo, or drop
    a run (the neighbour takes its frames). Lengths stay multiples of `grain` where they were; every edit keeps the total
    length, so frames after the edited runs keep their keys."""
    keys = np.asarray(keys, np.uint32)
    total = len(keys)
    runs = to_runs(keys)
    if not runs:
        return keys.copy()
    for _ in range(int(n_edits)):
        starts = np.cumsum([0] + [n for _, n in runs[:-1]])
        cand = [i for i, s in enumerate(starts) if s < window] or [0]
        i = int(rng.choice(cand))
        kind = rng.choice(("shift", "rotate", "release", "insert", "drop"), p=(0.35, 0.3, 0.05, 0.2, 0.1))
        if kind == "shift" and i + 1 < len(runs):
            d = int(rng.integers(1, 13)) * grain * (1 if rng.random() < 0.5 else -1)
            d = max(-runs[i][1] + grain, min(d, runs[i + 1][1] - grain))
            runs[i][1] += d
            runs[i + 1][1] -= d
        elif kind == "rotate":
            runs[i][0] = rotate(runs[i][0], int(rng.choice((-2, -1, 1, 2))))
        elif kind == "release":
            runs[i][0] = runs[i][0] & ~ARROWS
        elif kind == "insert":  # overwrite part of the run (later runs keep their frames)
            at = int(rng.integers(0, max(runs[i][1] // grain, 1))) * grain
            n = min(int(rng.integers(1, 17)) * grain, runs[i][1] - at)
            k = int(rng.choice(COMBOS))
            tail = runs[i][1] - at - n
            runs[i:i + 1] = [r for r in ([runs[i][0], at], [k, n], [runs[i][0], tail]) if r[1] > 0]
        elif kind == "drop" and len(runs) > 1:
            j = i + 1 if i + 1 < len(runs) else i - 1
            runs[j][1] += runs[i][1]
            del runs[i]
    out = from_runs(runs)
    if len(out) >= total:
        return out[:total].copy()
    return np.r_[out, np.full(total - len(out), out[-1] if len(out) else 0, np.uint32)]


def random_holds(n: int, rng: np.random.Generator, bias=None, grain: int = 2) -> np.ndarray:
    """n frames of random combos held 4..32 frames; with `bias` (a combo) half the holds are it or a 45-degree
    neighbour."""
    out = []
    while sum(len(x) for x in out) < n:
        if bias is not None and rng.random() < 0.5:
            k = rotate(int(bias), int(rng.choice((-1, 0, 1))))
        else:
            k = int(rng.choice(COMBOS))
        out.append(np.full(int(rng.integers(2, 17)) * grain, k, np.uint32))
    return np.concatenate(out)[:n] if out else np.zeros(0, np.uint32)


# ------------------------------------------------------------------------------------------------- search bookkeeping
@dataclass
class Node:
    """A branch that crossed gate `gate` at `time` frames after A (fractional); its keys are keys[:end] of rollout
    `rollout` (end = the first observed frame at or past the gate)."""
    rollout: int
    end: int
    gate: int
    time: float
    pos: np.ndarray
    vel: np.ndarray
    counts: dict = field(default_factory=dict)  # pickup events so far (ExtraPoint, PreLifeUp)
    expanded: int = 0
    type: int = -1  # ball type at the crossing
    reach: int = -1  # the last gate the rollout that made this branch crossed (its own lookahead)


def select(nodes: list[Node], k: int, k_speed: int = 0, slack: float = 132.0, dup_pos: float = 0.25,
           dup_vel: float = 0.25, k_surv: int = 0, ahead: int = 2) -> list[Node]:
    """k nodes: first up to k_surv earliest "survivors" (their own rollout crossed at least `ahead` more gates), then the
    earliest others; near-duplicates (position and velocity both within dup_*) skipped. Plus up to k_speed
    fastest-moving ones among those within `slack` frames of the earliest."""
    if not nodes:
        return []
    order = sorted(nodes, key=lambda n: n.time)
    chosen: list[Node] = []

    def dup(n):
        return any(np.linalg.norm(n.pos - c.pos) < dup_pos and np.linalg.norm(n.vel - c.vel) < dup_vel for c in chosen)

    for n in order:
        if len(chosen) >= min(k_surv, k):
            break
        if n.reach >= n.gate + ahead and not dup(n):
            chosen.append(n)
    for n in order:
        if len(chosen) >= k:
            break
        if not dup(n):
            chosen.append(n)
    if k_speed:
        t0 = order[0].time
        fast = sorted((n for n in order if n.time <= t0 + slack and n not in chosen),
                      key=lambda n: -float(np.linalg.norm(n.vel)))
        added = 0
        for n in fast:
            if added >= k_speed:
                break
            if not dup(n):
                chosen.append(n)
                added += 1
    return chosen


@dataclass
class EndCheck:
    """End-state limits at gate B relative to the TAS there (docs/segsearch.md): speed at least TAS - dv, horizontal
    heading within dheading degrees, distance to the TAS point at most dpos, height at least TAS - dy; same ball type;
    at least the TAS's pickups in the stretch; no ball-off."""
    dv: float = 0.5
    dheading: float = 20.0
    dpos: float = 3.0
    dy: float = 0.5

    def check(self, ref: dict, got: dict) -> dict:
        v0, v1 = np.asarray(ref["vel"], float), np.asarray(got["vel"], float)
        h0, h1 = v0[[0, 2]], v1[[0, 2]]
        ang = float(np.degrees(np.arccos(np.clip(h0 @ h1 / max(np.linalg.norm(h0) * np.linalg.norm(h1), 1e-9), -1, 1))))
        p0, p1 = np.asarray(ref["pos"], float), np.asarray(got["pos"], float)
        res = {
            "speed": float(np.linalg.norm(v1)), "speed_tas": float(np.linalg.norm(v0)),
            "heading_deg": ang, "dist": float(np.linalg.norm(p1 - p0)), "dy": float(p1[1] - p0[1]),
        }
        fails = []
        if res["speed"] < res["speed_tas"] - self.dv:
            fails.append("speed")
        if ang > self.dheading:
            fails.append("heading")
        if res["dist"] > self.dpos:
            fails.append("position")
        if res["dy"] < -self.dy:
            fails.append("height")
        if got.get("type") != ref.get("type"):
            fails.append("ball_type")
        for k, n in (ref.get("counts") or {}).items():
            if (got.get("counts") or {}).get(k, 0) < n:
                fails.append(f"pickup:{k}")
        if got.get("ball_off"):
            fails.append("ball_off")
        res["fails"] = fails
        res["ok"] = not fails
        return res

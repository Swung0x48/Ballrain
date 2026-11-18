"""The per-life draws of pool tasks, from the task's server-side life_seed so the verifier replays them:
the learner's WorkerSpec.counter_rand (ballrain/train/worker.py _counter_draw / _counter_apply) and its early-respawn
rule (Worker._skip_respawn, WorkerSpec.p_early_respawn / respawn_level_starts).

In a game process not started from level load (task start_frame > 0: TAS and at-rest starts) every life draws one set
of counter-feature values that every observation of that life carries instead of the true ones (raw feature units):
  {"lives": {"values": [...], "p": [...] | None}}  the "lives" entry replaced by a value drawn with probabilities p
                                                    (uniform when p is absent)
  {"sr_s": [lo, hi], "points": [lo, hi]}            uniform offsets added to those entries
Life i of the task (0 = the hand-over life, then each respawn) uses u_j = sampling.sample_u(life_seed, 4 i + j):
j = 0 for lives (inverse cdf), 1 for sr_s, 2 for points (lo + (hi - lo) u), 3 for the early-respawn draw when life i
ends in a death. The learner's own worker draws from numpy; the distribution is the same, the numbers are not (they
only have to be the same for client and verifier).

Early respawns: a death (BallOff, the process going on) of life i whose respawn sector (the sector at death) the start
record enters before the task's respawn_frontier skips the respawn life when u_3 >= p_early_respawn: the task ends
there ("respawn_skipped"). Never at p_early_respawn >= 1, without a respawn_frontier, or with respawn_level_starts in
a task started from level load. The learner compares with its live frontier, the task with the frontier at issue.
The native client implements the same (native/core/src/task_runner.cpp CounterRand).
"""
from __future__ import annotations

import numpy as np

from .sampling import sample_u


def active(spec: dict | None, start_frame: int) -> bool:
    return bool(spec) and int(start_frame) > 0


def draw(spec: dict, seed: int, life: int, slices: dict) -> dict | None:
    """{slice: ("set" | "add", value)} of life `life`, or None when nothing applies."""
    out = {}
    if "lives" in spec and "lives" in slices:
        vals = [float(v) for v in spec["lives"]["values"]]
        p = spec["lives"].get("p")
        p = np.full(len(vals), 1.0 / len(vals)) if p is None else np.asarray(p, np.float64) / float(np.sum(p))
        c = np.cumsum(p)
        u = sample_u(seed, 4 * life)
        i = int(min(np.searchsorted(c, u, side="right"), len(vals) - 1))
        out[slices["lives"]] = ("set", vals[i])
    for j, k in ((1, "sr_s"), (2, "points")):
        if k in spec and k in slices:
            lo, hi = float(spec[k][0]), float(spec[k][1])
            out[slices[k]] = ("add", lo + (hi - lo) * sample_u(seed, 4 * life + j))
    return out or None


def skip_respawn(rollout: dict, task, life: int, sector: int, tas_cross_frame) -> bool:
    """True when the respawn after life `life`'s death in `sector` is not played (Worker._skip_respawn's rule).
    tas_cross_frame(sector): the record frame where the record enters that sector (None: never)."""
    p = float(rollout.get("p_early_respawn", 1.0))
    fr = getattr(task, "respawn_frontier", None)
    if p >= 1.0 or fr is None:
        return False
    if rollout.get("respawn_level_starts") and int(task.start_frame) == 0:
        return False
    cross = tas_cross_frame(int(sector))
    if cross is None or int(cross) >= int(fr):
        return False
    return sample_u(int(getattr(task, "life_seed", 0) or 0), 4 * int(life) + 3) >= p


def apply(o, cr: dict | None):
    """The observation with the life's counter values (float32, as the worker's _counter_apply)."""
    if not cr or o is None:
        return o
    o = np.array(o, dtype=np.float32, copy=True)
    for s_, (op, v) in cr.items():
        o[s_] = v if op == "set" else o[s_] + v
    return o

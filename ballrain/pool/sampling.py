"""Deterministic action sampling for pool tasks (docs/pool-prototype.md 10.6).

The gateway puts a random sample_seed (53 bits, server-side, unpredictable) into every task. The action of task step t
is drawn by inverting the policy's cumulative distribution at u_t = sample_u(sample_seed, t), a fixed function of the
seed and the step. The player's sampler and the verifier's replay both call these functions, so the replay recomputes
which action the policy had to take and refuses any other one (replay_action_mismatch), except where u_t lies within
ACTION_EPS of a boundary of the cumulative distribution (different CPUs / BLAS round the probabilities slightly
differently). This closes two gaps: a client choosing its own actions while reporting honest log-probs, and a cached
trajectory re-signed onto another task with the same start (its actions follow another seed).

The u_t stream is uniform on [0, 1), so the behaviour distribution is the policy's own at temperature 1 (the learner
still gets the behaviour log-prob of every action).
"""
from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
SEED_BITS = 53  # protocol integers stay within 2**53 (JSON-safe)
ACTION_EPS = 1e-4  # tolerance at a boundary of the cumulative distribution


def splitmix64(x: int) -> int:
    x = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def sample_u(seed: int, t: int) -> float:
    """The uniform draw of task step t (0-based over the whole task): a float64 in [0, 1)."""
    z = splitmix64((int(seed) & MASK64) ^ splitmix64(int(t) & MASK64))
    return (z >> 11) * (1.0 / (1 << 53))


def cdf(log_probs) -> np.ndarray:
    """Cumulative distribution (float64) of a log-probability vector."""
    p = np.exp(np.asarray(log_probs, np.float64))
    return np.cumsum(p / p.sum())


def sample_action(log_probs, u: float) -> int:
    """The action the inverse-cdf draw u picks (the same rule as NumpyPolicy.act's random draw)."""
    c = cdf(log_probs)
    return int(min(np.searchsorted(c, float(u), side="right"), len(c) - 1))


def action_consistent(log_probs, u: float, a: int, eps: float = ACTION_EPS) -> bool:
    """True when action a is the one u picks under these probabilities, up to eps at a cdf boundary."""
    c = cdf(log_probs)
    n = len(c)
    a = int(a)
    if not 0 <= a < n:
        return False
    lo = float(c[a - 1]) if a > 0 else 0.0
    hi = float(c[a]) if a < n - 1 else 1.0
    return lo - eps <= float(u) < hi + eps

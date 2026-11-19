"""Verification at scale: tiers and the cross-check comparison (ballrain/pool/cross_check.py), server rewards gating."""
from types import SimpleNamespace

import numpy as np

from ballrain.pool import cross_check as X
from ballrain.pool import protocol as P
from ballrain.pool.server_rewards import refuse_reason


def _chunk(seq=0, steps=4, obs=0.0, action=1, final=True):
    h = {"task_id": "a" * 32, "seq": seq, "step0": 0, "steps": steps, "n_obs": 3, "n_act": 9, "policy_sha": "0" * 64,
         "chain_prev": "c" * 64, "final": final, "end": "max_steps" if final else None}
    arr = {"actions": np.full(steps, action, np.uint8), "rewards": np.zeros(steps, "<f4"), "logp": np.full(steps, -1, "<f4"),
           "values": np.zeros(steps + 1, "<f4"), "flags": np.array([0] * (steps - 1) + [P.FLAG_TRUNCATED], np.uint8),
           "obs": np.full((steps + 1, 3), obs, "<f2"), "obs_hash": np.zeros((steps + 1, 32), np.uint8)}
    return P.decode_chunk(P.encode_chunk(h, arr))


def test_compare_ignores_the_task_seeded_chain_and_finds_data_differences():
    a = _chunk()
    b_h = {k: v for k, v in a.header.items() if k not in ("arrays", "protocol", "type")}
    arr = {k: np.array(v, copy=True) for k, v in a.arrays.items()}
    arr["obs_hash"][:] = 7  # another task id seeds another chain: not a difference
    b = P.decode_chunk(P.encode_chunk(b_h, arr))
    assert X.compare([a], [b]) is None
    assert "obs differ" in X.compare([a], [_chunk(obs=0.5)])
    assert "actions differ" in X.compare([a], [_chunk(action=2)])
    assert "chunk count" in X.compare([a], [a, a])


class _Store:
    def __init__(self, probation=10 ** 9, suspect=False, passed=0, escalated=False):
        self.p, self.s, self.passed, self.e = probation, suspect, passed, escalated

    def probation_steps(self, f):
        return self.p

    def is_suspect(self, f):
        return self.s

    def take_escalation(self, f):
        return self.e

    def passed_tasks(self, f):
        return self.passed


def _cfg(**kw):
    c = dict(probation_chunks=200, task_sample=0.15, trusted_after=50, trusted_sample=0.03, cross_fraction=0.0)
    c.update(kw)
    return SimpleNamespace(**c)


def test_tiers():
    assert X.assign(_Store(probation=0), _cfg(), 1, 256)[:2] == ("trial", True)
    assert X.assign(_Store(suspect=True), _cfg(), 1, 256)[:2] == ("suspect", True)
    assert X.assign(_Store(escalated=True), _cfg(), 1, 256)[:2] == ("escalated", True)
    assert X.assign(_Store(), _cfg(task_sample=1.0), 1, 256) == ("normal", True, False)
    assert X.assign(_Store(), _cfg(task_sample=0.0), 1, 256) == ("normal", False, False)
    assert X.assign(_Store(passed=50), _cfg(trusted_sample=1.0), 1, 256) == ("trusted", True, False)
    # every drawn task to a peer: never our replay, never for trial / suspect accounts
    assert X.assign(_Store(), _cfg(task_sample=1.0, cross_fraction=1.0), 1, 256) == ("normal", False, True)
    assert X.assign(_Store(suspect=True), _cfg(cross_fraction=1.0), 1, 256)[1:] == (True, False)
    n = sum(X.assign(_Store(passed=50), _cfg(), 1, 256)[1] for _ in range(4000))
    assert 60 < n < 190  # trusted: about 3 %


def test_server_rewards_refuses_what_it_cannot_recompute():
    assert refuse_reason({"reward": {"mode": "sr"}, "rollout": {}}) is None
    assert refuse_reason({"reward": {"mode": "hs"}, "rollout": {}}) == "HS rewards"
    assert refuse_reason({"reward": {}, "rollout": {"gates": {"g": 1}}}) == "rollout gates"

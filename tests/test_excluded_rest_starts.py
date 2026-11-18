"""At-rest starts with a drive target handed over in an exclude_starts range (TasStarts.exclude / nearest_any): the
curriculum and nearest_valid keep skipping the excluded frames, the targeted rest start may use them, expected() knows
their state (hand-over check), and the worker gives such a life no split-bonus reference. No game is launched."""
from types import SimpleNamespace

import numpy as np

from ballrain.curriculum import TasStarts
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.train.worker import Worker


def _starts():
    st = object.__new__(TasStarts)
    st.valid = np.arange(936, 8000, dtype=np.int64)
    n = len(st.valid)
    st.sector = np.where(st.valid < 7817, 1, 2)
    st.sr_double = st.valid * (1000 / 264)
    st.points, st.lives = np.full(n, 900), np.full(n, 3)
    st.pos = np.c_[st.valid * 0.1, np.zeros(n), -st.valid * 0.1]
    st.checkpoints = np.where(st.valid < 7817, 0, 1)
    v = st.valid
    st.exclude(~((v >= 4450) & (v < 6018)))
    return st


def test_exclude_keeps_excluded_frames_as_hand_over_states():
    st = _starts()
    assert st.index_of(5100) is None and st.nearest_valid(5100) in (4449, 6018)
    assert st.nearest_any(5100) == 5100 and st.nearest_any(4300) == 4300 and st.nearest_any(0) == 0
    e = st.expected(5100)
    assert e["frame"] == 5100 and np.allclose(e["pos"], [510.0, 0.0, -510.0]) and e["lives"] == 3
    assert st.expected(4300) == {"frame": 4300, "sector": 1, "checkpoints": 0, "points": 900, "lives": 3,
                                 "sr_double": float(4300 * 1000 / 264), "pos": [430.0, 0.0, -430.0]}
    assert st.expected(100) is None
    st2 = object.__new__(TasStarts)  # restrict alone (no exclude): nearest_any falls back to the kept frames
    st2.valid = np.arange(10, 20, dtype=np.int64)
    assert st2.nearest_any(30) == 19


def _env(rest_starts, st):
    env = object.__new__(BallanceEnv)
    env.cfg = EnvConfig(rest_starts=rest_starts, p_rest=1.0)
    env.starts = st
    env.curriculum = None
    env._start_rng = np.random.default_rng(0)
    return env


def test_targeted_rest_start_may_use_excluded_frames_plain_ones_not():
    st = _starts()
    env = _env([[5090, 5110, -207.0, -225.0, 0.8]], st)
    for _ in range(30):
        k = env._draw_start()
        assert 5090 <= k <= 5110 and isinstance(env._rest_next, tuple)
        x, z = env._rest_next
        assert -207.8 <= x <= -206.2 and -225.8 <= z <= -224.2
    env = _env([[5090, 5110]], st)
    assert env._draw_start() in (4449, 6018)


def test_no_split_reference_for_an_excluded_hand_over():
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(split_k=5.0, split_rest_allow_s=1.0)
    w.env = SimpleNamespace(starts=_starts())
    for k, has_ref in ((5100, False), (4300, True)):
        ep = SimpleNamespace(d={"kind": "tas_rest", "start_frame": k, "start_sector": 1, "start_sr_ms": 1000.0}, seg="x")
        w._split_start(ep, {})
        assert (ep.seg is not None) == has_ref
        if has_ref:
            assert np.isclose(ep.seg[1], 4300 * 1000 / 264 - 1000.0)


def test_entry_drive_limit():
    import sys
    from ballrain import instances
    sys.path.insert(0, str(instances.REPO_ROOT / "scripts"))
    import train_m2
    assert train_m2.parse_rest_starts("5060-5080@-205.5:-225:0.5:1200") == [[5060, 5080, -205.5, -225.0, 0.5, 1200]]
    assert isinstance(train_m2.parse_rest_starts("5060-5080@-205.5:-225:0.5:1200")[0][5], int)
    env = _env([[5060, 5080, -205.5, -225.0, 0.5, 1200]], _starts())
    assert 5060 <= env._draw_start() <= 5080 and len(env._rest_next) == 3 and env._rest_next[2] == 1200
    env = _env([[5060, 5080, -205.5, -225.0, 0.5]], _starts())
    env._draw_start()
    assert len(env._rest_next) == 2

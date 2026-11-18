"""Guards added after the r8 NaN incident (2026-11-18): (a) reach patches and the learner's rows are finite, (b)
absurd finite magnitudes are refused at upload ([guards]), (c) a policy that computes nonsense on a fixed probe batch
is never published."""
from types import SimpleNamespace

import numpy as np
import pytest

from ballrain.pool import learner_link as LL
from ballrain.pool import server_verify as SV

torch = pytest.importorskip("torch")


def test_reach_patches_and_learner_rows_are_finite(tmp_path):
    f = tmp_path / "000001.brpc"
    with pytest.raises(ValueError, match="non-finite"):
        SV.write_reach_patch(f, {"rows": [1], "reach": np.array([[np.nan, 0.0]], np.float32), "start": 2})
    assert not SV.reach_patch_path(f).exists()
    from test_pool_learner import _chunk
    c = _chunk(n_obs=4)
    LL.chunk_message(c, 1, (np.array([1]), np.array([[0.5, 0.5]], np.float32), 2))  # a finite patch: fine
    with pytest.raises(ValueError, match="non-finite"):
        LL.chunk_message(c, 1, (np.array([1]), np.array([[np.inf, 0.5]], np.float32), 2))


class _App:
    def __init__(self, **cfg):
        from ballrain.pool.server_app import PoolApp
        self.cfg = SimpleNamespace(**cfg)
        self._magnitudes = PoolApp._magnitudes.__get__(self)


def test_upload_magnitude_bounds(tmp_path):
    from ballrain.pool.server_config import load_config
    app = _App(guard_obs_slack=0.5, guard_max_abs_value=1000.0, guard_max_abs_reward=1000.0, pool_dir=tmp_path)
    man = {"features": {"clip": 10.0}}
    ok = {"obs": np.full((3, 4), 10.0, np.float16), "values": np.array([2.0, -3.0, 1.0], np.float32),
          "rewards": np.array([-25.6, 9.9], np.float32), "env_rewards": np.array([-25.6, 9.9], np.float32)}
    assert app._magnitudes(SimpleNamespace(arrays=ok), man, "x") is None  # the live run's extremes pass
    for k, bad, kind in (("obs", np.full((3, 4), 12.0, np.float16), "obs"),
                         ("values", np.array([0.0, 5e3, 0.0], np.float32), "values"),
                         ("rewards", np.array([0.0, -2e3], np.float32), "rewards"),
                         ("env_rewards", np.array([1e4, 0.0], np.float32), "env_rewards")):
        got = app._magnitudes(SimpleNamespace(arrays=dict(ok, **{k: bad})), man, "x")
        assert got is not None and got[0] == kind and got[1] > got[2]
    cfg = tmp_path / "c.toml"
    cfg.write_text("[guards]\nmax_abs_reward = 50\n", encoding="utf-8")
    c = load_config(cfg, pool_dir=tmp_path, environ={})
    assert c.guard_max_abs_reward == 50.0 and c.guard_max_abs_value == 1000.0 and c.guard_obs_slack == 0.5
    for bad in ("[guards]\nmax_reward = 5\n", "[guards]\nobs_slack = 0\n"):
        cfg.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(cfg, pool_dir=tmp_path, environ={})


def test_probe_refuses_a_collapsed_policy(tmp_path):
    from test_pool_weights import _state
    from ballrain.pool import weights as W
    from ballrain.pool.server_tasks import probe_policy
    state, model, _ = _state()
    m = W.export_state(state, tmp_path / "ok", "r8")
    probe_policy(tmp_path / "ok", m)  # a normal policy passes
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    last_w = sorted(k for k in sd if k.startswith("pi.") and k.endswith(".weight"))[-1]
    sd[last_w].zero_()
    sd[last_w.replace("weight", "bias")].copy_(torch.tensor([200.0] + [0.0] * (sd[last_w].shape[0] - 1)))
    m2 = W.export_state(dict(state, model=sd), tmp_path / "collapsed", "r8")
    with pytest.raises(ValueError, match="entropy"):
        probe_policy(tmp_path / "collapsed", m2)
    sd3 = {k: v.clone() for k, v in model.state_dict().items()}
    for k in sd3:  # every first-layer weight zero: the outputs no longer depend on the input
        if k in ("pi.0.weight", "v.0.weight"):
            sd3[k].zero_()
    m3 = W.export_state(dict(state, model=sd3), tmp_path / "flat", "r8")
    with pytest.raises(ValueError, match="depend"):
        probe_policy(tmp_path / "flat", m3)

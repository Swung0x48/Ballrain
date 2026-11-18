"""Counter randomisation (WorkerSpec.counter_rand, train_m2 --counter-rand): in a process not started from level load
each life draws one set of counter-feature values (lives replaced, sr_s / points offsets) carried by every observation of
that life; level-start processes keep the true counters. No game is launched (stub env)."""
import json
import sys
from types import SimpleNamespace

import numpy as np

from ballrain import instances
from ballrain.train.worker import Worker, WorkerSpec

CR = {"lives": {"values": [0.0, 0.1, 0.2, 0.3], "p": [0.2, 0.2, 0.2, 0.4]}, "sr_s": [0.0, 0.3], "points": [-0.05, 0.45]}
SL = {"valid": slice(0, 1), "lives": slice(1, 2), "points": slice(2, 3), "sr_s": slice(3, 4), "sector": slice(4, 5)}


class _Fx:
    def __init__(self):
        self.t = 0

    def slices(self):
        return SL

    def compute(self, tracker):
        self.t += 1  # the true sr_s / points move every frame
        return np.array([1.0, 0.3, 0.9 - 0.001 * self.t, 0.5 + 0.01 * self.t, 1.0], np.float32), {}


class _Env:
    def __init__(self, cfg, launcher=None):
        self.cfg = SimpleNamespace(trace=False)
        self.fx = _Fx()
        self.starts = None
        self.curriculum = None
        self.psf = cfg["psf"]
        self._start = {}

    def reset(self, options=None):
        self._start = {"kind": "tas" if self.psf else "level", "process_start_frame": self.psf}
        return self.fx.compute(None)[0], {"start": self._start}

    def step(self):
        return self.fx.compute(None)[0]


def _worker(monkeypatch, psf, cr=CR, seed=0):
    import ballrain.env
    monkeypatch.setattr(ballrain.env, "BallanceEnv", _Env)
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(env={"psf": psf}, launcher_factory=None, counter_rand=cr, trace_sample=0.0,
                             max_reset_failures=1)
    w.curriculum_state = None
    w.fx_info = {}
    w._counter_rng = np.random.default_rng(seed)
    w._cr = None
    w._collect_runs = lambda: None
    w._make_env()
    return w


def _true(t):
    return 0.3, 0.9 - 0.001 * t, 0.5 + 0.01 * t


def test_values_in_range_and_constant_within_a_life(monkeypatch):
    w = _worker(monkeypatch, 4100)
    lives = []
    for _ in range(2000):
        obs, info = w._reset()
        _, pt, sr = _true(w.env.fx.t)
        d_pt, d_sr = float(obs[2]) - pt, float(obs[3]) - sr
        assert float(obs[1]) in (0.0, 0.1, 0.2, 0.3) or np.isclose(float(obs[1]), [0.0, 0.1, 0.2, 0.3]).any()
        assert -0.05 - 1e-5 <= d_pt <= 0.45 + 1e-5 and -1e-5 <= d_sr <= 0.3 + 1e-5
        assert float(obs[0]) == 1.0 and float(obs[4]) == 1.0  # other features (sector) untouched
        dr = info["start"]["counter_rand"]
        assert np.isclose(dr["lives"], obs[1]) and np.isclose(dr["sr_s"], d_sr, atol=1e-5)
        for _ in range(3):  # the same life: the same lives value and offsets on every observation
            o = w.env.step()
            _, pt, sr = _true(w.env.fx.t)
            assert np.isclose(o[1], obs[1]) and np.isclose(o[2] - pt, d_pt, atol=1e-5) and np.isclose(o[3] - sr, d_sr, atol=1e-5)
        lives.append(round(float(obs[1]), 1))
    lv = np.array(lives)
    assert 0.36 < np.mean(lv == 0.3) < 0.44 and all(0.17 < np.mean(lv == v) < 0.23 for v in (0.0, 0.1, 0.2))


def test_new_life_draws_new_values(monkeypatch):
    w = _worker(monkeypatch, 4100)
    offs = set()
    for _ in range(20):
        obs, _ = w._reset()
        offs.add(round(float(obs[3]) - _true(w.env.fx.t)[2], 5))
    assert len(offs) == 20


def test_level_start_processes_and_option_off_keep_true_counters(monkeypatch):
    for psf, cr in ((0, CR), (4100, {})):
        w = _worker(monkeypatch, psf, cr)
        state = w._counter_rng.bit_generator.state
        for _ in range(5):
            obs, info = w._reset()
            assert np.allclose(obs[1:4], _true(w.env.fx.t)) and "counter_rand" not in info["start"]
            o = w.env.step()
            assert np.allclose(o[1:4], _true(w.env.fx.t))
        assert w._counter_rng.bit_generator.state == state  # nothing drawn
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).counter_rand == {}


def test_option_reaches_spec_and_remote_template():
    sys.path.insert(0, str(instances.REPO_ROOT / "scripts"))
    import train_m2
    assert train_m2.parse(["--run", "x"]).counter_rand == ""
    a = train_m2.parse(["--run", "x", "--counter-rand", json.dumps(CR)])
    assert json.loads(a.counter_rand) == CR
    src = (train_m2.REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert "counter_rand=json.loads(a.counter_rand)" in src and '"counter_rand": json.loads(a.counter_rand)' in src

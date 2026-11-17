"""From-spawn training options: at-rest starts driven to a target (EnvConfig.rest_starts [lo, hi, x, z, jitter],
--rest-starts lo-hi@x:z:jitter) and respawns played in processes started from level load
(WorkerSpec.respawn_level_starts). No game is launched."""
import sys
from types import SimpleNamespace

import numpy as np

from ballrain import instances
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.train.worker import Worker, WorkerSpec

sys.path.insert(0, str(instances.REPO_ROOT / "scripts"))
import train_m2  # noqa: E402


def test_parse_rest_starts():
    assert train_m2.parse_rest_starts("") is None
    assert train_m2.parse_rest_starts("13450-13650,18800-18900") == [[13450, 13650], [18800, 18900]]
    assert train_m2.parse_rest_starts("1940-1940@3.5:-170:1.5") == [[1940, 1940, 3.5, -170.0, 1.5]]
    try:
        train_m2.parse_rest_starts("1940-1940@3.5:-170")
    except SystemExit:
        pass
    else:
        raise AssertionError("an incomplete target was accepted")
    a = train_m2.parse(["--run", "x", "--respawn-level-starts"])
    assert a.respawn_level_starts and not train_m2.parse(["--run", "x"]).respawn_level_starts
    src = (train_m2.REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert "respawn_level_starts=a.respawn_level_starts" in src and '"respawn_level_starts": a.respawn_level_starts' in src


def _env(rest_starts):
    env = object.__new__(BallanceEnv)
    env.cfg = EnvConfig(rest_starts=rest_starts, p_rest=1.0)
    env.starts = SimpleNamespace(nearest_valid=lambda k: int(k))
    env.curriculum = None
    env._start_rng = np.random.default_rng(0)
    return env


def test_draw_start_with_and_without_target():
    env = _env([[1940, 1940, 3.5, -170.0, 1.5]])
    for _ in range(50):
        assert env._draw_start() == 1940
        x, z = env._rest_next
        assert 2.0 <= x <= 5.0 and -171.5 <= z <= -168.5
    env = _env([[13450, 13650]])
    k = env._draw_start()
    assert 13450 <= k <= 13650 and env._rest_next is True


def _decider(level_start, flag, p=0.0):
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(p_early_respawn=p, respawn_level_starts=flag)
    valid = np.arange(936, 20587)
    sector = np.select([valid < 7817, valid < 11756, valid < 14232], [1, 2, 3], 4)
    w.env = SimpleNamespace(_after_death=True, curriculum=SimpleNamespace(frontier=4100),
                            starts=SimpleNamespace(valid=valid, sector=sector),
                            _start={"kind": "respawn", "process_start_frame": 0 if level_start else 4150})
    w._respawn_rng = np.random.default_rng(0)
    return w


def test_respawns_played_in_level_start_processes():
    assert not _decider(True, True)._skip_respawn("BallOff", {"sector": 1})  # level-start process: played
    assert _decider(False, True)._skip_respawn("BallOff", {"sector": 1})  # TAS-start process: skipped as before
    assert _decider(True, False)._skip_respawn("BallOff", {"sector": 1})  # option off: skipped as before
    w = _decider(True, True)
    w.spec = SimpleNamespace(p_early_respawn=0.0)  # an older pickled WorkerSpec without the field
    assert w._skip_respawn("BallOff", {"sector": 1})
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).respawn_level_starts is False

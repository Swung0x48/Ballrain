"""Level options of the M2 scripts (train_m2 --level / --record, eval_m2 taking the checkpoint's level and start record)
and rehearsal starts gated by the curriculum frontier (EnvConfig.rehearse_behind_frontier). No game is launched."""
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from ballrain import instances
from ballrain.env import BallanceEnv, EnvConfig

sys.path.insert(0, str(instances.REPO_ROOT / "scripts"))
import eval_m2  # noqa: E402
import train_m2  # noqa: E402


def test_level_and_record_options():
    a = train_m2.parse(["--run", "x"])
    # --level defaults to None at parse time and resolves to the checkpoint's level (else 1) in main()
    assert a.level is None and not a.rehearse_behind_frontier
    a.level = 1
    assert train_m2.start_record(a) == "SR01_1_14_457_264"
    a = train_m2.parse(["--run", "x", "--level", "2", "--rehearse-behind-frontier"])
    assert train_m2.start_record(a) == "SR02_1_28_992_264" and a.rehearse_behind_frontier
    a = train_m2.parse(["--run", "x", "--level", "2", "--record", "SR02_1_33_070_240"])
    assert train_m2.start_record(a) == "SR02_1_33_070_240"
    try:
        train_m2.start_record(train_m2.parse(["--run", "x", "--level", "7"]))
    except SystemExit:
        pass
    else:
        raise AssertionError("a level without a default record was accepted")
    src = (train_m2.REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert "asset_path(1" not in src and "rehearse_behind_frontier=a.rehearse_behind_frontier" in src


def _ck(level=None, record=None):
    from ballrain.features import FeatureConfig
    from ballrain.rewards import RewardConfig
    e = asdict(EnvConfig())
    e.update(action_repeat=2, world_actions=True, features=asdict(FeatureConfig()), reward=asdict(RewardConfig(mode="sr")))
    if level is not None:
        e.update(level=level, start_record=record)
    else:  # a checkpoint from before r8 (no level / start_record recorded)
        e.pop("level")
        e.pop("start_record")
    return {"env_config": e}


def test_eval_takes_the_checkpoint_level():
    c = eval_m2.build_env_config(_ck(2, "SR02_1_28_992_264"), "out", 5000, 30000, "headless", None)
    assert c.level == 2 and c.start_record == "SR02_1_28_992_264" and c.start_frame == 5000
    c = eval_m2.build_env_config(_ck(2, "SR02_1_28_992_264"), "out", 0, 30000, "headless", None)
    assert c.level == 2 and c.start_record is None
    c = eval_m2.build_env_config(_ck(), "out", 4100, 30000, "headless", None)
    assert c.level == 1 and c.start_record == "SR01_1_14_457_264"


def _env(behind, frontier):
    env = object.__new__(BallanceEnv)
    env.cfg = EnvConfig(rehearse_starts=[[5000, 6000], [0, 0]], p_rehearse=1.0, rehearse_behind_frontier=behind)
    env.starts = SimpleNamespace(nearest_valid=lambda k: int(k))
    env.curriculum = SimpleNamespace(frontier=frontier, sample=lambda rng: -1)
    env._start_rng = np.random.default_rng(0)
    return env


def test_rehearsal_behind_the_frontier():
    env = _env(False, 20000)
    ks = [env._draw_start() for _ in range(200)]
    assert any(5000 <= k <= 6000 for k in ks) and 0 in ks  # gate off: the whole range, level starts too
    env = _env(True, 20000)  # frontier after the range: skipped (falls through to the curriculum), level starts kept
    ks = [env._draw_start() for _ in range(200)]
    assert set(ks) <= {-1, 0} and -1 in ks and 0 in ks
    env = _env(True, 5600)  # frontier inside the range: only its part at or after the frontier
    ks = [env._draw_start() for _ in range(300)]
    r = [k for k in ks if k not in (-1, 0)]
    assert r and min(r) >= 5600 and max(r) <= 6000
    env = _env(True, 3000)  # frontier before the range: all of it
    r = [k for k in (env._draw_start() for _ in range(300)) if k not in (-1, 0)]
    assert min(r) < 5100 and max(r) > 5900

"""--frontier-floor (scripts/train_m2.py gated_curriculum): the frontier moves back in steps but never below the floor."""
import sys
from types import SimpleNamespace

import numpy as np

from ballrain import instances
from ballrain.curriculum import CurriculumConfig

sys.path.insert(0, str(instances.REPO_ROOT / "scripts"))
import train_m2  # noqa: E402


def _cu(floor):
    starts = SimpleNamespace(valid=np.r_[np.arange(936, 4450), np.arange(6018, 20000)])
    cu = train_m2.gated_curriculum(starts, CurriculumConfig(start_back=19999 - 4100, step=100, band=200, advance_at=0.6,
                                                            min_episodes=4), finish_gate=0.0)
    cu.floor = floor
    return cu


def _succeed(cu, n=4):
    moved = False
    for _ in range(n):
        moved |= cu.update_gated(cu.frontier, True, True)
    return moved


def test_floor_stops_the_frontier():
    cu = _cu(3800)
    assert cu.frontier == 4100
    assert _succeed(cu) and cu.frontier == 4000
    assert _succeed(cu) and cu.frontier == 3900
    assert _succeed(cu) and cu.frontier == 3800
    assert not _succeed(cu) and cu.frontier == 3800  # at the floor: stays
    cu = _cu(3850)
    for _ in range(3):
        _succeed(cu)
    assert cu.frontier == 3850  # 3800 would be below the floor: clamped
    cu = _cu(0)
    for _ in range(4):
        _succeed(cu)
    assert cu.frontier < 3800  # no floor


def test_option_parsed():
    assert train_m2.parse(["--run", "x"]).frontier_floor == 0
    assert train_m2.parse(["--run", "x", "--frontier-floor", "3800"]).frontier_floor == 3800

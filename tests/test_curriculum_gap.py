"""A gap in the valid start frames wider than the curriculum step must not stall the frontier (r8 L2: no valid
starts between 12157 and 12819 while the ball is unphysicalized during a transformation; step 200)."""
import numpy as np

from ballrain.curriculum import CurriculumConfig, ReverseCurriculum


class _Starts:
    def __init__(self, frames):
        self.valid = np.asarray(frames, dtype=np.int64)


def _cur(frames, frontier, step=200):
    cfg = CurriculumConfig(step=step, band=400, min_episodes=4, advance_at=0.6)
    try:
        cu = ReverseCurriculum(_Starts(frames), cfg)
    except Exception:
        cu = ReverseCurriculum.__new__(ReverseCurriculum)
        cu.cfg, cu.valid, cu.history = cfg, np.asarray(frames), []
    cu.frontier = frontier
    return cu


def test_next_frontier_crosses_a_wide_gap():
    frames = list(range(11000, 12158)) + list(range(12819, 14000))
    cu = _cur(frames, 12819)
    assert cu.next_frontier(12819 - 200) == 12157


def test_next_frontier_normal_step():
    frames = list(range(11000, 14000))
    cu = _cur(frames, 13000)
    assert cu.next_frontier(12800) == 12800


def test_next_frontier_at_first_frame_stays():
    frames = list(range(12819, 14000))
    cu = _cur(frames, 12819)
    assert cu.next_frontier(12619) == 12819

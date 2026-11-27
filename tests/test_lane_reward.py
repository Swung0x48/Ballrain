"""Opt-in route-adherence term (RewardConfig.lane_weight): paid per metre of new maximum route s inside the window,
scaled by closeness to the route line (3D distance to the projection point); off by default, bit-identical to the reward before the term existed."""
import math

import pytest

from ballrain.rewards import Reward, RewardConfig


class T:
    def __init__(self):
        self.points, self.lives, self.life_bonus = 1000, 3, 200
        self.sr_ms, self.checkpoints = 0.0, 0
        self.points_at_finish = self.lives_at_finish = self.followers_at_finish = None
        self.sector = 1

    def followers(self):
        return 0

    def hs_partial(self):
        return self.points + 200 * self.lives

    def ball_type(self):
        return 0


def sequence():
    """A fixed life: forward and backward route progress, unsupported (held) stretches, a checkpoint, pickups, a
    re-acquired projection and a death at the end. Yields (s, supported, rebase, died, points_delta, cp, finfo)."""
    s = 140.0
    for i in range(60):
        ds = 1.3 if i % 7 else -0.8
        s += ds
        supported = not (20 <= i < 24)
        rebase = i == 40
        if rebase:
            s += 6.0
        died = i == 59
        finfo = {"pos": [-90.0 + 0.01 * i, 25.0 + 0.3 * math.sin(i), -180.0],
                 "path": {"s": s, "point": [-90.0, 25.0, -180.0], "tangent": [0.0, 0.0, 1.0], "jump": rebase}}
        yield s, supported, rebase, died, (100 if i == 30 else -1 if i % 3 == 0 else 0), i == 45, finfo


def run(cfg, pass_features):
    t = T()
    r = Reward(cfg)
    r.begin(t, 140.0)
    out = []
    for s, sup, rb, died, dp, cp, finfo in sequence():
        t.sr_ms += 1000 / 132
        t.points += dp
        if cp:
            t.checkpoints += 1
        rew, terms = r.step(t, s, died, False, died, rebase=rb, supported=sup,
                            lane_features=finfo if pass_features else None)
        out.append((rew, terms))
    return out


RUN_CFG = {"death_refund": 0.4, "pickup_weight": 0.5}

# Per-step rewards of sequence() under RUN_CFG, computed with ballrain/rewards.py as of 058b08b (before the lane term).
BEFORE = [
    -0.08757575757575872, 0.12242424242424357, 0.12242424242424357, 0.12242424242424356,
    0.12242424242424356, 0.12242424242424356, 0.12242424242424356, -0.08757575757575872,
    0.12242424242424357, 0.12242424242424356, 0.12242424242424356, 0.12242424242424356,
    0.12242424242424356, 0.12242424242424356, -0.08757575757575872, 0.12242424242424356,
    0.12242424242424356, 0.12242424242424357, 0.12242424242424357, 0.12242424242424357,
    -0.007575757575757564, -0.007575757575757564, -0.007575757575757564, -0.007575757575757564,
    0.4324242424242459, 0.12242424242424357, 0.12242424242424357, 0.12242424242424357,
    -0.08757575757575871, 0.12242424242424357, 50.122424242424245, 0.12242424242424357,
    0.12242424242424357, 0.12242424242424357, 0.12242424242424357, -0.08757575757575871,
    0.12242424242424357, 0.12242424242424357, 0.12242424242424357, 0.12242424242424357,
    0.7224242424242436, 0.12242424242424357, -0.08757575757575871, 0.12242424242424357,
    0.12242424242424357, 0.12242424242424357, 0.12242424242424357, 0.12242424242424357,
    0.12242424242424357, -0.08757575757575871, 0.12242424242424357, 0.12242424242424357,
    0.12242424242424357, 0.12242424242424357, 0.12242424242424357, 0.12242424242424357,
    -0.08757575757575871, 0.12242424242424357, 0.12242424242424357, -105.56757575757577,
]


def test_off_is_bit_identical_to_before():
    for pass_features in (False, True):
        got = [rew for rew, _ in run(RewardConfig(mode="sr", **RUN_CFG), pass_features)]
        assert got == BEFORE
        assert all("lane" not in terms for _, terms in run(RewardConfig(mode="sr", **RUN_CFG), pass_features))


def _cfg(**kw):
    return RewardConfig(mode="sr", progress_weight=0.0, time_weight=0.0, death_penalty=0.0, death_refund=0.0,
                        hold_unsupported=False, lane_s=(150.0, 300.0), lane_width=0.5, **kw)


def _step(r, t, s, pos, supported=True, died=False, rebase=False):
    finfo = {"pos": pos, "path": {"s": s, "point": [0.0, 0.0, s], "tangent": [0.0, 0.0, 2.0]}}
    return r.step(t, s, died, False, died, rebase=rebase, supported=supported, lane_features=finfo)[1].get("lane", 0.0)


def test_pays_new_max_s_only_scaled_by_closeness():
    t = T()
    r = Reward(_cfg(lane_weight=0.1))
    r.begin(t, 149.0)
    assert _step(r, t, 151.0, [0.0, 0.0, 151.0]) == pytest.approx(0.1 * 1.0)  # only the metre inside the window
    assert _step(r, t, 152.0, [0.3, 0.4, 152.0]) == pytest.approx(0.0)  # 0.5 m off the line
    assert _step(r, t, 153.0, [0.0, 0.25, 153.0]) == pytest.approx(0.1 * 0.5 * 1.0)
    assert _step(r, t, 151.0, [0.0, 0.0, 151.0]) == 0.0  # rolled back: nothing, s_max stays
    assert _step(r, t, 153.0, [0.0, 0.0, 153.0]) == 0.0  # forward again up to s_max: nothing
    assert _step(r, t, 154.0, [0.0, 0.0, 154.0]) == pytest.approx(0.1)
    assert _step(r, t, 160.0, [0.0, 0.0, 160.0], rebase=True) == 0.0  # re-acquired projection: s_max only moves
    assert _step(r, t, 161.0, [0.0, 0.0, 161.0]) == pytest.approx(0.1)
    assert _step(r, t, 302.0, [0.0, 0.0, 302.0]) == pytest.approx(0.1 * (300.0 - 161.0))  # capped at the window end
    assert _step(r, t, 310.0, [0.0, 0.0, 310.0]) == 0.0


def test_held_without_support_and_forfeited_at_death():
    t = T()
    cfg = _cfg(lane_weight=0.1)
    cfg.hold_unsupported = True
    r = Reward(cfg)
    r.begin(t, 150.0)
    assert _step(r, t, 152.0, [0.0, 0.0, 152.0], supported=False) == 0.0
    assert _step(r, t, 153.0, [0.0, 0.0, 153.0], supported=True) == pytest.approx(0.3)  # held pay at the landing
    assert _step(r, t, 155.0, [0.0, 0.0, 155.0], supported=False) == 0.0
    assert _step(r, t, 156.0, [0.0, 0.0, 156.0], supported=False, died=True) == 0.0  # forfeited
    r.begin(t, 200.0)  # a new life: s_max from its own start
    assert _step(r, t, 201.0, [0.0, 0.0, 201.0]) == pytest.approx(0.1)


def test_validation():
    for kw in ({"lane_weight": -0.1}, {"lane_weight": float("nan")}, {"lane_weight": 0.1, "lane_width": 0.0},
               {"lane_weight": 0.1, "lane_s": (300.0, 150.0)}, {"lane_weight": 0.1, "lane_s": (-1.0, 10.0)},
               {"lane_weight": 0.1, "lane_s": ()}, {"lane_weight": 0.1, "mode": "hs"}):
        with pytest.raises(ValueError):
            Reward(RewardConfig(**{"lane_s": (150.0, 300.0), **kw}))
    r = Reward(RewardConfig(lane_weight=0.1, lane_s=[150.0, 300.0]))
    r.check_lane_window(2397.7)
    with pytest.raises(ValueError):
        r.check_lane_window(250.0)
    Reward(RewardConfig()).check_lane_window(10.0)  # off: no window check

"""Score / life ball tracking (ballrain.pickups), the pickup block layout and the pickup shaping."""
import numpy as np

from ballrain import pickups as P
from ballrain.features import FeatureConfig, layout_size, make_layout
from ballrain.rewards import Reward, RewardConfig


class T:
    def __init__(self, sector=1):
        self.points, self.lives, self.life_bonus, self.sector = 1000, 3, 200, sector
        self.sr_ms, self.checkpoints = 0.0, 0
        self.points_at_finish = self.lives_at_finish = self.followers_at_finish = None

    def followers(self):
        return 0

    def hs_partial(self):
        return self.points + 200 * self.lives


def item(name):
    return next(it for it in P.load(1) if it[0] == name)


def test_level1_placements():
    items = P.load(1)
    assert sum(k == "point" for _, k, _, _ in items) == 11 and sum(k == "life" for _, k, _, _ in items) == 4


def test_collect_by_counters_and_respawn():
    t = T(sector=3)
    pk = P.PickupState(1)
    pk.start(t)
    pos = np.asarray(item("P_Extra_Point_10")[3]) + [2.5, 0, 0]
    t.points += 100 - 1
    assert pk.update(t, pos) == ["P_Extra_Point_10"]
    t.points += 120  # the collect macro's small balls: no core
    assert pk.update(t, pos) == []
    life = np.asarray(item("P_Extra_Life_02")[3])
    t.lives += 1
    assert pk.update(t, life) == ["P_Extra_Life_02"]
    assert pk.missed([3]) == 1  # P_Extra_Point_03 still there
    pk.respawn(t)  # the sector's life ball comes back, the score ball does not
    assert "P_Extra_Life_02" not in pk.collected and "P_Extra_Point_10" in pk.collected


def test_nearest_in_sector_only():
    t = T(sector=3)
    pk = P.PickupState(1)
    pk.start(t, ["P_Extra_Point_10"])
    near = pk.nearest(item("P_Extra_Point_03")[3], 3, 2)
    assert near[0][0] == 0.0 and near[0][1] == "point" and len(near) == 2  # Point_03 and Life_02 left in sector 3


def test_layout_appends_pickup_block():
    a = FeatureConfig(version="f8", reach=True)
    b = FeatureConfig(version="f8", reach=True, pickups=True)
    la, lb = make_layout(a), make_layout(b)
    assert lb[:len(la)] == la and [n for n, _ in lb[len(la):]] == ["pickups", "pickups_left"]
    assert layout_size(b) - layout_size(a) == P_SLOTS * 6 + 2


P_SLOTS = 2


def test_shaping_is_potential_based():
    c = RewardConfig(pickup_shaping=10.0, pickup_shaping_range=30.0, gamma=1.0, progress_weight=0.0, time_weight=0.0)
    r = Reward(c)
    t = T()
    r.begin(t, 0.0, pickup_dist=30.0)
    tot = 0.0
    for d in (20.0, 10.0, 3.0, 40.0, 25.0):
        tot += r.step(t, 0.0, False, False, False, pickup_dist=d)[1]["pickup_shaping"]
    assert abs(tot - (10.0 * (1 - 25 / 30) - 0.0)) < 1e-9  # telescopes: Phi(end) - Phi(start)
    assert abs(r.step(t, 0.0, False, True, True, pickup_dist=5.0)[1]["pickup_shaping"] + 10.0 * (1 - 25 / 30)) < 1e-9

"""SR reward with score pickups (RewardConfig.pickup_weight): cores, small balls, life balls, finish, death."""
from ballrain.rewards import Reward, RewardConfig


class T:
    def __init__(self):
        self.points, self.lives, self.life_bonus = 1000, 3, 200
        self.sr_ms, self.checkpoints = 0.0, 0
        self.points_at_finish = self.lives_at_finish = self.followers_at_finish = None
        self.f = 0

    def followers(self):
        return self.f

    def hs_partial(self):
        return self.points + 200 * self.lives


def step(r, t, died=False, finished=False):
    return r.step(t, 0.0, died, finished, died or finished)[1]


def test_pickups_paid_once_and_decay_not_charged():
    t = T()
    r = Reward(RewardConfig(pickup_weight=0.1, time_weight=0.0, progress_weight=0.0))
    r.begin(t, 0.0)
    t.points -= 1  # a time decrement alone
    assert "pickup" not in step(r, t)
    t.points += 100 - 1  # core touched in a step with a decrement
    assert abs(step(r, t)["pickup"] - 10.0) < 1e-9
    t.points += 120  # collect macro: six small balls
    assert abs(step(r, t)["pickup"] - 12.0) < 1e-9
    t.lives += 1
    assert abs(step(r, t)["pickup"] - 20.0) < 1e-9


def test_finish_pending_followers_and_death():
    t = T()
    r = Reward(RewardConfig(pickup_weight=0.1))
    r.begin(t, 0.0)
    t.points_at_finish, t.lives_at_finish, t.followers_at_finish = 1000, 3, 2
    terms = step(r, t, finished=True)
    assert abs(terms["pickup_followers_finish"] - 4.0) < 1e-9 and "pickup" not in terms
    r.begin(T(), 0.0)
    assert abs(step(r, T(), died=True)["pickup_death_life"] + 20.0) < 1e-9


def test_off_by_default():
    t = T()
    r = Reward(RewardConfig())
    r.begin(t, 0.0)
    t.points += 100
    assert not any(k.startswith("pickup") for k in step(r, t))

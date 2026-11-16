"""Reward functions, the BML+ 0.3.12 SR timer and run recording, with fake trackers / frames."""
import json

import numpy as np
import pytest

from ballrain import protocol as P, tas
from ballrain.path import ReferencePath
from ballrain.protocol import Ev
from ballrain.recording import RunRecorder, sr_timer
from ballrain.rewards import Reward, RewardConfig
from ballrain.state import GameTracker

D = np.float32(1000 / 132)


class FakeTracker:
    def __init__(self):
        self.sr_ms = 0.0
        self.points, self.lives, self.life_bonus, self.level_bonus = 1000, 3, 200, 100
        self.checkpoints = 0
        self.n_followers = 0
        self.followers_at_finish = None

    def hs_partial(self):
        return self.points + self.lives * self.life_bonus

    def followers(self):
        return self.n_followers


def test_sr_reward_terms_and_shaping_telescopes():
    cfg = RewardConfig(mode="sr", progress_weight=0.1, time_weight=1.0, finish_bonus=10, death_penalty=5,
                       shaping_weight=0.5, gamma=1.0)
    r = Reward(cfg)
    t = FakeTracker()
    r.begin(t, 0.0)
    total = 0.0
    for k in range(1, 11):
        t.sr_ms = 100.0 * k
        rew, terms = r.step(t, 2.0 * k, died=False, finished=k == 10, terminal=k == 10)
        total += rew
    # progress 20 units * 0.1, time 1 s, finish 10; shaping with gamma 1 telescopes to Phi(T)=0 - Phi(0)=0
    assert total == pytest.approx(2.0 - 1.0 + 10.0 + 0.0)
    assert r.terms["shaping"] == pytest.approx(0.0)
    r.begin(t, 0.0)
    rew, terms = r.step(t, -4.0, died=True, finished=False, terminal=True)
    assert terms["death"] == -5 and terms["progress"] == pytest.approx(-0.4)


def test_sr_progress_without_support_is_held_until_landing_and_dropped_at_death():
    cfg = RewardConfig(mode="sr", progress_weight=1.0, time_weight=0.0, finish_bonus=0, death_penalty=5)
    t = FakeTracker()
    r = Reward(cfg)
    r.begin(t, 0.0)
    assert r.step(t, 10.0, False, False, False, supported=True)[0] == 10
    # a jump: 30 units in the air, paid at the landing
    assert r.step(t, 25.0, False, False, False, supported=False)[0] == 0
    assert r.step(t, 40.0, False, False, False, supported=False)[0] == 0 and r.held == 30
    assert r.step(t, 41.0, False, False, False, supported=True)[0] == 31 and r.held == 0
    # flying into the void: nothing is paid, the sector's progress (41) is taken back at BallOff
    r.step(t, 60.0, False, False, False, supported=False)
    rew, terms = r.step(t, 80.0, died=True, finished=False, terminal=True, supported=False)
    assert terms["progress"] == 0 and terms["death"] == -5 and terms["death_refund"] == -41
    assert r.total == pytest.approx(-5)
    # a held backward move is never forgiven; progress held at LevelFinish is paid
    r.begin(t, 100.0)
    r.step(t, 90.0, False, False, False, supported=False)
    assert r.step(t, 120.0, False, True, True, supported=False)[1]["progress"] == 20
    r.begin(t, 100.0)
    r.step(t, 90.0, False, False, False, supported=False)
    rew, terms = r.step(t, 95.0, True, False, True, supported=False)
    assert terms["progress"] == -5 and terms["death_refund"] == 0  # net held -5 is charged


def test_sr_death_refund_only_covers_the_current_sector():
    cfg = RewardConfig(mode="sr", progress_weight=0.1, time_weight=0.0, death_penalty=5)
    t = FakeTracker()
    r = Reward(cfg)
    r.begin(t, 0.0)
    r.step(t, 100.0, False, False, False, supported=True)
    t.checkpoints = 1  # a checkpoint: the respawn point moves, the 10 earned so far are kept
    r.step(t, 150.0, False, False, False, supported=True)
    rew, terms = r.step(t, 160.0, True, False, True, supported=True)
    # the step that crossed the checkpoint counts for sector 2; the dying step's own progress is not paid
    assert terms["progress"] == 0 and terms["death_refund"] == pytest.approx(-5.0)
    assert r.total == pytest.approx(15 - 5 - 5)
    off = Reward(RewardConfig(mode="sr", progress_weight=0.1, time_weight=0.0, hold_unsupported=False, death_refund=0))
    t.checkpoints = 0
    off.begin(t, 0.0)
    off.step(t, 30.0, False, False, False, supported=False)
    assert off.step(t, 40.0, True, False, True, supported=False)[1] == {"progress": 1.0, "time": 0.0, "death": -5}


def test_hs_reward_score_balls_lives_and_finish():
    r = Reward(RewardConfig(mode="hs", score_scale=1.0))
    t = FakeTracker()
    r.begin(t, None)
    t.points = 1100  # a score ball (+100)
    assert r.step(t, None, False, False, False)[0] == 100
    t.points, t.lives = 1099, 4  # time decay -1, life ball +200
    assert r.step(t, None, False, False, False)[0] == 199
    rew, terms = r.step(t, None, died=True, finished=False, terminal=True)
    assert terms["death"] == -200
    rew, terms = r.step(t, None, died=False, finished=True, terminal=True)
    assert terms["finish"] == 100 and terms["followers_finish"] == 0


def test_hs_reward_followers_arrivals_losses_and_finish_credit():
    """+100 on the hit, +20 per arrival (both through Points), lost balls never pay, the balls still
    following at LevelFinish are credited at finish (official readout)."""
    r = Reward(RewardConfig(mode="hs", score_scale=1.0))
    t = FakeTracker()
    r.begin(t, None)
    t.points, t.n_followers = 1100, 6
    assert r.step(t, None, False, False, False)[0] == 100
    t.points, t.n_followers = 1140, 4  # two arrivals
    assert r.step(t, None, False, False, False)[0] == 40
    t.n_followers = 0  # the other four are lost at a checkpoint
    assert r.step(t, None, False, False, False)[0] == 0
    t.points, t.n_followers = 1240, 6  # another score ball, nothing arrives before the finish
    t.followers_at_finish = 6
    rew, terms = r.step(t, None, died=False, finished=True, terminal=True)
    assert terms["followers_finish"] == 120 and rew == 100 + 100 + 120


def test_hs_follower_shaping_is_potential_based():
    cfg = RewardConfig(mode="hs", score_scale=1.0, hs_follower_shaping=1.0, gamma=1.0)
    r = Reward(cfg)
    t = FakeTracker()
    r.begin(t, None)
    t.points, t.n_followers = 1100, 6
    rew, terms = r.step(t, None, False, False, False)
    assert terms["follower_shaping"] == 120
    t.n_followers = 0  # lost: the shaping is taken back
    rew, terms = r.step(t, None, False, False, False)
    assert terms["follower_shaping"] == -120
    t.n_followers = 3
    r.step(t, None, False, False, False)
    t.followers_at_finish = 3
    rew, terms = r.step(t, None, died=False, finished=True, terminal=True)
    assert terms["follower_shaping"] == -60 and terms["followers_finish"] == 60
    assert r.terms["follower_shaping"] == 0  # telescopes to Phi(terminal) - Phi(0) = 0 with gamma 1


def test_reward_rejects_unknown_mode():
    with pytest.raises(ValueError):
        Reward(RewardConfig(mode="xx"))


def test_reference_path_projection_and_sector_window():
    p = ReferencePath([[0, 0, 0], [0, 0, 100], [100, 0, 100], [100, 0, 0]])
    assert p.length == 300
    pr = p.project([1, 5, 50])
    assert pr["s"] == pytest.approx(50) and pr["seg"] == 0 and pr["tangent"].tolist() == [0, 0, 1]
    # restricted to sector 3 (segment 2), the projection cannot jump to another part of the path
    pr = p.project([90, 0, 5], sector=3, margin=0)
    assert pr["seg"] == 2 and pr["s"] == pytest.approx(300 - 5)
    pr = p.project([90, 0, 5], sector=1, margin=0)  # sector 1 = segment 0 only
    assert pr["seg"] == 0 and pr["s"] == pytest.approx(5)
    assert p.point_at(150).tolist() == [50, 0, 100]
    assert p.next_nodes(2, 2).tolist() == [[100, 0, 0], [100, 0, 0]]
    assert p.points_ahead(290, 2, 5.0).tolist() == [[100, 0, 5], [100, 0, 0]]


def _ev(*pairs):
    return [(n, c, 0) for n, c in pairs]


def test_sr_timer_rules():
    frames = tas.make_frames(np.full(10, D), np.zeros(10, np.uint32))
    # CounterActive in frame index 2 (event number 3) counts that frame; CounterInactive in frame index 6
    # (number 7) stops before that frame's delta: frames 2..5 = 4 deltas
    r = sr_timer(frames, _ev((1, Ev.StartLevel), (3, Ev.CounterActive), (7, Ev.CounterInactive)))
    assert r["double"] == 4 * float(D) and r["ms"] == int(4 * float(D)) and not r["running"]
    # pause / unpause, and a StartLevel resets
    r = sr_timer(frames, _ev((1, Ev.CounterActive), (3, Ev.PauseLevel), (5, Ev.UnpauseLevel)))
    assert r["double"] == 8 * float(D)
    r = sr_timer(frames, _ev((1, Ev.CounterActive), (9, Ev.StartLevel)))
    assert r["double"] == 0.0
    # the accumulator is a double sum of f32 deltas, not an f32 sum
    many = tas.make_frames(np.full(20000, D), np.zeros(20000, np.uint32))
    r = sr_timer(many, _ev((1, Ev.CounterActive)))
    assert r["double"] == sum(float(D) for _ in range(20000))
    assert r["double"] != float(np.cumsum(np.full(20000, D, np.float32))[-1])


def test_run_recorder_exports_tas_and_meta(tmp_path):
    rec = RunRecorder(tmp_path, "run1", 3, mode="sr", seed=7, copy="train-03")
    a = tas.make_frames(np.full(5, D), [0, 1, 1, 2, 2])
    rec.add_frames(a, executed=3)  # stop-on-event: the last two frames are void
    rec.add_frames(tas.make_frames([D], [8]))
    rec.begin_life(0)
    rec.events = [(1, Ev.CounterActive, 0)]
    rec.end_life(4, "BallOff")
    mod = tas.make_frames(np.full(4, D), [0, 1, 1, 8])
    tas.save_raw(tmp_path / "framelog.bin", mod)
    meta_path = rec.finish(None, "GameOver", tmp_path / "framelog.bin", None)
    m = json.loads(meta_path.read_text())
    assert m["framelog_match"] is True and m["frames"] == 4 and m["copy"] == "train-03" and m["seed"] == 7
    assert m["lives"][0]["end_reason"] == "BallOff" and m["events"] == [[1, "CounterActive", 0]]
    assert tas.load(tmp_path / "run1.tas").tobytes() == mod.tobytes()
    # a differing mod log is reported, and the mod's log (what was injected) is exported
    rec2 = RunRecorder(tmp_path, "run2", 3)
    rec2.add_frames(tas.make_frames(np.full(4, D), [0, 1, 2, 8]))
    m2 = json.loads(rec2.finish(None, "GameOver", mod, None).read_text())
    assert m2["framelog_match"] is False and m2["framelog_mismatch"]["first_diff"] == 2
    assert tas.load(tmp_path / "run2.tas").tobytes() == mod.tobytes()


def test_tracker_events_arrays_and_hs():
    t = GameTracker(2)
    lm = P.LevelMessage(0, arrays=[
        P.ArrayDef(1, "Energy", [P.ArrayColumn(n, 1, None) for n in ("Points", "Lifes", "SP", "SL", "TF", "LifeBonus")],
                   [[np.int32(v).tobytes() for v in (0, 0, 1000, 3, 500, 200)]]),
        P.ArrayDef(2, "AllLevel", [P.ArrayColumn(f"c{i}", 1, None) for i in range(7)],
                   [[np.int32(0).tobytes()] * 6 + [np.int32(100 * (r + 1)).tobytes()] for r in range(13)])])
    o = P.Observation(5, arrs=[P.ArrayCell(1, 0, 0, 1, np.int32(990).tobytes())])  # delta before the level msg
    t.update(o)
    t.merge_level(lm)
    assert t.points == 990  # the newer obs delta is not overwritten by the level snapshot
    o = P.Observation(6, arrs=[P.ArrayCell(1, 0, 1, 1, np.int32(3).tobytes())],
                      events=np.array([(6, Ev.PostCheckpoint, 0), (6, Ev.LevelFinish, 0)], P.EVT_DT))
    t.update(o)
    assert t.checkpoints == 1 and t.sector == 2 and t.flags.finished
    assert t.hs_at_finish == 990 + 3 * 200 + 200 and t.levelfinish_frame == 6

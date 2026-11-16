"""HS mechanics through the env against ballrain.fakegame: followers, the Esc action, the collect macro,
the pause cap, losses at a checkpoint and the official HS readout at LevelFinish. No game is launched."""
import json
import sys

import numpy as np
import pytest

from ballrain import fakegame, tas
from ballrain.env import COLLECT, ESC, BallanceEnv, EnvConfig

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="named shared memory / events need Windows")

UP = 1


def _env(tmp_path, **kw):
    game = kw.pop("game", {})
    cfg = EnvConfig(level=2, out_root=str(tmp_path), action_repeat=4, mode="hs", **kw)
    return BallanceEnv(cfg, launcher=fakegame.FakeLauncher(**game))


def _until(env, action, cond, limit=2000):
    for _ in range(limit):
        obs, r, term, trunc, info = env.step(action)
        if cond(info) or term or trunc:
            return obs, r, term, trunc, info
    raise AssertionError("condition not reached")


def test_esc_action_defaults_by_mode(tmp_path):
    assert ESC in _env(tmp_path).actions and COLLECT not in _env(tmp_path).actions
    assert ESC not in BallanceEnv(EnvConfig(mode="sr")).actions
    assert COLLECT in _env(tmp_path, collect_action=True).actions
    with pytest.raises(ValueError):
        _env(tmp_path, esc_frames=2)


def test_collect_macro_pauses_collects_and_exports_the_pause(tmp_path):
    env = _env(tmp_path, collect_action=True)
    env.reset()
    _, _, _, _, info = _until(env, UP, lambda i: i["followers"] > 0)
    assert info["followers"] == 6
    t = env.session.tracker
    pts, sr = t.points, t.sr_ms
    obs, r, term, trunc, info = env.step(env.actions.index(COLLECT))
    c = info["collect"]
    assert c["result"] == "done" and c["pause"] == "paused" and c["unpause"] == "unpaused"
    assert c["followers_before"] == 6 and c["followers_after"] == 0 and info["followers"] == 0
    assert t.points == pts + 120 and r == pytest.approx(0.01 * 120)  # no time decay while paused
    names = [e for _, e in info["events"]]
    assert "PauseLevel" in names and "UnpauseLevel" in names and not t.flags.paused
    # the SR clock ran only on the frames outside the pause
    assert t.sr_ms - sr < (env.cfg.esc_frames * 2 + 1) * float(np.float32(1000 / 132))
    sl = env.fx.slices()
    assert obs[sl["followers"]][0] == 0 and obs[sl["paused"]][0] == 0
    _, _, _, _, info = _until(env, UP, lambda i: False and i)
    meta = json.loads(open(info["run_meta"], encoding="utf-8").read())
    assert meta["framelog_match"] is True
    frames = tas.load(tmp_path / "train-00" / meta["run_id"] / meta["tas_path"])
    esc = np.flatnonzero(frames["keys"] & tas.KEY_ESC)
    assert len(esc) == 2 and esc[1] - esc[0] > 10  # pause and unpause presses are in the exported .tas
    env.close()


def test_pause_masks_keys_and_is_capped(tmp_path):
    env = _env(tmp_path, max_pause_frames=40)
    env.reset()
    _until(env, UP, lambda i: i["frames_done"] > 60)
    obs, r, term, trunc, info = env.step(env.actions.index(ESC))
    assert info["esc"] == "paused" and info["paused"]
    sl = env.fx.slices()
    assert obs[sl["paused"]][0] == 1
    pos = env.session.tracker.obs.ball["pos"].copy()
    forced = None
    for _ in range(30):
        obs, r, term, trunc, info = env.step(UP)  # masked while paused: the ball does not move
        if "forced_unpause" in info:
            forced = info["forced_unpause"]
            break
        assert np.allclose(env.session.tracker.obs.ball["pos"], pos) and r == 0
    assert forced == "unpaused" and not info["paused"]
    frames = env.session.frame_log()
    keys = frames["keys"][-40:]
    assert not np.any(keys & tas.KEY_UP)  # pause_keys (0) were injected
    env.close()


def test_followers_lost_at_checkpoint_never_pay(tmp_path):
    env = _env(tmp_path)
    env.reset()
    _until(env, UP, lambda i: i["sector"] == 2 and i["frames_done"] > 0)
    t = env.session.tracker
    _until(env, UP, lambda i: t.frames_done > t.last_checkpoint_frame + 4)
    assert t.followers() == 0 and t.follower_state()["total"] == 0
    # score ball 2 (just before the checkpoint): +100 only, its six small balls were lost
    hits = sum(1 for _, c, _ in t.events if c == 18)
    assert hits >= 2
    env.close()


def test_levelfinish_credits_the_followers_still_pending(tmp_path):
    env = _env(tmp_path, game={"score_ball_2": (0.0, 2.0, -192.0)})
    env.reset()
    total, info = 0.0, None
    while True:
        obs, r, term, trunc, info = env.step(UP)
        total += r
        if term or trunc:
            break
    assert info["episode_end"] == "LevelFinish"
    meta = json.loads(open(info["run_meta"], encoding="utf-8").read())
    hp = meta["hs_parts"]
    assert hp["source"] == "tail_end" and hp["followers"] > 0 and hp["followers_at_hook"] == hp["followers"]
    assert meta["hs"] == hp["points"] + 20 * hp["followers"] + hp["lives"] * 200 + hp["level_bonus"]
    assert meta["hs"] == meta["hs_mod"] + 20 * hp["followers"] and not meta["notes"]
    assert info["reward_terms"]["followers_finish"] == pytest.approx(0.01 * 20 * hp["followers"])
    assert total == pytest.approx(0.01 * (meta["hs"] - (1000 + 3 * 200)), abs=1e-9)
    env.close()


def test_repeated_pauses_are_bounded_per_episode(tmp_path):
    """Forced unpauses alone do not bound a policy that pauses again at once: the episode pause budget does."""
    env = _env(tmp_path, max_pause_frames=30, max_episode_pause_frames=100, collect_action=True)
    env.reset()
    _until(env, UP, lambda i: i["frames_done"] > 60)
    esc = env.actions.index(ESC)
    results = []
    for _ in range(80):
        obs, r, term, trunc, info = env.step(esc if not env.session.tracker.flags.paused else UP)
        results.append(info.get("esc"))
        assert not term and not trunc
    assert info["episode_pause_frames"] <= 100 + 30 + 2 * env.cfg.esc_frames
    assert results[-1] == "masked" and not info["paused"]
    obs, r, term, trunc, info = env.step(env.actions.index(COLLECT))
    assert info["collect"]["result"] == "noop"
    env.close()

"""TAS start states, the reverse curriculum and the warm pool (ballrain.curriculum, ballrain.env) against
ballrain.fakegame. No game is launched."""
import json
import sys

import numpy as np
import pytest

from ballrain import curriculum as CU, dataset, fakegame, tas
from ballrain.env import BallanceEnv, EnvConfig

D = np.float32(1000 / 132)


def test_curriculum_moves_back_on_success_and_round_trips():
    valid = np.arange(1000, 20001, 10)
    cu = CU.ReverseCurriculum(valid=valid, config=CU.CurriculumConfig(start_back=1500, step=1000, band=1500,
                                                                       min_episodes=10, advance_at=0.5, seed=1))
    assert cu.frontier == 18500
    ks = [cu.sample() for _ in range(500)]
    assert min(ks) >= 18500 and max(ks) <= 20000 and all(k in set(valid.tolist()) for k in ks)
    assert not cu.update(5000, True)  # outside the band: ignored
    for _ in range(9):
        assert not cu.update(18600, True)
    assert cu.update(18600, True) and cu.frontier == 17500
    for _ in range(10):
        cu.update(17600, False)  # failures: stays
    assert cu.frontier == 17500
    st = cu.state_dict()
    cu2 = CU.ReverseCurriculum(valid=valid)
    cu2.load_state_dict(json.loads(json.dumps(st)))
    assert cu2.frontier == 17500 and cu2.cfg.step == 1000
    # all the way back: k = 0 (from level load) joins with p_level_start
    cu2.load_state_dict({"frontier": 0})
    assert cu2.at_start and 0 in {cu2.sample() for _ in range(200)}
    assert CU.episode_outcome({"start": {"kind": "tas", "frame": 18600}, "episode_end": "LevelFinish"}) == (18600, True)
    assert CU.episode_outcome({"start": {"kind": "respawn", "frame": 3}, "episode_end": "BallOff"}) is None


pytestmark_win = pytest.mark.skipif(sys.platform != "win32", reason="named shared memory / events need Windows")


@pytest.fixture
def fake_record(tmp_path):
    """A fake-level dataset record: 600 frames, up from frame 40 (built through the dataset replay policy)."""
    keys = np.zeros(600, np.uint32)
    keys[40:] = tas.KEY_UP
    rec = tmp_path / "SR01_0_00_000_132.tas"
    tas.save(rec, tas.make_frames(np.full(600, D), keys))
    res = dataset.build_dataset([rec], tmp_path / "ds", copies=["train-00"], launcher_factory=fakegame.FakeLauncher)
    assert res[0]["ok"]
    return tmp_path / "ds" / rec.stem


@pytestmark_win
def test_tas_starts_and_env_hand_over(tmp_path, fake_record):
    st = CU.TasStarts(fake_record)
    assert st.level == 1 and len(st.valid) > 100 and (fake_record / "starts.npz").is_file()
    k = st.nearest_valid(300)
    assert k == 300 and st.expected(k)["frame"] == 300
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path / "ep"), action_repeat=4, start_record=str(fake_record),
                                start_frame=300), launcher=fakegame.FakeLauncher())
    obs, info = env.reset()
    s = info["start"]
    assert s["kind"] == "tas" and s["frame"] == 300 and s["control_frame"] == 300 and s["new_process"]
    assert info["frames_done"] == 300
    # the hand-over state equals the recorded state at frame 300
    exp = st.expected(300)
    assert env.session.obs.ball["pos"] == pytest.approx(exp["pos"]) and info["sector"] == exp["sector"]
    assert info["sr_ms"] == pytest.approx(exp["sr_double"]) and s["sr_ms"] == pytest.approx(exp["sr_double"])
    done = False
    while not done:
        _, _, term, trunc, info = env.step(3)  # left: off the road
        done = term or trunc
    assert info["start"]["frame"] == 300
    env.close()
    meta = json.loads(open(env.finished_runs[0], encoding="utf-8").read())
    assert meta["start_state"]["frame"] == 300 and meta["start_state"]["record"] == fake_record.name
    assert meta["prefix"]["frames"] == 300 and meta["framelog_match"] and meta["timing"]["k"] == 300
    frames = tas.load(env.finished_runs[0].with_suffix(".tas"))
    rec = tas.load(tmp_path / "SR01_0_00_000_132.tas")
    assert frames[:300].tobytes() == rec[:300].tobytes() and len(frames) > 300  # prefix + agent frames


@pytestmark_win
def test_level_start_k0_presses_q_and_curriculum_api(tmp_path, fake_record):
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path / "ep"), action_repeat=4, start_record=str(fake_record),
                                curriculum={"start_back": 50, "seed": 3}), launcher=fakegame.FakeLauncher())
    obs, info = env.reset()
    assert info["start"]["kind"] == "tas" and info["start"]["frame"] >= env.starts.valid[-1] - 50
    env.set_start_state({"frontier": 0, "config": {**env.get_start_state()["config"], "p_level_start": 1.0}})
    env.reset(options={"new_process": True})
    assert env._start["frame"] == 0 and env.recorder.meta.prefix is None
    log = env.session.frame_log()
    assert (log["keys"] & tas.KEY_Q).any()  # level 1 from level load: the scripted Q
    env.close()


@pytestmark_win
def test_warm_pool_alternates_copies(tmp_path, fake_record):
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path / "ep"), action_repeat=4, start_record=str(fake_record),
                                curriculum={"start_back": 100, "seed": 1}, warm_pool=True, warm_copy="train-01"),
                      launcher=fakegame.FakeLauncher())
    copies, kinds = [], []
    while len(copies) < 3:
        obs, info = env.reset()  # a respawn in the same process, or the pre-launched process after GameOver
        kinds.append(info["start"]["kind"])
        if info["start"]["new_process"]:
            copies.append(info["start"]["copy"])
        done = False
        while not done:
            _, _, term, trunc, info = env.step(3)
            done = term or trunc
    env.close()
    assert copies == ["train-00", "train-01", "train-00"] and "respawn" in kinds
    assert len(env.finished_runs) == 3 and env._warm is None
    for p in env.finished_runs:
        assert json.loads(open(p, encoding="utf-8").read())["framelog_match"]


def test_vecenv_pairs_copies_for_the_warm_pool():
    from ballrain import vecenv
    fns = vecenv.env_fns(EnvConfig(level=1, warm_pool=True, curriculum={}), 4, seed=10)
    pairs = [f.args[1] for f in fns]
    assert pairs == [("train-00", "train-01"), ("train-02", "train-03"), ("train-04", "train-05"),
                     ("train-06", "train-07")]
    with pytest.raises(ValueError):
        vecenv.env_fns(EnvConfig(level=1, warm_pool=True), 2, copies=["train-00", "train-01", "train-02"])


@pytestmark_win
def test_mismatching_hand_over_is_relaunched(tmp_path, fake_record):
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path / "ep"), action_repeat=4, start_record=str(fake_record),
                                start_frame=300), launcher=fakegame.FakeLauncher())
    real = env.starts.expected
    calls = []

    def flaky(k):  # the first hand-over "diverged"
        e = real(k)
        calls.append(k)
        if len(calls) == 1:
            e = dict(e, pos=[0.0, 0.0, 0.0])
        return e
    env.starts.expected = flaky
    obs, info = env.reset()
    assert info["start"]["handover_mismatches"] == 1 and len(calls) == 2
    assert len(list((tmp_path / "ep" / "train-00").glob("*/start_mismatch.json"))) == 1
    env.close()
    env.starts.expected = lambda k: dict(real(k), lives=99)  # never matches: gives up after the retries
    env2 = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path / "ep2"), action_repeat=4, start_record=str(fake_record),
                                 start_frame=300, start_retries=1), launcher=fakegame.FakeLauncher())
    env2.starts.expected = lambda k: dict(real(k), lives=99)
    with pytest.raises(RuntimeError, match="differs from the dataset 2 times"):
        env2.reset()
    env2.close()


@pytestmark_win
def test_trainer_loop_through_a_vector_env(tmp_path, fake_record):
    """The documented trainer loop: vector_outcomes -> ReverseCurriculum.update_many -> venv.call."""
    import gymnasium as gym
    from ballrain import vecenv
    cfg = EnvConfig(level=1, out_root=str(tmp_path / "ep"), action_repeat=4, start_record=str(fake_record),
                    curriculum={"start_back": 50, "min_episodes": 2, "advance_at": 0.0, "step": 100, "band": 50},
                    max_episode_frames=8)
    venv = vecenv.make_vec_env(cfg, n=2, backend="sync", launcher_factory=fakegame.FakeLauncher, seed=1)
    cu = CU.ReverseCurriculum(CU.TasStarts(fake_record), CU.CurriculumConfig(**cfg.curriculum))
    f0 = cu.frontier
    venv.reset(seed=1)
    outcomes = []
    for _ in range(40):
        _, _, term, trunc, infos = venv.step(np.array([1, 1]))
        res = CU.vector_outcomes(infos)
        outcomes += res
        if cu.update_many(res):
            venv.call("set_start_state", cu.state_dict())
    venv.close()
    assert outcomes and all(not s for _, s in outcomes)  # truncated episodes are failures, but counted
    assert cu.frontier < f0
    assert CU.vector_outcomes([{"start": {"kind": "tas", "frame": 5}, "episode_end": "LevelFinish"}, {}]) == [(5, True)]


def test_failure_weighted_starts_keep_a_floor_per_bin():
    import numpy as np
    from ballrain.curriculum import CurriculumConfig, ReverseCurriculum
    cu = ReverseCurriculum(valid=np.arange(936, 20587), config=CurriculumConfig(
        weighted=1.0, bin_frames=2000, bin_floor=0.05, p_level_start=0.1, seed=3))
    cu.frontier = 936
    bins, p = cu.bin_probs()
    assert len(bins) == 10 and abs(p.sum() - 1) < 1e-9 and np.allclose(p, 0.1)  # no data: every bin counts as failing
    cu.set_bin_fail({b: 0.0 for b in range(10)} | {5: 1.0})
    _, p = cu.bin_probs()
    assert abs(p[5] - 0.55) < 1e-9 and np.allclose(np.delete(p, 5), 0.05)
    ks = np.array([cu.sample() for _ in range(20000)])
    assert 0.07 < np.mean(ks == 0) < 0.13  # k = 0 draws are kept at the start
    inb = np.mean([(cu.bin_of(k) == 5) for k in ks[ks > 0]])
    assert 0.5 < inb < 0.6
    st = cu.state_dict()
    cu2 = ReverseCurriculum(valid=np.arange(936, 20587))
    cu2.load_state_dict(st)
    assert cu2.bin_fail == cu.bin_fail and cu2.cfg.weighted == 1.0

"""Transport, agent session, gym env, vector env and dataset builder against ballrain.fakegame
(the mod's side of the real shared-memory transport, in a thread). No game is launched."""
import json
import sys
import threading

import gymnasium as gym
import numpy as np
import pytest

from ballrain import client, dataset, fakegame, protocol as P, tas, transport, vecenv
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.recording import sr_timer

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="named shared memory / events need Windows")

D = np.float32(1000 / 132)


def test_transport_lockstep_and_seq_gaps():
    name = transport.make_name("t")
    py = transport.PyChannel(name)
    mod = transport.ModChannel(name)
    try:
        assert py.header()["magic"] == P.SHM_MAGIC and py.header()["py_pid"] > 0
        mod.hello("Mod 1 abc")
        h = py.wait_hello(2)
        assert h["mod_state"] == P.ModState.HELLO and h["mod_version"] == "Mod 1 abc"
        # two observations without waiting (prefix events): Python only sees the latest
        mod.write_obs(P.build_message(P.OBS_MAGIC, 1, []), 1)
        mod.write_obs(P.build_message(P.OBS_MAGIC, 2, []), 2)
        o = P.parse_observation(py.wait_obs(2))
        assert o.frames_done == 2
        a = P.Action(tas.make_frames([D], [5]), repeat=3, stop_mask=P.event_mask("BallOff"))
        py.send_action(a)
        got = mod.wait_action(2)
        assert got.repeat == 3 and got.frames["keys"].tolist() == [5] and got.stop_mask == 1 << 12
        assert mod.wait_action(0.05) is None
        with pytest.raises(transport.Timeout):
            py.wait_obs(0.05)
        mod.set_state(P.ModState.ENDED, P.EndReason.PY_STOP)
        assert py.wait_obs(1, final_grace_s=0.1) is None
    finally:
        mod.close()
        py.close()


def _session(tmp_path, **kw):
    return client.AgentSession(client.SessionConfig(level=1, out_dir=tmp_path / "run", **kw), fakegame.FakeLauncher())


def test_session_env_vars_and_stop_on_event(tmp_path):
    s = _session(tmp_path, max_frames=5000)
    o = s.start()
    try:
        env = s.launcher.envs[0]
        assert env["BALLRAIN_MODE"] == "agent" and env["BALLRAIN_SHM"].startswith("Local\\ballrain-train-00-")
        assert env["BALLRAIN_FRAMES"] is None and env["BALLRAIN_MAX_FRAMES"] == "5000"
        assert o.frames_done == 0 and o.event_codes() == [P.Ev.PreLoadLevel]
        o = s.act(tas.make_frames([D], [0]), repeat=1000, stop=["StartLevel"])
        assert o.frames_done == fakegame.LOAD_FRAMES  # stopped right after the StartLevel frame
        assert s.n_logged == fakegame.LOAD_FRAMES  # the rest of the block is void
        assert s.tracker.level.by_name["A01_Floor_01"] == fakegame.ENT_FLOOR  # level message loaded
        assert s.tracker.arrays.get("Energy", 0, 0) == 1000
        s.run_until(0, ["BallNavActive"])
        assert s.tracker.flags.ball_nav and s.frames_done == fakegame.BALLNAV_AT
        s.stop()
        assert s.ended and s.end_reason == P.EndReason.PY_STOP
        assert tas.load_raw(s.mod_framelog()).tobytes() == s.frame_log().tobytes()
    finally:
        s.close()


def test_session_prefix_and_timeout(tmp_path):
    prefix = tas.make_frames(np.full(40, D), np.zeros(40, np.uint32))
    tas.save_raw(tmp_path / "prefix.bin", prefix)
    s = _session(tmp_path, prefix=tmp_path / "prefix.bin", timeout_ms=300)
    o = s.start()
    try:
        assert o.frames_done == 40 and s.n_logged == 40  # prefix replayed by the mod, logged by Python
        assert s.tracker.flags.ball_nav
        o = s.chan.wait_obs(5)  # no action: the mod times out and ends the run
        assert o is not None and s.chan.header()["end_reason"] == P.EndReason.TIMEOUT
    finally:
        s.close()


def test_session_prefix_events_are_acknowledged(tmp_path):
    prefix = tas.make_frames(np.full(40, D), np.zeros(40, np.uint32))
    tas.save_raw(tmp_path / "prefix.bin", prefix)
    s = _session(tmp_path, prefix=tmp_path / "prefix.bin", prefix_events=True, timeout_ms=3000)
    seen = []
    s.on_prefix_obs = lambda o: seen.append(o.frames_done)
    o = s.start()
    try:
        assert seen == list(range(1, 40)) and s.n_prefix_obs == 39  # one per prefix frame but the last
        assert o.frames_done == 40 and not o.flags & P.OBS_PREFIX and s.n_logged == 40
        assert s.launcher.envs[0]["BALLRAIN_PREFIX_EVENTS"] == "1"
        assert s.launcher.handles[0].game.prefix_acks == 39
        o = s.act(tas.make_frames([D], [0]), repeat=5)
        assert o.frames_done == 45
        s.stop()
        assert tas.load_raw(s.mod_framelog()).tobytes() == s.frame_log().tobytes()
    finally:
        s.close()


def _run_episode(env, action):
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(action)
        assert env.observation_space.contains(obs)
        total += r
        steps += 1
        if term or trunc:
            return total, steps, info, term, trunc


def test_env_lives_gameover_and_recording(tmp_path):
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), action_repeat=4), launcher=fakegame.FakeLauncher(lives=1))
    obs, info = env.reset(seed=3)
    assert info["valid"] and info["lives"] == 1 and info["life"] == 0
    # life 1: roll left off the road
    _, _, info, term, _ = _run_episode(env, 3)
    assert term and info["episode_end"] == "BallOff" and "run_meta" not in info
    obs, info = env.reset()  # same process, after the respawn
    assert info["life"] == 1 and info["lives"] == 0
    _, _, info, term, _ = _run_episode(env, 3)
    assert info["episode_end"] == "BallOff"
    obs, info = env.reset()  # GameOver during the fast-forward: a new process
    assert info["life"] == 0 and info["lives"] == 1
    assert len(env.finished_runs) == 1
    meta = json.loads(open(env.finished_runs[0], encoding="utf-8").read())
    assert meta["end_reason"] == "GameOver" and meta["framelog_match"] is True
    assert [l["end_reason"] for l in meta["lives"]] == ["BallOff", "BallOff"]
    assert meta["sr_ms"] is None and meta["hs"] is None
    frames = tas.load(env.finished_runs[0].with_suffix(".tas"))
    assert len(frames) == meta["frames"]
    assert meta["lives"][1]["start_frame"] > meta["lives"][0]["end_frame"]
    env.close()
    assert len(env.finished_runs) == 2


def test_env_levelfinish_sr_and_hs(tmp_path):
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), action_repeat=8, mode="hs"), launcher=fakegame.FakeLauncher())
    env.reset()
    total, steps, info, term, _ = _run_episode(env, 1)
    assert term and info["episode_end"] == "LevelFinish"
    meta = json.loads(open(info["run_meta"], encoding="utf-8").read())
    assert meta["end_reason"] == "LevelFinish" and meta["framelog_match"]
    lf = meta["levelfinish_frame"]
    assert meta["frames"] == lf  # the SR tail frames are not exported
    frames = tas.load(tmp_path / "train-00" / meta["run_id"] / meta["tas_path"])
    assert len(frames) == lf
    ev = [(n, P.EVENT_CODES[name]) for n, name, _ in meta["events"]]
    py = sr_timer(frames, ev)
    assert py["double"] == meta["sr_python_at_levelfinish"]["double"]
    # the mod's SREN block: the SR at the LevelFinish frame equals the Python recomputation exactly
    assert meta["sr_mod_at_levelfinish"] == py["double"]
    assert meta["sr_tail_frames"] == fakegame.SR_TAIL and meta["hs_mod"] == meta["hs"] and not meta["notes"]
    # the official value includes the tail until CounterInactive
    assert meta["sr_double"] == pytest.approx(py["double"] + (fakegame.SR_TAIL - 1) * float(D))
    assert meta["sr_ms"] == int(meta["sr_double"])
    assert meta["hs"] == meta["hs_parts"]["points"] + 3 * 200 + 100
    # HS reward telescopes to final HS minus the score potential at the episode start
    assert total == pytest.approx(0.01 * (meta["hs"] - (1000 + 3 * 200)), abs=1e-9)
    env.close()


def test_env_check_and_action_repeat(tmp_path):
    from gymnasium.utils.passive_env_checker import env_reset_passive_checker, env_step_passive_checker
    env = BallanceEnv(EnvConfig(level=2, out_root=str(tmp_path), action_repeat=3, camera_actions=True,
                                max_episode_frames=30), launcher=fakegame.FakeLauncher())
    assert env.action_space.n == 11
    obs, info = env.reset()
    f0 = info["frames_done"]
    _, _, _, _, info = env.step(1)
    assert info["frames_done"] == f0 + 3
    _, _, info, term, trunc = _run_episode(env, 0)
    assert trunc and not term and info["episode_end"] == "truncated"
    env.close()
    # gymnasium's checks (check_env's determinism check compares info, which names the run, so only
    # the space / API checks run here; same seed => same trajectory is checked explicitly)
    env2 = BallanceEnv(EnvConfig(level=2, out_root=str(tmp_path / "c")), launcher=fakegame.FakeLauncher())
    env_reset_passive_checker(env2, seed=5)
    o1 = [env_step_passive_checker(env2, a)[0] for a in (1, 1, 4, 0)]
    env2.reset(seed=5)  # an explicit seed starts a fresh process
    o2 = [env2.step(a)[0] for a in (1, 1, 4, 0)]
    assert all(np.array_equal(a, b) for a, b in zip(o1, o2))
    env2.reset()  # reset in the middle of a life: new process too
    assert len(env2.finished_runs) == 2
    env2.close()


def test_sync_vector_env_two_copies(tmp_path):
    cfg = EnvConfig(level=1, out_root=str(tmp_path), action_repeat=4)
    venv = vecenv.make_vec_env(cfg, n=2, backend="sync", launcher_factory=fakegame.FakeLauncher)
    obs, info = venv.reset(seed=1)
    assert obs.shape == (2, venv.single_observation_space.shape[0])
    seen = set()
    for _ in range(80):
        obs, r, term, trunc, info = venv.step(np.array([1, 3]))
        seen.update(np.flatnonzero(term).tolist())
    assert 1 in seen  # the env rolling left died and was auto-reset
    venv.close()
    copies = {p.parent.name for p in tmp_path.rglob("*.json")}
    assert {p.name for p in tmp_path.iterdir()} == {"train-00", "train-01"} and copies
    with pytest.raises(ValueError):
        vecenv.env_fns(cfg, 2, copies=["train-00", "train-00"])


def test_async_vector_env(tmp_path):
    cfg = EnvConfig(level=1, out_root=str(tmp_path), action_repeat=4)
    venv = vecenv.make_vec_env(cfg, n=2, backend="async", launcher_factory=fakegame.FakeLauncher)
    try:
        obs, _ = venv.reset(seed=0)
        for _ in range(5):
            obs, r, term, trunc, info = venv.step(np.array([1, 1]))
        assert obs.shape[0] == 2 and np.isfinite(obs).all()
    finally:
        venv.close()


def test_dataset_replay_policy(tmp_path):
    keys = np.zeros(400, np.uint32)
    keys[40:] = tas.KEY_UP
    rec = tmp_path / "SR01_0_00_000_132.tas"
    tas.save(rec, tas.make_frames(np.full(400, D), keys))
    res = dataset.build_dataset([rec], tmp_path / "ds", copies=["train-00"], launcher_factory=fakegame.FakeLauncher)
    m = res[0]
    assert m["ok"] and m["executed_frames"] == 400 and m["framelog_equals_record_prefix"]
    assert m["observations"] == 401 and m["end_reason"] == "python_stop"
    d = tmp_path / "ds" / rec.stem
    z = np.load(d / "raw.npz")
    assert z["frames_done"].tolist() == list(range(401))
    obs = list(dataset.load_observations(d / "raw.npz"))
    assert obs[-1].frames_done == 400 and obs[-1].ball["pos"][2] == pytest.approx(z["ball_pos"][-1][2])
    assert (d / "level.msg").exists() and json.loads((tmp_path / "ds" / "index.json").read_text())[0]["ok"]


def test_final_observation_not_lost_when_ended_is_set_first(tmp_path):
    """BallrainIO sets ENDED before it publishes the final observation; Python must still get it."""
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), action_repeat=8),
                      launcher=fakegame.FakeLauncher(final_delay_s=0.6))
    env.reset()
    _, _, info, term, _ = _run_episode(env, 1)
    assert term and info["episode_end"] == "LevelFinish" and info["framelog_match"]
    meta = json.loads(open(info["run_meta"], encoding="utf-8").read())
    assert meta["sr_ms"] is not None and meta["hs"] is not None and meta["hs_parts"]["source"] == "tail_end"
    env.close()


def test_hs_reward_telescopes_over_lives(tmp_path):
    """Die once, then finish: the HS rewards of both episodes sum to HS minus the start potential."""
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), action_repeat=4, mode="hs"),
                      launcher=fakegame.FakeLauncher())
    env.reset()
    t1, _, info, _, _ = _run_episode(env, 3)
    assert info["episode_end"] == "BallOff"
    env.reset()
    t2, _, info, _, _ = _run_episode(env, 1)
    assert info["episode_end"] == "LevelFinish"
    meta = json.loads(open(info["run_meta"], encoding="utf-8").read())
    assert meta["hs_parts"]["lives"] == 2
    assert t1 + t2 == pytest.approx(0.01 * (meta["hs"] - (1000 + 3 * 200)), abs=1e-9)
    env.close()


def test_hello_must_name_the_2002_physics(tmp_path, monkeypatch):
    def other_hello(self, version, pid=None, obs_versions=2):
        b = b"OtherMod 1.0".ljust(P.SHM_VERSION_LEN, b"\0")
        self.m.mm[P.H_MOD_VERSION:P.H_MOD_VERSION + P.SHM_VERSION_LEN] = b
        self.set_state(P.ModState.HELLO)

    monkeypatch.setattr(transport.ModChannel, "hello", other_hello)
    s = _session(tmp_path)
    with pytest.raises(client.SessionError):
        s.start()
    s.close()

"""The headless backend's Python side without the engine: the socket transport (ballrain.transport.SocketChannel /
SocketModChannel), AgentSession / BallanceEnv over it with ballrain.fakegame speaking the headless hello, the
backend field of the run metadata, ballrain.headless.HeadlessLauncher's environment, and the replay stub's framing.
Runs on every OS (TCP on Windows, unix sockets elsewhere)."""
import json
import os
import struct
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from ballrain import client, fakegame, headless, protocol as P, tas, transport
from ballrain.env import BallanceEnv, EnvConfig

D = np.float32(1000 / 132)
KINDS = ["tcp"] + (["unix"] if transport.default_socket_kind() == "unix" else [])


@pytest.mark.parametrize("kind", KINDS)
def test_socket_transport_roundtrip(kind):
    py = transport.SocketChannel(kind)
    box = {}

    def mod_side():
        m = transport.SocketModChannel(py.name)
        m.hello("Fake 1 headless")
        m.set_state(P.ModState.LOADING)
        m.write_obs(P.build_message(P.OBS_MAGIC, 0, [("ZZZZ", b"x" * 4)]), 0)
        box["a"] = m.wait_action(5)
        box["none"] = m.wait_action(0.05)  # nothing sent: timeout
        m.set_state(P.ModState.ENDED, P.EndReason.PY_STOP, 7)
        m.write_obs(P.build_message(P.OBS_MAGIC, 7, [], flags=P.OBS_FINAL), 7)
        m.close()

    th = threading.Thread(target=mod_side)
    th.start()
    try:
        assert py.name.startswith(kind + ":")
        h = py.wait_hello(10)
        assert h["mod_state"] == P.ModState.HELLO and h["mod_version"] == "Fake 1 headless" and h["mod_pid"] == os.getpid()
        o = P.parse_observation(py.wait_obs(5))
        assert o.frames_done == 0 and py.header()["mod_state"] == P.ModState.LOADING and py.header()["mod_seq"] == 1
        a = P.Action(tas.make_frames([D, D], [1, 2]), repeat=4, stop_mask=P.event_mask("BallOff"), flags=P.ACT_WANT_TRAC,
                     obs_mask=P.OBSMASK_BODY)
        assert py.send_action(a) == 1
        o = P.parse_observation(py.wait_obs(5))
        assert o.final and o.frames_done == 7
        h = py.header()
        assert h["mod_state"] == P.ModState.ENDED and h["end_reason"] == P.EndReason.PY_STOP and h["frames_done"] == 7
        assert py.wait_obs(5) is None  # EOF after the end: no PeerGone
    finally:
        th.join(10)
        py.close()
    got = box["a"]
    assert got.repeat == 4 and got.frames["keys"].tolist() == [1, 2] and got.stop_mask == 1 << P.Ev.BallOff
    assert got.flags == P.ACT_WANT_TRAC and got.obs_mask == P.OBSMASK_BODY and box["none"] is None


def test_socket_peer_gone_and_timeout():
    py = transport.SocketChannel("tcp")
    m = transport.SocketModChannel(py.name)
    th = threading.Thread(target=lambda: m.hello("x headless"))
    th.start()
    py.wait_hello(5)
    th.join()
    try:
        with pytest.raises(transport.Timeout):
            py.wait_obs(0.05)
        m.close()  # the game dies without ending the run
        with pytest.raises(transport.PeerGone):
            py.wait_obs(5)
    finally:
        py.close()


def test_socket_hello_timeout_when_nobody_connects():
    py = transport.SocketChannel("tcp")
    try:
        with pytest.raises(transport.Timeout):
            py.wait_hello(0.2)
    finally:
        py.close()


def _session(tmp_path, launcher=None, **kw):
    return client.AgentSession(client.SessionConfig(level=1, out_dir=tmp_path / "run", **kw),
                               launcher or fakegame.FakeLauncher(transport="socket"))


def test_session_over_socket_stop_on_event_and_framelog(tmp_path):
    s = _session(tmp_path, max_frames=5000)
    o = s.start()
    try:
        env = s.launcher.envs[0]
        assert env["BALLRAIN_SOCK"].split(":")[0] in ("tcp", "unix") and env["BALLRAIN_SHM"] is None
        assert s.backend == "headless" and "headless" in s.hello["mod_version"]
        assert o.frames_done == 0 and o.event_codes() == [P.Ev.PreLoadLevel]
        o = s.act(tas.make_frames([D], [0]), repeat=1000, stop=["StartLevel"])
        assert o.frames_done == fakegame.LOAD_FRAMES and s.n_logged == fakegame.LOAD_FRAMES
        assert s.tracker.level.by_name["A01_Floor_01"] == fakegame.ENT_FLOOR
        s.run_until(0, ["BallNavActive"])
        assert s.tracker.flags.ball_nav
        s.stop()
        assert s.ended and s.end_reason == P.EndReason.PY_STOP
        assert tas.load_raw(s.mod_framelog()).tobytes() == s.frame_log().tobytes()
    finally:
        s.close()


def test_session_over_socket_prefix_events_and_timeout(tmp_path):
    prefix = tas.make_frames(np.full(40, D), np.zeros(40, np.uint32))
    tas.save_raw(tmp_path / "prefix.bin", prefix)
    s = _session(tmp_path, prefix=tmp_path / "prefix.bin", prefix_events=True, timeout_ms=300)
    seen = []
    s.on_prefix_obs = lambda o: seen.append(o.frames_done)
    o = s.start()
    try:
        assert seen == list(range(1, 40)) and o.frames_done == 40 and s.n_logged == 40
        o = s.chan.wait_obs(5)  # no action: the game times out and ends the run
        assert o is not None and s.chan.header()["end_reason"] == P.EndReason.TIMEOUT
    finally:
        s.close()


def test_hello_must_match_the_backend(tmp_path):
    # a retail launcher must not accept a headless hello, a headless launcher not a retail one
    s = _session(tmp_path / "a", launcher=fakegame.FakeLauncher(transport="socket", backend="retail",
                                                                hello="FakeGame 0.1 headless"))
    with pytest.raises(client.SessionError):
        s.start()
    s.close()
    s = _session(tmp_path / "b", launcher=fakegame.FakeLauncher(transport="socket", backend="headless",
                                                                hello="FakeGame 0.1 e72e4afcfa5c33a7"))
    with pytest.raises(client.SessionError):
        s.start()
    s.close()


def _run_episode(env, action):
    while True:
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            return info, term


def test_env_headless_backend_lives_finish_and_metadata(tmp_path):
    cfg = EnvConfig(level=1, out_root=str(tmp_path), action_repeat=4, backend="headless")
    env = BallanceEnv(cfg, launcher=fakegame.FakeLauncher(transport="socket", lives=1))
    env.reset(seed=1)
    info, term = _run_episode(env, 3)
    assert term and info["episode_end"] == "BallOff"
    env.reset()  # respawn in the same process
    info, term = _run_episode(env, 1)
    assert info["episode_end"] == "LevelFinish" and info["framelog_match"]
    meta = json.loads(Path(info["run_meta"]).read_text(encoding="utf-8"))
    assert meta["backend"] == "headless" and "headless" in meta["mod_version"]
    assert meta["trace_hash"] is not None and len(meta["trace_hash"]) == 16  # the final observation's HASH section
    assert meta["sr_ms"] is not None and [l["end_reason"] for l in meta["lives"]] == ["BallOff", "LevelFinish"]
    env.close()


def test_env_backend_checks(tmp_path):
    with pytest.raises(ValueError):
        BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), backend="nope"))
    with pytest.raises(ValueError):  # a retail launcher under a headless config
        BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), backend="headless"), launcher=fakegame.FakeLauncher())
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), backend="headless",
                                headless={"binary": str(tmp_path / "missing"), "root": str(tmp_path)}))
    assert isinstance(env.launcher, headless.HeadlessLauncher)
    with pytest.raises(FileNotFoundError):  # binary / game folder checked at launch
        env.reset()
    env.close()


def test_retail_runs_are_marked_retail(tmp_path):
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), action_repeat=8), launcher=fakegame.FakeLauncher(
        transport="socket", backend="retail", hello="FakeGame 0.1 e72e4afcfa5c33a7"))
    env.reset()
    info, _ = _run_episode(env, 1)
    meta = json.loads(Path(info["run_meta"]).read_text(encoding="utf-8"))
    assert meta["backend"] == "retail" and meta["trace_hash"] is None
    env.close()


def test_headless_launcher_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BALLRAIN_TRACE", "/leaked")  # parent BALLRAIN_* never leak into the engine
    monkeypatch.setenv("PATH_KEEP_ME", "1")
    la = headless.HeadlessLauncher(headless.HeadlessConfig(binary="b", root="r", reset_seed=5, hidden_ticks=34,
                                                           legacy_edit=False, extra_env={"BALLRAIN_HL_VERBOSE": 1}))
    env = la.environment({"BALLRAIN_LEVEL": "3", "BALLRAIN_SHM": None, "BALLRAIN_SOCK": "unix:/x"})
    assert env["BALLRAIN_LEVEL"] == "3" and "BALLRAIN_SHM" not in env and "BALLRAIN_TRACE" not in env
    assert env["BALLRAIN_HL_SEED"] == "5" and env["BALLRAIN_HL_HIDDEN_TICKS"] == "34" and env["BALLRAIN_HL_LEGACY"] == "0"
    assert env["BALLRAIN_HL_VERBOSE"] == "1" and env["PATH_KEEP_ME"] == "1" and "BALLRAIN_HL_NMC" not in env
    assert headless.headless_copy_names(3) == ["hl-00", "hl-01", "hl-02"]
    ch = la.make_channel(None)
    try:
        assert ch.env_var == "BALLRAIN_SOCK" and ch.name.split(":")[0] == transport.default_socket_kind()
    finally:
        ch.close()


def test_replay_stub_framing_matches_transport():
    """headless/tools/replay_stub.py (stdlib only) uses the transport's framing constants."""
    src = (Path(__file__).resolve().parents[1] / "headless" / "tools" / "replay_stub.py").read_text(encoding="utf-8")
    for name, value in (("BR_PING", transport.BR_PING), ("BR_ACT", transport.BR_ACT)):
        assert f"{value:#x}" in src
    assert transport.OBS_HDR.format in src and transport.FRAME_HDR.format in src
    assert struct.calcsize(transport.OBS_HDR.format) == 24


def test_trace_hash_repeats_with_the_same_frames(tmp_path):
    """The run fingerprint: the same frames give the same hash (the determinism detector of docs/headless.md)."""
    hashes = []
    for i in range(2):
        env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path / str(i)), action_repeat=4, backend="headless"),
                          launcher=fakegame.FakeLauncher(transport="socket"))
        env.reset()
        info, _ = _run_episode(env, 1)
        hashes.append(json.loads(Path(info["run_meta"]).read_text(encoding="utf-8"))["trace_hash"])
        env.close()
    assert hashes[0] == hashes[1] and hashes[0] is not None

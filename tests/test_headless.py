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
    assert env["BALLRAIN_HL_VERBOSE"] == "1" and env["PATH_KEEP_ME"] == "1" and env["BALLRAIN_HL_BALL_ORDER"] == "ckid"
    # the recipe is SimTool's flags, passed only where the config differs from their defaults
    assert not [k for k in env if k in ("BALLRAIN_HL_SEED", "BALLRAIN_HL_HIDDEN_TICKS", "BALLRAIN_HL_LEGACY")]
    assert la.arguments() == ["b", "--ballrain-agent", "--root", str(Path("r").resolve()), "--tas-offset", "34",
                              "--tas-seed", "5", "--tas-no-legacy"]
    assert headless.HeadlessLauncher(headless.HeadlessConfig(binary="b", root="r")).arguments()[1:3] == ["--ballrain-agent",
                                                                                                         "--root"]
    full = headless.HeadlessLauncher(headless.HeadlessConfig(binary="b", root="r", boot_ticks=400, reset_nmc=-1,
                                                             reset_frame=0, reset_precision=True, orig_random=True))
    assert full.arguments()[4:] == ["--boot-ticks", "400", "--tas-movement-check", "-1", "--tas-reset-frame", "0",
                                    "--tas-reset-precision", "--tas-orig-random"]
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


def test_ball_order_parsing():
    P = headless.parse_ball_order
    assert P(None) == P("") == P("ckid") == "ckid" and P("reverse") == "reverse"
    assert P("perm:007") == "perm:7" and P("retail:18446744073709551615") == "retail:18446744073709551615"
    assert P("table:Ball_Wood=0,P_Ball_Stone_MF004=1,P_Ball_Stone_MF002=02") == \
        "table:Ball_Wood=0,P_Ball_Stone_MF004=1,P_Ball_Stone_MF002=2"
    assert P("perm", allow_seedless=True) == "perm" and P("retail", allow_seedless=True) == "retail"
    for bad in ("nope", "ckid:1", "reverse:2", "perm", "perm:", "perm:-1", "perm:1x", "retail:18446744073709551616",
                "table:", "table:a", "table:=1", "table:a=1,a=2", "table:a=1,b=1", "table:a=1048576", "table:a=1,,b=2"):
        with pytest.raises(ValueError):
            P(bad)


def test_ball_order_resolution_and_draws():
    import random
    rng = random.Random(5)
    a, b = headless.resolve_ball_order("retail", rng), headless.resolve_ball_order("retail", rng)
    assert a.startswith("retail:") and b.startswith("retail:") and a != b  # a fresh seed per process
    assert headless.resolve_ball_order("perm:3") == "perm:3" and headless.resolve_ball_order(None) == "ckid"
    # the draw (splitmix64, as the engine's RetailDrawL1): fixed per seed, Ball_Wood first, the measured blocks
    names = {f"P_Ball_Stone_MF00{i}" for i in range(1, 5)}
    stone_first, mf4_before_mf2 = 0, 0
    for seed in range(2000):
        d = headless.retail_draw_l1(seed)
        assert d == headless.retail_draw_l1(seed) and d[0] == "Ball_Wood"
        assert set(d[1:3]) == {"P_Ball_Stone_MF", "P_Ball_Wood_MF"} and set(d[3:]) == names
        stone_first += d[1] == "P_Ball_Stone_MF"
        mf4_before_mf2 += d.index("P_Ball_Stone_MF004") < d.index("P_Ball_Stone_MF002")
    assert 0.62 < stone_first / 2000 < 0.70 and 0.45 < mf4_before_mf2 / 2000 < 0.55
    assert headless.retail_draw_l1(0) == ["Ball_Wood", "P_Ball_Stone_MF", "P_Ball_Wood_MF", "P_Ball_Stone_MF003",
                                          "P_Ball_Stone_MF004", "P_Ball_Stone_MF002", "P_Ball_Stone_MF001"]
    assert sorted(headless.perm_draw(9, list("abcd"))) == list("abcd")
    # table / retail: the ranked balls swap among their own CK-id places; Ball_Stone / Ball_Paper / the paper pool
    # keep theirs (the L1 order under ckid, as the engine logs it)
    ckid = ["Ball_Wood", "Ball_Stone", "Ball_Paper", "P_Ball_Paper_MF", "P_Ball_Stone_MF", "P_Ball_Wood_MF",
            "P_Ball_Paper_MF001", "P_Ball_Stone_MF001", "P_Ball_Stone_MF002", "P_Ball_Stone_MF003", "P_Ball_Stone_MF004"]
    got = headless.ranked_order(ckid, headless.retail_draw_l1(0))
    assert got == ["Ball_Wood", "Ball_Stone", "Ball_Paper", "P_Ball_Paper_MF", "P_Ball_Stone_MF", "P_Ball_Wood_MF",
                   "P_Ball_Paper_MF001", "P_Ball_Stone_MF003", "P_Ball_Stone_MF004", "P_Ball_Stone_MF002",
                   "P_Ball_Stone_MF001"]
    assert headless.ranked_order(ckid, headless.retail_draw_l1(12)) == ckid  # the draw equal to ckid changes nothing
    assert headless.ranked_order(ckid, ["P_Ball_Stone_MF004", "Missing", "Ball_Wood"]) ==         ["P_Ball_Stone_MF004"] + ckid[1:-1] + ["Ball_Wood"]
    assert headless.effective_ball_order("retail:5", "1") == "retail:5"
    assert headless.effective_ball_order("retail:5", 2) == "ckid" and headless.effective_ball_order("perm:5", 2) == "perm:5"


def test_ball_order_extra_env_wins_and_is_validated():
    la = headless.HeadlessLauncher(headless.HeadlessConfig(binary="b", root="r", ball_order="retail:1",
                                                           extra_env={"BALLRAIN_HL_BALL_ORDER": "reverse"}))
    assert la.environment({})["BALLRAIN_HL_BALL_ORDER"] == "reverse"
    with pytest.raises(ValueError):
        headless.HeadlessLauncher(headless.HeadlessConfig(binary="b", root="r", ball_order="sideways")).environment({})


def test_headless_handle_carries_the_ball_order(tmp_path):
    (tmp_path / "base.cmo").write_bytes(b"")
    la = headless.HeadlessLauncher(headless.HeadlessConfig(binary=sys.executable, root=str(tmp_path), log=False,
                                                           ball_order="perm"))
    h = la.start({"BALLRAIN_LEVEL": "1"})  # python rejects the agent's arguments and exits: only the handle matters
    try:
        assert h.ball_order.startswith("perm:") and int(h.ball_order[5:]) < 1 << 32
    finally:
        headless.HeadlessLauncher.stop(h, 10)
    la.cfg.ball_order = "retail"
    h = la.start({"BALLRAIN_LEVEL": "3"})  # retail draws are measured on L1 only: the engine runs ckid
    try:
        assert h.ball_order == "ckid"
    finally:
        headless.HeadlessLauncher.stop(h, 10)


class _OrderedFake(fakegame.FakeLauncher):
    def start(self, env):
        h = super().start(env)
        h.ball_order = "retail:42"
        return h


def test_env_records_the_ball_order_in_the_metadata(tmp_path):
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), action_repeat=4, backend="headless"),
                      launcher=_OrderedFake(transport="socket"))
    env.reset(seed=1)
    info, _ = _run_episode(env, 1)
    meta = json.loads(Path(info["run_meta"]).read_text(encoding="utf-8"))
    assert meta["backend"] == "headless" and meta["ball_order"] == "retail:42"
    env.close()
    env = BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path / "r"), action_repeat=8), launcher=fakegame.FakeLauncher(
        transport="socket", backend="retail", hello="FakeGame 0.1 e72e4afcfa5c33a7"))
    env.reset()
    info, _ = _run_episode(env, 1)
    assert json.loads(Path(info["run_meta"]).read_text(encoding="utf-8"))["ball_order"] is None
    env.close()


def test_phys2002_host_packaging():
    """The Windows 2002-DLL host (headless/README.md): patch 0003 touches product files only (its own sources are in
    headless/src, the layout is BallrainIO's Ivp2002.h), the Mac build applies 0002 alone, the agent's 2002 code is
    compiled only under the 0003 define."""
    root = Path(__file__).resolve().parents[1]
    hl = root / "headless"
    p3 = (hl / "patches" / "0003-hlwin-phys2002.diff").read_text(encoding="utf-8")
    files = [line[6:].strip() for line in p3.splitlines() if line.startswith("+++ b/")]
    assert files and not any("phys2002" in f or "ballrain_agent" in f for f in files)
    assert "BALLRAIN_PHYS2002_HOST" in p3 and "sim/phys2002.cpp" in p3 and "sim/Ivp2002.h" in p3
    for f in ("phys2002.cpp", "phys2002.hpp"):
        assert (hl / "src" / f).is_file()
    assert (root / "BallrainIO" / "src" / "Ivp2002.h").is_file()
    assert 'patches/0002-*.diff' in (hl / "build_mac.sh").read_text(encoding="utf-8")
    assert 'patches/0002-*.diff' in (hl / "mac" / "setup_mac.sh").read_text(encoding="utf-8")
    agent = (hl / "src" / "ballrain_agent.cpp").read_text(encoding="utf-8")
    assert '#ifdef BALLRAIN_PHYS2002_HOST\n#include "phys2002.hpp"' in agent
    win = (hl / "build_win.ps1").read_text(encoding="utf-8")
    assert "0003-hlwin-phys2002.diff" in win and "Ivp2002.h" in win
    # the sha the host checks is the one the retail mod checks
    sha = (hl / "src" / "phys2002.hpp").read_text(encoding="utf-8").split('kSha256 = "')[1][:64]
    assert sha.startswith("e72e4afc") and len(sha) == 64


def test_gate_phys2002_counts_the_ivp_fields_as_exact():
    """scripts/headless_gate.py --phys2002: only per-process addresses and IVP residue stay apart."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import headless_gate as G
    assert G.APPROX_PHYS2002 < G.APPROX
    for key in (("FRIC", "lt_normal"), ("FRIC", "lt_point"), ("FRIC", "lt_valid"), ("GLCT", "groups"),
                ("BALL", "core_flags_raw"), ("BALL", "core_state_raw"), ("BODY", "core_state_raw")):
        assert key in G.APPROX and key not in G.APPROX_PHYS2002
    assert ("MIND", "ledge") in G.APPROX_PHYS2002 and ("SCOR", "energy_param_ptr") in G.APPROX_PHYS2002

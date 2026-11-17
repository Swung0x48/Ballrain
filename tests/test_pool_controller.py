"""The volunteer's controller (ballrain.pool.client_*, tools/pool/controller.py) against the stub gateway
(test_pool_stub_gateway.py): game-folder messages, start / pause, the leaderboard switch round trip, refusal and
revocation banners shown verbatim, the hard-kill lifecycle (no worker or engine outlives the controller), and a full
sampling run of real worker processes on the fake game."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ballrain.pool import client_game as G
from ballrain.pool import protocol as P
from ballrain.pool import reasons as R
from ballrain.pool import task as T
from ballrain.pool.client_core import PAUSE_PLAYER, ClientConfig, Controller
from ballrain.pool.client_lifecycle import kill_pid, pid_alive, python_command
from test_pool_stub_gateway import TOKEN, StubGateway

REPO = Path(__file__).resolve().parents[1]
CONTROLLER = REPO / "tools" / "pool" / "controller.py"


def wait_until(cond, timeout=20.0, step=0.1):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return cond()


@pytest.fixture
def gw():
    g = StubGateway()
    yield g
    g.close()


def _ctl(tmp_path, gw, **kw) -> Controller:
    cfg = ClientConfig(gateway_url=gw.url, edition="all", instances=1)
    kw.setdefault("auto_pause", False)
    c = Controller(tmp_path / "data", cfg, **kw)
    c.set_token(TOKEN)
    return c


def test_controller_and_worker_modules_do_not_import_torch():
    """The player's processes run numpy only (the policy is weights.NumpyPolicy; the worker reuses the learner's
    rollout helpers, whose module does not import torch)."""
    code = ("import sys; import ballrain.pool.client_core, ballrain.pool.client_worker, ballrain.pool.client_sampler, "
            "ballrain.pool.client_api, ballrain.pool.client_game, ballrain.pool.client_lifecycle, ballrain.env; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------------------------- game folder

def _ballance(root: Path, physics: bytes | None = None) -> Path:
    for f, data in (("base.cmo", b"b"), ("Database.tdb", b"d"), ("Bin/VxMath.dll", b"v"), ("3D Entities/a.nmo", b"a"),
                    ("Text/t.txt", b"t")):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_bytes(data)
    if physics is not None:
        (root / "BuildingBlocks").mkdir(exist_ok=True)
        (root / "BuildingBlocks" / "physics_RT.dll").write_bytes(physics)
    return root


def test_game_dir_messages_name_the_file(tmp_path):
    assert G.check_game_dir("", "all")[0].message == "还没有选择 Ballance 安装目录。"
    assert "找不到 Ballance 安装目录" in G.check_game_dir(tmp_path / "nope", "all")[0].message
    empty = tmp_path / "empty"
    empty.mkdir()
    msgs = [p.message for p in G.check_game_dir(empty, "all")]
    assert msgs == ["缺少文件：base.cmo。游戏文件不完整，请检查安装目录。",
                    "缺少文件：Database.tdb。游戏文件不完整，请检查安装目录。",
                    "缺少文件：Bin/VxMath.dll。游戏文件不完整，请检查安装目录。",
                    "缺少文件夹：3D Entities（应在 Ballance 安装目录下）", "缺少文件夹：Text（应在 Ballance 安装目录下）"]
    root = _ballance(tmp_path / "Ballance")
    assert G.check_game_dir(root, "all") == []  # the x87-soft edition does not load physics_RT.dll
    (root / "Text" / "t.txt").unlink()
    assert [p.message for p in G.check_game_dir(root, "all")] == ["文件夹 Text 是空的，游戏文件不完整。"]
    if sys.platform != "win32":
        with pytest.raises(ValueError, match="Windows"):
            G.check_game_dir(root, "windows")
        return
    root2 = _ballance(tmp_path / "B2")
    p = G.check_game_dir(root2, "windows")
    assert [x.name for x in p] == ["BuildingBlocks/physics_RT.dll"] and p[0].kind == "missing"
    assert "缺少文件：BuildingBlocks/physics_RT.dll" in p[0].message and "全平台版" in p[0].message
    root3 = _ballance(tmp_path / "B3", physics=b"not the 2002 dll")
    p = G.check_game_dir(root3, "windows")
    sha = T.file_sha256(root3 / "BuildingBlocks" / "physics_RT.dll")
    assert p[0].kind == "wrong" and p[0].message == (
        f"BuildingBlocks/physics_RT.dll 不是 2002 年原版（sha256 {sha[:12]}…，应为 e72e4afcfa5c…）。"
        "Windows 版只能加载原版物理 DLL，请换回原版文件，或者改用全平台版。")
    eng = G.check_engine(tmp_path / "SimTool.exe", "windows")
    assert eng[0].message.startswith("找不到无头引擎文件")


def test_gateway_game_file_refusal_is_shown_verbatim(tmp_path, gw):
    pl = R.payload(R.Reason.GAME_FILES_MISMATCH, file="Database.tdb", got="ab" * 32)
    gw.hello_error = pl
    c = _ctl(tmp_path, gw, simulate=True)
    try:
        assert not c.connect()
        assert c.banner == {"kind": "rejected", "code": "game_files_mismatch", "message": pl["message"]}
        assert c.banner["message"].startswith("游戏文件校验不通过：Database.tdb 与原版不一致")
    finally:
        c.shutdown()


def test_hello_reports_hashes_only(tmp_path, gw):
    root = _ballance(tmp_path / "Ballance")
    eng = tmp_path / "BallanceMMOSimTool.exe"
    eng.write_bytes(b"engine")
    eng.chmod(0o755)
    c = _ctl(tmp_path, gw)
    c.cfg.ballance_dir, c.cfg.engine_binary = str(root), str(eng)
    try:
        assert c.problems() == [] and c.connect()
        h = gw.hellos[-1]
        assert set(h["game_files"]) == set(P.GAME_FILE_LIST)
        assert h["engine"] == {"kind": "x87soft", "sha256": T.file_sha256(eng), "version": f"dev-{T.file_sha256(eng)[:12]}"}
        assert str(tmp_path) not in json.dumps(h, ensure_ascii=False)  # no local paths leave the machine
    finally:
        c.shutdown()


# ---------------------------------------------------------------------------------------------- leaderboard, banners

def test_leaderboard_switch_round_trip(tmp_path, gw):
    c = _ctl(tmp_path, gw, simulate=True)
    try:
        assert c.leaderboard is None and c.connect()
        assert c.leaderboard is True  # on by default (the gateway's state)
        assert c.set_leaderboard(False) is False and gw.me_updates == [False] and gw.leaderboard is False
    finally:
        c.shutdown()
    c2 = _ctl(tmp_path, gw, simulate=True)  # a restart reads the gateway's state, it does not re-send its own
    try:
        assert c2.connect() and c2.leaderboard is False and gw.me_updates == [False]
        assert c2.set_leaderboard(True) is True and gw.me_updates == [False, True]
        c2.refresh_me()
        assert c2.snapshot()["leaderboard"] is True
    finally:
        c2.shutdown()


def test_revocation_banner_shows_the_reason_verbatim(tmp_path, gw):
    c = _ctl(tmp_path, gw, simulate=True)
    try:
        assert c.connect()
        c.start()
        pl = R.payload(R.Reason.TOKEN_REVOKED_REPLAY, task_id="t" * 8, chunk=3, step=1234)
        gw.revoked = pl
        c.heartbeat_once()  # heartbeat_ok carries "revoked" ... and every authenticated call now fails with it
        s = c.snapshot()
        assert s["banner"] == {"kind": "revoked", "code": "token_revoked_replay", "message": pl["message"]}
        assert s["banner"]["message"] == ("你的算力池令牌已被停用：任务 tttttttt 的第 3 段数据在第 1234 步重放结果与提交不一致。"
                                          "这个令牌还没进入训练的数据已全部隔离。")
        assert not s["running"] and not s["connected"] and not s["want_running"]
        # an admin revocation arriving as a refused call (TOKEN_FATAL), with the admin's note
        pl2 = R.payload(R.Reason.TOKEN_REVOKED, note="测试用的吊销 <b>不是标记</b>")
        gw.revoked = pl2
        c.refresh_me()
        assert c.snapshot()["banner"]["message"] == "你的算力池令牌已被管理员吊销。说明：测试用的吊销 <b>不是标记</b>"
        # a worker's refusal event (a chunk rejected, not fatal): banner, the token stays usable
        c.token_dead = False
        gw.revoked = None
        pl3 = R.payload(R.Reason.CHAIN_BROKEN, task_id="abc", chunk=2, step=40)
        c._refused(pl3, False)
        assert c.snapshot()["banner"] == {"kind": "rejected", "code": "chain_broken", "message": pl3["message"]}
        assert c.connected
    finally:
        c.shutdown()


# ---------------------------------------------------------------------------------------------- instances

def _inst(c):
    s = c.snapshot()
    return s["instances"][0] if s["instances"] else None


def test_start_pause_and_auto_pause_with_real_child_processes(tmp_path, gw):
    playing = [False]
    c = _ctl(tmp_path, gw, simulate=True, auto_pause=True, probes={"player": lambda: playing[0], "battery": lambda: False})
    c.probe_s = 0.2
    try:
        assert c.connect()
        c.start_service()
        assert _inst(c) is None  # nothing runs before the player presses start
        c.start()
        i = wait_until(lambda: (x := _inst(c)) and x["state"] == "running" and x["engine_pid"] and x)
        assert i and pid_alive(i["pid"]) and pid_alive(i["engine_pid"])
        eng = i["engine_pid"]
        c.pause()
        assert wait_until(lambda: _inst(c)["state"] == "paused" and _inst(c)["engine_pid"] is None)
        assert wait_until(lambda: not pid_alive(eng), 5)  # pausing ends the engine; the worker waits
        assert pid_alive(i["pid"])
        c.start()
        assert wait_until(lambda: _inst(c)["state"] == "running" and _inst(c)["engine_pid"])
        playing[0] = True  # the player starts Ballance: automatic pause, with the reason shown
        assert wait_until(lambda: c.snapshot()["auto_pause"] == PAUSE_PLAYER and _inst(c)["state"] == "paused")
        assert c.snapshot()["want_running"] and not c.snapshot()["running"]
        playing[0] = False
        assert wait_until(lambda: _inst(c)["state"] == "running")
        eng = _inst(c)["engine_pid"]
        assert wait_until(lambda: gw.heartbeats and gw.heartbeats[-1]["instances"], 12)
        hb = gw.heartbeats[-1]
        assert hb["instances"][0]["slot"] == 0 and hb["running"] is True
    finally:
        c.shutdown()
    assert wait_until(lambda: not pid_alive(i["pid"]) and not pid_alive(eng), 5)


def _controller_proc(tmp_path, gw, *extra):
    exe, env_extra = python_command()
    status = tmp_path / "status.json"
    cmd = exe + [str(CONTROLLER), "--headless", "--data-dir", str(tmp_path / "data"), "--gateway", gw.url,
                 "--edition", "all", "--instances", "1", "--simulate", "--start", "--no-auto-pause",
                 "--status-file", str(status), "--token-stdin", *extra]
    env = dict(os.environ, **env_extra, PYTHONIOENCODING="utf-8")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, **kw)
    p.stdin.write((TOKEN + "\n").encode())
    p.stdin.flush()

    def ready():
        try:
            s = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        i = s["instances"][0] if s["instances"] else None
        return (s, i) if i and i["engine_pid"] and i["state"] == "running" else None
    got = wait_until(ready, 60)
    if not got:
        p.kill()
        pytest.fail(f"controller did not start an instance: {p.stderr.read().decode('utf-8', 'replace')[-2000:]}")
    return p, got


def test_hard_kill_of_the_controller_ends_every_child(tmp_path, gw):
    p, (s, i) = _controller_proc(tmp_path, gw)
    ctl_pid, worker, engine = s["pid"], i["pid"], i["engine_pid"]
    assert ctl_pid == p.pid and pid_alive(worker) and pid_alive(engine)
    try:
        kill_pid(ctl_pid)  # TerminateProcess / SIGKILL: no handler, no cleanup code runs in the controller
        assert wait_until(lambda: not pid_alive(ctl_pid), 5)
        t0 = time.monotonic()
        gone = wait_until(lambda: not pid_alive(worker) and not pid_alive(engine), 5)
        assert gone, f"still alive after 5 s: worker {pid_alive(worker)}, engine {pid_alive(engine)}"
        assert time.monotonic() - t0 < 5
    finally:
        for pid in (worker, engine):  # only our own processes, by pid
            if pid_alive(pid):
                kill_pid(pid)
        p.wait(5)


def test_normal_exit_of_the_controller_ends_every_child(tmp_path, gw):
    p, (s, i) = _controller_proc(tmp_path, gw, "--duration", "4")
    worker, engine = i["pid"], i["engine_pid"]
    try:
        assert p.wait(30) == 0
        assert wait_until(lambda: not pid_alive(worker) and not pid_alive(engine), 5)
    finally:
        for pid in (worker, engine):
            if pid_alive(pid):
                kill_pid(pid)


@pytest.mark.skipif(sys.platform != "win32", reason="CTRL_BREAK_EVENT")
def test_ctrl_break_of_the_controller_ends_every_child(tmp_path, gw):
    import signal
    p, (s, i) = _controller_proc(tmp_path, gw)
    worker, engine = i["pid"], i["engine_pid"]
    try:
        os.kill(p.pid, signal.CTRL_BREAK_EVENT)
        assert p.wait(30) == 0
        assert wait_until(lambda: not pid_alive(worker) and not pid_alive(engine), 5)
    finally:
        for pid in (worker, engine):
            if pid_alive(pid):
                kill_pid(pid)


# ---------------------------------------------------------------------------------------------- full run, fake game

@pytest.mark.skipif(sys.platform != "win32", reason="the fake record is built over named shared memory")
def test_workers_sample_a_task_end_to_end_on_the_fake_game(tmp_path, gw, request):
    pytest.importorskip("torch")
    pool = request.getfixturevalue("pool")
    m, tmp = pool["manifest"], pool["tmp"]
    sha = m["policy_sha"]
    gw.files[f"/pool/weights/{sha}.json"] = (tmp / "weights" / f"{sha}.json").read_bytes()
    gw.files[f"/pool/weights/{sha}.npz"] = (tmp / "weights" / f"{sha}.npz").read_bytes()
    for a in pool["task"]["assets"].values():
        gw.files[f"/pool/assets/{a}.npz"] = (tmp / "assets" / f"{a}.npz").read_bytes()
    task = T.Task(task_id=T.new_task_id(), **dict(pool["task"], issued_at=int(time.time()), chunk_steps=32))
    gw.task_queue.append(task.to_dict())
    c = _ctl(tmp_path, gw, fake_game=True)
    try:
        assert c.connect()
        c.start_service()
        c.start()
        assert wait_until(lambda: gw.chunks.get(task.task_id) and gw.chunks[task.task_id][-1].header["final"], 90), \
            (c.snapshot()["logs"], c.snapshot()["instances"])
        chunks = gw.chunks[task.task_id]
        steps = sum(ch.steps for ch in chunks)
        assert chunks[0].header["policy_sha"] == sha and chunks[-1].header["end"] in P.END_REASONS
        assert wait_until(lambda: c.snapshot()["session_accepted"] == steps, 10)
        # the next task's chunks come back "rejected, token revoked": the banner shows the reason as sent
        pl = R.payload(R.Reason.TOKEN_REVOKED_REPLAY, task_id=task.task_id[:8], chunk=0, step=5)
        gw.revoke_on_chunk = pl
        gw.task_queue.append(T.Task(task_id=T.new_task_id(), **dict(pool["task"], issued_at=int(time.time()))).to_dict())
        assert wait_until(lambda: (c.snapshot()["banner"] or {}).get("kind") == "revoked", 60), c.snapshot()["logs"]
        assert c.snapshot()["banner"]["message"] == pl["message"] and not c.snapshot()["running"]
    finally:
        c.shutdown()


# ============================================================================================== review fixes

def test_a_crashing_instance_backs_off_and_gives_up(tmp_path, gw, monkeypatch):
    """A worker that exits at once (a broken bundle): restarts wait 2 s, 4 s, ... and stop after CRASH_LIMIT exits
    in a row with a visible message; the player's start gives it another try."""
    from ballrain.pool import client_core as C
    monkeypatch.setattr(C, "python_command", lambda: ([sys.executable, "-c", "import sys; sys.exit(1)"], {}))
    c = _ctl(tmp_path, gw, simulate=True)
    spawned = []
    real = c._spawn

    def spawn(slot, restarts):
        spawned.append(time.monotonic())
        real(slot, restarts)
    c._spawn = spawn
    try:
        assert c.connect()
        c.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8:
            c.tick()
            time.sleep(0.1)
        assert 2 <= len(spawned) <= 4, len(spawned)  # 0 s, ~2 s, ~6 s (was: one every tick)
        assert all(b - a >= 1.9 for a, b in zip(spawned, spawned[1:]))
        monkeypatch.setattr(C, "CRASH_BACKOFF_S", (0.0, 0.0))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10 and not (c.banner or {}).get("message", "").startswith("实例 1 连续"):
            c.tick()
            time.sleep(0.05)
        n = len(spawned)
        assert "已停止重启" in c.banner["message"] and "logs/instance-01.log" in c.banner["message"]
        for _ in range(20):
            c.tick()
            time.sleep(0.05)
        assert len(spawned) == n  # given up
        c.start()  # the player tries again
        for _ in range(5):
            c.tick()
            time.sleep(0.05)
        assert len(spawned) > n
    finally:
        c.shutdown()


def test_session_expired_reconnects(tmp_path, gw):
    """The gateway forgot the session (a long sleep): the refusal's promise holds, the controller says hello again,
    hands the new session to the workers, and the banner goes away."""
    c = _ctl(tmp_path, gw, simulate=True)
    try:
        assert c.connect()
        sid = c.api.session_id
        gw.sessions.clear()
        c.heartbeat_once()
        assert (c.banner or {}).get("code") == "session_expired"
        c.tick()
        assert c.api.session_id != sid and c.api.session_id in gw.sessions and c.banner is None
        assert len(gw.hellos) == 2
        gw.sessions.clear()  # a second expiry within RECONNECT_S waits (no hello storm)
        c.heartbeat_once()
        c.tick()
        assert len(gw.hellos) == 2
    finally:
        c.shutdown()


def test_too_many_instances_on_hello_retries_after_the_player_lowers_the_count(tmp_path, gw):
    c = _ctl(tmp_path, gw, simulate=True)
    try:
        gw.hello_error = R.payload(R.Reason.TOO_MANY_INSTANCES, max_instances=2)
        c.cfg.instances = 6
        assert not c.connect() and c.banner["code"] == "too_many_instances"
        gw.hello_error = None
        assert c.set_instances(2) == 2
        c.tick()
        assert c.connected and c.banner is None and gw.hellos[-1]["instances"] == 2
    finally:
        c.shutdown()


def test_instances_count_survives_a_revocation(tmp_path, gw):
    c = _ctl(tmp_path, gw, simulate=True)
    try:
        assert c.connect()
        assert c.set_instances(6) == 6
        gw.revoked = R.payload(R.Reason.TOKEN_REVOKED, note="x")
        c.heartbeat_once()
        c.max_instances = 0  # what the gateway's heartbeat_ok says to a revoked account (the stub answers 403)
        assert not c.connected
        assert c.set_instances(c.cfg.instances - 1) == 5  # not clamped to 1
        assert json.loads((tmp_path / "data" / "config.json").read_text(encoding="utf-8"))["instances"] == 5
    finally:
        c.shutdown()


def test_a_bad_cached_manifest_is_dropped(tmp_path, gw):
    """A cached manifest whose npz_sha256 does not match what the gateway serves: the copy is deleted (fetched
    again next time) instead of re-downloading the weights every few seconds forever."""
    from ballrain.pool.client_api import Cache, GatewayClient, GatewayUnavailable
    sha = "a" * 64
    d = tmp_path / "cache" / "weights"
    d.mkdir(parents=True)
    (d / f"{sha}.json").write_text(json.dumps({"policy_sha": sha, "npz_sha256": "b" * 64, "npz_bytes": 3}), "utf-8")
    gw.files[f"/pool/weights/{sha}.npz"] = b"xyz"
    cache = Cache(tmp_path / "cache", GatewayClient(gw.url))
    with pytest.raises((GatewayUnavailable, ValueError)):
        cache.policy(sha)
    assert not (d / f"{sha}.json").exists() and not (d / f"{sha}.npz").exists()
    (d / f"{sha}.json").write_text(json.dumps({"policy_sha": "c" * 64, "npz_sha256": "b" * 64}), "utf-8")  # another
    with pytest.raises((GatewayUnavailable, ValueError)):
        cache.policy(sha)
    assert not (d / f"{sha}.json").exists()


def test_engine_crash_dumps_leave_the_players_folder(tmp_path):
    from ballrain.pool.client_worker import Worker
    root, data = tmp_path / "Ballance", tmp_path / "data"
    (root / "Bin").mkdir(parents=True)
    w = Worker()
    w.cfg = {"root": str(root), "data_dir": str(data)}
    w.emit = lambda *a, **k: None
    old = root / "Bin" / "crash-111.dmp"
    old.write_bytes(b"old")
    os.utime(old, (1, 1))  # not ours: older than the task, another pid
    for i in range(7):
        (root / "Bin" / f"crash-{1000 + i}.dmp").write_bytes(b"d")
    w.collect_crash_dumps(1000, time.time() - 60, keep=5)
    assert [p.name for p in (root / "Bin").glob("*.dmp")] == ["crash-111.dmp"]
    assert len(list((data / "logs").glob("crash-*.dmp"))) == 5


try:  # the fake-game task fixture (needs torch to export a synthetic policy; without it only the end-to-end test skips)
    from test_pool_sampler import pool  # noqa: E402,F401
except pytest.skip.Exception:
    pass

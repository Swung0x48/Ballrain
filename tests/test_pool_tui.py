"""The controller's Textual UI (tools/pool/controller.py build_app) driven headless with Textual's pilot against the
stub gateway: the first-run screen's game-folder messages, start / pause, the leaderboard switch synced to the
gateway, and the revocation banner showing the gateway's text verbatim (no markup)."""
import asyncio
import importlib.util
import time
from pathlib import Path

import pytest

pytest.importorskip("textual")

from ballrain.pool import reasons as R  # noqa: E402
from ballrain.pool.client_core import ClientConfig, Controller  # noqa: E402
from test_pool_stub_gateway import TOKEN, StubGateway  # noqa: E402

CONTROLLER = Path(__file__).resolve().parents[1] / "tools" / "pool" / "controller.py"


def _module():
    spec = importlib.util.spec_from_file_location("pool_controller_ui", CONTROLLER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _text(w) -> str:
    return str(w.render())


async def _until(pilot, cond, timeout=15.0):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if cond():
            return True
        await pilot.pause(0.1)
    return cond()


@pytest.fixture
def gw():
    g = StubGateway()
    yield g
    g.close()


def test_setup_screen_lists_the_missing_files(tmp_path, gw):
    c = Controller(tmp_path / "d", ClientConfig(gateway_url=gw.url, edition="all", ballance_dir=str(tmp_path),
                                                engine_binary=str(tmp_path / "SimTool.exe"), instances=1),
                   auto_pause=False)
    app = _module().build_app(c)

    async def go():
        async with app.run_test(size=(150, 50)) as pilot:
            await pilot.pause(0.3)
            assert type(app.screen).__name__ == "SetupScreen"
            await pilot.click("#verify")
            await pilot.pause(0.3)
            txt = _text(app.screen.query_one("#problems"))
            assert "缺少文件：base.cmo" in txt and "缺少文件夹：Text" in txt and "找不到无头引擎文件" in txt
    try:
        asyncio.run(go())
    finally:
        c.shutdown()


def test_main_screen_start_pause_leaderboard_and_revocation_banner(tmp_path, gw):
    c = Controller(tmp_path / "d", ClientConfig(gateway_url=gw.url, edition="all", instances=1), simulate=True,
                   auto_pause=False)
    c.set_token(TOKEN)
    app = _module().build_app(c)
    pl = R.payload(R.Reason.TOKEN_REVOKED, note="[b]不是标记[/b] 请联系管理员")

    async def go():
        async with app.run_test(size=(150, 50)) as pilot:
            assert await _until(pilot, lambda: c.connected)
            scr = app.screen
            assert type(scr).__name__ == "MainScreen"
            sw = scr.query_one("#leaderboard")
            assert await _until(pilot, lambda: sw.value is True and not sw.disabled)
            await pilot.click("#leaderboard")  # the player turns the leaderboard off: one POST, the gateway's answer shown
            assert await _until(pilot, lambda: gw.me_updates == [False] and c.leaderboard is False)
            await pilot.pause(0.6)
            assert sw.value is False and gw.me_updates == [False]
            await pilot.click("#toggle")
            assert c.want_running
            assert await _until(pilot, lambda: c.snapshot()["instances"] and
                                c.snapshot()["instances"][0]["state"] == "running")
            assert await _until(pilot, lambda: "暂停" in str(scr.query_one("#toggle").label))
            pid = c.snapshot()["instances"][0]["pid"]
            for _ in range(2):  # settings and back, twice: still one main screen, one service, one instance per slot
                await pilot.click("#setup")
                await pilot.pause(0.3)
                assert type(app.screen).__name__ == "SetupScreen"
                await pilot.click("#go")
                await pilot.pause(0.5)
                assert app.screen is scr
            await pilot.pause(1.5)
            assert len(c._threads) == 2 and sum(type(s).__name__ == "MainScreen" for s in app.screen_stack) == 1
            assert [i["pid"] for i in c.snapshot()["instances"]] == [pid] and len(c.children.alive()) == 1
            await pilot.click("#toggle")
            assert not c.want_running
            gw.revoked = pl
            c.heartbeat_once()
            await pilot.pause(0.8)
            banner = scr.query_one("#banner")
            assert banner.display and _text(banner) == f"【令牌已停用】{pl['message']}"
            assert "[b]不是标记[/b]" in _text(banner)  # shown as sent, not interpreted as markup
    try:
        asyncio.run(go())
    finally:
        c.shutdown()


def test_quitting_during_a_pending_forum_login_ends_everything_at_once(tmp_path, gw, monkeypatch):
    """The player starts sampling, opens the settings, starts a forum login and quits before finishing it in the
    browser: the login wait stops, and the instance and its engine end right away (not when the event loop has joined
    the login thread minutes later)."""
    import threading
    import webbrowser

    from ballrain.pool.client_api import GatewayClient
    from ballrain.pool.client_lifecycle import pid_alive
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)

    class Pending(GatewayClient):
        def login_poll(self, *a, **k):
            return {"type": "login_status", "status": "pending"}
    c = Controller(tmp_path / "d", ClientConfig(gateway_url=gw.url, edition="all", instances=1), simulate=True,
                   auto_pause=False, api_factory=Pending)
    c.set_token(TOKEN)
    app = _module().build_app(c)
    info = {}

    async def go():
        async with app.run_test(size=(150, 50)) as pilot:
            assert await _until(pilot, lambda: c.connected)
            await pilot.click("#toggle")
            assert await _until(pilot, lambda: c.snapshot()["instances"] and c.snapshot()["instances"][0]["engine_pid"])
            i = c.snapshot()["instances"][0]
            info["pids"] = (i["pid"], i["engine_pid"])
            await pilot.click("#setup")
            await pilot.pause(0.3)
            await pilot.click("#login")
            await pilot.pause(1.0)
            assert "ABCD-EF23" in _text(app.screen.query_one("#token_state"))  # the login code is shown
            info["t"] = time.monotonic()
            await pilot.press("ctrl+q")
    th = threading.Thread(target=lambda: asyncio.run(go()), daemon=True)
    try:
        th.start()
        th.join(30)
        assert not th.is_alive(), "the app did not return (the login wait kept it)"
        assert time.monotonic() - info["t"] < 10
        assert not any(pid_alive(p) for p in info["pids"])
    finally:
        c.shutdown()

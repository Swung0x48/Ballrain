"""Ballrain 算力池主控程序 (the volunteer's controller; docs/pool-prototype.md section 9).

    python tools/pool/controller.py                      # the TUI (Textual; uv sync --extra pool)
    python tools/pool/controller.py --headless --start   # no UI: status lines on stdout (scripts, tests)

Options (also stored in <data dir>/config.json once given): --gateway URL, --ballance-dir DIR, --edition
windows|all, --engine FILE, --instances N (1..8, default half the cores). --data-dir DIR (default: the user's
application-data folder). The pool token lives in <data dir>/token (log in through the forum in the TUI, or
--token-stdin to read it from stdin). Closing the controller in any way -- the window, Ctrl+C, a crash, a kill --
ends every instance and engine it started (ballrain.pool.client_lifecycle).

Headless extras: --start (sample right away), --status-file FILE (the snapshot as JSON every second), --duration S
(exit after S seconds), --login (print the forum login link and wait), --leaderboard on|off (set the switch),
--no-auto-pause, --simulate (stand-in engines, tests), --fake-game (the in-process fake game, tests).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ballrain.pool import client_game as G  # noqa: E402
from ballrain.pool.client_core import BANNER_TITLES, MAX_INSTANCES, ClientConfig, Controller, default_data_dir  # noqa: E402

TITLE = "Ballrain 算力池"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ballrain 算力池主控程序")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--gateway", default=None, help="网关地址（https://…）；也可用环境变量 BALLRAIN_POOL_GATEWAY")
    p.add_argument("--ballance-dir", default=None, help="Ballance 安装目录")
    p.add_argument("--edition", choices=sorted(G.EDITIONS), default=None, help="windows：2002 原版物理；all：全平台版")
    p.add_argument("--engine", default=None, help="无头引擎文件（BallanceMMOSimTool）")
    p.add_argument("--instances", type=int, default=None, help=f"实例数 1..{MAX_INSTANCES}")
    p.add_argument("--headless", "--no-tui", dest="headless", action="store_true", help="不用界面")
    p.add_argument("--start", action="store_true", help="headless: 立即开始采样")
    p.add_argument("--status-file", default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--token-stdin", action="store_true", help="从标准输入读取算力池令牌")
    p.add_argument("--login", action="store_true", help="headless: 通过论坛登录领取令牌")
    p.add_argument("--leaderboard", choices=("on", "off"), default=None)
    p.add_argument("--no-auto-pause", action="store_true", help="不因 Player.exe / 电池自动暂停")
    p.add_argument("--simulate", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--fake-game", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dev-hello", default=None, help=argparse.SUPPRESS)  # a dev gateway's dev.json (engine / file hashes)
    return p.parse_args(argv)


def make_controller(a) -> Controller:
    data = Path(a.data_dir) if a.data_dir else default_data_dir()
    cfg = ClientConfig.load(data)
    gw = a.gateway or os.environ.get("BALLRAIN_POOL_GATEWAY")
    if gw:
        cfg.gateway_url = gw
    if a.ballance_dir:  # absolute: the instances run with the data directory as their working directory
        cfg.ballance_dir = str(Path(a.ballance_dir).resolve())
    if a.edition:
        cfg.edition = a.edition
    if a.engine:
        cfg.engine_binary = str(Path(a.engine).resolve())
    if a.instances is not None:
        cfg.instances = max(1, min(MAX_INSTANCES, a.instances))
    cfg.save(data)
    dev_hello = json.loads(Path(a.dev_hello).read_text(encoding="utf-8")) if a.dev_hello else None
    ctl = Controller(data, cfg, simulate=a.simulate, fake_game=a.fake_game, auto_pause=not a.no_auto_pause,
                     dev_hello=dev_hello)
    if a.token_stdin:
        tok = sys.stdin.readline().strip()
        ctl.set_token(tok)
    return ctl


def install_exit_handlers(on_signal, on_close=None) -> None:
    """Ctrl+C, Ctrl+Break, SIGTERM / SIGHUP: on_signal(); on Windows the console window being closed (or log-off /
    shut-down): on_close() -- the process ends right after it returns, and the job object kills whatever is left."""
    on_close = on_close or on_signal

    def handler(signum, frame):
        on_signal()
    for s in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"):
        if hasattr(signal, s):
            try:
                signal.signal(getattr(signal, s), handler)
            except (ValueError, OSError):
                pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def console_handler(event):
            if event in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
                on_close()
                return True
            return False
        install_exit_handlers._keep = HandlerRoutine(console_handler)  # keep the callback alive
        ctypes.windll.kernel32.SetConsoleCtrlHandler(install_exit_handlers._keep, True)


# ============================================================================================== headless

def run_headless(a, ctl: Controller) -> int:
    done = threading.Event()
    stopped = threading.Event()

    def stop_all():
        if not stopped.is_set():
            stopped.set()
            ctl.shutdown()
        done.set()
    install_exit_handlers(done.set, stop_all)
    ctl.log_echo = lambda line: print(line, flush=True)
    status = Path(a.status_file) if a.status_file else None
    try:
        if a.login:
            st = ctl.login_start()
            print(f"请在浏览器中打开并登录论坛：{st['browser_url']}", flush=True)
            print(f"登录代码：{st['user_code']}（浏览器里显示的代码必须和这里一致）", flush=True)
            try:
                webbrowser.open(st["browser_url"])
            except Exception:  # noqa: BLE001
                pass
            res = ctl.login_wait(st, should_cancel=done.is_set)
            if res.get("status") != "ok":
                print(f"登录没有完成：{res.get('status')}", flush=True)
                return 2
        for p in ctl.problems():
            print(f"[问题] {p.message}", flush=True)
        if not ctl.connect():
            b = ctl.banner or {}
            print(f"【{BANNER_TITLES.get(b.get('kind'), '')}】{b.get('message')}", flush=True)
            return 3
        if a.leaderboard is not None:
            ctl.set_leaderboard(a.leaderboard == "on")
        ctl.start_service()
        if a.start:
            ctl.start()
        t_end = time.monotonic() + a.duration if a.duration else None
        last_banner = None
        while not done.is_set():
            snap = ctl.snapshot()
            if status is not None:
                tmp = status.with_name(status.name + ".tmp")
                tmp.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, status)
            if snap["banner"] != last_banner and snap["banner"]:
                print(f"【{BANNER_TITLES.get(snap['banner']['kind'], '')}】{snap['banner']['message']}", flush=True)
            last_banner = snap["banner"]
            if t_end is not None and time.monotonic() >= t_end:
                break
            done.wait(1.0)
    finally:
        stop_all()
    return 0


# ============================================================================================== TUI

def build_app(ctl: Controller):
    """The Textual app over a Controller (raises ImportError without textual)."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, VerticalScroll
    from textual.screen import Screen
    from textual.widgets import Button, DataTable, Footer, Header, Input, Label, RichLog, Select, Static, Switch

    class SetupScreen(Screen):
        """First run: gateway, forum login / token, Ballance folder, edition, engine."""

        def compose(self) -> ComposeResult:
            c = ctl.cfg
            yield Header()
            with VerticalScroll(id="setup"):
                yield Label("第一步：网关地址")
                yield Input(c.gateway_url, placeholder="https://…/", id="gateway")
                yield Label("第二步：登录论坛，领取算力池令牌")
                with Horizontal(classes="row"):
                    yield Button("在浏览器中登录论坛", id="login", variant="primary")
                    yield Input(placeholder="或者把令牌粘贴到这里", password=True, id="token")
                    yield Button("保存令牌", id="save_token")
                yield Static("已保存令牌。" if ctl.token else "还没有令牌。", id="token_state", markup=False)
                yield Label("第三步：Ballance 安装目录")
                yield Input(c.ballance_dir, placeholder="例如 C:\\Games\\Ballance", id="ballance_dir")
                yield Label("版本")
                yield Select([(v["label"], k) for k, v in G.EDITIONS.items()], value=c.edition, allow_blank=False,
                             id="edition")
                yield Label("无头引擎文件（随主控程序附带时可留空）")
                yield Input(c.engine_binary, placeholder="BallanceMMOSimTool", id="engine")
                with Horizontal(classes="row"):
                    yield Button("校验游戏文件", id="verify")
                    yield Button("进入主界面", id="go", variant="success")
                yield Static("", id="problems", markup=False)
            yield Footer()

        def _save(self) -> None:
            c = ctl.cfg
            c.gateway_url = self.query_one("#gateway", Input).value.strip()
            c.ballance_dir = self.query_one("#ballance_dir", Input).value.strip()
            c.edition = str(self.query_one("#edition", Select).value)
            c.engine_binary = self.query_one("#engine", Input).value.strip()
            c.save(ctl.data_dir)

        def _show_problems(self) -> bool:
            self._save()
            probs = ctl.problems()
            box = self.query_one("#problems", Static)
            if probs:
                box.update("\n".join(f"✗ {p.message}" for p in probs))
            else:
                box.update("✓ 游戏文件和引擎都找到了。其余文件是否为原版由网关在连接时核对。")
            return not probs

        def on_button_pressed(self, ev: Button.Pressed) -> None:
            bid = ev.button.id
            if bid == "verify":
                self._show_problems()
            elif bid == "save_token":
                try:
                    ctl.set_token(self.query_one("#token", Input).value.strip())
                    self.query_one("#token_state", Static).update("已保存令牌。")
                except ValueError as e:
                    self.query_one("#token_state", Static).update(str(e))
            elif bid == "login":
                self._save()
                self.app.run_worker(self._login, thread=True, exclusive=True)
            elif bid == "go":
                if self._show_problems() and ctl.token:
                    main = getattr(self.app, "main_screen", None)
                    if main is not None and main in self.app.screen_stack:  # back to the one main screen
                        self.app.pop_screen()
                        main.reconnect()
                    else:
                        self.app.push_screen(MainScreen())
                elif not ctl.token:
                    self.query_one("#token_state", Static).update("还没有令牌：请先登录论坛或粘贴令牌。")

        def _login(self) -> None:
            st_box = self.query_one("#token_state", Static)
            try:
                st = ctl.login_start()
            except Exception as e:  # noqa: BLE001
                self.app.call_from_thread(st_box.update, f"无法开始登录：{e}")
                return
            self.app.call_from_thread(st_box.update, f"登录代码 {st['user_code']}：已在浏览器中打开登录页，请核对页面上的代码"
                                                     f"与此一致后继续。如果没有打开，请手动访问：{st['browser_url']}")
            try:
                webbrowser.open(st["browser_url"])
            except Exception:  # noqa: BLE001
                pass
            res = ctl.login_wait(st)
            if res.get("status") == "ok":
                self.app.call_from_thread(st_box.update, "登录成功，已保存令牌。")
            elif res.get("reason"):
                self.app.call_from_thread(st_box.update, res["reason"]["message"])
            else:
                self.app.call_from_thread(st_box.update, "登录已过期，请重新开始。")

    class MainScreen(Screen):
        BINDINGS = [Binding("space", "toggle", "开始 / 暂停"), Binding("minus", "fewer", "减少实例"),
                    Binding("plus,equals_sign", "more", "增加实例")]

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("", id="banner", markup=False)
            with Horizontal(id="controls"):
                yield Button("开始", id="toggle", variant="success")
                yield Button("−", id="fewer")
                yield Static("", id="slider", markup=False)
                yield Button("+", id="more")
                yield Label("上榜")
                yield Switch(value=True, id="leaderboard", disabled=True)
                yield Button("设置", id="setup")
                yield Button("退出", id="quit", variant="error")
            yield Static("", id="summary", markup=False)
            yield DataTable(id="table", zebra_stripes=True, cursor_type="none")
            yield RichLog(id="log", markup=False, wrap=True, max_lines=500)
            yield Footer()

        def on_mount(self) -> None:
            t = self.query_one("#table", DataTable)
            t.add_columns("实例", "状态", "当前任务", "本任务步数", "累计步数", "已被接收", "步/秒", "说明")
            self._logn = 0
            self._lb_pending = False
            self.app.main_screen = self
            self.reconnect()
            self.set_interval(0.5, self.refresh_view)

        def reconnect(self) -> None:
            """hello with the current settings (first time, or back from the settings screen)."""
            self.app.run_worker(self._connect, thread=True, exclusive=True, group="connect")

        def _connect(self) -> None:
            if ctl.connect():
                ctl.start_service()  # once per controller (later calls do nothing)

        def refresh_view(self) -> None:
            s = ctl.snapshot()
            b = self.query_one("#banner", Static)
            if s["banner"]:
                kind = BANNER_TITLES.get(s["banner"]["kind"], "")
                b.update(f"【{kind}】{s['banner']['message']}")
                b.display = True
                b.set_class(s["banner"]["kind"] == "revoked", "revoked")
            elif s["auto_pause"]:
                b.update(s["auto_pause"])
                b.display = True
                b.remove_class("revoked")
            else:
                b.display = False
            tb = self.query_one("#toggle", Button)
            tb.label = "暂停" if s["want_running"] else "开始"
            tb.variant = "warning" if s["want_running"] else "success"
            tb.disabled = not s["connected"]
            n, mx = s["instances_wanted"], max(1, s["max_instances"])
            self.query_one("#slider", Static).update(f" 实例数 {n} / {mx}  " + "■" * n + "□" * (mx - n) + " ")
            sw = self.query_one("#leaderboard", Switch)
            if s["leaderboard"] is not None and sw.value != s["leaderboard"] and not self._lb_pending:
                sw.value = s["leaderboard"]  # the gateway's state (its Changed event matches ctl.leaderboard: ignored)
            sw.disabled = not s["connected"] or self._lb_pending  # one change at a time (no click lost in flight)
            user = s["user"] or {}
            st = s["stats"] or {}
            name = user.get("nickname") or user.get("username") or "未连接"
            self.query_one("#summary", Static).update(
                f"玩家：{name}    状态：{'采样中' if s['running'] else ('自动暂停' if s['auto_pause'] else '已暂停')}    "
                f"本次贡献 {s['session_steps']} 步（已接收 {s['session_accepted']} 步）    "
                f"累计已接收 {st.get('steps_accepted', '-')} 步，已通过验证 {st.get('steps_verified', '-')} 步    "
                f"{s['steps_per_s']} 步/秒")
            t = self.query_one("#table", DataTable)
            t.clear()
            for i in s["instances"]:
                t.add_row(str(i["slot"] + 1), i["state_text"], (i["task_id"] or "-")[:12], str(i["task_steps"]),
                          str(i["total_steps"]), str(i["accepted_steps"]), str(i["steps_per_s"]), i["detail"][:60])
            log = self.query_one("#log", RichLog)
            logs = list(ctl.logs)
            for line in logs[self._logn:] if self._logn <= len(logs) else logs:
                log.write(line)
            self._logn = len(logs)

        def action_toggle(self) -> None:
            if ctl.want_running:
                ctl.pause()
            else:
                ctl.start()

        def action_fewer(self) -> None:
            ctl.set_instances(ctl.cfg.instances - 1)

        def action_more(self) -> None:
            ctl.set_instances(ctl.cfg.instances + 1)

        def on_button_pressed(self, ev: Button.Pressed) -> None:
            bid = ev.button.id
            if bid == "toggle":
                self.action_toggle()
            elif bid == "fewer":
                self.action_fewer()
            elif bid == "more":
                self.action_more()
            elif bid == "setup":
                self.app.push_screen(SetupScreen())
            elif bid == "quit":
                ctl.request_stop()  # as PoolApp.action_quit
                self.app.exit()

        def on_switch_changed(self, ev: Switch.Changed) -> None:
            if ev.switch.id != "leaderboard" or ctl.leaderboard is None or bool(ev.value) == ctl.leaderboard:
                return  # not the player's change (the view following the gateway's state)
            want = bool(ev.value)
            self._lb_pending = True
            ev.switch.disabled = True
            self.app.run_worker(lambda: self._set_lb(want), thread=True)

        def _set_lb(self, want: bool) -> None:
            try:
                ctl.set_leaderboard(want)
            except Exception as e:  # noqa: BLE001  (the switch returns to the gateway's state)
                ctl.log(f"上榜设置失败：{e}")
            finally:
                self._lb_pending = False

    class PoolApp(App):
        TITLE = TITLE
        CSS = """
        #setup { padding: 1 2; }
        .row { height: auto; }
        .row Input { width: 1fr; }
        #banner { background: $warning 30%; color: $text; padding: 0 1; margin: 0 1; }
        #banner.revoked { background: $error 40%; }
        #controls { height: auto; padding: 0 1; }
        #controls Button { min-width: 6; }
        #controls Label { padding: 1 1 0 2; }
        #slider { padding: 1 1 0 1; width: auto; }
        #summary { padding: 0 1; height: auto; }
        #table { height: 1fr; min-height: 6; }
        #log { height: 12; border: tall $panel; }
        """
        BINDINGS = [Binding("ctrl+q", "quit", "退出", priority=True), Binding("ctrl+c", "quit", "退出", priority=True)]

        def on_mount(self) -> None:
            ready = ctl.token and not ctl.problems() and ctl.cfg.gateway_url
            self.push_screen(MainScreen() if ready else SetupScreen())

        async def action_quit(self) -> None:
            # stop first: a pending forum-login wait (a thread worker the event loop would join for minutes) returns
            # within poll_s, and the instances and engines end now, not when the event loop has finished
            ctl.request_stop()
            self.exit()

        def on_unmount(self) -> None:
            ctl.shutdown()

    return PoolApp()


def run_tui(a, ctl: Controller) -> int:
    try:
        app = build_app(ctl)
    except ImportError:
        print("界面需要 textual：uv sync --extra pool（或 pip install textual）。也可以用 --headless。", file=sys.stderr)
        return 4

    def hard_exit():
        ctl.shutdown()
        os._exit(0)
    install_exit_handlers(hard_exit)
    try:
        app.run()
    finally:
        ctl.shutdown()
    return 0


def main(argv=None) -> int:
    a = parse_args(argv)
    ctl = make_controller(a)
    try:
        if a.headless:
            return run_headless(a, ctl)
        return run_tui(a, ctl)
    finally:
        ctl.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

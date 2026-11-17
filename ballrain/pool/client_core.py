"""The controller's core, without any UI (docs/pool-prototype.md section 9): configuration and token, the game-folder
check, the gateway session (hello, heartbeat, "me", the leaderboard switch), the sampling instances (one
client_worker child process each, all inside client_lifecycle.ChildGroup) and the automatic pause. The TUI and the
--headless driver (tools/pool/controller.py) are thin layers over Controller.snapshot() and its actions.

Rules from docs/volunteer-compute.md 11:
- start / pause is the player's; closing the controller ends every instance and engine (ChildGroup);
- a refusal or revocation shows the gateway's reason.message exactly as received (banner); a TOKEN_FATAL reason
  (or heartbeat_ok.revoked, or chunk_result.token_revoked) stops every instance until the player logs in again;
- the leaderboard switch: the gateway's state is the truth (hello_ok.leaderboard, then POST /pool/me on the player's
  click only), default on at the gateway;
- the controller pauses itself while Player.exe runs or the machine is on battery, and resumes after.
"""
from __future__ import annotations

import collections
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import client_game as G
from .client_api import GatewayClient, GatewayError, GatewayRefusal, GatewayUnavailable, check_gateway_url
from .client_lifecycle import ChildGroup, on_battery, player_running, python_command

MAX_INSTANCES = 8
CRASH_LIMIT = 8  # unexpected exits of one slot in a row before the controller stops restarting it
CRASH_BACKOFF_S = (2.0, 300.0)  # first wait, longest wait (doubling in between)
STABLE_S = 120.0  # an instance that ran this long resets its slot's failure count
RECONNECT_S = 30.0  # at most one automatic hello per this many seconds (session expired, instance count changed)
RECONNECT_CODES = ("session_expired", "too_many_instances")
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32,128}")
REPO_ROOT = Path(__file__).resolve().parents[2]

PAUSE_PLAYER = "检测到 Ballance（Player.exe）正在运行，已自动暂停；关闭游戏后会自动继续。"
PAUSE_BATTERY = "电脑正在使用电池，已自动暂停；接上电源后会自动继续。"
BANNER_TITLES = {"revoked": "令牌已停用", "rejected": "网关拒收", "error": "无法连接"}
STATE_NAMES = {"idle": "空闲", "starting": "准备中", "running": "采样中", "paused": "已暂停", "error": "出错",
               "exited": "已退出"}


def default_instances() -> int:
    return max(1, min(MAX_INSTANCES, (os.cpu_count() or 2) // 2))


def default_data_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "BallrainPool"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "BallrainPool"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "ballrain-pool"


# ============================================================================================== configuration

@dataclass
class ClientConfig:
    gateway_url: str = ""
    ballance_dir: str = ""
    edition: str = field(default_factory=G.default_edition)
    engine_binary: str = ""
    instances: int = field(default_factory=default_instances)
    auto_pause_player: bool = True
    auto_pause_battery: bool = True

    @classmethod
    def load(cls, data_dir) -> "ClientConfig":
        p = Path(data_dir) / "config.json"
        c = cls()
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                d = {}
            for k in asdict(c):
                if k in d and isinstance(d[k], type(getattr(c, k))):
                    setattr(c, k, d[k])
        c.instances = max(1, min(MAX_INSTANCES, int(c.instances)))
        return c

    def save(self, data_dir) -> None:
        d = Path(data_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / ".config.json.tmp"
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, d / "config.json")


def load_token(data_dir) -> str | None:
    p = Path(data_dir) / "token"
    try:
        t = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return t or None


def save_token(data_dir, token: str | None) -> None:
    p = Path(data_dir) / "token"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not token:
        p.unlink(missing_ok=True)
        return
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("令牌格式不对（应为 32–128 个字母、数字、- 或 _）")
    tmp = p.with_name(".token.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(token)
    os.replace(tmp, p)


# ============================================================================================== instances

@dataclass
class Instance:
    slot: int
    proc: subprocess.Popen | None = None
    pid: int | None = None
    state: str = "starting"
    detail: str = ""
    task_id: str | None = None
    task_steps: int = 0
    total_steps: int = 0
    accepted_steps: int = 0
    steps_per_s: float = 0.0
    engine_pid: int | None = None
    chunks: int = 0
    stopping: bool = False
    sent: str = ""  # the last run / pause command sent
    restarts: int = 0
    log_file: object = None
    started: float = 0.0  # time.monotonic() at spawn

    def view(self) -> dict:
        return {"slot": self.slot, "pid": self.pid, "state": self.state, "state_text": STATE_NAMES.get(self.state, self.state),
                "detail": self.detail, "task_id": self.task_id, "task_steps": self.task_steps,
                "total_steps": self.total_steps, "accepted_steps": self.accepted_steps,
                "steps_per_s": self.steps_per_s, "engine_pid": self.engine_pid, "chunks": self.chunks}


# ============================================================================================== controller

def _close_pipes(p) -> None:
    """Close a child's pipes quietly (a dead child's stdin cannot take the buffered bytes any more)."""
    for f in (getattr(p, "stdin", None), getattr(p, "stdout", None)):
        if f is not None:
            try:
                f.close()
            except OSError:
                pass


class Controller:
    def __init__(self, data_dir, config: ClientConfig | None = None, *, api_factory=GatewayClient, probes=None,
                 simulate: bool = False, fake_game: bool = False, auto_pause: bool = True, log_echo=None,
                 dev_hello: dict | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = config or ClientConfig.load(self.data_dir)
        self.token = load_token(self.data_dir)
        self.api_factory = api_factory
        self.api: GatewayClient | None = None
        self.probes = probes if probes is not None else {"player": player_running, "battery": on_battery}
        self.auto_pause_enabled = auto_pause
        self.simulate, self.fake_game = simulate, fake_game
        self.dev_hello = dev_hello  # tests / development: {"engine_sha256", "game_files"} to report in simulate / fake-game mode
        self.log_echo = log_echo
        self.children = ChildGroup()
        self.events: queue.Queue = queue.Queue()
        self.lock = threading.RLock()
        self.instances: dict[int, Instance] = {}
        self.logs: collections.deque = collections.deque(maxlen=500)
        self.hello: dict | None = None
        self.user: dict | None = None
        self.leaderboard: bool | None = None
        self.stats: dict | None = None
        self.max_instances = MAX_INSTANCES
        self.heartbeat_s = 30
        self.banner: dict | None = None  # {"kind": "revoked" | "rejected" | "error", "code", "message"}
        self.token_dead = False
        self.want_running = False
        self.auto_pause: str | None = None
        self.notice: str | None = None
        self.engine: dict | None = None
        self.game_files: dict | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._spawn_cfg = None  # (edition, engine, game folder, gateway) the instances were started with
        self.probe_s = 5.0  # how often the automatic-pause probes run
        self._last_probe = 0.0
        self._last_hb = 0.0
        self._last_me = 0.0
        self._crash: dict[int, tuple] = {}  # slot -> (unexpected exits in a row, monotonic time it may start again)
        self._reconnect_wanted = False
        self._next_reconnect = 0.0

    # ------------------------------------------------------------------------------------------ logging
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.lock:
            self.logs.append(line)
        if self.log_echo is not None:
            self.log_echo(line)

    def set_banner(self, kind: str, reason: dict | None = None, message: str | None = None) -> None:
        with self.lock:
            self.banner = {"kind": kind, "code": (reason or {}).get("code"),
                           "message": (reason or {}).get("message") if reason else message}
        self.log(f"【{BANNER_TITLES.get(kind, kind)}】{self.banner['message']}")

    def clear_banner(self) -> None:
        with self.lock:
            self.banner = None

    # ------------------------------------------------------------------------------------------ setup checks
    def problems(self) -> list:
        """Game folder + engine problems for the configured edition (empty = ready to connect)."""
        out = []
        try:
            G.edition_engine(self.cfg.edition)
        except ValueError as e:
            return [G.Problem("", "engine", str(e))]
        if self.simulate or self.fake_game:  # development / test modes: no engine, no game folder
            return out
        out += G.check_game_dir(self.cfg.ballance_dir, self.cfg.edition)
        out += G.check_engine(self.engine_binary(), self.cfg.edition)
        return out

    def engine_binary(self) -> str:
        if self.cfg.engine_binary:  # absolute (the instances' working directory is the data directory)
            return str(Path(self.cfg.engine_binary).resolve())
        b = G.bundled_engine(self.cfg.edition)
        return str(b) if b else ""

    def set_token(self, token: str | None) -> None:
        save_token(self.data_dir, token)
        with self.lock:
            self.token = token or None
            self.token_dead = False
            if self.banner and self.banner.get("kind") == "revoked":
                self.banner = None
        if self.api is not None:
            self.api.token = self.token

    # ------------------------------------------------------------------------------------------ login
    def login_start(self) -> dict:
        api = self.api_factory(self.cfg.gateway_url)
        return api.login_start()

    def login_wait(self, started: dict, should_cancel=lambda: False) -> dict:
        """Poll until the player finished in the browser; returns the final login_status (status ok: the token is saved)."""
        api = self.api_factory(self.cfg.gateway_url)
        t_end = time.monotonic() + int(started["expires_in_s"])
        while time.monotonic() < t_end and not should_cancel() and not self._stop.is_set():
            try:
                st = api.login_poll(started["login_id"], started["poll_secret"])
            except GatewayRefusal as e:
                if e.code != "rate_limited":
                    self.set_banner("rejected", e.reason)
                    return {"status": "refused", "reason": e.reason}
                self._stop.wait(max(int(started["poll_s"]), int((e.reason.get("params") or {}).get("retry_s") or 1)))
                continue
            except GatewayUnavailable as e:
                self.log(f"登录状态查询失败，稍后重试：{e}")
                self._stop.wait(int(started["poll_s"]))
                continue
            if st["status"] == "ok" and st.get("token"):
                self.set_token(st["token"])
                self.log("论坛登录成功，已保存算力池令牌。")
                return st
            if st["status"] in ("refused", "expired"):
                if st.get("reason"):
                    self.set_banner("rejected", st["reason"])
                return st
            self._stop.wait(int(started["poll_s"]))
        return {"status": "expired"}

    # ------------------------------------------------------------------------------------------ session
    def connect(self) -> bool:
        """hello with this machine's engine and game-file hashes. False (with a banner) when refused."""
        try:
            check_gateway_url(self.cfg.gateway_url)
        except ValueError as e:
            self.set_banner("error", message=str(e))
            return False
        if not self.token:
            self.set_banner("error", message="还没有算力池令牌：请先在浏览器里登录论坛，然后把令牌粘贴到主控程序。")
            return False
        probs = self.problems()
        if probs:
            self.set_banner("error", message=probs[0].message)
            return False
        if self.simulate or self.fake_game:  # no engine / game folder: hashes from dev_hello (a dev gateway's) or zeros
            from . import protocol as P
            dh = self.dev_hello or {}
            kind = G.edition_engine(self.cfg.edition)
            self.engine = {"kind": kind, "sha256": dh.get("engine_sha256", "0" * 64), "version": "simulate"}
            names = P.GAME_FILE_LIST_2002 if kind == "2002" else P.GAME_FILE_LIST
            self.game_files = {n: (dh.get("game_files") or {}).get(n, "0" * 64) for n in names}
        else:
            self.engine = G.engine_info(self.engine_binary(), self.cfg.edition)
            self.game_files = G.game_hashes(self.cfg.ballance_dir, self.cfg.edition)
        self.api = self.api_factory(self.cfg.gateway_url, self.token)
        try:
            h = self.api.hello(G.platform_name(), self.engine, self.game_files, self.cfg.instances)
        except GatewayRefusal as e:
            self._refused(e.reason, e.fatal)
            return False
        except GatewayUnavailable as e:
            self.set_banner("error", message=str(e))
            return False
        with self.lock:
            self.hello = h
            self.user = h["user"]
            self.leaderboard = bool(h["leaderboard"])
            self.max_instances = min(MAX_INSTANCES, int(h["max_instances"]))
            self.heartbeat_s = int(h["heartbeat_s"])
            self.notice = h.get("notice")
            self._reconnect_wanted = False
            if self.banner and (self.banner.get("kind") == "error" or self.banner.get("code") in RECONNECT_CODES):
                self.banner = None
        spawn_cfg = (self.cfg.edition, self.engine_binary(), self.cfg.ballance_dir, self.cfg.gateway_url)
        if self._spawn_cfg is not None and spawn_cfg != self._spawn_cfg:  # settings changed: restart the instances
            for inst in list(self.instances.values()):
                inst.stopping = True
                self._send(inst, {"cmd": "stop"})
        self._spawn_cfg = spawn_cfg
        name = self.user.get("nickname") or self.user.get("username")
        self.log(f"已连接网关：{name}，上榜 {'开' if self.leaderboard else '关'}，最多 {self.max_instances} 个实例。")
        for inst in list(self.instances.values()):  # a new session: tell the running workers
            self._send(inst, {"cmd": "token", "token": self.token, "session_id": self.api.session_id})
        self._last_me = 0.0
        return True

    @property
    def connected(self) -> bool:
        return self.hello is not None and not self.token_dead

    def _refused(self, reason: dict, fatal: bool) -> None:
        self.set_banner("revoked" if fatal else "rejected", reason)
        if not fatal and (reason or {}).get("code") == "session_expired":
            with self.lock:  # the gateway forgot the session (a long sleep, a gateway restart): hello again
                self._reconnect_wanted = True
        if fatal:
            with self.lock:
                self.token_dead = True
                self.want_running = False
            for inst in list(self.instances.values()):
                self._send(inst, {"cmd": "pause"})

    # ------------------------------------------------------------------------------------------ player actions
    def start(self) -> None:
        with self.lock:
            self.want_running = True
            self._crash.clear()  # the player asks again: slots that gave up get a new try
        self.log("开始采样。")

    def request_stop(self) -> None:
        """The controller is closing: every wait (login polling, the service loops) ends now."""
        self._stop.set()

    def pause(self) -> None:
        with self.lock:
            self.want_running = False
        self.log("已暂停。")

    def set_instances(self, n: int) -> int:
        # max_instances is 0 after a revocation (and unknown before hello): never clamp the player's choice to that
        n = max(1, min(int(n), self.max_instances or MAX_INSTANCES, MAX_INSTANCES))
        with self.lock:
            self.cfg.instances = n
            if self.banner and self.banner.get("code") == "too_many_instances":
                self._reconnect_wanted, self._next_reconnect = True, 0.0  # hello again with the new count
        self.cfg.save(self.data_dir)
        return n

    def set_leaderboard(self, on: bool) -> bool:
        """POST /pool/me; the gateway's answer is the new state. Raises GatewayError (the caller shows it)."""
        if self.api is None or not self.connected:
            raise GatewayUnavailable("还没有连接网关")
        try:
            r = self.api.set_leaderboard(bool(on))
        except GatewayRefusal as e:
            self._refused(e.reason, e.fatal)
            raise
        with self.lock:
            self.leaderboard = bool(r["leaderboard"])
            self.stats = r["stats"]
            self.user = r["user"]
        self.log(f"上榜：{'开' if self.leaderboard else '关'}。")
        return self.leaderboard

    @property
    def running(self) -> bool:
        return self.want_running and self.connected and self.auto_pause is None

    # ------------------------------------------------------------------------------------------ service loop
    def start_service(self) -> None:
        """Start the service and heartbeat threads (once; later calls do nothing)."""
        if self._threads:
            return
        for fn, name in ((self._service, "service"), (self._heartbeat_loop, "heartbeat")):
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def _service(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                self.log(f"内部错误：{type(e).__name__}: {e}")
            self._stop.wait(0.25)

    def tick(self) -> None:
        self._drain_events()
        now = time.monotonic()
        if now - self._last_probe >= self.probe_s:
            self._last_probe = now
            self._probe()
        if self._reconnect_wanted and not self.token_dead and now >= self._next_reconnect:
            self._next_reconnect = now + RECONNECT_S
            self.log("重新连接网关……")
            self.connect()
        self._reconcile()

    def _probe(self) -> None:
        reason = None
        if self.auto_pause_enabled:
            try:
                if self.cfg.auto_pause_player and self.probes.get("player") and self.probes["player"]():
                    reason = PAUSE_PLAYER
                elif self.cfg.auto_pause_battery and self.probes.get("battery") and self.probes["battery"]():
                    reason = PAUSE_BATTERY
            except Exception:  # noqa: BLE001
                reason = None
        if reason != self.auto_pause:
            self.log(reason or "自动暂停的条件已解除。")
        self.auto_pause = reason

    def _crashed(self, slot: int, inst: Instance, now: float) -> None:
        """An instance exited without being asked to: wait before the slot starts again (2 s doubling to 5 min), and
        give up after CRASH_LIMIT exits in a row (a broken installation), with a visible message."""
        fails, _ = self._crash.get(slot, (0, 0.0))
        fails = 1 if now - inst.started >= STABLE_S else fails + 1
        if fails >= CRASH_LIMIT:
            self._crash[slot] = (fails, float("inf"))
            self.set_banner("error", message=f"实例 {slot + 1} 连续 {fails} 次意外退出，已停止重启。请查看主控程序数据目录下的 "
                                             f"logs/instance-{slot + 1:02d}.log；按“开始”可以再试。")
            return
        lo, hi = CRASH_BACKOFF_S
        wait = min(lo * 2 ** (fails - 1), hi)
        self._crash[slot] = (fails, now + wait)
        self.log(f"实例 {slot + 1} 意外退出，{wait:.0f} 秒后重新启动（连续第 {fails} 次）。")

    def _reconcile(self) -> None:
        with self.lock:
            running = self.running
            n = self.cfg.instances if self.connected else 0
        now = time.monotonic()
        for slot, inst in list(self.instances.items()):
            alive = inst.proc is not None and inst.proc.poll() is None
            if not alive:
                _close_pipes(inst.proc)
                if inst.log_file is not None:
                    inst.log_file.close()
                    inst.log_file = None
                del self.instances[slot]
                if not inst.stopping and slot < n:
                    self._crashed(slot, inst, now)
                continue
            if slot in self._crash and now - inst.started >= STABLE_S:
                del self._crash[slot]  # it has been up for a while: the failures are behind it
            if slot >= n and not inst.stopping:
                inst.stopping = True
                self._send(inst, {"cmd": "stop"})
        if running or self.instances:
            for slot in range(n):
                if slot not in self.instances and running and now >= self._crash.get(slot, (0, 0.0))[1]:
                    self._spawn(slot, self._crash.get(slot, (0, 0.0))[0])
        for inst in self.instances.values():
            if inst.stopping:
                continue
            want = "run" if running else "pause"
            if inst.sent != want:
                self._send(inst, {"cmd": want})
                inst.sent = want

    def _spawn(self, slot: int, restarts: int) -> None:
        exe, extra = python_command()
        env = dict(os.environ, **extra)
        env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["BALLRAIN_POOL_PARENT_PID"] = str(os.getpid())
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        logs = self.data_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        lf = open(logs / f"instance-{slot + 1:02d}.log", "ab")
        try:
            p = self.children.spawn(exe + ["-m", "ballrain.pool.client_worker"], env=env, cwd=str(self.data_dir),
                                    stderr=lf)
        except OSError as e:
            lf.close()
            self.log(f"实例 {slot + 1} 无法启动：{e}")
            return
        inst = Instance(slot=slot, proc=p, pid=p.pid, restarts=restarts, log_file=lf, started=time.monotonic())
        self.instances[slot] = inst
        cfg = {"cmd": "config", "slot": slot, "gateway": self.cfg.gateway_url, "token": self.token,
               "session_id": self.api.session_id if self.api else None, "edition": self.cfg.edition,
               "engine_kind": G.edition_engine(self.cfg.edition), "binary": self.engine_binary(),
               "root": str(Path(self.cfg.ballance_dir).resolve()) if self.cfg.ballance_dir else "",
               "data_dir": str(self.data_dir.resolve()), "simulate": self.simulate,
               "fake_game": self.fake_game}
        self._send(inst, cfg)
        threading.Thread(target=self._reader, args=(inst,), daemon=True, name=f"instance-{slot}").start()
        self.log(f"实例 {slot + 1} 已启动。")

    def _send(self, inst: Instance, msg: dict) -> None:
        p = inst.proc
        if p is None or p.stdin is None:
            return
        try:
            p.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            p.stdin.flush()
        except (OSError, ValueError):
            pass

    def _reader(self, inst: Instance) -> None:
        p = inst.proc
        try:
            for raw in p.stdout:
                try:
                    ev = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    continue
                if isinstance(ev, dict):
                    self.events.put((inst.slot, inst.pid, ev))
        except (OSError, ValueError):  # the pipe was closed under us (shutdown)
            pass
        self.events.put((inst.slot, inst.pid, {"ev": "exited"}))

    def _drain_events(self) -> None:
        while True:
            try:
                slot, pid, ev = self.events.get_nowait()
            except queue.Empty:
                return
            inst = self.instances.get(slot)
            if inst is None or inst.pid != pid:
                continue
            kind = ev.get("ev")
            if kind == "hello":
                inst.pid = ev.get("pid") or inst.pid
            elif kind == "state":
                inst.state = str(ev.get("state"))
                inst.detail = str(ev.get("detail") or "")
                inst.task_id = ev.get("task_id")
            elif kind == "engine":
                inst.engine_pid = ev.get("pid")
            elif kind == "progress":
                inst.task_id = ev.get("task_id") or inst.task_id
                inst.task_steps = int(ev.get("steps") or 0)
                inst.total_steps = int(ev.get("total_steps") or 0)
                inst.accepted_steps = int(ev.get("accepted_steps") or 0)
                inst.steps_per_s = float(ev.get("steps_per_s") or 0.0)
            elif kind == "chunk":
                inst.chunks += 1
                if ev.get("final"):
                    self.log(f"实例 {slot + 1}：任务 {str(ev.get('task_id'))[:8]} 已交完。")
            elif kind == "refusal":
                reason = ev.get("reason") or {}
                self._refused(reason, bool(ev.get("fatal")))
            elif kind == "log":
                self.log(f"实例 {slot + 1}：{ev.get('msg')}")
            elif kind == "exited":
                inst.state = "exited"

    # ------------------------------------------------------------------------------------------ heartbeat
    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            if self.connected and self.api is not None and now - self._last_hb >= self.heartbeat_s:
                self._last_hb = now
                self.heartbeat_once()
            if self.connected and self.api is not None and now - self._last_me >= 60.0:
                self._last_me = now
                self.refresh_me()
            self._stop.wait(0.5)

    def heartbeat_once(self) -> None:
        with self.lock:
            insts = [{"slot": i.slot, "state": i.state if i.state in ("idle", "starting", "running", "paused", "error")
                      else "error", "steps": int(i.total_steps), "task_id": i.task_id}
                     for i in sorted(self.instances.values(), key=lambda x: x.slot)]
            sps = sum(i.steps_per_s for i in self.instances.values())
            running = self.running
        try:
            r = self.api.heartbeat(running, insts, sps)
        except GatewayRefusal as e:
            self._refused(e.reason, e.fatal)
            return
        except GatewayError as e:
            self.log(f"心跳失败：{e}")
            return
        if r.get("revoked"):
            self._refused(r["revoked"], True)
        with self.lock:
            self.max_instances = max(0, min(MAX_INSTANCES, int(r["max_instances"])))
            if self.max_instances and self.cfg.instances > self.max_instances:
                self.cfg.instances = self.max_instances
            if r.get("notice") and r["notice"] != self.notice:
                self.notice = r["notice"]
                self.log(f"网关通知：{self.notice}")

    def refresh_me(self) -> None:
        try:
            r = self.api.me()
        except GatewayRefusal as e:
            self._refused(e.reason, e.fatal)
            return
        except GatewayError:
            return
        with self.lock:
            self.stats = r["stats"]
            self.leaderboard = bool(r["leaderboard"])
            self.user = r["user"]
        if r.get("token_state") == "revoked":
            self._refused(r.get("revoke_reason") or {"code": "token_revoked", "message": "你的算力池令牌已被吊销。",
                                                    "params": {}}, True)

    # ------------------------------------------------------------------------------------------ view / shutdown
    def snapshot(self) -> dict:
        with self.lock:
            insts = [i.view() for i in sorted(self.instances.values(), key=lambda x: x.slot)]
            return {
                "pid": os.getpid(), "connected": self.connected, "running": self.running,
                "want_running": self.want_running, "auto_pause": self.auto_pause, "banner": self.banner,
                "user": self.user, "leaderboard": self.leaderboard, "stats": self.stats,
                "instances_wanted": self.cfg.instances, "max_instances": self.max_instances, "instances": insts,
                "session_steps": sum(i["total_steps"] for i in insts),
                "session_accepted": sum(i["accepted_steps"] for i in insts),
                "steps_per_s": round(sum(i["steps_per_s"] for i in insts), 1),
                "notice": self.notice, "logs": list(self.logs)[-50:],
            }

    def shutdown(self, grace_s: float = 3.0) -> None:
        """Stop every instance and engine (and the service threads, and a pending login wait). Idempotent."""
        self._stop.set()
        for inst in list(self.instances.values()):
            inst.stopping = True
            self._send(inst, {"cmd": "stop"})
        self.children.shutdown(grace_s)
        for inst in list(self.instances.values()):
            _close_pipes(inst.proc)
            if inst.log_file is not None:
                try:
                    inst.log_file.close()
                except OSError:
                    pass
        self.instances.clear()

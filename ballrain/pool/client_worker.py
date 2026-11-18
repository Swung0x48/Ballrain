"""One sampling instance (a child process of the controller): get a task, get its weights and assets (sha-checked), run
the game with the numpy policy, upload the chunks. docs/pool-prototype.md section 9.

    python -m ballrain.pool.client_worker      (started by the controller only)

Talks to the controller in JSON lines: commands on stdin (the first line is the configuration, then "run" / "pause" /
"stop" / "token"), events on stdout ("hello", "state", "engine", "progress", "chunk", "refusal", "log"). Library
output goes to stderr (the controller's per-instance log file). The token arrives on stdin, never on the command line.

Lifecycle: the worker exits -- and takes its engine with it -- when the controller goes away: stdin EOF, the parent
pid changing / its handle signalled (client_lifecycle.ParentWatch); on Windows the controller's job object kills the
whole tree anyway, on Linux PR_SET_PDEATHSIG (set by the worker itself, client_lifecycle.bind_to_parent) delivers
SIGTERM, which kills our process group. Nothing is started
before the configuration line arrives.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

DEADLINE_MARGIN_S = 30
# how a task ended, for the player's log (protocol.END_REASONS; terms of docs/terminology.md)
END_TEXT = {"LevelFinish": "通关", "GameOver": "游戏结束，生命用完", "cap": "超过这条命的帧数上限", "stall": "停滞不前",
            "max_steps": "达到任务步数上限", "max_frames": "达到任务帧数上限", "deadline": "任务到期", "stopped": "已暂停",
            "error": "引擎出错", "truncated": "游戏进程结束"}


class Worker:
    def __init__(self):
        self._out_lock = threading.Lock()
        self.cfg: dict = {}
        self.running = threading.Event()  # the controller wants this instance to sample
        self.wake = threading.Event()  # any command arrived
        self.stopping = False
        self.state = "idle"
        self.total_steps = 0
        self.accepted_steps = 0
        self.task_id = None
        self.engine_pid = None
        self._sim_child = None
        self.api = None
        self.fatal = False

    # ------------------------------------------------------------------------------------------ I/O
    def emit(self, ev: str, **kw):
        line = json.dumps(dict(kw, ev=ev), ensure_ascii=False)
        with self._out_lock:
            try:
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            except (OSError, ValueError):
                self.gone("stdout closed")

    def set_state(self, state: str, detail: str = ""):
        self.state = state
        self.emit("state", state=state, task_id=self.task_id, detail=detail[:300])

    def log(self, msg: str):
        self.emit("log", msg=str(msg)[:500])

    def gone(self, why: str):
        """The controller is gone: end now, with our engines."""
        try:
            sys.stderr.write(f"controller gone ({why}); exiting\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        self._kill_sim_child()
        from .client_lifecycle import kill_own_group
        kill_own_group()

    def read_commands(self, stream):
        for line in stream:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            cmd = msg.get("cmd")
            if cmd == "run":
                self.running.set()
            elif cmd == "pause":
                self.running.clear()
            elif cmd == "stop":
                self.stopping = True
                self.running.clear()
            elif cmd == "token":
                self.cfg["token"] = msg.get("token")
                self.cfg["session_id"] = msg.get("session_id")
                if self.api is not None:
                    self.api.token, self.api.session_id = self.cfg["token"], self.cfg["session_id"]
                self.fatal = False
            self.wake.set()
        self.gone("stdin closed")

    def wait(self, seconds: float) -> None:
        """Sleep until a command arrives or the time is up."""
        self.wake.clear()
        self.wake.wait(seconds)

    # ------------------------------------------------------------------------------------------ main
    def main(self) -> int:
        from .client_lifecycle import bind_to_parent
        bind_to_parent()  # POSIX: die with the controller (PR_SET_PDEATHSIG, then the race-free parent check)
        line = _read_line_unbuffered(0)
        if not line:
            return 0
        self.cfg = json.loads(line)
        from .client_lifecycle import ParentWatch, lower_priority
        lower_priority()
        ParentWatch(self.gone)
        threading.Thread(target=self.read_commands, args=(_take_stdin(),), daemon=True, name="commands").start()
        self.emit("hello", pid=os.getpid(), slot=self.cfg.get("slot"))
        if self.cfg.get("simulate"):
            return self.simulate()
        from .client_api import Cache, GatewayClient
        self.api = GatewayClient(self.cfg["gateway"], self.cfg.get("token"), self.cfg.get("session_id"))
        self.cache = Cache(Path(self.cfg["data_dir"]) / "cache", self.api)
        self.loop()
        return 0

    def loop(self):
        from .client_api import GatewayRefusal, GatewayUnavailable
        backoff = 5.0
        fails = 0  # tasks in a row that failed before sampling (downloads, sha checks, broken assets)
        while not self.stopping:
            if not self.running.is_set() or self.fatal:
                if self.state not in ("paused", "error"):
                    self.set_state("paused")
                self.wait(1.0)
                continue
            try:
                msg = self.api.task()
            except GatewayRefusal as e:
                self.refusal(e, "task")
                self.wait(30.0)
                continue
            except GatewayUnavailable as e:
                self.set_state("error", str(e))
                self.wait(backoff)
                backoff = min(backoff * 2, 120.0)
                continue
            backoff = 5.0
            if msg["type"] == "no_task":
                self.task_id = None
                self.set_state("idle", msg["reason"]["message"])
                self.wait(float(msg["retry_s"]))
                continue
            try:
                self.run_task(msg["task"])
                fails = 0
            except GatewayRefusal as e:
                self.refusal(e, "task")
            except GatewayUnavailable as e:
                fails += 1
                self.set_state("error", str(e))
                self.wait(min(5.0 * 2 ** min(fails - 1, 6), 300.0))
            except Exception as e:  # noqa: BLE001  (a broken task must not end the instance)
                fails += 1
                traceback.print_exc()
                self.set_state("error", f"任务出错：{type(e).__name__}: {e}")
                self.log(f"任务出错：{type(e).__name__}: {str(e)[:300]}")
                self.wait(min(10.0 * 2 ** min(fails - 1, 5), 300.0))
            finally:
                self.task_id = None
                self.engine_pid = None
        self.set_state("idle", "stopped")

    def refusal(self, e, where: str):
        if e.fatal:
            self.fatal = True
            self.running.clear()
        self.emit("refusal", reason=e.reason, fatal=bool(e.fatal), where=where)
        self.set_state("error", e.message)

    def run_task(self, task_dict: dict):
        from .client_sampler import PolicyActions, TaskAborted, TaskSampler, Uploader
        from .task import LocalGame, Task, build_env
        task = Task.from_dict(task_dict)
        self.task_id = task.task_id
        t_recv = time.monotonic()
        if task.engine != self.cfg["engine_kind"]:
            self.log(f"任务 {task.task_id[:8]} 要求 {task.engine} 引擎，本实例运行的是 {self.cfg['engine_kind']} 引擎，已跳过。")
            self.wait(10.0)
            return
        self.set_state("starting", "下载策略权重和开局数据")
        policy, manifest = self.cache.policy(task.policy_sha)
        sdir = self.cache.starts(task.assets["starts"])
        pdir = self.cache.path(task.assets["path"], task.level)
        f, bsha = manifest["features"], task.assets.get("branch")
        bdir = self.cache.branch(bsha, task.level, f["branch_sha256"], f.get("path_sha256")) \
            if f.get("branch_sha256") and bsha else None
        slot = int(self.cfg.get("slot", 0))
        copy = f"pool-{slot:02d}"
        out = Path(self.cfg["data_dir"]) / "runs"
        local = LocalGame(engine=self.cfg["engine_kind"], binary=self.cfg.get("binary", ""), root=self.cfg.get("root", ""))
        launcher = None
        if self.cfg.get("fake_game"):  # tests: the in-process fake game (ballrain.fakegame)
            from ballrain.fakegame import FakeLauncher
            launcher = FakeLauncher(backend="headless")
        env = build_env(task, manifest, local, sdir, pdir, out, copy=copy, launcher=launcher, branch_dir=bdir)
        last = [time.monotonic(), 0]

        def on_result(res, header):
            self.emit("chunk", task_id=header["task_id"], seq=header["seq"], steps=int(res.get("steps", 0)),
                      status=res.get("status"), verify=res.get("verify"), final=bool(header["final"]))
            if res.get("status") != "rejected":
                self.accepted_steps += int(res.get("steps", 0))

        up = Uploader(self.api.chunk, on_result=on_result)
        engine_pid = [None]
        t_start = time.time()
        deadline = t_recv + task.deadline_s - min(DEADLINE_MARGIN_S, task.deadline_s // 4)

        def should_stop():
            if up.refusal is not None:
                raise TaskAborted("refused", up.refusal)
            if up.error is not None:
                raise TaskAborted(up.error)
            if self.stopping or not self.running.is_set():
                return "stopped"
            if time.monotonic() >= deadline:
                return "deadline"
            return None

        def on_start(e):
            h = getattr(getattr(e, "session", None), "handle", None)
            self.engine_pid = engine_pid[0] = getattr(h, "pid", None)
            self.emit("engine", pid=self.engine_pid)
            self.set_state("running", f"第 {task.level} 关，开局第 {task.start_frame} 帧")

        def on_step(n):
            self.total_steps += 1
            now = time.monotonic()
            if now - last[0] >= 1.0:
                self.emit("progress", task_id=task.task_id, steps=n, total_steps=self.total_steps,
                          accepted_steps=self.accepted_steps, steps_per_s=round((self.total_steps - last[1]) / (now - last[0]), 1))
                last[0], last[1] = now, self.total_steps

        try:
            sampler = TaskSampler(env, manifest["rollout"])
            summary = sampler.run_task(task, PolicyActions.for_task(policy, task), up.submit, should_stop,
                                       on_start=on_start, on_step=on_step, n_act=int(manifest["arch"]["n_act"]))
            self.log(f"任务 {task.task_id[:8]} 结束（{END_TEXT.get(summary['end'], summary['end'])}）：{summary['steps']} 步，"
                     f"{summary['chunks']} 段，{summary['episodes']} 条命")
        except TaskAborted as e:
            if e.refusal is not None:
                self.refusal(e.refusal, "chunk")
            else:
                self.log(f"任务 {task.task_id[:8]} 已放弃：{e}")
        finally:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
            self.engine_pid = None
            self.emit("engine", pid=None)
            self.collect_crash_dumps(engine_pid[0], t_start)
        drained = up.close()
        if up.refusal is not None:
            self.refusal(up.refusal, "chunk")
        elif not drained:
            self.log(f"任务 {task.task_id[:8]} 有数据段没有交上（{up.error or '超时'}）。")
        self.emit("progress", task_id=task.task_id, steps=0, total_steps=self.total_steps,
                  accepted_steps=self.accepted_steps, steps_per_s=0.0)
        shutil.rmtree(out / copy, ignore_errors=True)

    def collect_crash_dumps(self, engine_pid, since: float, keep: int = 5) -> None:
        """The engine writes crash-<pid>.dmp into its working directory, <Ballance>/Bin (the player's folder): move our
        engine's dump (and any other crash-*.dmp written there since this task started: only our engines write that
        name) into <data_dir>/logs, keeping the newest `keep`. Nothing of ours stays in the player's folder."""
        root = self.cfg.get("root") or ""
        if not root:
            return
        bin_dir = Path(root) / "Bin"
        try:
            found = [p for p in bin_dir.glob("crash-*.dmp")
                     if (engine_pid is not None and p.name == f"crash-{engine_pid}.dmp") or p.stat().st_mtime >= since]
        except OSError:
            return
        if not found:
            return
        logs = Path(self.cfg["data_dir"]) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for p in found:
            try:
                shutil.move(str(p), str(logs / p.name))
                self.log(f"引擎崩溃转储 {p.name} 已移到主控程序的 logs 目录。")
            except OSError:
                try:
                    p.unlink()
                except OSError:
                    pass
        dumps = sorted(logs.glob("crash-*.dmp"), key=lambda q: q.stat().st_mtime, reverse=True)
        for q in dumps[keep:]:
            try:
                q.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------------------------------ simulate (tests)
    def _kill_sim_child(self):
        p = self._sim_child
        if p is not None and p.poll() is None:
            p.kill()
        self._sim_child = None

    def simulate(self) -> int:
        """No game: a stand-in 'engine' child (a sleeping interpreter) while running, none while paused. For the
        lifecycle tests: the controller -> worker -> engine tree must die with the controller."""
        from .client_lifecycle import CREATE_NO_WINDOW, IS_WIN, python_command
        exe, extra = python_command()
        while not self.stopping:
            if self.running.is_set():
                if self._sim_child is None or self._sim_child.poll() is not None:
                    kw = {"creationflags": CREATE_NO_WINDOW} if IS_WIN else {}
                    self._sim_child = subprocess.Popen(
                        exe + ["-c", "import time\nwhile True: time.sleep(1)"], env=dict(os.environ, **extra),
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
                    self.engine_pid = self._sim_child.pid
                    self.task_id = "simulated"
                    self.emit("engine", pid=self.engine_pid)
                    self.set_state("running", "模拟实例")
                self.total_steps += 10
                self.emit("progress", task_id=self.task_id, steps=self.total_steps, total_steps=self.total_steps,
                          accepted_steps=self.accepted_steps, steps_per_s=10.0)
            else:
                if self._sim_child is not None:
                    self._kill_sim_child()
                    self.engine_pid = None
                    self.task_id = None
                    self.emit("engine", pid=None)
                if self.state != "paused":
                    self.set_state("paused")
            self.wait(1.0)
        self._kill_sim_child()
        self.set_state("idle", "stopped")
        return 0


def _read_line_unbuffered(fd: int) -> bytes:
    """One line from a file descriptor without read-ahead (the rest stays in the pipe for the command reader)."""
    out = bytearray()
    while len(out) < 1 << 20:
        b = os.read(fd, 1)
        if not b or b == b"\n":
            break
        out += b
    return bytes(out)


def _take_stdin():
    """The command pipe, moved off the standard input handle. On Windows a thread blocked in a synchronous read of a
    pipe blocks every other call on that file object -- including the GetFileType(stdin) that a C runtime runs while a
    DLL loads (numpy, the env's imports): the worker would hang in its first import. So the pipe is read through a
    duplicate handle, and the standard input becomes NUL."""
    fd = os.dup(0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        k = ctypes.WinDLL("kernel32")
        k.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
        k.SetStdHandle(ctypes.c_ulong(-10 & 0xFFFFFFFF), ctypes.c_void_p(msvcrt.get_osfhandle(0)))  # STD_INPUT_HANDLE
    sys.stdin = open(os.devnull, encoding="utf-8")
    return os.fdopen(fd, "rb")


def main() -> int:
    return Worker().main()


if __name__ == "__main__":
    raise SystemExit(main())

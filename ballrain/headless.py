"""Headless backend: the x87-soft headless engine (BallanceMMO SimTool with headless/src/ballrain_agent.cpp) as
an agent-mode game process, over the socket transport (docs/headless.md).

APPROXIMATE environment: it is not bit-exact with the 2002 retail game on every route (bit-identical to the end
on some TAS records, diverging on others; docs/headless.md). Bulk training only; retail Player.exe stays the
authority for evaluation and .tas export. Every run it produces carries backend = "headless" in its metadata
(ballrain.recording) and every observation the OBS_HEADLESS flag.

    from ballrain.env import BallanceEnv, EnvConfig
    env = BallanceEnv(EnvConfig(level=1, backend="headless", copy="hl-00"))   # HeadlessConfig from the env

HeadlessConfig fields default from the environment: BALLRAIN_HL_BIN (the SimTool binary), BALLRAIN_HL_ROOT (a
Ballance game folder), BALLRAIN_HL_SOCKET (unix | tcp). One process per run, as for the retail game; the
process exits by itself when the run ends or the socket closes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .transport import SocketChannel

BACKEND = "headless"


def _env_path(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class HeadlessConfig:
    binary: str = field(default_factory=lambda: _env_path("BALLRAIN_HL_BIN"))
    root: str = field(default_factory=lambda: _env_path("BALLRAIN_HL_ROOT"))
    socket: str = field(default_factory=lambda: _env_path("BALLRAIN_HL_SOCKET"))  # "" = unix where available, else tcp
    host: str = "127.0.0.1"
    port: int = 0  # tcp: 0 = any free port (DEFAULT_TCP_PORT 27787 + i for fixed ports)
    boot_ticks: int | None = None  # engine ticks before the level is requested (default 400)
    hidden_ticks: int | None = None  # fixed ticks between "Load Level" and frame 0 (default 34)
    legacy_edit: bool = True  # TASSupport legacy mode: delete Set Position in Ball_Explosion_* (as BallrainIO)
    reset_nmc: int | None = None  # next_movement_check at the clock reset (default 4, Player.exe's)
    reset_seed: int | None = None  # ivp seed at the clock reset (default 3560118427 = 75^21, Player.exe's)
    log: bool = True  # the engine's stderr into <run dir>/headless.log
    extra_env: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class HeadlessHandle:
    def __init__(self, proc: subprocess.Popen, log_file=None):
        self.proc = proc
        self.pid = proc.pid
        self.wait_handle = None
        self._log = log_file

    def wait(self, timeout_s: float) -> bool:
        try:
            self.proc.wait(timeout=max(0.0, timeout_s))
            return True
        except subprocess.TimeoutExpired:
            return False

    def exit_code(self):
        return self.proc.poll()

    def terminate(self, code: int = 1):
        if self.proc.poll() is None:
            self.proc.kill()

    def close(self):
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None


class HeadlessLauncher:
    """Drop-in for client.GameLauncher: starts one headless engine process per run."""

    backend = BACKEND

    def __init__(self, config: HeadlessConfig | None = None, **overrides):
        self.cfg = config or HeadlessConfig()
        for k, v in overrides.items():
            setattr(self.cfg, k, v)

    def check(self) -> None:
        c = self.cfg
        if not c.binary or not Path(c.binary).is_file():
            raise FileNotFoundError(f"headless binary not found: {c.binary!r} (HeadlessConfig.binary / BALLRAIN_HL_BIN)")
        if not c.root or not (Path(c.root) / "base.cmo").is_file():
            raise FileNotFoundError(f"no base.cmo in the game root {c.root!r} (HeadlessConfig.root / BALLRAIN_HL_ROOT)")

    def make_channel(self, session_cfg) -> SocketChannel:
        kind = self.cfg.socket or None
        return SocketChannel(kind, host=self.cfg.host, port=self.cfg.port)

    def environment(self, env: dict) -> dict:
        c = self.cfg
        out = {k: v for k, v in os.environ.items() if not k.upper().startswith("BALLRAIN_")}
        for k, v in env.items():
            if v is not None:
                out[k] = str(v)
        for name, v in (("BALLRAIN_HL_BOOT_TICKS", c.boot_ticks), ("BALLRAIN_HL_HIDDEN_TICKS", c.hidden_ticks),
                        ("BALLRAIN_HL_NMC", c.reset_nmc), ("BALLRAIN_HL_SEED", c.reset_seed)):
            if v is not None:
                out[name] = str(int(v))
        out["BALLRAIN_HL_LEGACY"] = "1" if c.legacy_edit else "0"
        for k, v in c.extra_env.items():
            out[k] = str(v)
        return out

    def start(self, env: dict) -> HeadlessHandle:
        self.check()
        log = None
        stderr = subprocess.DEVNULL
        if self.cfg.log and env.get("BALLRAIN_FRAMELOG"):
            log = open(Path(env["BALLRAIN_FRAMELOG"]).parent / "headless.log", "wb")
            stderr = log
        args = [self.cfg.binary, "--ballrain-agent", "--root", str(Path(self.cfg.root).resolve())]
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kw["start_new_session"] = False
        proc = subprocess.Popen(args, env=self.environment(env), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=stderr, **kw)
        return HeadlessHandle(proc, log)  # not kept here: one launcher starts thousands of processes per hour

    @staticmethod
    def stop(handle: HeadlessHandle, wait_s: float = 10.0):
        if not handle.wait(wait_s):
            handle.terminate()
            handle.wait(5.0)
        handle.close()
        return handle.exit_code()


def headless_copy_names(n: int, prefix: str = "hl") -> list[str]:
    """Instance names for n headless sub-envs (the run directories live under <out_root>/<name>/)."""
    return [f"{prefix}-{i:02d}" for i in range(n)]


def wait_all(handles, timeout_s: float = 10.0) -> None:
    t = time.monotonic() + timeout_s
    for h in handles:
        h.wait(max(0.0, t - time.monotonic()))

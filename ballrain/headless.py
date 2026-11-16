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

The frame recipe is SimTool's --tas one (BallanceMMO x87-soft): its flags, whose defaults are the verified Player.exe
state; HeadlessConfig passes a flag only when a field differs from that default. The ball-pair contact order
(HeadlessConfig.ball_order, BALLRAIN_HL_BALL_ORDER) is the one setting retail itself does not fix: see
parse_ball_order and docs/headless.md.
"""
from __future__ import annotations

import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .transport import SocketChannel

BACKEND = "headless"


def _env_path(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------------------------- ball-pair order
# Retail orders the two balls of a ball-ball contact by their CK3dEntity heap addresses, which differ per Player.exe
# process (on L1 the pooled P_Ball_Stone_MF001-004: their contact at the level load decides the run). The headless
# engine orders by a key (smaller = first), BALLRAIN_HL_BALL_ORDER (headless/src/ballrain_agent.cpp):
#   ckid                   the product's CK-id key (default; the gate's recipe)
#   reverse                the CK ids reversed
#   perm:<seed>            a seeded permutation of the pooled balls (P_Ball_*_MF*) over their own CK-id places
#   table:<name=rank,...>  the named entities by rank (distinct), every other ball after them by CK id
#   retail:<seed>          a draw of retail's measured order (L1 only, retail_draw_l1; other levels: ckid)
# "perm" / "retail" without a seed: the launcher draws a fresh seed for every process (recorded in the run metadata).
BALL_ORDER_ENV = "BALLRAIN_HL_BALL_ORDER"
BALL_ORDER_MODES = ("ckid", "reverse", "perm", "table", "retail")
_U64 = re.compile(r"[0-9]{1,20}")
_RANK_LIMIT = 1 << 20


def parse_ball_order(spec: str | None, *, allow_seedless: bool = False) -> str:
    """Validates a ball-order spec the way the engine does; returns it normalized ("" / None -> "ckid").
    allow_seedless accepts "perm" / "retail" (the launcher seeds them per process)."""
    s = (spec or "ckid").strip()
    mode, colon, arg = s.partition(":")
    if mode not in BALL_ORDER_MODES:
        raise ValueError(f"ball order {s!r}: unknown mode {mode!r} (one of {', '.join(BALL_ORDER_MODES)})")
    if mode in ("ckid", "reverse"):
        if colon:
            raise ValueError(f"ball order {s!r}: {mode} takes no argument")
        return mode
    if mode in ("perm", "retail"):
        if not colon and allow_seedless:
            return mode
        if not _U64.fullmatch(arg) or int(arg) >= 1 << 64:
            raise ValueError(f"ball order {s!r}: {mode}:<seed> needs a decimal seed (0 .. 2^64-1)")
        return f"{mode}:{int(arg)}"
    items = arg.split(",") if colon else [""]
    names, ranks, out = set(), set(), []
    for item in items:
        name, eq, rank = item.rpartition("=")
        if not eq or not name or not _U64.fullmatch(rank) or int(rank) >= _RANK_LIMIT:
            raise ValueError(f"ball order {s!r}: table entry {item!r} is not <name>=<rank> (rank 0 .. 2^20-1)")
        if name in names or int(rank) in ranks:
            raise ValueError(f"ball order {s!r}: table entry {item!r} repeats a name or a rank")
        names.add(name)
        ranks.add(int(rank))
        out.append(f"{name}={int(rank)}")
    return "table:" + ",".join(out)


def resolve_ball_order(spec: str | None, rng: random.Random | None = None) -> str:
    """The spec one process gets: a seedless "perm" / "retail" gets a fresh 32-bit seed."""
    s = parse_ball_order(spec, allow_seedless=True)
    if s in ("perm", "retail"):
        s = f"{s}:{(rng or random.SystemRandom()).getrandbits(32)}"
    return s


def effective_ball_order(spec: str, level) -> str:
    """What the engine runs for a resolved spec on a level: retail:<seed> is measured on level 1 only (other levels
    run ckid, the engine logs a warning)."""
    return "ckid" if spec.startswith("retail:") and str(level) != "1" else spec


_M64 = (1 << 64) - 1


def _splitmix64(state: int):
    state = (state + 0x9E3779B97F4A7C15) & _M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return state, z ^ (z >> 31)


def _below(state: int, n: int):
    limit = _M64 - (_M64 % n)
    while True:
        state, x = _splitmix64(state)
        if x < limit:
            return state, x % n


def _shuffle(items: list, state: int):
    v = list(items)
    for i in range(len(v), 1, -1):
        state, j = _below(state, i)
        v[i - 1], v[j] = v[j], v[i - 1]
    return v, state


def retail_draw_l1(seed: int) -> list[str]:
    """retail:<seed> on L1, lowest address first (the engine's RetailDrawL1): Ball_Wood; P_Ball_Stone_MF before
    P_Ball_Wood_MF with probability 0.66; a uniform permutation of P_Ball_Stone_MF001-004 (50 fresh Player.exe
    processes, runtime/analysis/ball_order/summary.json). Every other ball comes after these, by CK id."""
    state, coin = _below(int(seed) & _M64, 100)
    order = ["Ball_Wood"] + (["P_Ball_Stone_MF", "P_Ball_Wood_MF"] if coin < 66 else ["P_Ball_Wood_MF", "P_Ball_Stone_MF"])
    mf, _ = _shuffle([f"P_Ball_Stone_MF00{i}" for i in range(1, 5)], state)
    return order + mf


def perm_draw(seed: int, pooled_by_ckid: list) -> list:
    """perm:<seed>: the order the pooled balls (given in CK-id order, as the engine finds them) take their places in."""
    v, _ = _shuffle(list(pooled_by_ckid), int(seed) & _M64)
    return v


@dataclass
class HeadlessConfig:
    binary: str = field(default_factory=lambda: _env_path("BALLRAIN_HL_BIN"))
    root: str = field(default_factory=lambda: _env_path("BALLRAIN_HL_ROOT"))
    socket: str = field(default_factory=lambda: _env_path("BALLRAIN_HL_SOCKET"))  # "" = unix where available, else tcp
    host: str = "127.0.0.1"
    port: int = 0  # tcp: 0 = any free port (DEFAULT_TCP_PORT 27787 + i for fixed ports)
    # SimTool's --tas recipe; None / the default = SimTool's own default (the verified Player.exe state)
    boot_ticks: int | None = None  # --boot-ticks: engine ticks before the level is requested (400)
    hidden_ticks: int | None = None  # --tas-offset: fixed ticks between "Load Level" and frame 0 (34)
    legacy_edit: bool = True  # False: --tas-no-legacy (TASSupport legacy mode deletes Set Position in Ball_Explosion_*)
    reset_nmc: int | None = None  # --tas-movement-check: next_movement_check at the clock reset (4)
    reset_seed: int | None = None  # --tas-seed: ivp seed at the clock reset (3560118427 = 75^21)
    reset_frame: int | None = None  # --tas-reset-frame: the clock reset before this frame (1)
    reset_precision: bool = False  # True: --tas-reset-precision (BMMO's per-level x87 precision reset)
    orig_random: bool = False  # True: --tas-orig-random (the engine's own Random blocks)
    ball_order: str = "ckid"  # BALLRAIN_HL_BALL_ORDER (parse_ball_order); extra_env's value wins
    log: bool = True  # the engine's stderr into <run dir>/headless.log
    extra_env: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class HeadlessHandle:
    def __init__(self, proc: subprocess.Popen, log_file=None, ball_order: str | None = None):
        self.proc = proc
        self.pid = proc.pid
        self.wait_handle = None
        self.ball_order = ball_order  # the order this process runs (RunMeta.ball_order; effective_ball_order)
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

    def arguments(self) -> list[str]:
        """The engine's command line: SimTool's flags only where the config differs from their defaults."""
        c = self.cfg
        args = [c.binary, "--ballrain-agent", "--root", str(Path(c.root).resolve())]
        for flag, v in (("--boot-ticks", c.boot_ticks), ("--tas-offset", c.hidden_ticks),
                        ("--tas-movement-check", c.reset_nmc), ("--tas-seed", c.reset_seed),
                        ("--tas-reset-frame", c.reset_frame)):
            if v is not None:
                args += [flag, str(int(v))]
        for flag, on in (("--tas-no-legacy", not c.legacy_edit), ("--tas-reset-precision", c.reset_precision),
                         ("--tas-orig-random", c.orig_random)):
            if on:
                args.append(flag)
        return args

    def environment(self, env: dict, rng: random.Random | None = None) -> dict:
        """The engine's environment; BALLRAIN_HL_BALL_ORDER resolved for this process (a seedless perm / retail
        drawn here)."""
        c = self.cfg
        out = {k: v for k, v in os.environ.items() if not k.upper().startswith("BALLRAIN_")}
        for k, v in env.items():
            if v is not None:
                out[k] = str(v)
        for k, v in c.extra_env.items():
            out[k] = str(v)
        out[BALL_ORDER_ENV] = resolve_ball_order(out.get(BALL_ORDER_ENV, c.ball_order), rng)
        return out

    def start(self, env: dict) -> HeadlessHandle:
        self.check()
        log = None
        stderr = subprocess.DEVNULL
        if self.cfg.log and env.get("BALLRAIN_FRAMELOG"):
            log = open(Path(env["BALLRAIN_FRAMELOG"]).parent / "headless.log", "wb")
            stderr = log
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kw["start_new_session"] = False
        penv = self.environment(env)
        proc = subprocess.Popen(self.arguments(), env=penv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=stderr, **kw)
        # not kept here: one launcher starts thousands of processes per hour
        return HeadlessHandle(proc, log, ball_order=effective_ball_order(penv[BALL_ORDER_ENV], penv.get("BALLRAIN_LEVEL")))

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

"""Engine side of the segment search: one rollout = one headless process. The state at gate A (and at any branch point
after it) is restored by the engine's own prefix replay (BALLRAIN_PREFIX: TAS[:A] + the branch's keys, replayed without
observations, ~13k frames/s); only the new tail is stepped with observations. The engine has no save / restore of a
state over the agent protocol, so this replay is the price of every branch (docs/segsearch.md)."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import tas
from ..client import AgentSession, SessionConfig
from ..headless import HeadlessLauncher
from ..protocol import Ev
from .core import crossing

END_EVENTS = {int(Ev.BallOff): "ball_off", int(Ev.Dead): "ball_off", int(Ev.GameOver): "game_over",
              int(Ev.LevelFinish): "level_finish"}
PICKUPS = {int(Ev.ExtraPoint): "ExtraPoint", int(Ev.PreLifeUp): "PreLifeUp"}


@dataclass
class EngineSpec:
    binary: str
    root: str
    ball_order: str = "ckid"  # pinned: a seedless "retail" draws a new order per process (runs would not repeat)
    level: int = 1


def keys_to_frames(keys, delta) -> np.ndarray:
    keys = np.asarray(keys, np.uint32)
    return tas.make_frames(np.full(len(keys), np.float32(delta), np.float32), keys)


def rollout(eng: EngineSpec, work: Path, copy: str, tas_prefix: np.ndarray, base_keys, cont_keys, *, delta: float,
            block: int = 4, gates=(), start_gate: int = 0, best=None, slack: float = 1e9, max_gap: float = 1e9,
            y_floor: float = -1e9, stop_at_last_gate: bool = True, record_sr: bool = False) -> dict:
    """Restore TAS[:A] + base_keys by prefix replay, then step cont_keys in blocks of `block` frames with an observation
    after each. Times are frames after A. Stops at: a ball-off / game over / level finish, a position below y_floor, the
    last gate crossed (stop_at_last_gate), a gate crossed later than best[gate] + slack ("behind"), no gate crossed for
    max_gap frames ("stuck"), or the end of cont_keys ("keys_end")."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    base = np.asarray(base_keys, np.uint32)
    cont = np.asarray(cont_keys, np.uint32)
    A = len(tas_prefix)
    pf = work / "prefix.bin"
    tas.save_raw(pf, np.concatenate([tas_prefix, keys_to_frames(base, delta)]))
    launcher = HeadlessLauncher(binary=eng.binary, root=eng.root, socket="tcp", log=False, ball_order=eng.ball_order)
    s = AgentSession(SessionConfig(eng.level, work / "run", copy=copy, prefix=pf, hello_timeout_s=90, obs_timeout_s=90,
                                   events_txt=False), launcher)
    t0 = time.perf_counter()
    T, P, V, TY, SR, EV = [], [], [], [], [], []
    end = "keys_end"
    cross = {}
    pid = None
    try:
        o = s.start()
        pid = s.handle.pid if s.handle is not None else None
        t_start = time.perf_counter() - t0
        if s.frames_done != A + len(base):
            raise RuntimeError(f"prefix replay stopped at frame {s.frames_done}, expected {A + len(base)}")

        def rec(o):
            b = o.ball
            T.append(s.frames_done - A)
            P.append(np.asarray(b["pos"], float).copy())
            V.append((np.asarray(b["speed"], float) + np.asarray(b["speed_change"], float)).copy())
            TY.append(int(b["type"]))
            if record_sr:
                SR.append(float(o.time["sr_double"]) if o.time is not None else float("nan"))
            for c in o.events["code"]:
                EV.append((s.frames_done - A, int(c)))

        rec(o)
        frames = keys_to_frames(cont, delta)
        g = start_gate
        last_cross = T[0]
        k = 0
        while k < len(frames):
            o = s.act(frames[k:k + block])
            k = s.frames_done - A - len(base)
            if o is None or o.ball is None:
                end = "engine_end"
                break
            rec(o)
            ev = [END_EVENTS[int(c)] for c in o.events["code"] if int(c) in END_EVENTS]
            if ev:
                end = ev[0]
                break
            if P[-1][1] < y_floor:
                end = "fell"
                break
            while g < len(gates):
                c = crossing(gates[g], T[-2:], P[-2:])
                if c is None:
                    break
                cross[g] = c[0]
                last_cross = c[0]
                g += 1
            if g >= len(gates) and stop_at_last_gate and gates:
                end = "last_gate"
                break
            if best is not None and g < len(gates) and T[-1] > best.get(g, 1e18) + slack:
                end = "behind"
                break
            if T[-1] - last_cross > max_gap:
                end = "stuck"
                break
            if s.ended:
                end = "engine_end"
                break
        wall = time.perf_counter() - t0
    finally:
        s.close(kill=True)
    used = len(T) and (T[-1] - len(base))
    out = {"t": np.asarray(T, np.int64), "pos": np.asarray(P, float), "vel": np.asarray(V, float),
           "type": np.asarray(TY, np.int16), "events": EV, "end": end, "cross": cross,
           "keys": np.r_[base, cont[:max(int(used), 0)]].astype(np.uint32), "base_len": len(base),
           "wall": wall, "wall_start": t_start, "pid": pid}
    if record_sr:
        out["sr"] = np.asarray(SR, float)
    return out


def pickup_counts(events, upto: float) -> dict:
    """{ExtraPoint: n, PreLifeUp: n} among (time, code) events at times <= upto."""
    out = {v: 0 for v in PICKUPS.values()}
    for t, c in events:
        if c in PICKUPS and t <= upto:
            out[PICKUPS[c]] += 1
    return out


def ball_off(events, upto: float) -> bool:
    return any(c in (int(Ev.BallOff), int(Ev.Dead)) and t <= upto for t, c in events)


def default_engine() -> EngineSpec:
    repo = Path(__file__).resolve().parents[2]
    return EngineSpec(os.environ.get("BALLRAIN_SEG_BIN", str(repo / "runtime/hl-win/bin-x86p2002f-pressure/BallanceMMOSimTool.exe")),
                      os.environ.get("BALLRAIN_SEG_ROOT", str(repo / "runtime/hl-win/game")))

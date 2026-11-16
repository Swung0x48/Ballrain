"""Episode recording: every game process (one run) writes its frames as a legacy .tas plus metadata.

The mod's BALLRAIN_FRAMELOG is authoritative (it is what was injected). Python keeps its own log of
the frames it sent, truncated to what the mod reports as executed (stop-on-event voids the rest of a
block); at the end both must be byte-identical, otherwise the run is marked framelog_match=false and
the mod's log is exported (nothing is silently reconciled).

SR per BML+ 0.3.12 (docs/benchmarks.md): StartLevel resets; CounterActive / UnpauseLevel start;
CounterInactive / PauseLevel pause; while running every frame adds (double) of its f32 delta in
PostProcess, i.e. after that frame's events. Official value = int64 truncation of the double sum.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import protocol as P
from . import tas
from .protocol import Ev

META_VERSION = 3  # 3: start_state (TAS start states), timing; 2: hs = official end-screen readout


def sr_timer(frames, events) -> dict:
    """Recompute the BML+ 0.3.12 SR timer from frames (FRAME_DTYPE) and events [(number, code)].
    Event number n belongs to frame index n-1. Returns dict(double, float32, ms, running)."""
    frames = tas.as_frames(frames)
    by_frame: dict = {}
    for n, code, *_ in events:
        by_frame.setdefault(int(n) - 1, []).append(int(code))
    acc = 0.0
    running = False
    deltas = frames["delta"]
    for i in range(len(frames)):
        for code in by_frame.get(i, ()):
            if code == Ev.StartLevel:
                acc, running = 0.0, False
            elif code in (Ev.CounterActive, Ev.UnpauseLevel):
                running = True
            elif code in (Ev.CounterInactive, Ev.PauseLevel):
                running = False
        if running:
            acc += float(deltas[i])
    return {"double": acc, "float32": float(np.float32(acc)), "ms": int(math.trunc(acc)), "running": running}


@dataclass
class LifeRecord:
    index: int
    start_frame: int  # frames_done at the start of control (first action of this life)
    end_frame: int | None = None  # frames_done at the end (the observation that ended the life)
    end_reason: str | None = None  # BallOff, LevelFinish, GameOver, truncated, ...
    sr_ms_end: float | None = None
    points_end: int | None = None
    lives_end: int | None = None
    steps: int = 0
    reward: float = 0.0


@dataclass
class RunMeta:
    meta_version: int = META_VERSION
    run_id: str = ""
    level: int = 0
    mode: str = ""
    seed: int | None = None
    copy: str = ""
    started_at: float = 0.0
    wall_s: float = 0.0
    prefix: dict | None = None
    start_state: dict | None = None  # TAS start state: record, record_sha256, frame (the prefix = its first frames)
    timing: dict | None = None  # process start: launch_s, handover_s (prefix replay), intro_s, total_s
    mod_version: str = ""
    feature_version: str = ""
    feature_config: dict | None = None
    reward_config: dict | None = None
    frames: int = 0
    end_reason: str | None = None
    levelfinish_frame: int | None = None
    lives: list = field(default_factory=list)
    sr_ms: int | None = None  # official: trunc of the mod's double sum (after the SR tail)
    sr_double: float | None = None
    sr_float32: float | None = None
    sr_python_at_levelfinish: dict | None = None
    sr_mod_at_levelfinish: float | None = None
    sr_tail_frames: int | None = None
    sr_bml_float: float | None = None  # IBML::GetSRScore() at the end (float of the double sum)
    hs: int | None = None  # official: the end-screen score (docs/benchmarks.md, HS readout)
    hs_parts: dict | None = None
    hs_mod: int | None = None  # the mod's snapshot in the LevelFinish hook (SREN), without the followers
    framelog_source: str = "python"
    framelog_match: bool | None = None
    framelog_mismatch: dict | None = None
    tas_path: str = ""
    tas_sha256: str = ""
    events: list = field(default_factory=list)  # [(number, name, param)]
    notes: list = field(default_factory=list)


class RunRecorder:
    """Records one game process."""

    def __init__(self, out_dir, run_id: str, level: int, **meta):
        self.out_dir = Path(out_dir)
        self.meta = RunMeta(run_id=run_id, level=level, started_at=time.time(), **meta)
        self._chunks: list = []
        self.n_frames = 0
        self.events: list = []  # (number, code, param)
        self.lives: list[LifeRecord] = []
        self._t0 = time.monotonic()

    # frames -----------------------------------------------------------------------------
    def add_frames(self, frames, executed: int | None = None) -> None:
        """Frames sent in one action block (already expanded by repeat); executed = how many the mod
        ran (from frames_done), the rest are void."""
        frames = tas.as_frames(frames)
        if executed is not None:
            if not 0 <= executed <= len(frames):
                raise ValueError(f"executed {executed} outside 0..{len(frames)}")
            frames = frames[:executed]
        if len(frames):
            self._chunks.append(frames.copy())
            self.n_frames += len(frames)

    def add_prefix(self, frames) -> None:
        self.add_frames(frames)

    def frames(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, tas.FRAME_DTYPE)
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        return self._chunks[0]

    def add_events(self, events) -> None:
        for e in events:
            self.events.append((int(e["frame"]), int(e["code"]), int(e["param"])))

    # lives ------------------------------------------------------------------------------
    def begin_life(self, frames_done: int) -> LifeRecord:
        lr = LifeRecord(len(self.lives), frames_done)
        self.lives.append(lr)
        return lr

    def end_life(self, frames_done: int, reason: str, tracker=None, steps: int = 0, reward: float = 0.0):
        if not self.lives or self.lives[-1].end_frame is not None:
            return None
        lr = self.lives[-1]
        lr.end_frame, lr.end_reason, lr.steps, lr.reward = frames_done, reason, steps, float(reward)
        if tracker is not None:
            lr.sr_ms_end, lr.points_end, lr.lives_end = tracker.sr_ms, tracker.points, tracker.lives
        return lr

    # finish -----------------------------------------------------------------------------
    def finish(self, tracker=None, end_reason: str | None = None, mod_framelog=None, final_time=None,
               sren=None) -> Path:
        """Write <run_id>.tas and <run_id>.json. mod_framelog: path or frames of the mod's log.
        final_time: TIME section of the last observation; sren: the mod's SREN block (preferred)."""
        m = self.meta
        self.out_dir.mkdir(parents=True, exist_ok=True)
        py = self.frames()
        export = py
        if mod_framelog is not None:
            mod = tas.load_raw(mod_framelog) if isinstance(mod_framelog, (str, Path)) else tas.as_frames(mod_framelog)
            same = len(mod) == len(py) and mod.tobytes() == py.tobytes()
            m.framelog_match = bool(same)
            if not same:
                n = min(len(mod), len(py))
                diff = np.flatnonzero(mod[:n].view("<u8") != py[:n].view("<u8"))
                m.framelog_mismatch = {"mod_frames": int(len(mod)), "python_frames": int(len(py)),
                                       "first_diff": int(diff[0]) if len(diff) else n}
            export = mod
            m.framelog_source = "mod"
        m.frames = int(len(export))
        m.end_reason = end_reason
        m.wall_s = time.monotonic() - self._t0
        m.lives = [asdict(lr) for lr in self.lives]
        m.events = [(n, P.EVENT_NAMES.get(c, str(c)), p) for n, c, p in self.events]
        if tracker is not None:
            m.levelfinish_frame = tracker.levelfinish_frame
            if tracker.hs_at_finish is not None:
                m.hs = int(tracker.hs_at_finish)
                m.hs_parts = {"points": tracker.points_at_finish, "lives": tracker.lives_at_finish,
                              "life_bonus": tracker.life_bonus_at_finish, "level_bonus": tracker.level_bonus,
                              "followers": tracker.followers_at_finish, "followers_at_hook": tracker.followers_at_hook,
                              "source": tracker.hs_source, "levelfinish_hook": tracker.hs_levelfinish_hook}
        finished = tracker is not None and tracker.levelfinish_frame is not None
        if sren is not None:
            m.sr_double = float(sren["sr_double"])
            m.sr_float32 = float(sren["sr_f32sum"])
            m.sr_bml_float = float(sren["sr_bml"])
            m.sr_tail_frames = int(sren["tail_frames"])
            if sren["level_finished"]:
                m.sr_mod_at_levelfinish = float(sren["sr_double_at_finish"])
                if int(sren["hs_valid"]) == 1:
                    m.hs_mod = int(sren["hs"])
            if not sren["counter_inactive_seen"] and finished:
                m.notes.append("SR tail ended without CounterInactive (BALLRAIN_SR_TAIL too small?)")
        elif final_time is not None:
            m.sr_double = float(final_time["sr_double"])
            m.sr_float32 = None
            m.sr_bml_float = float(final_time["sr_float"])
        if finished and m.sr_double is not None:
            m.sr_ms = int(math.trunc(m.sr_double))
        hp = m.hs_parts
        if hp and hp.get("levelfinish_hook") is not None and hp.get("followers_at_hook") is not None:
            hook = hp["levelfinish_hook"] + 20 * hp["followers_at_hook"]
            if hook != m.hs:
                m.notes.append(f"HS readout at the end of the SR tail ({m.hs}) != LevelFinish hook snapshot + "
                               f"20 x followers ({hook})")
        if tracker is not None and tracker.levelfinish_frame is not None:
            lf = tracker.levelfinish_frame
            m.sr_python_at_levelfinish = sr_timer(export[:lf], self.events)
        tas_path = self.out_dir / f"{m.run_id}.tas"
        tas.save(tas_path, export)
        m.tas_path = tas_path.name
        import hashlib
        m.tas_sha256 = hashlib.sha256(tas_path.read_bytes()).hexdigest()
        meta_path = self.out_dir / f"{m.run_id}.json"
        meta_path.write_text(json.dumps(asdict(m), indent=1, default=_json_default), encoding="utf-8")
        return meta_path


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))

"""Agent-mode client for one game process (docs/agent-protocol.md).

AgentSession owns: the shared-memory channel, the game process (through a launcher), the level
message files, a GameTracker and a RunRecorder-compatible frame log. One session = one process =
one run; RL episodes (lives) are layered on top by ballrain.env.

Launchers: GameLauncher starts Player.exe of a runtime copy (ballrain.instances safety rules);
ballrain.fakegame.FakeLauncher runs a simulated mod in a thread over the same transport;
ballrain.headless.HeadlessLauncher starts the headless engine (docs/headless.md). A launcher may provide
make_channel(cfg) (its transport: the headless backend uses the socket transport) and a `backend` name
("retail" when absent), which decides what the hello must name.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import instances, protocol as P, tas
from .protocol import Action, Ev
from .state import GameTracker
from .transport import PeerGone, PyChannel, Timeout, TransportError, make_name

HEADLESS_HELLO = "headless"  # every headless hello names it (headless/src/ballrain_agent.cpp)

DEFAULT_DELTA = np.float32(1000 / 132)


class SessionError(RuntimeError):
    pass


@dataclass
class SessionConfig:
    level: int
    out_dir: Path
    copy: str = instances.TRAINING_COPY
    timeout_ms: int = 30 * 60 * 1000  # mod's wait for an action (a PPO update can take minutes)
    max_frames: int | None = None
    body_radius: float = 60.0
    sr_tail: int = 32
    skip_render: bool = True
    exit_game: bool = True
    prefix: Path | None = None  # headerless frames replayed by the mod before hand-over
    prefix_events: bool = False
    trace: bool = False
    events_txt: bool = True
    hello_timeout_s: float = 120.0
    obs_timeout_s: float = 300.0
    debug: Path | None = None
    post_finish: int = 0  # diagnostic (BALLRAIN_POST_FINISH): idle frames run after the SR tail
    post_log: Path | None = None  # diagnostic (BALLRAIN_POST_LOG)
    launch: str = "direct"  # instances.LAUNCH_RECIPES key (direct = Popen with cwd <copy>/Bin, as in M0/M1)

    def paths(self) -> dict:
        d = Path(self.out_dir).resolve()
        return {"framelog": d / "framelog.bin", "levelmsg": d / "level.msg", "trace": d / "trace.txt",
                "events": d / "events.txt"}


def agent_env(cfg: SessionConfig, shm_name: str, channel_var: str = "BALLRAIN_SHM") -> dict:
    """Environment of an agent-mode launch; every managed variable is set or explicitly cleared.
    channel_var: the variable that names the channel (BALLRAIN_SHM, or BALLRAIN_SOCK for the socket transport)."""
    p = cfg.paths()
    env = {k: None for k in instances.MANAGED_VARS}
    env.update({
        "BALLRAIN_MODE": "agent",
        "BALLRAIN_LEVEL": str(instances._check_level(cfg.level)),
        channel_var: shm_name,
        "BALLRAIN_PARENT_PID": str(os.getpid()),
        "BALLRAIN_TIMEOUT_MS": str(int(cfg.timeout_ms)),
        "BALLRAIN_LEVELMSG": str(p["levelmsg"]),
        "BALLRAIN_FRAMELOG": str(p["framelog"]),
        "BALLRAIN_BODY_RADIUS": repr(float(cfg.body_radius)),
        "BALLRAIN_SR_TAIL": str(int(cfg.sr_tail)),
        "BALLRAIN_SKIP_RENDER": "1" if cfg.skip_render else None,
        "BALLRAIN_EXIT": "1" if cfg.exit_game else None,
        "BALLRAIN_LOADS": "1",
    })
    if cfg.max_frames is not None:
        env["BALLRAIN_MAX_FRAMES"] = str(int(cfg.max_frames))
    if cfg.prefix is not None:
        env["BALLRAIN_PREFIX"] = str(Path(cfg.prefix).resolve())
        env["BALLRAIN_PREFIX_EVENTS"] = "1" if cfg.prefix_events else None
    if cfg.trace:
        env["BALLRAIN_TRACE"] = str(p["trace"])
    if cfg.events_txt:
        env["BALLRAIN_EVENTS"] = str(p["events"])
    if cfg.debug is not None:
        env["BALLRAIN_DEBUG"] = str(cfg.debug)
    if cfg.post_finish:
        env["BALLRAIN_POST_FINISH"] = str(int(cfg.post_finish))
        if cfg.post_log is not None:
            env["BALLRAIN_POST_LOG"] = str(Path(cfg.post_log).resolve())
    return env


class GameLauncher:
    """Starts Player.exe in runtime/<copy> (never anything else)."""

    def __init__(self, copy: str = instances.TRAINING_COPY, runtime=None, recipe=None):
        self.copy = copy
        self.runtime = runtime
        self.recipe = recipe

    def start(self, env: dict):
        gc = instances.get_copy(self.copy, self.runtime)
        if gc.has_tassupport():
            raise SessionError(f"{gc.root} contains TASSupport.bmodp; agent copies must not")
        if not gc.has_ballrainio():
            raise SessionError(f"{gc.mods_dir} has no BallrainIO mod")
        return instances.start_copy(gc, env, self.recipe)

    @staticmethod
    def stop(handle, wait_s: float = 10.0):
        return instances.stop_handle(handle, wait_s)


class AgentSession:
    def __init__(self, cfg: SessionConfig, launcher=None):
        self.cfg = cfg
        self.launcher = launcher or GameLauncher(cfg.copy, recipe=instances.LAUNCH_RECIPES[cfg.launch])
        self.paths = cfg.paths()
        self.tracker = GameTracker(cfg.level)
        self.chan: PyChannel | None = None
        self.handle = None
        self.hello: dict | None = None
        self.obs: P.Observation | None = None
        self.frames_done = 0
        self.ended = False
        self.end_reason: int | None = None
        self.final_time = None  # TIME section of the last observation
        self.sren = None  # SREN section (final SR / HS block of the mod), when sent
        self.sr_at_levelfinish = None
        self.trace_hash = None  # headless backend: FNV-1a over the exported frames' ball matrices (final observation)
        self.log_chunks: list = []
        self.n_logged = 0
        self._level_loaded = False
        self._inc_pos = 0
        self.prefix_frames = None
        self.on_prefix_obs = None  # callable(Observation) for each prefix-frame observation
        self.prefix_obs_mask = 0  # obs_mask sent with the prefix acknowledgements
        self.n_prefix_obs = 0
        self.timing: dict = {}  # perf_counter stamps: launch, spawned, hello, first_obs
        self.backend = getattr(self.launcher, "backend", "retail")  # "retail" | "headless"

    # lifecycle ----------------------------------------------------------------------------
    def start(self) -> P.Observation:
        d = Path(self.cfg.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        for k in ("framelog", "levelmsg", "trace", "events"):
            for p in (self.paths[k], Path(str(self.paths[k]) + ".inc")):
                if p.exists():
                    p.unlink()
        if self.cfg.prefix is not None:
            self.prefix_frames = tas.load_raw(self.cfg.prefix)
        make_channel = getattr(self.launcher, "make_channel", None)
        self.chan = make_channel(self.cfg) if make_channel is not None else PyChannel(make_name(self.cfg.copy))
        env = agent_env(self.cfg, self.chan.name, getattr(self.chan, "env_var", "BALLRAIN_SHM"))
        try:
            self.timing["launch"] = time.perf_counter()
            self.handle = self.launcher.start(env)
            self.timing["spawned"] = time.perf_counter()
            wh = self._wait_handle()
            self.hello = self.chan.wait_hello(self.cfg.hello_timeout_s, wh)
            self.timing["hello"] = time.perf_counter()
            if self.backend == "retail":
                want = instances.PHYSICS_RT_SHA256[:16]
                if want not in self.hello["mod_version"]:
                    raise SessionError(f"mod hello {self.hello['mod_version']!r} does not name physics_RT {want}")
            elif HEADLESS_HELLO not in self.hello["mod_version"]:
                raise SessionError(f"{self.backend} backend: hello {self.hello['mod_version']!r} is not a headless one")
            obs = self._next_obs()
        except Exception:
            self.close(kill=True)
            raise
        self.timing["first_obs"] = time.perf_counter()
        if obs is None:
            raise SessionError(f"game ended before the first observation (header {self.chan.header()})")
        if self.prefix_frames is not None:
            self._log(self.prefix_frames[:obs.frames_done])
        return obs

    def _wait_handle(self):
        h = self.handle
        return getattr(h, "wait_handle", None) or getattr(h, "_h", None)

    def _next_obs(self) -> P.Observation | None:
        """Next ordinary observation. Prefix-frame observations (BALLRAIN_PREFIX_EVENTS, flags bit1) are
        ingested, handed to on_prefix_obs and acknowledged with an empty action (agent-protocol.md 5, 8.1:
        the mod waits for that acknowledgement before it runs the next prefix frame)."""
        while True:
            data = self.chan.wait_obs(self.cfg.obs_timeout_s, self._wait_handle())
            if "first_obs_raw" not in self.timing:
                self.timing["first_obs_raw"] = time.perf_counter()  # before parsing / level message load
            if data is None:
                self._mark_ended()
                h = self.chan.header()
                if h["mod_state"] == P.ModState.ERROR:
                    raise SessionError(f"mod error (last_error {h['last_error']}); see ModLoader.log")
                return None
            obs = P.parse_observation(data)
            self._ingest(obs)
            if obs.flags & P.OBS_PREFIX and not obs.final and not self.ended:
                self.n_prefix_obs += 1
                if self.on_prefix_obs is not None:
                    self.on_prefix_obs(obs)
                self.chan.send_action(Action(np.zeros(0, tas.FRAME_DTYPE), 1, 0, 0, self.prefix_obs_mask))
                continue
            return obs

    def _ingest(self, obs: P.Observation) -> None:
        if obs.frames_done < self.frames_done:
            raise SessionError(f"frames_done went back from {self.frames_done} to {obs.frames_done}")
        self.obs = obs
        self.frames_done = obs.frames_done
        if obs.time is not None:
            self.final_time = obs.time
        if obs.sren is not None:
            self.sren = obs.sren
        if obs.hash is not None:  # headless backend: run fingerprint
            self.trace_hash = f"{int(obs.hash['trace_hash']):016x}"
        # level message and increments first: the tracker resolves this observation's Physicalize /
        # Unphysicalize events through their records (ball transformations, state.GameTracker.transforming)
        if obs.has_event(Ev.StartLevel) or not self._level_loaded:
            self._load_level(full=obs.has_event(Ev.StartLevel))
        if self._level_loaded and (len(obs.phys) or obs.has_event(Ev.StartLevel)):
            self._load_increments()
        evs = self.tracker.update(obs)
        if obs.has_event(Ev.LevelFinish) and self.sr_at_levelfinish is None:
            # the mod's SREN block has the value at the LevelFinish frame (before the SR tail); without it,
            # the TIME of the observation carrying LevelFinish (which may already include the tail)
            if obs.sren is not None and obs.sren["level_finished"]:
                self.sr_at_levelfinish = float(obs.sren["sr_double_at_finish"])
            elif obs.time is not None:
                self.sr_at_levelfinish = float(obs.time["sr_double"])
        h = self.chan.header()
        if h["mod_state"] in (P.ModState.ENDED, P.ModState.ERROR) or obs.has_event(Ev.RunEnd) or obs.final:
            self._mark_ended(h)
        return evs

    def _mark_ended(self, h=None):
        h = h or self.chan.header()
        self.ended = True
        self.end_reason = h["end_reason"]

    def _load_level(self, full: bool) -> None:
        p = self.paths["levelmsg"]
        if not p.exists() or (self._level_loaded and not full):
            return
        lm, _ = P.parse_level_message(p.read_bytes())
        self.tracker.merge_level(lm)
        self._level_loaded = True

    def _load_increments(self) -> None:
        p = Path(str(self.paths["levelmsg"]) + ".inc")
        if not p.exists() or p.stat().st_size <= self._inc_pos:
            return
        data = p.read_bytes()
        msgs, used = P.parse_level_stream(data[self._inc_pos:])
        for lm in msgs:
            self.tracker.merge_level(lm)
        self._inc_pos += used

    # stepping -----------------------------------------------------------------------------
    def step(self, action: Action) -> P.Observation | None:
        if self.ended:
            raise SessionError("session has ended")
        n0 = self.frames_done
        self.chan.send_action(action)
        try:
            obs = self._next_obs()
        except (PeerGone, Timeout):
            self._mark_ended()
            raise
        n1 = self.frames_done
        lf = self.tracker.levelfinish_frame
        if lf is not None:
            n1 = min(n1, lf)  # SR tail frames after LevelFinish are not in the frame log
        executed = max(0, n1 - n0)
        total = action.total_frames
        if executed > total:
            raise SessionError(f"mod reports {executed} executed frames for a block of {total}")
        if executed:
            self._log(action.expanded()[:executed])
        return obs

    def _log(self, frames) -> None:
        if len(frames):
            self.log_chunks.append(np.array(frames, dtype=tas.FRAME_DTYPE))
            self.n_logged += len(frames)

    def frame_log(self) -> np.ndarray:
        if not self.log_chunks:
            return np.zeros(0, tas.FRAME_DTYPE)
        if len(self.log_chunks) > 1:
            self.log_chunks = [np.concatenate(self.log_chunks)]
        return self.log_chunks[0]

    def act(self, frames, repeat: int = 1, stop=(), flags: int = 0, obs_mask: int = 0) -> P.Observation | None:
        frames = np.asarray(frames, tas.FRAME_DTYPE).reshape(-1)
        return self.step(Action(frames, repeat, P.event_mask(*stop), flags, obs_mask))

    def run_until(self, keys: int, stop, delta=DEFAULT_DELTA, chunk: int = 4096, limit: int = 200000):
        """Inject constant (delta, keys) frames until one of the stop events (or the end of the run)."""
        mask = P.event_mask(*stop)
        frames = tas.make_frames([delta], [keys])
        done = 0
        obs = self.obs
        while not self.ended and done < limit:
            n0 = self.frames_done
            obs = self.step(Action(frames, min(chunk, limit - done), mask))
            done += self.frames_done - n0
            if obs is None or obs.events["code"].size and any(obs.has_event(e) for e in stop):
                break
        return obs

    def stop(self, wait_s: float = 30.0) -> None:
        """End the run (STOP), wait for the mod's final state and for the process to exit."""
        if self.chan is None:
            return
        if not self.ended:
            try:
                self.chan.send_action(Action(np.zeros(0, tas.FRAME_DTYPE), 1, 0, P.ACT_STOP))
                obs = None
                try:
                    obs = self._next_obs()
                except (Timeout, PeerGone):
                    pass
                if obs is not None and not self.ended:
                    self._mark_ended()
            except TransportError:
                pass
            self.ended = True
        self._finish_process(wait_s)

    def _finish_process(self, wait_s: float) -> None:
        if self.handle is not None:
            stop = getattr(self.launcher, "stop", None)
            if stop is not None:
                stop(self.handle, wait_s)
            self.handle = None

    def close(self, kill: bool = False) -> None:
        if self.handle is not None:
            self._finish_process(0.0 if kill else 30.0)
        if self.chan is not None:
            self.chan.close()
            self.chan = None

    def mod_framelog(self):
        p = self.paths["framelog"]
        return p if p.exists() else None

    @property
    def end_reason_name(self) -> str | None:
        if self.end_reason is None:
            return None
        return P.EndReason.NAMES[self.end_reason] if self.end_reason < len(P.EndReason.NAMES) else str(self.end_reason)

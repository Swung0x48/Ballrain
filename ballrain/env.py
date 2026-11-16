"""Gymnasium environment for one game instance (one runtime copy) in agent mode.

Process model: one game process per run (same-process reload is not exact). Inside a process every
life is one RL episode: BallOff terminates the episode; the next reset() fast-forwards the death
animation with explicit idle frames (stop on BallNavActive) and continues in the same process. The
process ends at LevelFinish, GameOver, a frame limit, truncation or close(); then its frames are
exported as a legacy .tas plus metadata (ballrain.recording) and the next reset() starts a new process.

Actions: Discrete over the arrow combinations (+ optional camera turns), plus in HS mode (esc_action)
"esc" -- Esc goes down for one frame and is released (esc_frames frames in total; the PauseLevel /
UnpauseLevel toggle follows two frames after the press) -- and optionally "collect" (collect_action):
Esc, idle frames until no small score ball is following any more (or collect_max_frames), Esc. While
paused the game runs no physics and no clock; arrow keys are masked to pause_keys (mask_keys_while_paused)
and a pause longer than max_pause_frames is ended by an Esc the env injects; once an episode has spent
max_episode_pause_frames paused frames, pausing is masked (Esc only unpauses, collect is a no-op), so an
episode cannot stall in the pause menu. Every injected frame,
pauses included, is in the frame log and the exported .tas.
"""
from __future__ import annotations

import hashlib
import itertools
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np

from . import instances, protocol as P, tas
from .client import AgentSession, SessionConfig
from .features import FeatureConfig, FeatureExtractor
from .protocol import Ev
from .recording import RunRecorder
from .rewards import Reward, RewardConfig
from .transport import PeerGone, Timeout

ARROWS9 = (0, tas.KEY_UP, tas.KEY_DOWN, tas.KEY_LEFT, tas.KEY_RIGHT, tas.KEY_UP | tas.KEY_LEFT,
           tas.KEY_UP | tas.KEY_RIGHT, tas.KEY_DOWN | tas.KEY_LEFT, tas.KEY_DOWN | tas.KEY_RIGHT)
CAMERA_ACTIONS = (tas.KEY_SHIFT | tas.KEY_LEFT, tas.KEY_SHIFT | tas.KEY_RIGHT)
ESC, COLLECT = "esc", "collect"  # special actions (see the module docstring)
EPISODE_STOP = ("BallOff", "LevelFinish", "GameOver")
_RUN_COUNTER = itertools.count()


@dataclass
class EnvConfig:
    level: int = 1
    mode: str = "sr"  # "sr" | "hs" (selects the default reward)
    copy: str = instances.TRAINING_COPY
    out_root: str | None = None  # default <runtime>/episodes
    delta_ms: float = float(np.float32(1000 / 132))
    action_repeat: int = 1  # frames per env step (each frame enters the .tas)
    camera_actions: bool = False  # add shift+left / shift+right
    esc_action: bool | None = None  # add the Esc (pause / unpause) action; None: on in HS mode, off in SR mode
    collect_action: bool = False  # add the "collect" macro (Esc, wait for the followers, Esc)
    esc_frames: int = 4  # frames of one Esc action: Esc down on the first, released after
    collect_max_frames: int = 1200  # longest wait of the collect macro (paused frames)
    collect_chunk: int = 8  # frames per idle block while the collect macro waits
    mask_keys_while_paused: bool = True
    pause_keys: int = 0  # keys injected instead of the action's keys while paused
    esc_requires_control: bool = True  # Esc is a no-op unless the player has control or the game is paused
    max_pause_frames: int | None = 3000  # consecutive paused frames before the env unpauses (None = no cap)
    max_episode_pause_frames: int | None = 20000  # paused frames per episode before pausing is masked (None = no cap)
    per_life: bool = True
    max_episode_frames: int | None = None
    max_process_frames: int | None = None  # BALLRAIN_MAX_FRAMES
    idle_keys: int = 0  # keys injected while the player has no control
    skip_tutorial: bool = True  # level 1: pulse Q until CamNavActive
    prefix: str | None = None  # headerless frames replayed by the mod before hand-over
    timeout_ms: int = 30 * 60 * 1000
    obs_timeout_s: float = 300.0
    hello_timeout_s: float = 120.0
    sr_tail: int = 32
    respawn_limit_frames: int = 20000  # BallOff -> BallNavActive takes ~1300 frames at 264 fps
    body_radius: float = 60.0
    skip_render: bool = True
    trace: bool = False
    seed: int | None = None
    launch: str = "direct"  # instances.LAUNCH_RECIPES key
    features: FeatureConfig = field(default_factory=FeatureConfig)
    reward: RewardConfig | None = None

    def reward_config(self) -> RewardConfig:
        return self.reward if self.reward is not None else RewardConfig(mode=self.mode)


class BallanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, launcher=None, **overrides):
        self.cfg = config or EnvConfig()
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        self.launcher = launcher
        self.fx = FeatureExtractor(self.cfg.features)
        self.reward = Reward(self.cfg.reward_config())
        if self.cfg.esc_frames < 3:
            raise ValueError("esc_frames must be >= 3 (Esc must be released, the toggle lands 2 frames later)")
        esc = self.cfg.esc_action if self.cfg.esc_action is not None else self.cfg.mode == "hs"
        self.actions = (ARROWS9 + (CAMERA_ACTIONS if self.cfg.camera_actions else ()) + ((ESC,) if esc else ())
                        + ((COLLECT,) if self.cfg.collect_action else ()))
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(-self.cfg.features.clip, self.cfg.features.clip,
                                                (self.fx.size,), np.float32)
        self.delta = np.float32(self.cfg.delta_ms)
        self.session: AgentSession | None = None
        self.recorder: RunRecorder | None = None
        self.run_dir: Path | None = None
        self.finished_runs: list = []  # meta json paths
        self.last_run_ok = True
        self._ep_steps = 0
        self._ep_frames = 0
        self._ep_pause_frames = 0
        self._ep_start = 0
        self._after_death = False  # the last episode ended with BallOff and the process continues
        self._last_info: dict = {}

    # ------------------------------------------------------------------------------ process
    def _out_root(self) -> Path:
        return Path(self.cfg.out_root) if self.cfg.out_root else instances.runtime_dir() / "episodes"

    def _start_process(self):
        c = self.cfg
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"L{c.level:02d}-{c.mode}-{c.copy}-{stamp}-{os.getpid()}-{next(_RUN_COUNTER)}"
        self.run_dir = self._out_root() / c.copy / run_id
        sc = SessionConfig(level=c.level, out_dir=self.run_dir, copy=c.copy, timeout_ms=c.timeout_ms,
                           max_frames=c.max_process_frames, body_radius=c.body_radius, sr_tail=c.sr_tail,
                           skip_render=c.skip_render, prefix=Path(c.prefix) if c.prefix else None, trace=c.trace,
                           obs_timeout_s=c.obs_timeout_s, hello_timeout_s=c.hello_timeout_s, launch=c.launch)
        prefix_meta = None
        if c.prefix:
            data = Path(c.prefix).read_bytes()
            prefix_meta = {"path": str(c.prefix), "frames": len(data) // tas.FRAME_SIZE,
                           "sha256": hashlib.sha256(data).hexdigest()}
        self.recorder = RunRecorder(self.run_dir, run_id, c.level, mode=c.mode, seed=c.seed, copy=c.copy,
                                    prefix=prefix_meta, feature_version=self.cfg.features.version,
                                    feature_config=self.cfg.features.to_dict(),
                                    reward_config=self.cfg.reward_config().to_dict())
        self.session = AgentSession(sc, self.launcher)
        self.session.start()
        self.recorder.meta.mod_version = (self.session.hello or {}).get("mod_version", "")
        self._intro()

    def _idle(self):
        return tas.make_frames([self.delta], [self.cfg.idle_keys])

    def _intro(self):
        s = self.session
        if not s.tracker.flags.started:
            s.run_until(self.cfg.idle_keys, ["StartLevel"], self.delta)
        if self.cfg.level == 1 and self.cfg.skip_tutorial and not s.tracker.flags.cam_nav and not s.ended:
            pulse = tas.make_frames([self.delta, self.delta], [tas.KEY_Q, 0])
            for _ in range(200):
                if s.ended or s.tracker.flags.cam_nav or s.tracker.flags.ball_nav:
                    break
                s.act(pulse, repeat=1, stop=["CamNavActive", "BallNavActive", "GameOver"])
        if not s.tracker.flags.ball_nav and not s.ended:
            s.run_until(self.cfg.idle_keys, ["BallNavActive", "GameOver", "LevelFinish"], self.delta)

    def _finish_process(self, reason: str | None = None):
        s, rec = self.session, self.recorder
        if s is None:
            return None
        try:
            s.stop()
        finally:
            s.close()
        rec.add_frames(s.frame_log())
        rec.events = list(s.tracker.events)
        if rec.lives and rec.lives[-1].end_frame is None:
            rec.end_life(s.frames_done, reason or s.end_reason_name or "stopped", s.tracker)
        rec.meta.sr_mod_at_levelfinish = s.sr_at_levelfinish
        meta = rec.finish(s.tracker, s.end_reason_name, s.mod_framelog(), s.final_time, s.sren)
        self.finished_runs.append(meta)
        self.last_run_ok = rec.meta.framelog_match is not False
        if not self.last_run_ok:
            warnings.warn(f"{rec.meta.run_id}: the mod's frame log differs from the frames Python sent "
                          f"({rec.meta.framelog_mismatch}); the .tas holds the mod's log", RuntimeWarning)
        self.session = self.recorder = None
        self._after_death = False
        return meta

    # ------------------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.cfg.seed = seed
        # an explicit seed (or options={"new_process": True}) asks for a fresh run: the game has no RNG
        # we could reseed, so "same seed => same trajectory" needs a new process
        fresh = seed is not None or bool((options or {}).get("new_process"))
        s = self.session
        if s is not None and not s.ended and self._after_death and not fresh:
            # per-life episodes: fast-forward the death and respawn with explicit idle frames
            s.run_until(self.cfg.idle_keys, ["BallNavActive", "GameOver", "LevelFinish"], self.delta,
                        limit=self.cfg.respawn_limit_frames)
            self._after_death = False
            if s.ended or not s.tracker.flags.ball_nav:
                self._finish_process()
        elif s is not None:
            self._finish_process("reset")  # reset in the middle of a life: the life cannot be restarted
        if self.session is None and s is not None and (options or {}).get("relaunch") is False:
            # driver convenience (gates): the process ended (e.g. GameOver during the respawn fast-forward);
            # do not start a new one. Returns (None, info) instead of an observation.
            return None, {"run_meta": str(self.finished_runs[-1]) if self.finished_runs else None,
                          "framelog_match": self.last_run_ok, "process_ended": True}
        if self.session is None:
            self._start_process()
            if self.session.ended or not self.session.tracker.flags.ball_nav:
                raise RuntimeError(f"run ended before the player got control ({self.session.end_reason_name})")
        s = self.session
        self.recorder.begin_life(s.frames_done)
        self._ep_steps = self._ep_frames = self._ep_pause_frames = 0
        self._ep_start = s.frames_done
        obs, finfo = self.fx.compute(s.tracker)
        self.reward.begin(s.tracker, self._arc(finfo))
        info = self._info(finfo, [])
        return obs, info

    @staticmethod
    def _arc(finfo):
        p = finfo.get("path")
        return p["s"] if p else None

    def step(self, action):
        s = self.session
        if s is None or s.ended:
            raise RuntimeError("step() after the run ended; call reset()")
        a = self.actions[int(action)]
        n0 = s.frames_done
        ev0 = len(s.tracker.events)
        error = None
        extra: dict = {}
        paused0 = s.tracker.flags.paused
        try:
            if a == ESC:
                if self._esc_allowed():
                    extra["esc"] = self._esc()
                else:
                    extra["esc"] = "masked"
                    s.act(self._idle(), repeat=self.cfg.action_repeat, stop=EPISODE_STOP)
            elif a == COLLECT:
                extra["collect"] = self._collect()
            else:
                keys = a
                if s.tracker.flags.paused and self.cfg.mask_keys_while_paused:
                    keys = self.cfg.pause_keys
                s.act(tas.make_frames([self.delta], [keys]), repeat=self.cfg.action_repeat, stop=EPISODE_STOP)
            if paused0 or s.tracker.flags.paused:
                self._ep_pause_frames += s.frames_done - n0
            cap = self.cfg.max_pause_frames
            if not s.ended and s.tracker.flags.paused and (
                    (cap is not None and s.tracker.paused_frames() >= cap) or self._pause_budget_spent()):
                n1 = s.frames_done
                extra["forced_unpause"] = self._esc()
                self._ep_pause_frames += s.frames_done - n1
        except (PeerGone, Timeout) as e:  # the game died or hung: truncate, record what we have, relaunch on reset
            error = f"{type(e).__name__}: {e}"
            s.ended = True
        self._ep_steps += 1
        self._ep_frames += s.frames_done - n0
        new = s.tracker.events[ev0:]
        codes = {c for _, c, _ in new}
        died = Ev.BallOff in codes
        finished = Ev.LevelFinish in codes
        game_over = Ev.GameOver in codes
        terminated = died or finished or game_over
        truncated = False
        if not terminated and (s.ended or (self.cfg.max_episode_frames is not None
                                           and self._ep_frames >= self.cfg.max_episode_frames)):
            truncated = True
        obs, finfo = self.fx.compute(s.tracker)
        r, terms = self.reward.step(s.tracker, self._arc(finfo), died, finished, terminated)
        info = self._info(finfo, new, terms)
        info.update(extra)
        if error:
            info["error"] = error
        if terminated or truncated:
            reason = "BallOff" if died else "LevelFinish" if finished else "GameOver" if game_over else "truncated"
            self.recorder.end_life(s.frames_done, reason, s.tracker, self._ep_steps, self.reward.total)
            info["episode_end"] = reason
            self._after_death = died and self.cfg.per_life and not s.ended
            if not self._after_death:
                info["run_meta"] = str(self._finish_process(reason))
                info["framelog_match"] = self.last_run_ok
        return obs, float(r), terminated, truncated, info

    # ------------------------------------------------------------------------------ Esc / pause
    def _pause_budget_spent(self) -> bool:
        b = self.cfg.max_episode_pause_frames
        return b is not None and self._ep_pause_frames >= b

    def _esc_allowed(self) -> bool:
        t = self.session.tracker
        f = t.flags
        if f.paused:
            return True  # unpausing is always allowed
        if self._pause_budget_spent():
            return False
        if not self.cfg.esc_requires_control:
            return True
        return f.ball_nav and not f.ball_off and not f.finished and not t.transforming

    def _esc(self) -> str:
        """One Esc press (down one frame, then released), esc_frames frames, stopping on episode events.
        Returns "paused" / "unpaused" / "no_toggle" from the PauseLevel / UnpauseLevel events."""
        s = self.session
        was = s.tracker.flags.paused
        n = self.cfg.esc_frames
        s.act(tas.make_frames([self.delta] * n, [tas.KEY_ESC] + [0] * (n - 1)), repeat=1, stop=EPISODE_STOP)
        now = s.tracker.flags.paused
        return "no_toggle" if now == was else ("paused" if now else "unpaused")

    def _collect(self) -> dict:
        """Collect macro: Esc, idle frames until no small ball is following (or collect_max_frames), Esc.
        A no-op block (keys 0) when nothing is following, the game is paused or Esc is not allowed."""
        s = self.session
        t = s.tracker
        out = {"followers_before": t.followers(), "waited": 0}
        if out["followers_before"] == 0 or t.flags.paused or not self._esc_allowed():
            s.act(self._idle(), repeat=self.cfg.action_repeat, stop=EPISODE_STOP)
            out["result"] = "noop"
            return out
        out["pause"] = self._esc()
        if t.flags.paused:
            while not s.ended and t.followers() > 0 and out["waited"] < self.cfg.collect_max_frames:
                n0 = s.frames_done
                s.act(tas.make_frames([self.delta], [self.cfg.pause_keys]),
                      repeat=min(self.cfg.collect_chunk, self.cfg.collect_max_frames - out["waited"]),
                      stop=EPISODE_STOP)
                out["waited"] += s.frames_done - n0
                if s.frames_done == n0:
                    break
            if not s.ended and t.flags.paused:
                out["unpause"] = self._esc()
        out["followers_after"] = t.followers()
        out["result"] = "done"
        return out

    def _info(self, finfo, events, terms=None) -> dict:
        s = self.session
        t = s.tracker
        info = {"frames_done": s.frames_done, "sr_ms": t.sr_ms, "episode_pause_frames": self._ep_pause_frames, "points": t.points, "lives": t.lives,
                "sector": t.sector, "events": [(n, P.EVENT_NAMES.get(c, str(c))) for n, c, _ in events],
                "valid": t.control_valid(), "surface": finfo.get("surface"), "gap": finfo.get("gap"),
                "paused": t.flags.paused, "followers": t.followers(),
                "edge_fan_min": float(np.min(finfo["edge_fan"])) if "edge_fan" in finfo else None,
                "life": len(self.recorder.lives) - 1 if self.recorder else None,
                "run_id": self.recorder.meta.run_id if self.recorder else None}
        if terms:
            info["reward_terms"] = terms
        return info

    def close(self):
        if self.session is not None:
            self._finish_process("closed")
        super().close()

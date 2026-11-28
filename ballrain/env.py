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
import json
import math
import os
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np

from . import instances, protocol as P, tas
from .branch_assets import bundle_rules, ordinary_start_frames, rest_member
from .branch_move import uses_branch_move
from .branch_rules import DEFAULT_RULES
from .client import AgentSession, SessionConfig
from .curriculum import CurriculumConfig, ReverseCurriculum, TasStarts
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
ARROW_BITS = (tas.KEY_UP, tas.KEY_DOWN, tas.KEY_LEFT, tas.KEY_RIGHT)
ARROW_MASK = tas.KEY_UP | tas.KEY_DOWN | tas.KEY_LEFT | tas.KEY_RIGHT
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
    # arrow actions are world directions (UP = +z, RIGHT = +x, as FeatureConfig(camera="world")), mapped to the arrow keys
    # of the current Cam_OrientRef basis at every step (world_to_camera_keys); with camera-relative keys a policy trained
    # under one camera orientation sees a rotated level under another (L1: the TAS turns the camera, a k = 0 run does not)
    world_actions: bool = False
    # own_camera: write the agent's own (spawn) camera into cam_yaw_ref / cam_yaw / cam_rotating on every step, also on
    # TAS hand-overs after a TAS camera turn (ballrain.own_camera); needs world_actions, the "world" feature frame, no
    # camera actions and a start_record (without one the lives start at spawn and keep that camera anyway)
    own_camera: bool = False
    # an arrow key's force takes Cam_OrientRef at its latest Create: each press, and for held keys the end of each camera
    # turn (the game re-creates them); death and pause shut all forces down (docs/observations.md 5). True (conservative):
    # a key held since before a Cam_OrientRef change, a respawn or an unpause is released for the first frame of the step
    # and pressed again on the second (costs one frame of that key's push). None = on with world_actions
    repress_stale_keys: bool | None = None
    # camera guard (always on, user rule): Cam_OrientRef off a quarter turn by more than camera_quarter_tol radians while
    # no camera turn is in progress (more than camera_turn_grace_ms of game time since the last shift + left / right
    # press) ends the episode ("camera_off_quarter"), marks the run invalid and takes back the episode's reward
    camera_quarter_tol: float = 1e-4
    camera_turn_grace_ms: float = 400.0
    esc_action: bool | None = None  # add the Esc (pause / unpause) action; None: on in HS mode, off in SR mode
    collect_action: bool = False  # add the "collect" macro (Esc, wait for the followers, Esc)
    # the harness collects the small score balls itself: after any step that leaves small balls following (a score
    # ball touched) it runs the collect macro (Esc, wait until none follows, Esc) within the same step, so the
    # policy never presses Esc (HS training without an Esc action; pausing stops the clock and the score decay)
    auto_collect: bool = False
    esc_frames: int = 4  # frames of one Esc action: Esc down on the first, released after
    collect_max_frames: int = 1200  # longest wait of the collect macro (paused frames)
    # collect_hold_arrows (off: the macro as before, bit-identical): the auto-collect macro keeps the arrow keys of the
    # step before it on the frames of its two Esc actions that the game still simulates (the pause Esc's frames before
    # the toggle, the unpause Esc's frames after it; the toggle lands two frames after the press), as a player holding
    # the arrows would; without it those frames run with no key, a push lost at every collect. Never on paused frames
    # (arrows act on the pause menu). The macro ends with the arrows held (pressed again after the unpause)
    collect_hold_arrows: bool = False
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
    # TAS start states (ballrain.curriculum, docs/observations.md "TAS 起点"): each new process starts at frame k
    # of this dataset record (the record's first k frames are the BALLRAIN_PREFIX); k from the curriculum
    # (curriculum = CurriculumConfig fields as a dict, {} for the defaults) or the fixed start_frame
    start_record: str | None = None
    start_frame: int | None = None
    curriculum: dict | None = None
    # at-rest starts: with probability p_rest a new process starts at a valid frame drawn from one of rest_starts
    # ([lo, hi] record frames), and after the hand-over the env brakes the ball (world arrows against its horizontal
    # velocity) until it is slower than rest_speed or rest_max_frames passed; the episode's start kind is "tas_rest"
    # (not a curriculum outcome). For skills a respawn at a checkpoint (ball at rest) needs, e.g. a run-up.
    # An entry [lo, hi, x, z, jitter] first drives the ball (world arrows) to a target drawn uniformly in
    # [x +- jitter] x [z +- jitter] and then brakes it there (e.g. a ball at rest in a guard-rail pocket beside the line);
    # its hand-over frame may lie in an exclude_starts range (TasStarts.nearest_any); a sixth element
    # [lo, hi, x, z, jitter, max_frames] sets that entry's drive limit (default rest_max_frames)
    rest_starts: list | None = None
    p_rest: float = 0.0
    # rehearsal starts: with probability p_rehearse a new process starts at a valid frame drawn uniformly from one of
    # rehearse_starts ([lo, hi] record frames), a normal TAS start otherwise (kind "tas"): keeps a hard section in the
    # draw after the curriculum frontier moved past it
    rehearse_starts: list | None = None
    p_rehearse: float = 0.0
    # rehearse_behind_frontier: draw a rehearsal start only from the part of its range at or after the curriculum frontier
    # (a range the frontier has not reached yet is skipped: the draw falls through to the curriculum), so every section
    # is rehearsed once the reverse curriculum has passed it (r8 / level 2)
    rehearse_behind_frontier: bool = False
    # rehearse_always: more [lo, hi] ranges drawn like rehearse_starts entries (same per-entry share of p_rehearse) but
    # never clipped by rehearse_behind_frontier: a hard section behind (upstream of) the frontier keeps being trained
    # (r8 L2 小节 3 cluster). A separate field so older workers (pickled EnvConfig) just ignore it
    rehearse_always: list | None = None
    # record frames [lo, hi) that are never used as start states (e.g. the TAS on a route the training does not
    # follow: L1 shortcut 2 while the reference line takes shortcut 1); applied to TasStarts, so curriculum draws,
    # rehearsal and at-rest starts all skip them
    exclude_starts: list | None = None
    # an at-rest start given by the caller instead of drawn (pool tasks: the server draws it): {"target": [x, z] |
    # None, "max_frames": n} applied to the start_frame hand-over (target None = brake only)
    start_rest: dict | None = None
    rest_speed: float = 0.5
    rest_max_frames: int = 600
    warm_pool: bool = False  # pre-launch the next process on warm_copy while the current one runs
    warm_copy: str | None = None  # second runtime copy of this env (vecenv.make_vec_env pairs train-2i, train-2i+1)
    start_check: bool = True  # compare the hand-over state with the dataset; relaunch on a mismatch
    start_retries: int = 3  # relaunches after a mismatching hand-over before giving up
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
    # "retail" (Player.exe + BallrainIO, the authority) | "headless" (the approximate x87-soft engine,
    # ballrain.headless; headless = HeadlessConfig fields as a dict, {} / None for the environment defaults)
    backend: str = "retail"
    headless: dict | None = None
    features: FeatureConfig = field(default_factory=FeatureConfig)
    reward: RewardConfig | None = None

    branch_assets: dict | None = None

    def __reduce_ex__(self, protocol):
        if uses_branch_move(self):
            return _restore_branch_move_env_v1, (vars(self),)
        if self.branch_assets is not None:
            # An older spawned worker must not silently drop an unfamiliar dataclass field.
            if self.branch_assets.get("version") == 2:
                return _restore_branch_env_v2, (vars(self),)
            if bundle_rules(self.branch_assets) != DEFAULT_RULES:
                return _restore_branch_env_rules_v1, (vars(self),)
            return _restore_branch_env_v1, (vars(self),)
        return super().__reduce_ex__(protocol)

    def reward_config(self) -> RewardConfig:
        return self.reward if self.reward is not None else RewardConfig(mode=self.mode)


def _restore_branch_move_env_v1(fields):
    return EnvConfig(**fields)


def _restore_branch_env_v2(fields):
    return EnvConfig(**fields)


def _restore_branch_env_v1(fields):
    return EnvConfig(**fields)


def _restore_branch_env_rules_v1(fields):
    return EnvConfig(**fields)


def key_push_dirs(cam_matrix) -> dict:
    """Horizontal world direction each arrow key pushes the ball under the camera reference cam_matrix (Cam_OrientRef,
    CK rows x, y, z axis, pos): UP / DOWN along +- the z row, RIGHT / LEFT along +- the x row, each projected on the
    horizontal plane and normalised (the actual rows: off a quarter turn they are not axis-aligned)."""
    m = np.asarray(cam_matrix, np.float64)
    out = {}
    for bit, row, sign in ((tas.KEY_UP, 2, 1.0), (tas.KEY_DOWN, 2, -1.0), (tas.KEY_RIGHT, 0, 1.0), (tas.KEY_LEFT, 0, -1.0)):
        v = sign * m[row].copy()
        v[1] = 0.0
        n = np.linalg.norm(v)
        out[bit] = v / n if n > 1e-9 else np.zeros(3)
    return out


_KEY_SETS = tuple(k for k in ARROWS9 if k)  # the 8 non-empty arrow combinations (each pair of keys is a sum of forces)


def world_to_camera_keys(keys: int, cam_matrix) -> int:
    """Arrow keys whose push best realises the world direction of `keys` (UP = +z, DOWN = -z, RIGHT = +x, LEFT = -x)
    under the camera reference cam_matrix: of the 8 arrow combinations, the one whose summed push (key_push_dirs; two
    keys are two independent forces) has the largest cosine with the wanted direction (ties: the first in ARROWS9 order,
    single keys before diagonals). At rest Cam_OrientRef is a quarter turn and this is the exact rotation of the action;
    off a quarter turn (mid-turn or after an interrupted turn) it is the nearest of the 8 directions the keys can push.
    Other bits pass through."""
    if not keys & ARROW_MASK:
        return keys
    w = np.array([bool(keys & tas.KEY_RIGHT) - bool(keys & tas.KEY_LEFT), 0.0,
                  bool(keys & tas.KEY_UP) - bool(keys & tas.KEY_DOWN)], np.float64)
    w /= np.linalg.norm(w)
    dirs = key_push_dirs(cam_matrix)
    best, best_c = 0, -np.inf
    for ks in _KEY_SETS:
        v = sum(dirs[b] for b in ARROW_BITS if ks & b)
        n = np.linalg.norm(v)
        c = float(v @ w) / n if n > 1e-9 else -np.inf
        if c > best_c + 1e-9:
            best, best_c = ks, c
    return (keys & ~ARROW_MASK) | best


class BallanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, launcher=None, **overrides):
        self.cfg = config or EnvConfig()
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        if self.cfg.backend not in ("retail", "headless"):
            raise ValueError(f"unknown backend {self.cfg.backend!r}")
        if launcher is None and self.cfg.backend == "headless":
            from .headless import HeadlessConfig, HeadlessLauncher
            launcher = HeadlessLauncher(HeadlessConfig(**(self.cfg.headless or {})))
        if launcher is not None and getattr(launcher, "backend", "retail") != self.cfg.backend:
            raise ValueError(f"launcher backend {getattr(launcher, 'backend', 'retail')!r} != EnvConfig.backend "
                             f"{self.cfg.backend!r}")
        self.launcher = launcher
        from .branch_assets import install
        fc = self.cfg.features
        if fc.reach and (fc.camera == "world") != bool(self.cfg.world_actions):
            # the reach block's 8 directions are the arrow keys under FeatureExtractor.action_quarter: world actions
            # need the world frame (quarter 0), camera-relative keys a camera frame (the Cam_OrientRef quarter)
            raise ValueError(f"FeatureConfig.reach with camera {fc.camera!r} and world_actions={self.cfg.world_actions}: "
                             "the reach directions would not be the action directions (world actions need "
                             "camera='world', camera-relative actions a camera frame)")
        self._branch_temp = install(self.cfg.branch_assets, self.cfg.level, fc.path_sha256, fc.branch_sha256)
        self.branch_rules = bundle_rules(self.cfg.branch_assets) if self._branch_temp else DEFAULT_RULES
        if self.cfg.branch_assets and self.cfg.branch_assets.get("version") == 2:
            for start in self.cfg.rest_starts or []:
                if len(start) == 3 and isinstance(start[2], dict):
                    rest_member(self.cfg.branch_assets, start[2].get("id"))
        if uses_branch_move(self.cfg):
            if (not self.cfg.start_check or not self.cfg.start_record or self.cfg.branch_assets is None
                    or self.cfg.branch_assets.get("version") != 2):
                raise ValueError("moving branch starts require a TAS record, start_check and pinned v2 branch assets")
            fixed = self.cfg.start_rest or {}
            if (fixed.get("branch_move") and (not fixed.get("branch_id") or fixed.get("branch_s") is None
                    or not self.cfg.start_frame or self.cfg.start_frame <= 0)):
                raise ValueError("fixed moving start requires a branch member, station and positive prefix")
        self.fx = FeatureExtractor(fc, branch_root=self._branch_temp.name if self._branch_temp else None)
        from .pickups import PickupState
        # score / life balls still present: tracked only where something reads them (the pickup block, the pickup
        # shaping); otherwise empty, so other runs never need the start record's level data
        rc = self.cfg.reward
        want = bool(fc.pickups or (rc is not None and getattr(rc, "pickup_shaping", 0.0)))
        self.pickups = PickupState(self.cfg.level if want else 0)
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
        self.branch_rest_failures = 0
        self._ep_steps = 0
        self._ep_frames = 0
        self._ep_pause_frames = 0
        self._ep_start = 0
        self._after_death = False  # the last episode ended with BallOff and the process continues
        self._last_info: dict = {}
        c = self.cfg
        if c.start_record and c.prefix:
            raise ValueError("start_record and prefix are exclusive")
        self.starts = (TasStarts(c.start_record, camera_tol=c.camera_quarter_tol, camera_grace_ms=c.camera_turn_grace_ms)
                       if c.start_record else None)
        self._own_cam = None
        if c.own_camera:
            if not c.world_actions or c.camera_actions or c.features.camera != "world":
                raise ValueError("own_camera needs world_actions, no camera_actions and the 'world' feature frame")
            if self.starts is not None:
                from . import own_camera
                self._own_cam = own_camera.spawn_quarter(self.starts)
        if self.starts is not None and c.exclude_starts:
            v = self.starts.valid
            keep = np.ones(len(v), bool)
            for lo, hi in c.exclude_starts:
                keep &= ~((v >= int(lo)) & (v < int(hi)))
            # exclude: curriculum / rehearsal draws skip these frames; at-rest starts with a drive target may use them
            (self.starts.exclude if hasattr(self.starts, "exclude") else self.starts.restrict)(keep)
        if self.starts is not None and self.starts.level != c.level:
            raise ValueError(f"start record {c.start_record} is level {self.starts.level}, not {c.level}")
        self.curriculum = None
        if self.starts is not None and c.curriculum is not None:
            self.curriculum = ReverseCurriculum(self.starts, CurriculumConfig(**c.curriculum),
                                                valid=ordinary_start_frames(self.starts, self.cfg.branch_assets))
        self._start_rng = np.random.default_rng(c.seed)
        self._copies = [c.copy] + ([c.warm_copy] if c.warm_pool else [])
        if c.warm_pool and (not c.warm_copy or c.warm_copy == c.copy):
            raise ValueError("warm_pool needs a second runtime copy (warm_copy)")
        self._copy_i = 0
        self._warm = None  # (thread, result box) of the pre-launched process
        self._start: dict = {}  # how the current episode started (info["start"])
        self._entry_context_next = "ordinary"
        self._rest_next = False  # the drawn start is an at-rest start (rest_starts)
        self._source_next = None  # which start list the drawn frame came from (_draw_start)
        self._held = 0  # arrow keys down in the last injected frame (repress_stale_keys)
        self._press_q: dict = {}  # arrow bit -> Cam_OrientRef quarter turn when it went down
        self._repress = c.world_actions if c.repress_stale_keys is None else bool(c.repress_stale_keys)
        self._ms_since_turn = float("inf")  # game ms since the last camera turn press (camera guard)
        self.camera_violations = 0
        # pool tasks (ballrain.pool.engine_chain): the digest every session of this env feeds; reset at each launch, so
        # a process discarded by the hand-over check leaves nothing in it. None = not recorded
        self.engine_digest = None

    # ------------------------------------------------------------------------------ process
    def _out_root(self) -> Path:
        return Path(self.cfg.out_root) if self.cfg.out_root else instances.runtime_dir() / "episodes"

    def _draw_start(self) -> int | None:
        """Start frame of the next process: None (no TAS start states), 0 (from level load) or a valid frame
        of the start record (curriculum draw, or the fixed start_frame snapped to the nearest valid one).
        _source_next: which list the frame came from ({"src": "rehearse" | "rest", "entry": the config entry} or
        {"src": "curriculum"}; None otherwise), info["start"]["source"] of the process (the learner's mastered-state
        anchor, scripts/train_m2.py --anchor-mastered). Recording it draws nothing: the draw order is unchanged."""
        self._rest_next = False
        self._source_next = None
        self._entry_context_next = "ordinary"
        if self.starts is None:
            return None
        c = self.cfg
        entries = list(c.rehearse_starts or []) + list(getattr(c, "rehearse_always", None) or [])
        if entries and c.p_rehearse and self._start_rng.random() < c.p_rehearse:
            j = int(self._start_rng.integers(len(entries)))
            lo, hi = entries[j][0], entries[j][1]
            always = j >= len(c.rehearse_starts or [])
            if c.rehearse_behind_frontier and not always and self.curriculum is not None and int(lo) > 0:
                lo = max(int(lo), int(self.curriculum.frontier))
            if int(lo) <= int(hi):
                self._source_next = {"src": "rehearse", "entry": [int(entries[j][0]), int(entries[j][1])]}
                self._entry_context_next = "drill"
                return self.starts.nearest_valid(int(self._start_rng.integers(int(lo), int(hi) + 1)))
        if c.rest_starts and c.p_rest and self._start_rng.random() < c.p_rest:
            self._entry_context_next = "drill"
            e = c.rest_starts[int(self._start_rng.integers(len(c.rest_starts)))]
            lo, hi = e[0], e[1]
            self._source_next = {"src": "rest", "entry": [dict(x) if isinstance(x, dict) else x for x in e]}
            if len(e) == 3 and isinstance(e[2], dict):
                self._rest_next = e[2]
                if e[2].get("branch_move"):
                    if lo != hi or int(lo) != lo or lo <= 0 or not e[2].get("id"):
                        raise ValueError("moving start requires an exact positive prefix and a branch member")
                    return int(lo)  # never snap a pinned replay prefix
                return self.starts.nearest_any(int(self._start_rng.integers(int(lo), int(hi)+1)))
            self._rest_next = True
            if len(e) >= 5:  # drive to a target first: (x, z) drawn in the jitter box
                j = float(e[4])
                self._rest_next = (float(e[2]) + float(self._start_rng.uniform(-j, j)),
                                   float(e[3]) + float(self._start_rng.uniform(-j, j)))
                if len(e) >= 6:  # [.., max_frames]: this entry's drive limit instead of rest_max_frames
                    self._rest_next += (int(e[5]),)
                # the hand-over frame may lie in an excluded range (exclude_starts): the ball is driven off it anyway
                near = getattr(self.starts, "nearest_any", self.starts.nearest_valid)
                return near(int(self._start_rng.integers(int(lo), int(hi) + 1)))
            return self.starts.nearest_valid(int(self._start_rng.integers(int(lo), int(hi) + 1)))
        if self.curriculum is not None:
            self._source_next = {"src": "curriculum"}
            return self.curriculum.sample(self._start_rng)
        if c.start_rest is not None and (c.start_frame or 0) > 0:  # a given at-rest start (pool task): not snapped
            self._entry_context_next = "drill"
            if c.start_rest.get("branch_s") is not None:  # the stone branch's measured drive (rest_starts lo-hi@branch:s)
                self._rest_next = {"branch_s": float(c.start_rest["branch_s"])}
                if c.start_rest.get("branch_id"):  # a multi-branch member (rest_starts lo-hi@branch:<id>:<s>)
                    self._rest_next["id"] = str(c.start_rest["branch_id"])
                if c.start_rest.get("branch_move"):
                    if not self._rest_next.get("id") or int(c.start_frame) != c.start_frame:
                        raise ValueError("moving start requires an exact positive prefix and a branch member")
                    self._rest_next["branch_move"] = True
                return int(c.start_frame)
            t = c.start_rest.get("target")
            mf = c.start_rest.get("max_frames")
            self._rest_next = True if t is None else ((float(t[0]), float(t[1])) + ((int(mf),) if mf else ()))
            return int(c.start_frame)
        k = self.cfg.start_frame or 0
        return self.starts.nearest_valid(k) if k > 0 else 0

    def _handover_check(self, sess: AgentSession, k: int) -> dict | None:
        """The hand-over state against the dataset's replay of the start record at frame k (ball position
        bit-exact, sector, checkpoints, lives, points, SR double sum). None when it matches. The game does not
        always replay the same inputs the same way: 1 of 354 prefix replays of SR01_1_14_457_264 kept on disk reached
        the first checkpoint one frame late (7818 instead of 7817) with byte-identical inputs, and one M0 reference
        replay diverged too; cause unknown (docs/observations.md 5.1). Only the prefix is checked: frames the agent
        plays (all of a k = 0 run) have no reference, so an agent run can diverge unnoticed; the export check
        (gates.export_diagnosis) tells that apart from a fidelity bug."""
        exp = self.starts.expected(k)
        t = sess.tracker
        o = sess.obs
        if exp is None or o is None or o.ball is None or sess.ended:
            return {"reason": "no state", "ended": sess.ended}
        got = {"frame": int(sess.frames_done), "sector": int(t.sector), "checkpoints": int(t.checkpoints),
               "points": int(t.points), "lives": int(t.lives), "sr_double": float(t.sr_ms or 0.0),
               "pos": np.asarray(o.ball["pos"], np.float64).tolist()}
        return None if got == exp else {"got": got, "expected": exp}

    def _brake(self, p: dict, rules=DEFAULT_RULES) -> None:
        """At-rest start: push against the ball's horizontal velocity until it is slower than rest_speed."""
        from .features import yaw_basis  # noqa: F401  (world_to_camera_keys uses it)
        s, c = p["session"], self.cfg
        n0 = s.frames_done
        while s.frames_done - n0 < (rules.rest_max_brake_frames or c.rest_max_frames) and not s.ended:
            o = s.tracker.obs
            v = (np.asarray(o.ball["speed"], np.float64) + np.asarray(o.ball["speed_change"], np.float64)
                 if o is not None and o.ball is not None else np.zeros(3))  # the velocity as in the features
            vh = np.array([v[0], 0.0, v[2]])
            if np.linalg.norm(vh) < (rules.rest_stop_speed or c.rest_speed):
                break
            w = -vh / np.linalg.norm(vh)
            best = max(ARROWS9[1:], key=lambda a: np.dot(w, np.array([bool(a & tas.KEY_RIGHT) - bool(a & tas.KEY_LEFT), 0.0,
                                                                      bool(a & tas.KEY_UP) - bool(a & tas.KEY_DOWN)])
                                                     / np.linalg.norm([bool(a & tas.KEY_RIGHT) - bool(a & tas.KEY_LEFT),
                                                                       bool(a & tas.KEY_UP) - bool(a & tas.KEY_DOWN)])))
            keys = world_to_camera_keys(best, o.cams["cams"][0]["matrix"]) if o.cams is not None else best
            ev0 = len(s.tracker.events)
            s.act(tas.make_frames([self.delta], [keys]), repeat=rules.rest_control_repeat, stop=EPISODE_STOP)
            if any(c_ in (Ev.BallOff, Ev.LevelFinish, Ev.GameOver) for _, c_, _ in s.tracker.events[ev0:]):
                break
        s.act(tas.make_frames([self.delta], [0]), repeat=1, stop=EPISODE_STOP)  # release the arrows
        if s.ended or s.tracker.flags.ball_off or not s.tracker.flags.ball_nav:
            try:
                s.stop()
            finally:
                s.close()
            raise RuntimeError(f"at-rest start at frame {p['start'].get('frame')}: the ball was lost while braking")
        p["start"]["kind"] = "tas_rest"
        p["start"]["rest_frames"] = int(s.frames_done - n0)
        p["recorder"].meta.start_state = dict(p["recorder"].meta.start_state or {}, kind="tas_rest",
                                              rest_frames=int(s.frames_done - n0))

    def _prepare(self, copy: str, k: int | None, rest: bool = False) -> dict:
        """_launch with the hand-over check (start_check): a process whose prefix replay did not reach the
        recorded state is stopped (its run directory is kept, marked by a start_mismatch.json) and relaunched."""
        mismatches = []
        for _ in range(max(1, self.cfg.start_retries + 1)):
            p = self._launch(copy, k)
            if not k or not self.cfg.start_check:
                break
            bad = self._handover_check(p["session"], k)
            if bad is None:
                break
            mismatches.append(bad)
            s = p["session"]
            try:
                s.stop()
            finally:
                s.close()
            (p["run_dir"] / "start_mismatch.json").write_text(json.dumps(bad, default=str), encoding="utf-8")
        else:
            raise RuntimeError(f"hand-over at frame {k} of {self.starts.record} differs from the dataset "
                               f"{len(mismatches)} times: {mismatches[-1]}")
        p["start"]["handover_mismatches"] = len(mismatches)
        p["recorder"].meta.start_state = dict(p["recorder"].meta.start_state or {},
                                              handover_mismatches=len(mismatches))
        if rest and k:
            if isinstance(rest, dict):
                try:
                    if rest.get("branch_move"):
                        from .branch_move import replay
                        replay(self, p, k, rest["branch_s"], rest.get("id"))
                    else:
                        self._branch_rest(p, k, rest["branch_s"], key=rest.get("id"))
                except Exception as exc:
                    # The failed simulation is not a valid handover. Launch once without
                    # the drive, preserving the requested TAS frame (and curriculum band).
                    p["session"].close(kill=True)
                    self.branch_rest_failures += 1
                    if self._rules_for_rest(rest).rest_fallback == "error":
                        raise
                    p = self._prepare(copy, k, rest=False)
                    failure = dict(branch_rest_failures=self.branch_rest_failures,
                                   branch_rest_target=rest["branch_s"], branch_rest_error=str(exc))
                    if rest.get("branch_move"):
                        failure.update(kind="branch_moving_fallback", branch_move_fallback=True,
                                       branch_move_member=rest.get("id"), branch_move_station=rest["branch_s"])
                    p["start"].update(failure)
                    p["recorder"].meta.start_state.update(failure)
                except BaseException:
                    p["session"].close(kill=True)
                    raise
                return p
            if isinstance(rest, tuple):
                self._drive(p, rest)
            self._brake(p)
        return p

    def _rules_for_rest(self, rest):
        if self.cfg.branch_assets and self.cfg.branch_assets.get("version") == 2:
            _, member = rest_member(self.cfg.branch_assets, rest.get("id"))
            return bundle_rules(member)
        return self.branch_rules

    def _branch_rest(self, p, k, station, key=None):
        """Replay the measured drive through the trafo, then close the loop on an arc target.
        Never synthesize a transform, teleport, or accept the wrong material / surface.
        Use a private extractor: warm-preparation can run concurrently with training.
        """
        from .path import asset_path, file_sha256
        cfg = self.cfg.features
        if self._branch_temp is None:
            raise ValueError('branch rest start needs distributed assets')
        root = Path(self._branch_temp.name)
        s = p['session']
        fx = FeatureExtractor(cfg, branch_root=root)
        fx.reference_path(s.tracker)
        branch = fx.branch
        if branch is None:
            raise ValueError('no matching branch at handover')
        multi = None
        if hasattr(branch, 'by_id'):
            multi = branch
            key, _ = rest_member(self.cfg.branch_assets, key)
            entry, branch = multi.by_id[key]
            root = root / key
        with np.load(root / 'drive.npz', allow_pickle=False) as drive:
            if (str(drive['record_sha256']) != p['start'].get('record_sha256') or
                    str(drive['branch_sha256']) != file_sha256(asset_path(self.cfg.level, root))):
                raise ValueError('drive replay does not match TAS recording or branch asset')
            samples = drive['samples']
            if k != int(drive['prefix_frame']) or not samples[:, 1].min() <= station <= samples[:, 1].max():
                raise ValueError('prefix frame or target outside the recorded branch drive')
            offset = int(samples[np.argmin(abs(samples[:, 1] - station)), 0])
            frames = drive['frames'][:offset]
            phase = 0
            required = branch.ball_type
            required_sector = branch.sector
            if multi and not entry.get('legacy'):
                phase = max(i for i, piece in enumerate(entry['phases'])
                            if piece['local_s'][0] <= station <= piece['local_s'][1])
                condition = entry['phases'][phase]['when']
                required = condition['ball']
                seg = branch.alt._segments(station, station)[0]
                required_sector = condition.get('sector', int(branch.alt.seg_sector[seg]))
        for block in range(0, len(frames), 512):
            s.act(frames[block:block+512], stop=(Ev.BallOff, Ev.GameOver, Ev.LevelFinish))
            if s.ended or s.tracker.flags.ball_off:
                raise RuntimeError('ball lost during branch drive replay')
        target = branch.alt.point_at(station)
        if s.tracker.ball_type() != required or s.tracker.sector != required_sector:
            raise RuntimeError('branch drive did not produce the required ball type and sector')
        rules = branch.rules
        self._drive(p, target[[0, 2]], rules)
        self._brake(p, rules)
        if multi:
            multi.restore(key, phase, station)
        _, fi = fx.compute(s.tracker)
        o = s.tracker.obs.ball
        if ((rules.rest_require_support and not fi.get('supported')) or not fi.get('path', {}).get('branch') or
                np.linalg.norm(o['pos'] - target) > rules.rest_position_tolerance or
                np.linalg.norm(fi['vel'][[0, 2]]) > (rules.rest_stop_speed or self.cfg.rest_speed)):
            raise RuntimeError('branch rest target not reached at rest on the correct surface')
        if multi:
            if (s.tracker.ball_type() != required or s.tracker.sector != required_sector
                    or fi['path'].get('branch_id') != key or fi['path'].get('phase') != phase):
                raise RuntimeError('branch phase mismatch')
            p['start'].update(branch_id=key, phase=phase)
            p['recorder'].meta.start_state.update(branch_id=key, phase=phase)
        p['start']['branch_s'] = station
        p['recorder'].meta.start_state.update(branch_s=station)

    def _drive(self, p: dict, target, rules=DEFAULT_RULES) -> None:
        """At-rest start with a target (rest_starts entry [lo, hi, x, z, jitter]): push the ball with world arrows towards
        (x, z), aiming at a velocity proportional to the remaining distance (capped), until it is within 1 unit and
        slower than rest_speed or rest_max_frames passed; _brake then stops it there."""
        s, c = p["session"], self.cfg
        n0 = s.frames_done
        tgt = np.array([target[0], target[1]], np.float64)
        dirs = [(a, np.array([bool(a & tas.KEY_RIGHT) - bool(a & tas.KEY_LEFT), bool(a & tas.KEY_UP) - bool(a & tas.KEY_DOWN)],
                             np.float64)) for a in ARROWS9[1:]]
        max_frames = int(target[2]) if len(target) >= 3 else (rules.rest_max_drive_frames or c.rest_max_frames)
        while s.frames_done - n0 < max_frames and not s.ended:
            o = s.tracker.obs
            if o is None or o.ball is None:
                break
            pos = np.asarray(o.ball["pos"], np.float64)[[0, 2]]
            v = (np.asarray(o.ball["speed"], np.float64) + np.asarray(o.ball["speed_change"], np.float64))[[0, 2]]
            d = tgt - pos
            if np.linalg.norm(d) < rules.drive_arrival_radius and np.linalg.norm(v) < (rules.rest_stop_speed or c.rest_speed):
                break
            want = d * rules.drive_gain
            if np.linalg.norm(want) > rules.drive_speed_cap:
                want *= rules.drive_speed_cap / np.linalg.norm(want)
            w = want - v
            keys = 0
            if np.linalg.norm(w) > rules.drive_deadband:
                best = max(dirs, key=lambda kd: float(np.dot(w, kd[1])) / np.linalg.norm(kd[1]))[0]
                keys = world_to_camera_keys(best, o.cams["cams"][0]["matrix"]) if o.cams is not None else best
            ev0 = len(s.tracker.events)
            s.act(tas.make_frames([self.delta], [keys]), repeat=rules.rest_control_repeat, stop=EPISODE_STOP)
            if any(c_ in (Ev.BallOff, Ev.LevelFinish, Ev.GameOver) for _, c_, _ in s.tracker.events[ev0:]):
                break
        p["start"]["rest_target"] = [float(tgt[0]), float(tgt[1])]
        p["start"]["drive_frames"] = int(s.frames_done - n0)

    def _launch(self, copy: str, k: int | None) -> dict:
        """Launch one game process on `copy` and bring it to the player's control: the TAS prefix (k > 0) is
        replayed by the mod, then the intro (StartLevel, the level-1 Q press while CamNav is not active,
        BallNavActive) runs as from level load. Thread-safe (the warm pool calls it in a worker thread)."""
        c = self.cfg
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"L{c.level:02d}-{c.mode}-{copy}-{stamp}-{os.getpid()}-{next(_RUN_COUNTER)}"
        run_dir = self._out_root() / copy / run_id
        prefix = Path(c.prefix) if c.prefix else None
        start_meta = None
        if k:
            run_dir.mkdir(parents=True, exist_ok=True)
            prefix = run_dir / "prefix.bin"
            tas.save_raw(prefix, self.starts.prefix(k))
        if self.starts is not None:
            start_meta = {"kind": "tas", "record": self.starts.record,
                          "record_sha256": self.starts.meta.get("record_sha256"), "frame": int(k or 0)}
        sc = SessionConfig(level=c.level, out_dir=run_dir, copy=copy, timeout_ms=c.timeout_ms,
                           max_frames=c.max_process_frames, body_radius=c.body_radius, sr_tail=c.sr_tail,
                           skip_render=c.skip_render, prefix=prefix, trace=c.trace,
                           obs_timeout_s=c.obs_timeout_s, hello_timeout_s=c.hello_timeout_s, launch=c.launch)
        prefix_meta = None
        if prefix is not None:
            data = prefix.read_bytes()
            prefix_meta = {"path": prefix.name if k else str(prefix), "frames": len(data) // tas.FRAME_SIZE,
                           "sha256": hashlib.sha256(data).hexdigest()}
        env_options = {"action_repeat": c.action_repeat, "camera_actions": c.camera_actions,
                       "world_actions": c.world_actions, "repress_stale_keys": self._repress, "delta_ms": c.delta_ms}
        rec = RunRecorder(run_dir, run_id, c.level, mode=c.mode, seed=c.seed, copy=copy, prefix=prefix_meta,
                          backend=c.backend,
                          env_options=env_options,
                          start_state=start_meta, feature_version=self.cfg.features.version,
                          feature_config=self.cfg.features.to_dict(), reward_config=self.cfg.reward_config().to_dict())
        t0 = time.perf_counter()
        sess = AgentSession(sc, self.launcher)
        if self.engine_digest is not None:
            self.engine_digest.reset()
            sess.digest = self.engine_digest
        sess.start()  # closes (kills) its process on failure
        t1 = time.perf_counter()
        rec.meta.mod_version = (sess.hello or {}).get("mod_version", "")
        rec.meta.ball_order = getattr(sess.handle, "ball_order", None)  # headless: the contact order it ran with
        try:
            self._intro(sess)
        except BaseException:
            sess.close(kill=True)
            raise
        t2 = time.perf_counter()
        st = sess.timing
        timing = {"k": int(k or 0), "launch_s": st.get("hello", t1) - st.get("launch", t0),
                  "handover_s": st.get("first_obs_raw", t1) - st.get("hello", t0),
                  "handover_python_s": t1 - st.get("first_obs_raw", t1), "intro_s": t2 - t1, "total_s": t2 - t0,
                  "handover_frames": int(len(sess.prefix_frames)) if sess.prefix_frames is not None else 0,
                  "control_frame": int(sess.frames_done)}
        rec.meta.timing = timing
        start = dict(start_meta or {"kind": "level", "frame": 0})
        start.update(sr_ms=sess.tracker.sr_ms, control_frame=int(sess.frames_done), new_process=True,
                     copy=copy, timing=timing)
        return {"session": sess, "recorder": rec, "run_dir": run_dir, "start": start}

    def _pickups_start(self, start: dict) -> None:
        """A new process: what the start record's prefix collected before the hand-over frame is gone already."""
        pk = self.pickups
        if not pk or self.session is None:
            return
        k = int(start.get("frame") or 0)
        got = []
        if k > 0 and self.starts is not None and start.get("record") == self.starts.record:
            from .pickups import record_pickups
            got = [n for f, n in record_pickups(self.starts.dir, self.cfg.level) if f <= k]
        pk.start(self.session.tracker, got)
        self.session.tracker.pickups = pk

    def _pickup_dist(self):
        """Distance to the nearest score / life ball still present in the current sector (None: none / no ball)."""
        pos = self._ball_pos()
        if not self.pickups or pos is None:
            return None
        near = self.pickups.nearest(pos, self.session.tracker.sector, 1)
        return near[0][0] if near else None

    def _ball_pos(self):
        o = self.session.tracker.obs if self.session is not None else None
        b = o.ball if o is not None else None
        return np.asarray(b["pos"], np.float64) if b is not None and b["found"] else None

    def _start_process(self):
        c = self.cfg
        if self._warm is not None:
            th, box = self._warm
            tj = time.perf_counter()
            th.join()
            self._warm = None
            if "prepared" in box:
                box["prepared"]["start"]["adopt_wait_s"] = time.perf_counter() - tj
            if "error" in box:
                raise box["error"]
            prepared = box["prepared"]
        else:
            k = self._draw_start()
            source = self._source_next
            prepared = self._prepare(self._copies[self._copy_i], k, self._rest_next)
            if source is not None:
                prepared["start"]["source"] = source
            prepared["entry_context"] = self._entry_context_next
        self.fx.set_branch_entry_context(prepared["entry_context"])
        self.session, self.recorder = prepared["session"], prepared["recorder"]
        self.run_dir = prepared["run_dir"]
        self._start = prepared["start"]
        self._copy_i = self._copies.index(prepared["start"]["copy"])
        self._init_held(prepared["start"])
        self._pickups_start(prepared["start"])
        if c.warm_pool:
            self._launch_warm()

    def _launch_warm(self):
        """Pre-launch the next process on the other copy while this one runs (its start frame is drawn now,
        one process ahead of curriculum updates)."""
        nxt = self._copies[(self._copy_i + 1) % len(self._copies)]
        k = self._draw_start()
        rest, source = self._rest_next, self._source_next
        rest = self._rest_next
        context = self._entry_context_next
        box: dict = {}

        def run():
            try:
                box["prepared"] = self._prepare(nxt, k, rest)
                if source is not None:
                    box["prepared"]["start"]["source"] = source
                box["prepared"]["entry_context"] = context
            except BaseException as e:  # re-raised when the process is adopted
                box["error"] = e

        th = threading.Thread(target=run, name=f"ballrain-warm-{nxt}", daemon=True)
        th.start()
        self._warm = (th, box)

    def _drop_warm(self):
        if self._warm is None:
            return
        th, box = self._warm
        th.join()
        self._warm = None
        p = box.get("prepared")
        if p is not None:
            s = p["session"]
            try:
                s.stop()
            finally:
                s.close()

    def _idle(self):
        return tas.make_frames([self.delta], [self.cfg.idle_keys])

    def _intro(self, s: AgentSession):
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

    # ------------------------------------------------------------------------------ start states
    def set_start_state(self, state: dict) -> None:
        """Curriculum state from the trainer (ReverseCurriculum.state_dict()); see ballrain.curriculum."""
        if self.curriculum is None:
            raise RuntimeError("no curriculum configured (EnvConfig.start_record + curriculum)")
        self.curriculum.load_state_dict(state)

    def get_start_state(self) -> dict | None:
        return self.curriculum.state_dict() if self.curriculum is not None else None

    def set_start_frame(self, k: int | None) -> None:
        """Fixed start frame for the next processes (without a curriculum); 0 / None = from level load."""
        self.cfg.start_frame = k

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
        rec.meta.trace_hash = s.trace_hash
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
            f0 = s.frames_done
            s.run_until(self.cfg.idle_keys, ["BallNavActive", "GameOver", "LevelFinish"], self.delta,
                        limit=self.cfg.respawn_limit_frames)
            self._ms_since_turn += (s.frames_done - f0) * float(self.delta)
            self._set_held(self.cfg.idle_keys)
            self._invalidate_held()  # the game shuts every key force down at death: a held key needs a new press
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
        else:  # a respawn in the same process (per-life episodes)
            prev = self._start
            if self.pickups:
                self.pickups.respawn(self.session.tracker)
            self._start = {"kind": "respawn", "frame": int(self.session.frames_done),
                           "sr_ms": self.session.tracker.sr_ms, "new_process": False, "copy": prev.get("copy"),
                           "record": prev.get("record"), "process_start_frame": prev.get("process_start_frame")}
        if self._start.get("new_process"):
            self._start["process_start_frame"] = self._start.get("frame")
        s = self.session
        self.recorder.begin_life(s.frames_done)
        self._ep_steps = self._ep_frames = self._ep_pause_frames = 0
        self._ep_start = s.frames_done
        self.fx.reset_episode()  # path projection state: re-acquired at every episode start
        if self._start.get("branch_id"):
            self.fx.reference_path(s.tracker)
            self.fx.branch.restore(self._start["branch_id"], self._start["phase"], self._start["branch_s"])
        obs, finfo = self.fx.compute(s.tracker)
        self._apply_own_camera(obs)
        if self.reward.cfg.lane_weight and self.fx.path is not None:
            self.reward.check_lane_window(self.fx.path.length)
        self.reward.begin(s.tracker, self._arc(finfo), pickup_dist=self._pickup_dist())
        info = self._info(finfo, [])
        return obs, info

    def _apply_own_camera(self, obs) -> None:
        if self._own_cam is not None:
            from . import own_camera
            own_camera.write(obs, self.fx.slices(), self._own_cam)

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
            if not isinstance(a, int):
                self._held, self._press_q = 0, {}  # Esc / collect end with the arrows released
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
                if self.cfg.world_actions and a in ARROWS9:
                    cams = s.tracker.obs.cams if s.tracker.obs is not None else None
                    if cams is not None:
                        keys = world_to_camera_keys(a, cams["cams"][0]["matrix"])
                if s.tracker.flags.paused and self.cfg.mask_keys_while_paused:
                    keys = self.cfg.pause_keys
                if a in CAMERA_ACTIONS:
                    self._ms_since_turn = -float(self.delta) * self.cfg.action_repeat  # counts from the press
                stale = self._stale(keys) if self._repress else 0
                if stale and self.cfg.action_repeat > 1:
                    extra["repressed"] = int(stale)
                    R = self.cfg.action_repeat
                    s.act(tas.make_frames([self.delta] * R, [keys & ~stale] + [keys] * (R - 1)), repeat=1,
                          stop=EPISODE_STOP)
                    done = s.frames_done - n0
                    if done >= 1:
                        self._set_held(keys & ~stale)
                    if done >= 2:
                        self._set_held(keys)
                else:
                    s.act(tas.make_frames([self.delta], [keys]), repeat=self.cfg.action_repeat, stop=EPISODE_STOP)
                    if s.frames_done > n0:
                        self._set_held(keys)
            if (self.cfg.auto_collect and not s.ended and not paused0 and not s.tracker.flags.paused
                    and s.tracker.followers() > 0 and self._esc_allowed()
                    and not {c for _, c, _ in s.tracker.events[ev0:]} & {Ev.BallOff, Ev.LevelFinish, Ev.GameOver}):
                n1 = s.frames_done
                hold = (self._held & ARROW_MASK) if self.cfg.collect_hold_arrows else 0
                extra["auto_collect"] = self._collect(hold)
                self._held, self._press_q = 0, {}  # the macro ends with the arrows released (pausing shuts them down)
                if hold and extra["auto_collect"].get("unpause") == "unpaused":
                    self._set_held(hold)  # pressed again after the unpause toggle: held, with the current camera
                self._ep_pause_frames += s.frames_done - n1
            elif paused0 or s.tracker.flags.paused:
                self._ep_pause_frames += s.frames_done - n0
            if paused0 and not s.tracker.flags.paused:
                self._invalidate_held()  # pausing shuts the key forces down
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
        self._ms_since_turn += (s.frames_done - n0) * float(self.delta)
        cam_bad = not terminated and not s.ended and self._camera_off_quarter()
        if cam_bad:
            terminated, truncated = True, False
        if self.pickups:
            extra["pickups_new"] = self.pickups.update(s.tracker, self._ball_pos())
        obs, finfo = self.fx.compute(s.tracker)
        self._apply_own_camera(obs)
        p = finfo.get("path")
        r, terms = self.reward.step(s.tracker, self._arc(finfo), died, finished, terminated,
                                    rebase=bool(p and p.get("jump")), supported=finfo.get("supported", False),
                                    pickup_dist=self._pickup_dist(),
                                    s3_features=finfo if self.cfg.level == 2 else None,
                                    lane_features=finfo if self.reward.cfg.lane_weight else None)
        if cam_bad:  # no credit for the episode: its return becomes -death_penalty
            pen = -self.reward.total - self.reward.cfg.death_penalty
            terms = dict(terms, camera_off_quarter=pen)
            r += pen
            self.reward.total += pen
            self.camera_violations += 1
        info = self._info(finfo, new, terms)
        info.update(extra)
        if error:
            info["error"] = error
        if terminated or truncated:
            reason = ("camera_off_quarter" if cam_bad else "BallOff" if died else "LevelFinish" if finished
                      else "GameOver" if game_over else "truncated")
            if cam_bad:
                self.recorder.meta.invalid = "camera_off_quarter"
                self.recorder.meta.notes.append(f"camera_off_quarter at frame {s.frames_done}: Cam_OrientRef "
                                                f"{self._camera_dev():.3g} rad off a quarter turn, no turn in progress")
                info["camera_off_quarter"] = True
            self.recorder.end_life(s.frames_done, reason, s.tracker, self._ep_steps, self.reward.total)
            info["episode_end"] = reason
            self._after_death = died and self.cfg.per_life and not s.ended
            if not self._after_death:
                info["run_meta"] = str(self._finish_process(reason))
                info["framelog_match"] = self.last_run_ok
        return obs, float(r), terminated, truncated, info

    # ------------------------------------------------------------------------------ held arrow keys
    def _cam_quarter(self) -> int | None:
        """Cam_OrientRef yaw in quarter turns (0..3), None without a CAMS section."""
        o = self.session.tracker.obs if self.session is not None else None
        cams = o.cams if o is not None else None
        if cams is None:
            return None
        from .features import yaw_of
        return int(round(yaw_of(cams["cams"][0]["matrix"]) / (np.pi / 2))) % 4

    def _camera_dev(self) -> float:
        o = self.session.tracker.obs if self.session is not None else None
        cams = o.cams if o is not None else None
        if cams is None:
            return 0.0
        from .features import yaw_of
        return abs(math.remainder(yaw_of(cams["cams"][0]["matrix"]), math.pi / 2))

    def _camera_off_quarter(self) -> bool:
        """Camera guard: Cam_OrientRef off a quarter turn (camera_quarter_tol) with no turn in progress."""
        return (self._camera_dev() > self.cfg.camera_quarter_tol
                and self._ms_since_turn > self.cfg.camera_turn_grace_ms)

    def _invalidate_held(self) -> None:
        """Control returns after the game shut the key forces down (death, pause): every held arrow key is stale."""
        for b in ARROW_BITS:
            if self._held & b:
                self._press_q[b] = None

    def _set_held(self, keys: int) -> None:
        """The arrow keys of the frame just injected; a key that went down takes the current camera."""
        keys = int(keys) & ARROW_MASK
        q = self._cam_quarter()
        for b in ARROW_BITS:
            if keys & b and not self._held & b:
                self._press_q[b] = q
            elif not keys & b:
                self._press_q.pop(b, None)
        self._held = keys

    def _stale(self, keys: int) -> int:
        """Arrow keys of `keys` that are held since before the last Cam_OrientRef change."""
        q = self._cam_quarter()
        out = 0
        for b in ARROW_BITS:
            if keys & b and self._held & b and q is not None and self._press_q.get(b, q) != q:
                out |= b
        return out

    def _init_held(self, start: dict) -> None:
        """Held arrow keys at the start of a process: none from level load; after a TAS prefix, the keys of its last
        frame, each with the record's camera when it went down (TasStarts.camera_quarters)."""
        self._held, self._press_q = 0, {}
        self._ms_since_turn = float("inf")
        k = int(start.get("frame") or 0)
        if start.get("kind") == "branch_moving":
            self._set_held(start["branch_move_keys"])
            return
        if start.get("kind") not in ("tas", "branch_moving_fallback") or k <= 0 or self.starts is None:
            return
        fr = self.starts.frames[:k]
        turn = ((fr["keys"] & tas.KEY_SHIFT) != 0) & ((fr["keys"] & (tas.KEY_LEFT | tas.KEY_RIGHT)) != 0)
        if turn.any():
            e = int(np.flatnonzero(turn & ~np.r_[False, turn[:-1]])[-1])
            self._ms_since_turn = float(np.sum(fr["delta"][e:], dtype=np.float64))
        keys = self.starts.frames["keys"][:k]
        last = int(keys[-1]) & ARROW_MASK
        if not last:
            return
        cq = self.starts.camera_quarters()
        for b in ARROW_BITS:
            if last & b:
                j = k - 1
                while j > 0 and keys[j - 1] & b:
                    j -= 1
                self._press_q[b] = int(cq[j]) if j < len(cq) and cq[j] >= 0 else None  # state before frame j
        self._held = last

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

    def _esc(self, hold: int = 0) -> str:
        """One Esc press (down one frame, then released), esc_frames frames, stopping on episode events.
        Returns "paused" / "unpaused" / "no_toggle" from the PauseLevel / UnpauseLevel events.
        hold: arrow keys held on the frames the game still simulates (the toggle lands two frames after the press): the
        frames before the toggle when this Esc pauses, the frames after it when it unpauses."""
        s = self.session
        was = s.tracker.flags.paused
        n = self.cfg.esc_frames
        keys = [tas.KEY_ESC] + [0] * (n - 1)
        if hold:
            keys = keys[:2] + [int(hold)] * (n - 2) if was else [k | int(hold) for k in keys[:2]] + keys[2:]
        s.act(tas.make_frames([self.delta] * n, keys), repeat=1, stop=EPISODE_STOP)
        now = s.tracker.flags.paused
        return "no_toggle" if now == was else ("paused" if now else "unpaused")

    def _collect(self, hold: int = 0) -> dict:
        """Collect macro: Esc, idle frames until no small ball is following (or collect_max_frames), Esc.
        A no-op block (keys 0) when nothing is following, the game is paused or Esc is not allowed.
        hold: arrow keys held on the simulated frames of both Esc actions (collect_hold_arrows)."""
        s = self.session
        t = s.tracker
        out = {"followers_before": t.followers(), "waited": 0}
        if out["followers_before"] == 0 or t.flags.paused or not self._esc_allowed():
            s.act(self._idle(), repeat=self.cfg.action_repeat, stop=EPISODE_STOP)
            out["result"] = "noop"
            return out
        out["pause"] = self._esc(hold)
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
                out["unpause"] = self._esc(hold)
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
                "supported": finfo.get("supported"), "progress_held": self.reward.held,
                "life": len(self.recorder.lives) - 1 if self.recorder else None,
                "run_id": self.recorder.meta.run_id if self.recorder else None, "start": self._start}
        route = finfo.get("path") or {}
        key = route.get("branch_id")
        if "branch_id" in route:
            self.branch_rules = self.fx.branch.by_id[key][1].rules if key else DEFAULT_RULES
            info.update(branch_id=key, phase=route["phase"])
        if "pos" in finfo:
            info["ball_pos"] = tuple(map(float, finfo["pos"]))
        if "branch" in route:
            info.update({key: route[key] for key in ("branch", "local_s", "ref_elapsed_ms", "ref_remaining_ms")})
            if (route["branch"] and t.flags.ball_nav and not t.flags.ball_off and not t.flags.finished
                    and route["dist"] <= self.branch_rules.rest_position_tolerance):
                # Log the physical projection in curriculum coordinates, not credited progress
                # (a high-water mark that can advance while the ball is falling).
                follower = self.fx.branch.by_id[key][1] if key else self.fx.branch
                main_s, member_s = follower.stations
                info["branch_progress_sample"] = float(np.interp(route["local_s"], member_s, main_s))
        elif self.fx.branch is not None:
            # No IVP ball during transforms / finish: never compare the route clock to absolute TAS IGT.
            info.update(self.fx.branch.timing)
        if terms:
            info["reward_terms"] = terms
        return info

    def close(self):
        try:
            if self.session is not None:
                self._finish_process("closed")
        finally:
            self._drop_warm()
        super().close()

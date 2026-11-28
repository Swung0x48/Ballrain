"""Rollout worker: one OS process per game copy, collecting fixed-length chunks with its own policy snapshot.

Why not a vector env: gymnasium / SB3 vector envs step all copies in lockstep, so every new game process (3-5 s:
launch, level load, prefix replay) stalls all the others. Measured on L1 curriculum starts with a random policy,
8 copies: 80 decision steps/s in lockstep (AsyncVectorEnv) against 222 steps/s for 8 independent processes
(docs/observations.md 7.1). Each worker here steps its own BallanceEnv (including its resets) and sends chunks of
`chunk` steps to the learner; after each update the learner broadcasts the new weights, the observation
statistics and the curriculum state, which the worker picks up at its next chunk boundary (at most one update
stale; PPO's ratio uses the behaviour log-probs stored with the chunk).

Per-episode limits (worker-side truncation, bootstrapped like a time limit, logged as end "cap" / "stall"):
- frame cap: cap_mult x the start record's remaining time from the episode's start (TAS start k, or the start of the
  sector for a respawn) + cap_extra_s;
- stall: the projection's arc length has not grown by stall_progress units for stall_s seconds of game time.
A forced truncation ends the game process (env.reset() in the middle of a life starts a new process).

Early respawns (WorkerSpec.p_early_respawn < 1): after a BallOff whose respawn sector the start record enters before
the curriculum frontier, the respawn life is skipped with probability 1 - p_early_respawn: the game process ends and the
next episode is a new curriculum start; the dead episode's record carries respawn_skipped = True.
WorkerSpec.skip_respawn_from_sector > 0 skips every respawn life in that sector or later the same way (no draw).

Run directories: after a process ends its meta (.json) is read for the run statistics; with prune="finish"
non-finishing runs are deleted and finishing runs keep only the .json, .tas and events.txt (disk: a run
directory is about 1.8 MB, mostly level.msg); prune="none" keeps everything (evaluation).
"""
from __future__ import annotations

import json
import queue
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..branch_rules import DEFAULT_RULES

KEEP_FILES_SUFFIX = (".json", ".tas")
KEEP_FILES = ("events.txt", "trace.txt")


@dataclass
class WorkerSpec:
    wid: int
    copy: str
    env: object  # ballrain.env.EnvConfig (copy / seed set by the learner)
    seed: int
    chunk: int = 128
    hidden: tuple = (256, 256)
    activation: str = "tanh"
    cap_mult: float = 2.5
    cap_extra_s: float = 10.0
    stall_s: float = 10.0
    stall_terminal: bool = False  # a stall ends the episode as a failure (terminal, death penalty + refund) instead of a truncation
    stall_extra: float = 0.0  # added to the stall penalty
    # stall_compensate: divide the stall penalty by gamma ** (stall window in steps): the penalty arrives a stall window
    # after the decision to stop, so undiscounted it is cheaper than dying now (the "stall because of death_refund" hole)
    stall_compensate: bool = False
    # stall_speed: while the ball moves faster than this (units / s) the stall clock does not run, so a deliberate run-up
    # (back first, then forward: no new arc progress for a while) is not cut; stall_max_s bounds the time without new
    # progress regardless of speed (no endless oscillation). None = the stall clock only looks at arc progress
    stall_speed: float | None = None
    stall_max_s: float = 15.0
    gamma: float = 0.997
    respawn_stall_s: float | None = None  # stall limit of respawn lives (ball at rest at the checkpoint); None = stall_s
    stall_progress: float = 5.0
    prune: str = "finish"  # "finish" | "none"
    trace_sample: float = 0.0  # fraction of game processes run with an agent-mode trace and kept whatever the outcome
    #                            (export spot-checks of an approximate backend: scripts/sim2real_m2.py spotcheck)
    launcher_factory: object = None  # tests (ballrain.fakegame.FakeLauncher)
    max_reset_failures: int = 6
    # start_tags (the learner's mastered-state anchor, scripts/train_m2.py --anchor-mastered): every episode carries
    # "start_tag" {kind, frame, sector, source} and every chunk "start_tags" (the tags of the lives it holds), "start_ref"
    # (per row: index into start_tags, -1 = none) and "past_cp" (per row: the life had already reached a later sector
    # than it started in when the row was observed). Off: chunks and episodes exactly as before
    start_tags: bool = False
    extra: dict = field(default_factory=dict)
    # route probes: {name: [xmin, xmax, zmin, zmax]} world boxes; every episode records which boxes the ball entered and in
    # which box (if any) it was last seen before a BallOff (e.g. a TAS shortcut vs the detour around it)
    route_probes: dict = field(default_factory=dict)
    # sub-goal gates (training-side shaping, off by default): {name: {"center": [x, y, z], "radius": r, "bonus": b,
    # "min_speed": v (horizontal, ball speed units), "heading": deg (atan2(vz, vx)), "heading_tol": deg}}; the first time in
    # an episode the ball is within radius of center and meets the speed / heading conditions, the step's reward gets
    # +bonus (term "gate_bonus"); each gate pays at most once per episode. Box form instead of center / radius:
    # {"boxes": [[xmin, xmax, zmin, zmax, y_above], ...], "bonus": b}: the ball centre inside any box above its y_above
    gates: dict = field(default_factory=dict)
    # split bonus (off when split_k = 0): at every checkpoint / LevelFinish reached in a life, k * (t_ref - t_agent) /
    # max(t_ref, split_min_ref_s) clipped to [-k, k]; t_agent = IGT of the life since its start or the previous checkpoint,
    # t_ref = the start record's IGT over the same stretch (TAS starts: from the hand-over frame; respawn / at-rest
    # starts: from the checkpoint crossing / the start frame + split_rest_allow_s for starting at rest)
    split_k: float = 0.0
    split_min_ref_s: float = 5.0
    split_rest_allow_s: float = 1.0
    # extra reference time for segments that cross a section the reference line routes differently from the TAS:
    # [[lo_frame, hi_frame, seconds], ...]: a segment that starts before TAS frame lo and ends at or after hi gets +seconds
    split_extra: list = field(default_factory=list)
    # slow split = failure (off when empty): {"lag_s": X, "penalty": P (optional), "min_ref_s": M (optional)}. At a
    # checkpoint / LevelFinish, a split (the same stretch as the split bonus: from the life's hand-over or its previous
    # checkpoint, so inherited lag never counts) with t_agent - t_ref > X ends the life as a failure: reward P (default:
    # a death's, -(death_penalty + death_refund x sector progress paid)), term "slow_fail", end "slow_fail", the game
    # process ends like a terminal stall. Splits with t_ref < M (default 0) are not judged. getattr-read
    slow_fail: dict = field(default_factory=dict)
    # early respawns: after a BallOff whose respawn sector the start record enters before the curriculum frontier (TAS
    # cross frame < frontier: a stretch the curriculum has not reached), the respawn life is played with this probability;
    # otherwise the game process ends and the next episode is a new curriculum start (episode record respawn_skipped).
    # 1.0 = always play it (the old behaviour); respawns at or after the frontier are always played
    p_early_respawn: float = 1.0
    # penalty zones (off when empty): {name: {"box": [xmin, xmax, zmin, zmax], "y_below": y, "penalty": p}}: each time the
    # ball centre enters the box below y_below (e.g. L1's sector-2 pit, where the ball cannot reach the TAS line again),
    # the step's reward gets -penalty (default 1.5, term "zone_penalty") and the life goes on; the zone re-arms once the
    # ball has left it (outside the box or zone_rearm above y_below). Counted per episode (zone_entries)
    zones: dict = field(default_factory=dict)
    # A contact zone instead of a box: {"rail_touches": n, "s": [lo, hi], "penalty": p}: the ball touches at least n rails
    # (the contact block's n_rail_touch, needs FeatureConfig.contacts; a double rail is two rail contacts) while the
    # route s is in [lo, hi]. Once per life (no re-arm: contacts flicker while the ball rolls in a double rail).
    # "terminal": true on a zone (box or contact): entering it also ends the life there as a failure (terminal, no
    # bootstrap, end "zone_terminal"); the penalty still applies. Without the key the life goes on (unchanged).
    # "min_steps": n on a contact zone: it fires only after n consecutive steps with the rail-touch count reached (the
    # count resets on a step without it), so a brush of another rail while hopping between rails does not fire
    zone_rearm: float = 1.0
    # unclimbable-wall push (off when wall_push_rate = 0): while the ball touches a wall (mindist gap <= wall_gap, contact
    # normal |n_y| <= wall_max_ny) whose top is more than wall_min_rise above the ball's lowest point (user: a wall higher
    # than about half a ball can never be climbed; ball radius ~2), the action pushes into it (cosine >= wall_push_cos
    # with the wall's inward normal) and the route progress is below wall_max_progress units / s, each step pays
    # -wall_push_rate x its game seconds (term "wall_push"). Only inside the penalty zones marked "wall_push": true
    # (ball inside the zone's box below its y_below), unless wall_push_everywhere (the old behaviour: on the whole
    # route, where it also fired along the normal section-A line; it was first suspected of the r6 collapse at
    # alt-line s ~360, but the collapse came back with zone-limited wall push: see runtime EXPERIMENTS, 2026-11-17)
    wall_push_rate: float = 0.0
    wall_push_everywhere: bool = False
    # zone_stall_exempt: a life that has entered a penalty zone pays no stall penalty when it ends by stall (it still
    # ends there, terminal with --stall-terminal, and keeps the one-off zone penalty): a pit life costs less than a death
    zone_stall_exempt: bool = False
    # rail-end gate (off when empty): {"s": route arc length, "below": TAS frame (default 4000), "follow_frontier": bool}:
    # with follow_frontier the effective "below" is min(below, curriculum frontier), so band / curriculum starts at or
    # after the frontier always play through once the frontier moves below "below". A 小节-1 life that did not
    # start at or after TAS frame `below` (a TAS / at-rest start below it, a level start, or a 小节-1 respawn) is truncated
    # (bootstrapped, end "gate_truncate", not a failure) once its route arc length reaches s while still in 小节 1. L1 r6:
    # lives started on the early 小节 1 stretch stop at the entry rail end, before the flight onto platform A, so their
    # missed landings (middle-floor falls) cannot bleed into the 4100+ starts' landing; the value there comes from the
    # starts at / after `below`. Read with getattr: a bridge's older pickled WorkerSpec has no such field
    rail_gate: dict = field(default_factory=dict)
    # respawn_level_starts: in a process started from level load (k = 0) every respawn life is played, whatever
    # p_early_respawn says (from-spawn training: the agent's own approach, deaths and respawns included); getattr-read
    respawn_level_starts: bool = False
    # skip_respawn_from_sector (0 = off): a BallOff whose respawn sector (the sector at death) is >= this value never
    # plays its respawn life, in any process (level starts included): the process ends and a new curriculum start
    # follows (respawn_skipped). Set by a bridge for engines that crash on such respawns (remote_worker_m2.py
    # --skip-respawn-from-sector; L2 r8: the 2002-physics SimTool crashes on the 盘点-4 paper-ball respawn, see
    # runtime/m2-train/r8-prep/simtool-2002-respawn-crash.md); getattr-read
    skip_respawn_from_sector: int = 0
    # perturb_starts (off when empty): {"frames": [lo, hi], "steps": [a, b], "p": p}. A TAS start (kind "tas") handed
    # over at a record frame in [lo, hi] is, with probability p, followed by a prefix of n uniformly random actions (n in
    # [a, b] agent steps, drawn from the worker's own rng) played before the policy takes over: the life starts wherever
    # that real engine state lands (beside / off the TAS line, other speed and heading), not at an edited state. The
    # prefix steps are not part of the life (no rows, rewards discarded); the life records its TAS frame and the prefix
    # actions (episode perturb_frame / perturb_actions) so it replays exactly. A prefix that ends the life (death,
    # truncation) is dropped and a new start is drawn. Read with getattr: an older pickled WorkerSpec has no such field
    perturb_starts: dict = field(default_factory=dict)
    # counter randomisation (off when empty): in a game process not started from level load (TAS / at-rest starts and
    # their respawns) every life draws one set of counter-feature values and the observations of that life carry them
    # instead of the true ones: {"lives": {"values": [...], "p": [...]} (replaced), "sr_s": [lo, hi], "points": [lo, hi]
    # (uniform offsets added)}, raw feature units. Level-start processes keep their true counters. L1 r7: TAS starts
    # always see 3 lives, and a from-spawn life with fewer took the 小节-4 spiral corners fatally. getattr-read
    counter_rand: dict = field(default_factory=dict)
    wall_gap: float = 0.1
    wall_max_ny: float = 0.5
    wall_min_rise: float = 2.0
    wall_push_cos: float = 0.5
    wall_max_progress: float = 2.0
    actor_heads: str = "single"


def in_zone(pos, zone: dict, margin: float = 0.0) -> bool:
    """Ball centre inside the zone's box and below its y_below (+ margin)."""
    x0, x1, z0, z1 = zone["box"]
    return bool(x0 <= pos[0] <= x1 and z0 <= pos[2] <= z1 and pos[1] < zone["y_below"] + margin)


def push_direction(keys: int, world_actions: bool, cam_matrix=None) -> np.ndarray:
    """Horizontal unit world direction the arrow keys push (zeros without arrows). world_actions: the keys are world
    directions (UP +z, DOWN -z, RIGHT +x, LEFT -x, ballrain.env); otherwise camera arrows under cam_matrix."""
    from ballrain import tas
    from ballrain.env import key_push_dirs
    if not isinstance(keys, (int, np.integer)):
        return np.zeros(3)
    if world_actions or cam_matrix is None:
        v = np.array([bool(keys & tas.KEY_RIGHT) - bool(keys & tas.KEY_LEFT), 0.0,
                      bool(keys & tas.KEY_UP) - bool(keys & tas.KEY_DOWN)], np.float64)
    else:
        dirs = key_push_dirs(cam_matrix)
        v = sum((dirs[b] for b in (tas.KEY_UP, tas.KEY_DOWN, tas.KEY_LEFT, tas.KEY_RIGHT) if keys & b), np.zeros(3))
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def unclimbable_walls(pos, radius: float, normals, gaps, ray_down, gap: float = 0.1, max_ny: float = 0.5,
                      min_rise: float = 2.0, probe_up: float = 6.5) -> list:
    """Horizontal unit normals (pointing from the wall to the ball) of the walls the ball touches that rise more than
    min_rise above its lowest point. normals / gaps: the mindist entries (ballrain.features.mind_normals, MIND gap);
    ray_down(P) -> (y, tri) of the static geometry (TriangleSet.ray_down). The wall top: a vertical ray just behind the
    contact point, from probe_up above the ball's lowest point (walls taller than probe_up, or movable bodies, which
    are not in the static geometry, are not counted)."""
    pos = np.asarray(pos, np.float64)
    bottom = pos[1] - radius
    out = []
    for n, g in zip(np.asarray(normals, np.float64).reshape(-1, 3), np.asarray(gaps, np.float64).ravel()):
        if g > gap or abs(n[1]) > max_ny:
            continue
        nh = np.array([n[0], 0.0, n[2]])
        ln = np.linalg.norm(nh)
        if ln < 1e-9:
            continue
        nh /= ln
        cp = pos - n * (radius + g)
        probe = cp - nh * 0.25
        probe[1] = bottom + probe_up
        y, _ = ray_down(probe[None, :])
        top = float(y[0])
        if not np.isnan(top) and top > bottom + min_rise:
            out.append(nh)
    return out


def wall_push_hit(push, walls, cos_min: float = 0.5) -> bool:
    """The push direction points into one of the walls (cosine with the wall's inward normal >= cos_min)."""
    if not np.any(push):
        return False
    return any(float(np.dot(push, -w)) >= cos_min for w in walls)


def stall_age(ep, frame, s, branch, speed, progress, pause_speed, hard_limit, window, rules=DEFAULT_RULES):
    """Age in game frames. Never compare odometers across a route switch."""
    if s is not None and (branch != ep.stall_branch or s > ep.mark_s + progress):
        ep.mark_s, ep.mark_frame = s, frame
        ep.stall_branch = branch
    since = frame - ep.mark_frame
    if pause_speed is not None:
        # On the board a supported climb below 3 m/s is normal. Keep the hard
        # no-new-metres limit so lateral motion cannot postpone a stall forever.
        threshold = min(rules.stall_pause_speed, pause_speed) if branch else pause_speed
        if speed is not None and speed > threshold:
            ep.move_frame = frame
        since = min(since, frame - ep.move_frame)
        if frame - ep.mark_frame >= hard_limit:
            since = window
    return since


class _Episode:
    def __init__(self, info: dict, env, version: int, cap_frames: int | None):
        st = info.get("start") or {}
        self.d = {"kind": st.get("kind"), "start_frame": int(st.get("frame") or 0), "copy": st.get("copy"),
                  "start_sector": int(info.get("sector") or 0), "start_sr_ms": float(info.get("sr_ms") or 0.0),
                  "version_start": version, "run_id": info.get("run_id"), "new_process": bool(st.get("new_process")),
                  "cap_frames": cap_frames, "t_wall_start": time.time(),
                  # headless: the contact order of this process (e.g. "retail:<seed>"); None on retail
                  "ball_order": getattr(getattr(getattr(env, "session", None), "handle", None), "ball_order", None)}
        pt = st.get("perturb")
        if pt:  # WorkerSpec.perturb_starts: the TAS frame and the random prefix this life started after
            self.d["perturb_frame"] = int(pt["frame"])
            self.d["perturb_actions"] = "".join(str(int(x)) for x in pt["actions"])
        self.ret = 0.0
        self.steps = 0
        self.max_sector = self.d["start_sector"]
        self.terms: dict = {}
        self.jumps = 0
        self.s0 = env.reward.s
        self.end_branch = (bool(info.get("branch")), info.get("branch_id"))
        self.last_branch_s = info.get("branch_progress_sample")
        self.best_s = env.reward.s if env.reward.s is not None else -np.inf
        self.mark_s, self.mark_frame = self.best_s, 0
        self.stall_branch = bool(info.get("branch"))
        if self.stall_branch and env.branch_rules.stall_odometer == "local_s" and "local_s" in info:
            self.mark_s = info["local_s"]
        if "branch_rest_failures" in st:
            self.d["branch_rest_failures"] = st["branch_rest_failures"]
        if st.get("kind") in ("branch_moving", "branch_moving_fallback"):
            self.d.update(branch_move_member=st.get("branch_id", st.get("branch_move_member")),
                          branch_move_phase=st.get("phase"), branch_move_station=st.get("branch_move_station"),
                          branch_move_offset=st.get("branch_move_offset"),
                          branch_rest_error=st.get("branch_rest_error"))
        self.move_frame = 0
        self.actions = np.zeros(env.action_space.n, np.int64)
        self.probes_seen: set = set()
        self.last_probe = None
        self.gates_hit: set = set()
        self.splits: list = []  # (sector reached, t_agent s, t_ref s, bonus)
        self.max_sector_seen = self.d["start_sector"]
        self.seg = None  # (agent IGT ms at the segment start, TAS IGT ms at the segment start)
        self.zone_in: dict = {}  # penalty zone name -> the ball is inside (not re-armed yet)
        self.zone_counts: dict = {}  # penalty zone name -> entries this episode
        self.zone_terminal = None  # name of a "terminal" zone that fired (the life ends this step)

    def step(self, a, r, info, finfo):
        self.ret += r
        self.steps += 1
        self.actions[a] += 1
        self.max_sector = max(self.max_sector, int(info.get("sector") or 0))
        for k, v in (info.get("reward_terms") or {}).items():
            self.terms[k] = self.terms.get(k, 0.0) + float(v)
        if "branch" in info:
            branch = (bool(info["branch"]), info.get("branch_id"))
            if branch != self.end_branch:
                self.last_branch_s = None
                self.end_branch = branch
        if "branch_progress_sample" in info and info.get("episode_end") != "BallOff":
            self.last_branch_s = info["branch_progress_sample"]
        p = (finfo or {}).get("path")
        if p and p.get("jump"):
            self.jumps += 1

    def gate(self, pos, vel, gates: dict) -> float:
        bonus = 0.0
        for name, g in gates.items():
            if name in self.gates_hit:
                continue
            if "boxes" in g:  # box form: [[xmin, xmax, zmin, zmax, y_above], ...]: ball centre in any box, above its y
                if not any(b[0] <= pos[0] <= b[1] and b[2] <= pos[2] <= b[3] and pos[1] > b[4] for b in g["boxes"]):
                    continue
            elif np.linalg.norm(pos - np.asarray(g["center"], np.float64)) > g.get("radius", 2.0):
                continue
            vh = float(np.hypot(vel[0], vel[2]))
            if vh < g.get("min_speed", 0.0):
                continue
            if "heading" in g:
                hd = np.degrees(np.arctan2(vel[2], vel[0]))
                if abs((hd - g["heading"] + 180) % 360 - 180) > g.get("heading_tol", 180):
                    continue
            self.gates_hit.add(name)
            bonus += float(g.get("bonus", 0.0))
        return bonus

    def probe(self, pos, probes: dict):
        self.last_probe = None
        for name, (x0, x1, z0, z1) in probes.items():
            if x0 <= pos[0] <= x1 and z0 <= pos[2] <= z1:
                self.probes_seen.add(name)
                self.last_probe = name

    def finish(self, info: dict, end: str, env, frames: int) -> dict:
        d = self.d
        # A branch window always returns a nearest point, even far below the route.
        # If no usable branch sample existed, leave s_end blank rather than invent one.
        s_end = self.last_branch_s if self.end_branch[0] else env.reward.s
        x, y, z = info.get("ball_pos", (None, None, None))
        d.update(end=end, ret=self.ret, steps=self.steps, frames=frames, max_sector=self.max_sector,
                 end_sector=int(info.get("sector") or 0), sr_ms_end=float(info.get("sr_ms") or 0.0),
                 finish=end == "LevelFinish", terms=self.terms, jumps=self.jumps,
                 s_start=self.s0, s_end=s_end, t_wall=time.time() - d.pop("t_wall_start"),
                 end_x=x, end_y=y, end_z=z, points_end=info.get("points"), lives_end=info.get("lives"),
                 actions=self.actions.tolist())
        d["success_cp"] = bool(d["finish"] or self.max_sector > d["start_sector"])
        # score / life balls the life left behind: everything still present after a finish, else in the sectors it
        # passed (from its start sector up to the last one it left); None without placements for the level
        pk = getattr(env, "pickups", None)
        if pk:
            hi = len({it[2] for it in pk.items}) if d["finish"] else self.max_sector - 1
            d["pickups_missed"] = pk.missed(range(d["start_sector"], hi + 1))
        else:
            d["pickups_missed"] = None
        d["probes"] = sorted(self.probes_seen | {f"gate:{g}" for g in self.gates_hit})
        d["splits"] = [[int(a), round(b, 3), round(c, 3), round(e, 3)] for a, b, c, e in self.splits]
        d["death_probe"] = self.last_probe if end == "BallOff" else None
        d["zone_entries"] = int(sum(self.zone_counts.values()))
        d["zones"] = ",".join(f"{k}:{v}" for k, v in sorted(self.zone_counts.items()))
        d["respawn_skipped"] = False  # set by the worker when this BallOff's early respawn is not played
        return d


class Worker:
    def __init__(self, spec: WorkerSpec, ctrl_q, data_q):
        import torch
        from .ppo import ActorCritic
        torch.set_num_threads(1)
        self.torch = torch
        self.spec = spec
        self.ctrl_q, self.data_q = ctrl_q, data_q
        self.gen = torch.Generator().manual_seed(spec.seed)
        self._trace_rng = np.random.default_rng(spec.seed + 99991)
        self._respawn_rng = np.random.default_rng(spec.seed + 104729)
        self._counter_rng = np.random.default_rng(spec.seed + 130363)
        self._perturb_rng = np.random.default_rng(spec.seed + 155921)
        self._cr: dict | None = None  # feature slice -> ("set" | "add", value) of the current life (counter_rand)
        self.env = None
        self.fx_info = {}
        self.curriculum_state = None
        self.errors: list = []
        self.pending_runs: list = []
        self._make_env()
        self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n, spec.hidden,
                                 spec.activation, spec.actor_heads)
        self.model.eval()
        # the per-step policy in numpy (ppo.NumpyPolicy; BALLRAIN_TORCH_POLICY=1 keeps torch), refreshed with the weights
        import os as _os
        from .ppo import NumpyPolicy
        self.np_policy = None if _os.environ.get("BALLRAIN_TORCH_POLICY") else NumpyPolicy(self.model)
        self._act_rng = np.random.default_rng(spec.seed + 7919)
        self.version = -1
        self.rms = None
        self.stop = False

    # ---------------------------------------------------------------------------------------- env
    def _make_env(self):
        from ballrain.env import BallanceEnv
        launcher = self.spec.launcher_factory() if self.spec.launcher_factory is not None else None
        self.env = BallanceEnv(self.spec.env, launcher=launcher)
        self.runs_seen = 0
        orig = self.env.fx.compute

        def compute(tracker):
            o, fi = orig(tracker)
            self.fx_info = fi
            self._fx_raw = o
            return self._counter_apply(o), fi
        self.env.fx.compute = compute
        st = self.env.starts
        if st is not None:
            self.lf = st.levelfinish_frame or int(st.valid[-1])
            self.rec_ms = st.sr_final / max(self.lf - int(st.valid[0]), 1) if np.isfinite(st.sr_final) else 1000 / 264
        if self.curriculum_state is not None and self.env.curriculum is not None:
            self.env.set_start_state(self.curriculum_state)

    def _tas_igt(self, k: int) -> float | None:
        """The start record's SR double sum (ms) at frame k (nearest valid frame)."""
        st = self.env.starts
        if st is None:
            return None
        i = int(np.clip(np.searchsorted(st.valid, k), 0, len(st.valid) - 1))
        return float(st.sr_double[i])

    def _tas_cross(self, sector: int) -> float | None:
        """TAS IGT (ms) when the record entered `sector` (sector 1: its first control frame); LevelFinish = sector 0."""
        st = self.env.starts
        if st is None:
            return None
        if sector == 0:
            return float(st.sr_final)
        m = st.sector == sector
        return float(st.sr_double[m][0]) if m.any() else None

    def _tas_cross_frame(self, sector: int):
        st = self.env.starts
        m = st.sector == sector
        return int(st.valid[m][0]) if m.any() else None

    def _contact_zone(self, z: dict) -> bool:
        """Contact zone (WorkerSpec.zones with "rail_touches"): the latest observation's rail-touch count reaches
        z["rail_touches"] while the route s is inside z["s"]."""
        s = self.env.reward.s
        lo, hi = z["s"]
        o = getattr(self, "_fx_raw", None)
        if s is None or not (lo <= s <= hi) or o is None:
            return False
        sl = getattr(self, "_agg_slice", None)
        if sl is None:
            from ballrain.features import layout_slices, CONTACT_AGG_FIELDS
            ag = layout_slices(self.env.fx.cfg)["contact_agg"]
            sl = self._agg_slice = ag.start + CONTACT_AGG_FIELDS.index("n_rail_touch")
        return int(round(float(o[sl]) * 2)) >= int(z["rail_touches"])  # n_rail_touch = min(count, 2) / 2

    def _zone_wall(self, ep, a: int, s_before, frames: int) -> float:
        """Penalty zones and the unclimbable-wall push for the step just taken (WorkerSpec.zones / wall_push_rate);
        returns the reward to add (<= 0) and books the terms on the episode."""
        sp = self.spec
        o = self.env.session.tracker.obs
        if o is None or o.ball is None or not bool(o.ball["found"]):
            return 0.0
        pos = np.asarray(o.ball["pos"], np.float64)
        out = 0.0
        wp_here = bool(getattr(sp, "wall_push_everywhere", False))  # getattr: a WorkerSpec pickled by older code
        for name, z in sp.zones.items():
            if "rail_touches" in z:
                if ep.zone_in.get(name):
                    continue
                hit = self._contact_zone(z)
                need = int(z.get("min_steps", 1) or 1)
                if need > 1:  # only n consecutive steps in contact fire (a brush while changing rails does not)
                    runs = getattr(ep, "zone_run", None)
                    if runs is None:
                        runs = ep.zone_run = {}
                    runs[name] = runs.get(name, 0) + 1 if hit else 0
                    hit = runs[name] >= need
                if hit:
                    pen = -float(z.get("penalty", 1.5))
                    out += pen
                    ep.terms["zone_penalty"] = ep.terms.get("zone_penalty", 0.0) + pen
                    ep.zone_counts[name] = ep.zone_counts.get(name, 0) + 1
                    ep.zone_in[name] = True  # once per life
                    if z.get("terminal"):
                        ep.zone_terminal = name
                continue
            inside = in_zone(pos, z)
            if inside and z.get("wall_push"):
                wp_here = True
            if inside and not ep.zone_in.get(name):
                pen = -float(z.get("penalty", 1.5))
                out += pen
                ep.terms["zone_penalty"] = ep.terms.get("zone_penalty", 0.0) + pen
                ep.zone_counts[name] = ep.zone_counts.get(name, 0) + 1
                ep.zone_in[name] = True
                if z.get("terminal"):
                    ep.zone_terminal = name
            elif ep.zone_in.get(name) and not in_zone(pos, z, sp.zone_rearm):
                ep.zone_in[name] = False
        if sp.wall_push_rate and wp_here and len(o.mind) and "mind_normals" in self.fx_info:
            dt = frames * float(sp.env.delta_ms) / 1000.0
            s = self.env.reward.s
            prog = (s - s_before) / dt if (s is not None and s_before is not None and dt > 0) else 0.0
            keys = self.env.actions[int(a)]
            if dt > 0 and prog < sp.wall_max_progress and isinstance(keys, (int, np.integer)) and keys:
                from ballrain.features import level_geometry
                cam = o.cams["cams"][0]["matrix"] if o.cams is not None else None
                push = push_direction(keys, bool(sp.env.world_actions), cam)
                radius = float(o.ball["extra_radius"]) or 2.0
                walls = unclimbable_walls(pos, radius, self.fx_info["mind_normals"], o.mind["gap"],
                                          level_geometry(self.env.session.tracker).tris.ray_down, gap=sp.wall_gap,
                                          max_ny=sp.wall_max_ny, min_rise=sp.wall_min_rise)
                if walls and wall_push_hit(push, walls, sp.wall_push_cos):
                    pen = -sp.wall_push_rate * dt
                    out += pen
                    ep.terms["wall_push"] = ep.terms.get("wall_push", 0.0) + pen
        return out

    def _stall_penalty(self, ep, win_frames: int) -> float:
        """Reward (<= 0) of a stall that ends the life (WorkerSpec.stall_terminal): a death's penalty plus the
        sector's progress back (+ stall_extra), divided by gamma ** (stall window in steps) with stall_compensate;
        0 with zone_stall_exempt for a life that entered a penalty zone (its zone penalty was the price)."""
        sp = self.spec
        if getattr(sp, "zone_stall_exempt", False) and ep.zone_counts:  # getattr: a WorkerSpec pickled by older code
            return 0.0
        rc = self.env.reward.cfg
        pen = -(rc.death_penalty + rc.death_refund * max(self.env.reward.sector_paid, 0.0) + sp.stall_extra)
        if sp.stall_compensate:
            pen /= sp.gamma ** (win_frames / max(sp.env.action_repeat, 1))
        return pen

    def _skip_respawn(self, end, info) -> bool:
        """True when the respawn life after this episode end is not played (WorkerSpec.p_early_respawn): a BallOff
        with the process continuing, whose respawn sector (the sector at death: the last checkpoint reached) the start
        record enters before the curriculum frontier, and a draw >= p_early_respawn. No draw at p >= 1."""
        p = self.spec.p_early_respawn
        env = self.env
        k = int(getattr(self.spec, "skip_respawn_from_sector", 0) or 0)
        if k > 0 and end == "BallOff" and getattr(env, "_after_death", False) and int(info.get("sector") or 0) >= k:
            return True  # an engine that cannot play this respawn (bridge override): no draw
        if (p >= 1.0 or end != "BallOff" or not getattr(env, "_after_death", False) or env.curriculum is None
                or env.starts is None):
            return False
        if getattr(self.spec, "respawn_level_starts", False) and \
                int((getattr(env, "_start", None) or {}).get("process_start_frame") or 0) == 0:
            return False  # a process started from level load plays all its respawns (from-spawn training)
        cross = self._tas_cross_frame(int(info.get("sector") or 0))
        if cross is None or cross >= int(env.curriculum.frontier):
            return False
        return bool(self._respawn_rng.random() >= p)

    def _gate_hit(self, ep, s, info) -> bool:
        """WorkerSpec.rail_gate: this 小节-1 life (not started at or after TAS frame gate["below"]) has reached the gate's
        route arc length. With a "sector" key (one 小节 or a list) the gate applies to lives started in that 小节 / those 小节
        instead, and holds whatever 小节 they have reached since (a stretch specialist's lives stop past its stretch,
        checkpoint included)."""
        g = getattr(self.spec, "rail_gate", None) or {}
        if not g or s is None:
            return False
        if "sector" in g:  # an int, or a list of 小节 (lives started in any of them)
            secs = g["sector"] if isinstance(g["sector"], (list, tuple)) else (g["sector"],)
            if int(ep.d.get("start_sector") or 0) not in {int(x) for x in secs}:
                return False
        elif int(ep.d.get("start_sector") or 0) != 1 or int(info.get("sector") or 1) > 1:
            return False
        below = int(g.get("below", 4000))
        if g.get("follow_frontier"):  # starts in the curriculum's range [frontier, end] always play through
            cu = getattr(getattr(self, "env", None), "curriculum", None)
            if cu is not None:
                below = min(below, int(cu.frontier))
        k = ep.d.get("start_frame")
        if ep.d.get("kind") != "respawn" and k is not None and int(k) >= below:
            return False
        return float(s) >= float(g["s"])

    def _split_start(self, ep, info):
        """Segment start of a new life for the split bonus (and the slow-split failure)."""
        if not (self.spec.split_k or getattr(self.spec, "slow_fail", None)) or self.env.starts is None:
            return
        if "ref_elapsed_ms" in info:
            ep.seg = (ep.d["start_sr_ms"], info["ref_elapsed_ms"], None)
            return
        kind, k, sec = ep.d["kind"], ep.d["start_frame"], ep.d["start_sector"]
        agent = ep.d["start_sr_ms"]
        if kind == "tas_rest" and k > 0 and self.env.starts.index_of(k) is None:
            ep.seg = None  # an at-rest start handed over in an excluded range (off the TAS's route): no reference time
            return
        if kind == "tas" and k > 0:
            ref = self._tas_igt(k)
        elif kind == "tas_rest":
            ref = self._tas_igt(k) - 1000 * self.spec.split_rest_allow_s
        elif kind == "respawn":
            c = self._tas_cross(sec)
            ref = None if c is None else c - 1000 * self.spec.split_rest_allow_s
        else:
            ref = self._tas_cross(1)  # from level load: the TAS's first control
        k0 = {"respawn": self._tas_cross_frame(sec)}.get(kind, k if k > 0 else int(self.env.starts.valid[0]))
        ep.seg = None if ref is None else (agent, ref, k0)

    def _split_bonus(self, ep, info, finished: bool) -> float:
        """Split bonus of the stretch that ends here (0 with split_k 0); also leaves the stretch's lag t_agent - t_ref
        in ep.split_lag and its t_ref in ep.split_ref (None when no split was measured) for _slow_fail."""
        ep.split_lag = ep.split_ref = None
        if ep.seg is None or not (self.spec.split_k or getattr(self.spec, "slow_fail", None)):
            return 0.0
        sec = 0 if finished else int(info.get("sector") or 0)
        end_ref = info["ref_elapsed_ms"] if "ref_elapsed_ms" in info else self._tas_cross(sec)
        sr = info.get("sr_ms")
        if end_ref is None or sr is None:
            return 0.0
        t_agent = (float(sr) - ep.seg[0]) / 1000.0
        t_ref = (end_ref - ep.seg[1]) / 1000.0
        end_frame = self._tas_cross_frame(sec) if sec else (self.env.starts.levelfinish_frame or 10 ** 9)
        for lo, hi, extra in self.spec.split_extra:
            if ep.seg[2] is not None and end_frame is not None and ep.seg[2] < lo and end_frame >= hi:
                t_ref += float(extra)
        k = self.spec.split_k
        b = float(np.clip(k * (t_ref - t_agent) / max(t_ref, self.spec.split_min_ref_s), -k, k)) if k else 0.0
        ep.splits.append((sec if sec else 99, t_agent, t_ref, b))
        ep.seg = (float(sr), end_ref, end_frame)
        ep.split_lag, ep.split_ref = t_agent - t_ref, t_ref
        return b

    def _slow_fail(self, ep) -> float | None:
        """WorkerSpec.slow_fail: the failure reward (< 0) when the split just measured (_split_bonus) lost more than
        lag_s against the start record, else None. Default reward: a death's (no stall compensation)."""
        sf = getattr(self.spec, "slow_fail", None) or {}
        lag, ref = getattr(ep, "split_lag", None), getattr(ep, "split_ref", None)
        if not sf or lag is None or ref is None or ref < float(sf.get("min_ref_s", 0.0)):
            return None
        if lag <= float(sf["lag_s"]):
            return None
        if "penalty" in sf:
            return -abs(float(sf["penalty"]))
        rc = self.env.reward.cfg
        return -(rc.death_penalty + rc.death_refund * max(self.env.reward.sector_paid, 0.0))

    def _cap_frames(self, info) -> int | None:
        if "ref_remaining_ms" in info:
            budget = (info["ref_elapsed_ms"] + info["ref_remaining_ms"]) * self.spec.cap_mult
            return int((budget + 1000*self.spec.cap_extra_s) / self.spec.env.delta_ms)
        st = self.env.starts
        if st is None:
            return None
        s = info.get("start") or {}
        if s.get("kind") == "respawn":
            m = st.sector == int(info.get("sector") or 0)
            k = int(st.valid[m][0]) if m.any() else int(st.valid[0])
        else:
            k = int(s.get("frame") or 0)
        rem_ms = max(self.lf - k, 0) * self.rec_ms
        return int((self.spec.cap_mult * rem_ms + 1000 * self.spec.cap_extra_s) / self.spec.env.delta_ms)

    def _collect_runs(self):
        fr = self.env.finished_runs
        while self.runs_seen < len(fr):
            p = Path(fr[self.runs_seen])
            self.runs_seen += 1
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"meta {p}: {e}")
                continue
            ss = m.get("start_state") or {}
            self.pending_runs.append({"run_id": m.get("run_id"), "copy": m.get("copy"), "end_reason": m.get("end_reason"),
                                      "sr_ms": m.get("sr_ms"), "start_frame": ss.get("frame", 0),
                                      "frames": m.get("frames"), "lives": len(m.get("lives") or []),
                                      "deaths": sum(1 for x in m.get("lives") or [] if x.get("end_reason") == "BallOff"),
                                      "framelog_match": m.get("framelog_match"),
                                      "handover_mismatches": ss.get("handover_mismatches"),
                                      "invalid": m.get("invalid"), "ball_order": m.get("ball_order"),
                                      "dir": str(p.parent)})
            self._prune(p.parent, m.get("end_reason") == "LevelFinish" and not m.get("invalid"))

    def _prune(self, d: Path, finished: bool):
        if self.spec.prune == "none":
            return
        if (d / "trace.txt").is_file():  # a spot-check sample: keep it whatever the outcome (without the level files)
            finished = True
        try:
            if not finished:
                shutil.rmtree(d, ignore_errors=True)
                return
            for f in d.iterdir():
                if f.is_file() and not (f.suffix in KEEP_FILES_SUFFIX or f.name in KEEP_FILES):
                    f.unlink(missing_ok=True)
        except OSError as e:
            self.errors.append(f"prune {d}: {e}")

    def _counter_draw(self) -> dict | None:
        """counter_rand: the counter values of the life that just started (None: true counters). Off in processes started
        from level load (process_start_frame 0) and when the layout has no such feature."""
        cr = getattr(self.spec, "counter_rand", None) or {}
        st = getattr(self.env, "_start", None) or {}
        if not cr or int(st.get("process_start_frame") or 0) == 0:
            return None
        sl = self.env.fx.slices()
        draw, out = {}, {}
        if "lives" in cr and "lives" in sl:
            vals = [float(v) for v in cr["lives"]["values"]]
            p = cr["lives"].get("p")
            draw["lives"] = float(vals[int(self._counter_rng.choice(len(vals), p=p))])
            out[sl["lives"]] = ("set", draw["lives"])
        for k in ("sr_s", "points"):
            if k in cr and k in sl:
                draw[k] = float(self._counter_rng.uniform(float(cr[k][0]), float(cr[k][1])))
                out[sl[k]] = ("add", draw[k])
        st["counter_rand"] = draw
        return out or None

    def _counter_apply(self, o):
        if self._cr and o is not None:
            o = np.array(o, copy=True)
            for s_, (op, v) in self._cr.items():
                o[s_] = v if op == "set" else o[s_] + v
        return o

    def _perturb_prefix(self, info):
        """WorkerSpec.perturb_starts: None (no prefix: off, not a TAS start, outside the frame range, or not drawn),
        False (the prefix ended the life), else (obs, info) after the prefix, info["start"] carrying the TAS frame and
        the prefix actions ("perturb")."""
        ps = getattr(self.spec, "perturb_starts", None) or {}
        st = info.get("start") or {}
        if not ps or st.get("kind") != "tas":
            return None
        k = int(st.get("frame") or 0)
        lo, hi = (int(v) for v in ps["frames"])
        if not lo <= k <= hi or self._perturb_rng.random() >= float(ps.get("p", 1.0)):
            return None
        a, b = (int(v) for v in ps["steps"])
        n_act = int(self.env.action_space.n)
        acts = [int(x) for x in self._perturb_rng.integers(0, n_act, size=int(self._perturb_rng.integers(a, b + 1)))]
        obs, info2 = None, info
        for x in acts:
            obs, _r, term, trunc, info2 = self.env.step(x)
            if term or trunc:
                return False
        if obs is None:
            return None
        info2 = dict(info2)
        info2["start"] = {**st, "perturb": {"frame": k, "actions": acts}}
        return obs, info2

    def _reset(self, new_process: bool = False):
        """new_process: end the current game process (e.g. a skipped early respawn) and start a new curriculum start."""
        fails = 0
        if self.spec.trace_sample:  # read by the env when it launches the next process
            self.env.cfg.trace = bool(self._trace_rng.random() < self.spec.trace_sample)
        while True:
            try:
                self._cr = None  # the reset observation is computed with the true counters, then the new life's set
                obs, info = self.env.reset(options={"new_process": True} if new_process else None)
                pert = self._perturb_prefix(info)
                if pert is False:  # the prefix ended the life: draw a new start
                    new_process = True
                    self._collect_runs()
                    continue
                if pert is not None:
                    obs, info = pert
                self._cr = self._counter_draw()
                obs = self._counter_apply(obs)
                self._collect_runs()
                return obs, info
            except Exception as e:  # noqa: BLE001  launch failure, hand-over mismatches, crashed game
                fails += 1
                self.errors.append(f"reset: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
                try:
                    self.env.close()
                except Exception as e2:  # noqa: BLE001
                    self.errors.append(f"close: {e2}")
                self._collect_runs()
                if fails >= self.spec.max_reset_failures:
                    raise
                time.sleep(2.0 * fails)
                self._make_env()

    # ---------------------------------------------------------------------------------------- control
    def _apply(self, msg):
        if msg.get("cmd") == "stop":
            self.stop = True
            return
        if "model" in msg:
            self.model.load_state_dict({k: self.torch.as_tensor(v) for k, v in msg["model"].items()})
            if self.np_policy is not None:
                self.np_policy.refresh(self.model)
            self.version = int(msg["version"])
        if "rms" in msg:
            self.rms = msg["rms"]
        if msg.get("curriculum") is not None:
            self.curriculum_state = msg["curriculum"]
            if self.env.curriculum is not None:
                self.env.set_start_state(self.curriculum_state)

    @staticmethod
    def parent_alive() -> bool:
        """False once the process that started this worker (the learner, or a remote bridge) is gone (e.g. killed with
        -9): the worker must not keep launching game processes on its own."""
        import multiprocessing as mp
        p = mp.parent_process()
        return p is None or p.is_alive()

    def _drain(self, block=False):
        try:
            if block:
                while True:
                    try:
                        self._apply(self.ctrl_q.get(timeout=5.0))
                        break
                    except queue.Empty:
                        if not self.parent_alive():
                            self.stop = True
                            return
            while True:
                self._apply(self.ctrl_q.get_nowait())
        except queue.Empty:
            pass

    def _episode(self, info) -> _Episode:
        ep = _Episode(info, self.env, self.version, self._cap_frames(info))
        if self.spec.start_tags:  # how this life started, for the learner's start families (WorkerSpec.start_tags)
            st = info.get("start") or {}
            ep.d["start_tag"] = {"kind": ep.d["kind"], "frame": ep.d["start_frame"], "sector": ep.d["start_sector"],
                                 "source": st.get("source")}
        return ep

    @staticmethod
    def _tag_row(buf: dict, t: int, ep: _Episode, past_cp: bool) -> None:
        """Row t of the chunk belongs to life ep (WorkerSpec.start_tags)."""
        tags = buf["start_tags"]
        ref = getattr(ep, "_tag_ref", None)
        if ref is None or ref[0] is not tags:
            tags.append(ep.d["start_tag"])
            ref = ep._tag_ref = (tags, len(tags) - 1)
        buf["start_ref"][t] = ref[1]
        buf["past_cp"][t] = past_cp

    # ---------------------------------------------------------------------------------------- loop
    def run(self):
        from .ppo import normalize
        sp = self.spec
        self._drain(block=True)
        if self.stop:
            self.env.close()
            return
        n_obs = self.env.observation_space.shape[0]
        obs, info = self._reset()
        ep = self._episode(info)
        self._split_start(ep, info)
        stall_all = int(1000 * sp.stall_s / sp.env.delta_ms)
        stall_rsp = int(1000 * (sp.respawn_stall_s or sp.stall_s) / sp.env.delta_ms)
        stall_max = int(1000 * sp.stall_max_s / sp.env.delta_ms)
        while not self.stop:
            T = sp.chunk
            # "obs" (the normalised rows) is not sent: the learner normalises "raw" with its own statistics and only
            # checks the key (2026-11-26: half the bytes of a chunk, for the runner hosts' links)
            buf = {"obs": None, "raw": np.zeros((T, n_obs), np.float32),
                   "actions": np.zeros(T, np.int64), "logp": np.zeros(T, np.float32), "values": np.zeros(T, np.float32),
                   "rewards": np.zeros(T, np.float32), "terminated": np.zeros(T, bool), "truncated": np.zeros(T, bool),
                   "trunc_values": np.zeros(T, np.float32)}
            if sp.start_tags:
                buf.update(start_tags=[], start_ref=np.full(T, -1, np.int16), past_cp=np.zeros(T, bool))
            episodes = []
            version = self.version
            t0 = time.perf_counter()
            t_reset = 0.0
            frames = 0
            t = 0
            while t < T:
                x = normalize(obs, self.rms)
                if self.np_policy is not None:
                    a, logp, v = self.np_policy.act(x, self._act_rng)
                else:
                    a, logp, v = self.model.act(x, gen=self.gen)
                a = int(a)
                f0 = self.env.session.frames_done if self.env.session is not None else 0
                s_before = self.env.reward.s
                try:
                    obs2, r, term, trunc, info2 = self.env.step(a)
                except Exception as e:  # noqa: BLE001  should not happen (the env truncates on PeerGone / Timeout)
                    self.errors.append(f"step: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
                    if t > 0:
                        buf["truncated"][t - 1] = True
                        buf["trunc_values"][t - 1] = float(v)
                    ep_d = ep.finish(info, "error", self.env, self.env._ep_frames)
                    episodes.append(ep_d)
                    try:
                        self.env.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._make_env()
                    tr = time.perf_counter()
                    obs, info = self._reset()
                    t_reset += time.perf_counter() - tr
                    ep = self._episode(info)
                    self._split_start(ep, info)
                    continue
                frames += int(info2.get("frames_done", f0)) - f0
                past_cp = ep.max_sector > ep.d["start_sector"]  # before this step: the sector the row was observed in
                ep.step(a, r, info2, self.fx_info)
                if (sp.zones or sp.wall_push_rate) and not term and self.env.session is not None:
                    sh = self._zone_wall(ep, a, s_before, int(info2.get("frames_done", f0)) - f0)
                    if sh:
                        r += sh
                        ep.ret += sh
                zone_failed = getattr(ep, "zone_terminal", None) is not None and not term
                if zone_failed:  # a "terminal" zone fired: the life ends here as a failure
                    term, trunc = True, False
                slow_failed = False
                if (sp.split_k or getattr(sp, "slow_fail", None)) and (int(info2.get("sector") or 0) > ep.max_sector_seen
                                                                       or info2.get("episode_end") == "LevelFinish"):
                    sb = self._split_bonus(ep, info2, info2.get("episode_end") == "LevelFinish")
                    if sb:
                        r += sb
                        ep.ret += sb
                        ep.terms["split_bonus"] = ep.terms.get("split_bonus", 0.0) + sb
                    sf = self._slow_fail(ep)
                    if sf is not None:  # too slow over this split: a failure like a death (the life ends here)
                        r += sf
                        ep.ret += sf
                        ep.terms["slow_fail"] = ep.terms.get("slow_fail", 0.0) + sf
                        slow_failed = True
                        term, trunc = True, False
                ep.max_sector_seen = max(ep.max_sector_seen, int(info2.get("sector") or 0))
                if (sp.route_probes or sp.gates) and self.env.session is not None:
                    o = self.env.session.tracker.obs
                    if o is not None and o.ball is not None and bool(o.ball["found"]):
                        bpos = np.asarray(o.ball["pos"], np.float64)
                        if sp.route_probes:
                            ep.probe(bpos, sp.route_probes)
                        if sp.gates and not term:
                            gb = ep.gate(bpos, np.asarray(o.ball["speed"], np.float64)
                                         + np.asarray(o.ball["speed_change"], np.float64), sp.gates)
                            if gb:
                                r += gb
                                ep.ret += gb
                                ep.terms["gate_bonus"] = ep.terms.get("gate_bonus", 0.0) + gb
                end = info2.get("episode_end") if (term or trunc) else None
                if slow_failed:
                    end = "slow_fail"
                if zone_failed:
                    end = "zone_terminal"
                if "branch" in info2 and ep.d.get("cap_branch") != info2["branch"]:
                    ep.d["cap_frames"] = self._cap_frames(info2)
                    ep.d["cap_branch"] = info2["branch"]
                if not (term or trunc):
                    ef = self.env._ep_frames
                    s = self.env.reward.s
                    branch = bool(info2.get("branch"))
                    odometer = info2.get("local_s") if branch and self.env.branch_rules.stall_odometer == "local_s" else s
                    speed = None
                    if sp.stall_speed is not None:
                        o = self.env.session.tracker.obs if self.env.session is not None else None
                        if o is not None and o.ball is not None:
                            velocity = o.ball["speed"]
                            if branch and self.env.branch_rules.stall_speed_source == "speed+speed_change":
                                velocity = np.asarray(velocity) + o.ball["speed_change"]
                            speed = float(np.linalg.norm(velocity))
                    win = stall_rsp if ep.d["kind"] == "respawn" else stall_all
                    since = stall_age(ep, ef, odometer, branch, speed, sp.stall_progress,
                                      sp.stall_speed, stall_max, win, self.env.branch_rules)
                    if self._gate_hit(ep, s, info2):
                        trunc, end = True, "gate_truncate"
                    elif ep.d["cap_frames"] is not None and ef >= ep.d["cap_frames"]:
                        trunc, end = True, "cap"
                    elif s is not None and since >= win:
                        trunc, end = True, "stall"
                        if sp.stall_terminal:  # stuck = a failure like a death: penalty + the sector's progress back
                            pen = self._stall_penalty(ep, stall_rsp if ep.d["kind"] == "respawn" else stall_all)
                            r += pen
                            ep.ret += pen
                            ep.terms["stall_penalty"] = ep.terms.get("stall_penalty", 0.0) + pen
                            term, trunc = True, False
                buf["raw"][t] = obs
                if sp.start_tags:
                    self._tag_row(buf, t, ep, past_cp)
                buf["actions"][t], buf["logp"][t], buf["values"][t] = a, logp, v
                buf["rewards"][t], buf["terminated"][t] = r, term
                buf["truncated"][t] = trunc and not term
                if trunc and not term:
                    buf["trunc_values"][t] = (self.np_policy or self.model).value(normalize(obs2, self.rms))
                t += 1
                if term or trunc:
                    skip = self._skip_respawn(end, info2)
                    ep_d = ep.finish(info2, end or "truncated", self.env, self.env._ep_frames)
                    ep_d["respawn_skipped"] = skip
                    episodes.append(ep_d)
                    tr = time.perf_counter()
                    obs, info = self._reset(new_process=skip)
                    t_reset += time.perf_counter() - tr
                    ep = self._episode(info)
                    self._split_start(ep, info)
                else:
                    obs, info = obs2, info2
            buf["last_value"] = (self.np_policy or self.model).value(normalize(obs, self.rms))
            buf.update(wid=sp.wid, copy=sp.copy, version=version, episodes=episodes, runs=self.pending_runs,
                       backend=getattr(sp.env, "backend", "retail"),
                       errors=self.errors, t_chunk=time.perf_counter() - t0, t_reset=t_reset, frames=frames)
            self.pending_runs, self.errors = [], []
            self.data_q.put(buf)
            self._drain()
            if not self.parent_alive():
                self.stop = True
        self.env.close()
        self._collect_runs()
        if not self.parent_alive():  # nobody reads the queue any more: do not block the exit on it
            self.data_q.cancel_join_thread()
            return
        self.data_q.put({"wid": sp.wid, "closed": True, "runs": self.pending_runs, "errors": self.errors})


def worker_main(spec: WorkerSpec, ctrl_q, data_q):
    w = None
    try:
        w = Worker(spec, ctrl_q, data_q)
        w.run()
    except BaseException as e:  # noqa: BLE001
        data_q.put({"wid": spec.wid, "fatal": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        if w is not None and w.env is not None:
            try:
                w.env.close()
            except Exception:  # noqa: BLE001
                pass

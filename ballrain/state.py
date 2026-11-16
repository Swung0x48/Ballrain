"""Accumulated game state on the Python side: level message contents, data arrays and event-driven
flags. Fed by ballrain.client with every parsed observation; read by features, rewards, recording.

Frame numbering (state-spec.md section 1): obs.frames_done = n means frames 0..n-1 have executed;
an event with number n happened while the frame with index n-1 executed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from . import protocol as P
from .protocol import Ev

# group names (state-spec.md 2.5, 2.6, 2.8)
G_FLOORS = "Phys_Floors"
G_RAILS = "Phys_FloorRails"
G_STOPPER = "Phys_FloorStopper"
G_DEPTH = "DepthTestCubes"
SURFACE_NAMES = {0: "none", 1: "stone", 2: "wood", 3: "metal"}
BALL_NAME_TYPES = {"Ball_Wood": 0, "Ball_Paper": 1, "Ball_Stone": 2}
# score ball (k 0) and its six small balls (k 1..6) of the template or a numbered instance (state-spec.md 2.9)
FOLLOWER_NAME = re.compile(r"^P_Extra_Point_Ball([0-6])(\d{3})?$")
SMALL_BALL_POINTS = 20


class ArrayStore:
    """Current contents of the watched data arrays. Cells arrive as raw bytes (obs ARRS deltas, the
    level message's full ARRS) and are decoded on read with the column type."""

    def __init__(self):
        self.cells: dict = {}  # (array ckid, row, col) -> (col type, bytes)
        self.names: dict = {}  # name -> ckid
        self.columns: dict = {}  # ckid -> [ArrayColumn]
        self.nrows: dict = {}  # ckid -> rows seen

    def define(self, a: P.ArrayDef, seed_cells: bool = True) -> None:
        if a.name:
            self.names[a.name] = a.ckid
        self.columns[a.ckid] = a.columns
        self.nrows[a.ckid] = max(self.nrows.get(a.ckid, 0), len(a.cells))
        if seed_cells:
            for r, row in enumerate(a.cells):
                for c, data in enumerate(row):
                    key = (a.ckid, r, c)
                    if key not in self.cells:  # obs deltas are newer than the level snapshot
                        self.cells[key] = (a.columns[c].type, data)

    def apply(self, cells) -> list:
        """Apply obs ARRS cells; returns [(name, row, col)] of changed cells."""
        changed = []
        for c in cells:
            key = (c.array, c.row, c.col)
            new = (c.col_type, c.data)
            if self.cells.get(key) != new:
                self.cells[key] = new
                changed.append((self.name_of(c.array), c.row, c.col))
            self.nrows[c.array] = max(self.nrows.get(c.array, 0), c.row + 1)
        return changed

    def name_of(self, ckid: int):
        for n, k in self.names.items():
            if k == ckid:
                return n
        return ckid

    def ckid(self, name: str):
        return self.names.get(name)

    def rows(self, name: str) -> int:
        k = self.names.get(name)
        return self.nrows.get(k, 0) if k is not None else 0

    def col_index(self, name: str, col_name: str):
        k = self.names.get(name)
        for i, c in enumerate(self.columns.get(k, [])):
            if c.name == col_name:
                return i
        return None

    def raw(self, name: str, row: int, col: int):
        k = self.names.get(name)
        return None if k is None else self.cells.get((k, row, col))

    def get(self, name: str, row: int, col: int, default=None):
        k = self.names.get(name)
        if k is None:
            return default
        v = self.cells.get((k, row, col))
        if v is None:
            return default
        cols = self.columns.get(k)
        pt = cols[col].param_type if cols and col < len(cols) else None
        return P.decode_cell(v[1], v[0], pt)


class LevelData:
    """Union of the full level message and its increments."""

    def __init__(self):
        self.entities: dict = {}
        self.by_name: dict = {}
        self.groups: dict = {}
        self.ent_groups: dict = {}
        self.phys: dict = {}  # seq -> PhysRecord
        self.phys_by_ent: dict = {}  # ckid -> latest PhysRecord
        self.meshes: dict = {}  # (ckid, mesh) -> Mesh
        self.ivp: dict = {}
        self.depth: list = []
        self.watch: list = []
        self.array_defs: list = []
        self.version = 0  # bumped on every merge (geometry caches key on it)

    def merge(self, lm: P.LevelMessage) -> None:
        for e in lm.entities:
            if e.ckid == 0:
                continue
            self.entities[e.ckid] = e
            if e.name is not None:
                self.by_name.setdefault(e.name, e.ckid)
        for g in lm.groups:
            if g.name is not None:
                old = self.groups.get(g.name)
                if old is not None:
                    for m in old.members:
                        self.ent_groups.get(int(m), set()).discard(g.name)
                self.groups[g.name] = g
                for m in g.members:
                    self.ent_groups.setdefault(int(m), set()).add(g.name)
        for p in sorted(lm.phys, key=lambda p: p.seq):
            self.phys[p.seq] = p
            cur = self.phys_by_ent.get(p.ckid)
            if cur is None or p.seq >= cur.seq:
                self.phys_by_ent[p.ckid] = p
        for m in lm.meshes:
            self.meshes[(m.ckid, m.mesh)] = m
        for o in lm.ivp:
            self.ivp[o.ckid] = o
        if lm.depth:
            self.depth = list(lm.depth)
        if lm.watch:
            self.watch = list(lm.watch)
        self.array_defs.extend(lm.arrays)
        self.version += 1

    # ---- queries
    def name(self, ckid: int):
        e = self.entities.get(int(ckid))
        return e.name if e else None

    def in_group(self, ckid: int, group: str) -> bool:
        return group in self.ent_groups.get(int(ckid), ())

    def contact_id(self, ckid: int) -> int:
        e = self.entities.get(int(ckid))
        return e.contact_id if e else -1

    def is_physicalized(self, ckid: int) -> bool:
        p = self.phys_by_ent.get(int(ckid))
        return p is not None and not p.unphysicalize

    def is_fixed(self, ckid: int) -> bool:
        p = self.phys_by_ent.get(int(ckid))
        return bool(p and p.fixed)

    def meshes_of(self, ckid: int) -> list:
        return [m for (c, _), m in self.meshes.items() if c == int(ckid)]

    def static_entities(self) -> list:
        """Physicalized fixed entities that collide with the player ball (stoppers excluded)."""
        return sorted(c for c, p in self.phys_by_ent.items()
                      if not p.unphysicalize and p.fixed and not self.in_group(c, G_STOPPER))

    def static_key(self) -> str:
        """Content hash of the static collision geometry."""
        h = hashlib.sha256()
        for c in self.static_entities():
            for m in sorted(self.meshes_of(c), key=lambda m: m.mesh):
                h.update(np.asarray(m.matrix, "<f4").tobytes())
                h.update(np.asarray(m.vertices, "<f4").tobytes())
                h.update(np.asarray(m.faces, "<u4").tobytes())
        return h.hexdigest()

    def entity_position(self, name: str):
        k = self.by_name.get(name)
        if k is None:
            return None
        return np.asarray(self.entities[k].matrix[3], np.float64)


@dataclass
class LifeInfo:
    index: int
    start_frame: int  # frames_done when control (BallNavActive) began
    end_frame: int | None = None
    end_reason: str | None = None


@dataclass
class Flags:
    ball_nav: bool = False
    cam_nav: bool = False
    counter: bool = False
    paused: bool = False
    started: bool = False
    finished: bool = False
    game_over: bool = False
    ball_off: bool = False  # since the last BallOff, until the next BallNavActive


class GameTracker:
    """Event- and array-driven summary of the game. One per game process."""

    def __init__(self, level: int):
        self.level_no = level
        self.level = LevelData()
        self.arrays = ArrayStore()
        self.flags = Flags()
        self.events: list = []  # (event number, code, param)
        self.frames_done = 0
        self.obs: P.Observation | None = None
        self.checkpoints = 0
        self.extra_points = 0
        self.life_ups = 0
        self.deaths = 0
        self.levelfinish_frame = None
        self.hs_at_finish = None
        self.points_at_finish = None
        self.lives_at_finish = None
        self.life_bonus_at_finish = None
        self.hs_source = None  # how hs_at_finish was read (see _finish_readout)
        self.hs_arrays_at_finish = None  # Points + Lifes x LifeBonus + LevelBonus of the observation carrying LevelFinish
        self.hs_levelfinish_hook = None  # the mod's snapshot in the LevelFinish hook (SREN), without followers
        self.followers_at_finish = None  # small score balls still following at the end of the SR tail
        self.followers_at_hook = None  # ... at the LevelFinish hook (SREN extension)
        self.watch: dict = {}  # watched ckid -> (visible, position) from WTCH
        self._fslots = None  # ckid -> (instance, k) of the watched score-ball objects
        self._fslots_key = None
        self._trafo = False
        self.last_checkpoint_frame = None  # event number of the last PreCheckpoint
        self.control_start_frame = None  # event number of the first BallNavActive
        self.paused_since = None  # event number of the PauseLevel of the current pause

    # ---- feeding
    def merge_level(self, lm: P.LevelMessage) -> None:
        self.level.merge(lm)
        for a in lm.arrays:
            self.arrays.define(a)

    def update(self, obs: P.Observation) -> list:
        """Apply one observation; returns its events as (number, name, param)."""
        self.obs = obs
        self.frames_done = obs.frames_done
        self.arrays.apply(obs.arrs)
        for w in obs.wtch:
            self.watch[int(w["ckid"])] = (int(w["visible"]), w["pos"])
        out = []
        f = self.flags
        for e in obs.events:
            n, code, param = int(e["frame"]), int(e["code"]), int(e["param"])
            self.events.append((n, code, param))
            out.append((n, P.EVENT_NAMES.get(code, str(code)), param))
            if code == Ev.StartLevel:
                f.started = True
            elif code in (Ev.Unphysicalize, Ev.Physicalize):
                # ball transformation: the old ball is unphysicalized when it starts, the new ball is
                # physicalized when it ends (agent-protocol.md 8.7). A death also unphysicalizes the ball,
                # after BallOff, and is not a transformation.
                if self._is_ball_record(param):
                    self._trafo = (code == Ev.Unphysicalize and f.started and not f.ball_off
                                   and not f.finished)
            elif code == Ev.BallNavActive:
                f.ball_nav, f.ball_off = True, False
                self._trafo = False
                if self.control_start_frame is None:
                    self.control_start_frame = n
            elif code == Ev.BallNavInactive:
                f.ball_nav = False
            elif code == Ev.CamNavActive:
                f.cam_nav = True
            elif code == Ev.CamNavInactive:
                f.cam_nav = False
            elif code == Ev.CounterActive:
                f.counter = True
            elif code == Ev.CounterInactive:
                f.counter = False
            elif code == Ev.PauseLevel:
                f.paused = True
                self.paused_since = n
            elif code == Ev.UnpauseLevel:
                f.paused = False
                self.paused_since = None
            elif code == Ev.BallOff:
                f.ball_off = True
                self._trafo = False
                self.deaths += 1
            elif code == Ev.PreCheckpoint:
                self.last_checkpoint_frame = n
            elif code == Ev.PostCheckpoint:
                self.checkpoints += 1
            elif code == Ev.ExtraPoint:
                self.extra_points += 1
            elif code == Ev.PostLifeUp:
                self.life_ups += 1
            elif code == Ev.LevelFinish:
                f.finished = True
                self._trafo = False
                self.levelfinish_frame = n
            elif code == Ev.GameOver:
                f.game_over = True
        if f.finished and self.hs_arrays_at_finish is None:
            self._finish_readout(obs)
        return out

    def _finish_readout(self, obs: P.Observation) -> None:
        """Official HS (docs/benchmarks.md, "HS readout"): the score the end screen counts, i.e.
        Points + 20 x small balls still following + Lifes x LifeBonus + LevelBonus at the end of the SR tail
        (CounterInactive; the time-point counter has stopped there). Measured with the post-finish probe:
        every follower is credited (+20 to Points) within ~350 frames after LevelFinish, the end-screen
        drain starts ~2900+ frames after it and counts Points + Lifes x LifeBonus + LevelBonus.
        The real mod builds the observation carrying LevelFinish at the end of the SR tail, so its arrays
        and SCOR are the tail-end values. With the diagnostic post-finish window (SREN post_frames > 0)
        the arrays are already drained; then the LevelFinish-hook snapshot plus its follower count is used
        (equal to the tail-end readout on every probed record)."""
        self.hs_arrays_at_finish = self.hs()
        sr, ext = obs.sren, obs.sren_ext
        hook_ok = sr is not None and bool(sr["level_finished"]) and int(sr["hs_valid"]) == 1
        if hook_ok:
            self.hs_levelfinish_hook = int(sr["hs"])
        if ext is not None:
            self.followers_at_hook = int(ext["followers_at_finish"])
        post = int(ext["post_frames"]) if ext is not None else 0
        if post == 0 or not hook_ok:
            self.points_at_finish, self.lives_at_finish = self.points, self.lives
            self.life_bonus_at_finish = self.life_bonus
            self.followers_at_finish = (int(ext["followers_end"]) if ext is not None else self.followers())
            self.hs_source = "tail_end"
        else:
            e = sr["energy"]
            self.points_at_finish, self.lives_at_finish = int(np.int32(e[0])), int(np.int32(e[1]))
            self.life_bonus_at_finish = int(np.int32(e[5]))
            self.followers_at_finish = self.followers_at_hook
            self.hs_source = "levelfinish_hook"
        self.hs_at_finish = (self.points_at_finish + SMALL_BALL_POINTS * self.followers_at_finish
                             + self.lives_at_finish * self.life_bonus_at_finish + self.level_bonus)

    def _is_ball_record(self, seq: int) -> bool:
        rec = self.level.phys.get(int(seq))
        return rec is not None and self.level.name(rec.ckid) in BALL_NAME_TYPES

    # ---- small score balls (state-spec.md 2.9, docs/observations.md)
    def _follower_slots(self) -> dict:
        key = (id(self.level), self.level.version)
        if key != self._fslots_key:
            slots = {}
            for ck, _group in self.level.watch:
                m = FOLLOWER_NAME.match(self.level.name(ck) or "")
                if m:
                    slots[int(ck)] = (m.group(2) or "", int(m.group(1)))
            self._fslots, self._fslots_key = slots, key
        return self._fslots

    def follower_state(self) -> dict:
        """From WTCH: small balls k 1..6 still visible while the score ball k 0 of the same instance is hidden.
        Returns {"total", "per_source": {instance: n}, "positions": [(x, y, z)]}."""
        mains, smalls = {}, []
        for ck, (src, k) in self._follower_slots().items():
            w = self.watch.get(ck)
            if w is None:
                continue
            if k == 0:
                mains[src] = w[0]
            elif w[0]:
                smalls.append((src, w[1]))
        per, pos = {}, []
        for src, p in smalls:
            if mains.get(src) == 0:
                per[src] = per.get(src, 0) + 1
                pos.append(np.asarray(p, np.float64))
        return {"total": sum(per.values()), "per_source": per, "positions": pos}

    def followers(self) -> int:
        """Small score balls still following the player ball: the mod's counter (SCOR, BallrainIO >= 0.3.1),
        otherwise derived from WTCH (follower_state)."""
        o = self.obs
        if o is not None and o.scor_ext is not None:
            return int(o.scor_ext["followers"])
        return self.follower_state()["total"]

    def nearest_follower(self, pos) -> float | None:
        ps = self.follower_state()["positions"]
        if not ps:
            return None
        return float(min(np.linalg.norm(p - np.asarray(pos, np.float64)) for p in ps))

    def frames_since_checkpoint(self) -> int:
        """Frames since the last PreCheckpoint (or since control first started)."""
        ref = self.last_checkpoint_frame if self.last_checkpoint_frame is not None else self.control_start_frame
        return 0 if ref is None else max(0, self.frames_done - ref)

    def paused_frames(self) -> int:
        return 0 if self.paused_since is None else max(0, self.frames_done - self.paused_since)

    # ---- arrays (state-spec.md 2.10)
    @property
    def points(self) -> int:
        return int(self.arrays.get("Energy", 0, 0, 0))

    @property
    def lives(self) -> int:
        return int(self.arrays.get("Energy", 0, 1, 0))

    @property
    def life_bonus(self) -> int:
        return int(self.arrays.get("Energy", 0, 5, 200))

    @property
    def level_bonus(self) -> int:
        v = self.arrays.get("AllLevel", self.level_no - 1, 6, None)
        return int(v) if v is not None else 100 * self.level_no

    def hs(self) -> int:
        return self.points + self.lives * self.life_bonus + self.level_bonus

    def hs_partial(self) -> int:
        """Score potential while playing: Points + Lifes * LifeBonus (the level bonus and the followers
        still pending only at finish). After LevelFinish: the readout values (see _finish_readout)."""
        if self.points_at_finish is not None:
            return self.points_at_finish + self.lives_at_finish * self.life_bonus_at_finish
        return self.points + self.lives * self.life_bonus

    @property
    def sector(self) -> int:
        v = self.arrays.get("IngameParameter", 0, 1, 0)
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = 0
        return v if v > 0 else 1 + self.checkpoints

    @property
    def transforming(self) -> bool:
        """Inside a ball transformation: from the Unphysicalize of the active ball (a Ball_* entity, not after
        a BallOff) to the Physicalize of the new ball. The old ball has no IVP object in between
        (agent-protocol.md 8.7). Not IngameParameter[0,0] (Activetrafo): it is never cleared and BallNav
        is also inactive during pauses and after LevelFinish (feature version f2 got this wrong)."""
        return self._trafo

    @property
    def active_ball(self):
        v = self.arrays.get("CurrentLevel", 0, 1, 0)
        return int(v) if v else None

    @property
    def sr_ms(self) -> float | None:
        t = self.obs.time if self.obs is not None else None
        return None if t is None else float(t["sr_double"])

    def ball_type(self) -> int:
        o = self.obs
        if o is not None and o.ball is not None and int(o.ball["type"]) in (0, 1, 2):
            return int(o.ball["type"])
        name = self.level.name(self.active_ball) if self.active_ball else None
        return BALL_NAME_TYPES.get(name, 255)

    def control_valid(self) -> bool:
        """The ball is under player control and has an IVP object (not dead, not transforming)."""
        o = self.obs
        return (self.flags.ball_nav and not self.flags.ball_off and not self.transforming
                and o is not None and o.ball is not None and bool(o.ball["found"]))

"""A simulated BallrainIO agent mode for tests and offline development.

FakeGame speaks the real wire protocol over the real transport (it is the mod's side of
ballrain.transport) and follows the timing rules of agent-protocol.md section 4-5 with a toy level:
a straight wooden road along -z, one checkpoint, one score ball, a moving box, a death box below.
Its physics has nothing to do with Ballance; only the protocol behaviour is meant to be faithful:
frame numbering, stop-on-event, repeat, STOP, frame limit, SR timer, level messages, lives, GameOver,
LevelFinish with an SR tail that is not logged, and the frame log.

HS mechanics (rules from the user, timings toy): touching a score ball gives +100 and its six small balls
start following (visible, P_Extra_Point_Ball1..6 with the score ball P_Extra_Point_Ball0 hidden); they
arrive one by one (+20 each, hidden) FOLLOW_FIRST + FOLLOW_STEP * i frames later, every PAUSE_STEP
frames while paused; the ones still following 2 frames after a PreCheckpoint are lost. Esc: physics
runs on the press frame and freezes from the next one; PauseLevel (+ BallNavInactive) two frames after
the press; while paused no physics, no clock, no time-point decay; a second press resumes physics on
the following frame, UnpauseLevel two frames after it. SCOR / SREN carry the 0.3.1 follower fields.
"""
from __future__ import annotations

import os
import struct
import threading
from pathlib import Path

import numpy as np

from . import protocol as P, tas
from .path import SYNTHETIC_LEVEL_MARKER
from .protocol import Ev
from .transport import ModChannel, SocketChannel, SocketModChannel

ROAD_HALF_WIDTH = 5.0
ROAD_LENGTH = 200.0
CHECKPOINT_Z = -100.0
SCORE_BALL = np.array([0.0, 2.0, -50.0])
SCORE_BALL_2 = np.array([0.0, 2.0, -97.0])  # instance 001, just before the checkpoint
FOLLOW_FIRST, FOLLOW_STEP, PAUSE_STEP = 60, 15, 2
START = np.array([0.0, 2.0, -5.0])
LOAD_FRAMES = 20  # StartLevel is event number 20
CAMNAV_AT = 25
BALLNAV_AT = 30
DEATH_SUBLIFE = 40
DEATH_RESPAWN = 80
SR_TAIL = 3
ACCEL = 25.0
GRAVITY = 30.0

ENT_FLOOR, ENT_BOX, ENT_BALL, ENT_END, ENT_DEPTH, ENT_START = 1001, 1100, 2001, 3001, 4001, 5001
ENT_SYNTHETIC = 6901  # carries ballrain.path.SYNTHETIC_LEVEL_MARKER
ENT_SCORE = (2100, 2200)  # + k: P_Extra_Point_Ball<k> (template) / P_Extra_Point_Ball<k>001
ARR_ENERGY, ARR_CUR, ARR_INGAME, ARR_CP, ARR_RESET, ARR_ALL = 7001, 7002, 7003, 7004, 7005, 7006
CAM_IDS = (6001, 6002, 6003, 6004, 6005)

IDENT43 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], np.float32)


def mat44(pos) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[3, :3] = pos
    return m


def mat43(pos, rot=None) -> np.ndarray:
    m = IDENT43.copy()
    if rot is not None:
        m[:3] = rot
    m[3] = pos
    return m


def box_mesh(half):
    hx, hy, hz = half
    V = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], np.float32)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.uint32)
    return V, F


class FakeGame:
    def __init__(self, env: dict, lives: int = 3, cam_forward=(0.0, 0.0, -1.0), final_delay_s: float = 0.0,
                 score_ball_2=None, hello: str = "FakeGame 0.1 e72e4afcfa5c33a7"):
        self.final_delay_s = final_delay_s
        self.env = env
        self.hello_text = hello
        # the shared-memory transport (BALLRAIN_SHM) or the socket transport of the headless backend (BALLRAIN_SOCK)
        self.chan = SocketModChannel(env["BALLRAIN_SOCK"]) if env.get("BALLRAIN_SOCK") else ModChannel(env["BALLRAIN_SHM"])
        self.level = int(env.get("BALLRAIN_LEVEL") or 1)
        self.timeout_s = int(env.get("BALLRAIN_TIMEOUT_MS") or 60000) / 1000
        self.max_frames = int(env["BALLRAIN_MAX_FRAMES"]) if env.get("BALLRAIN_MAX_FRAMES") else None
        self.levelmsg = Path(env["BALLRAIN_LEVELMSG"]) if env.get("BALLRAIN_LEVELMSG") else None
        self.framelog = Path(env["BALLRAIN_FRAMELOG"]) if env.get("BALLRAIN_FRAMELOG") else None
        self.prefix = tas.load_raw(env["BALLRAIN_PREFIX"]) if env.get("BALLRAIN_PREFIX") else None
        self.prefix_events = env.get("BALLRAIN_PREFIX_EVENTS") == "1"
        self.prefix_acks = 0
        self.frames_done = 0
        self.log: list = []
        self.events: list = []
        self.kill = threading.Event()
        self.end_reason = P.EndReason.NONE
        # game state
        self.loaded = False
        self.pos = START.copy()
        self.vel = np.zeros(3)
        self.ball_nav = False
        self.counter = False
        self.sr = 0.0
        self.counter_ms = 0.0
        self.lives = lives
        self.points = 0
        self.sector = 1
        self.dead_at = None
        self.score_taken = False
        self.finished = False
        self.game_over_at = None
        self.box_pos = np.array([3.0, 1.0, -60.0])
        self.box_t = 0.0
        self.last_delta = np.float32(1000 / 132)
        self.tail_frames = 0
        self.sr_at_finish = None
        self.hs_at_finish = None
        self.energy_at_finish = None
        self.arrays: dict = {}
        self.sent_arrays: dict = {}
        # score balls: per instance, hit frame and the frames at which each small ball left (or None)
        self.sources = [{"pos": SCORE_BALL, "suffix": "", "base": ENT_SCORE[0], "hit": None, "gone": [None] * 6},
                        {"pos": SCORE_BALL_2 if score_ball_2 is None else np.asarray(score_ball_2, float), "suffix": "001", "base": ENT_SCORE[1], "hit": None, "gone": [None] * 6}]
        self.watch_sent: dict = {}
        self.prev_keys = 0
        self.frozen = False  # physics frozen by the pause menu
        self.paused = False
        self.pending: list = []  # (event number, action)
        self.pause_step_at = 0
        self.cp_loss_at = None
        self.followers_at_finish = 0
        f = np.asarray(cam_forward, np.float64)
        f = f / np.linalg.norm(f)
        r = np.array([f[2], 0.0, -f[0]])
        self.cam_rot = np.array([r, [0, 1, 0], f], np.float32)
        self._init_arrays(lives)

    # arrays ---------------------------------------------------------------------------------
    def _set(self, arr, row, col, ctype, value):
        self.arrays[(arr, row, col)] = (ctype, P.encode_cell(value, ctype))

    def _init_arrays(self, lives):
        self._set(ARR_ENERGY, 0, 0, P.ARR_INT, 0)
        self._set(ARR_ENERGY, 0, 1, P.ARR_INT, 0)
        self._set(ARR_ENERGY, 0, 2, P.ARR_INT, 1000)
        self._set(ARR_ENERGY, 0, 3, P.ARR_INT, 3)
        self._set(ARR_ENERGY, 0, 4, P.ARR_PARAMETER, 500.0)
        self._set(ARR_ENERGY, 0, 5, P.ARR_INT, 200)
        self._set(ARR_CUR, 0, 0, P.ARR_INT, self.level)
        self._set(ARR_CUR, 0, 1, P.ARR_OBJECT, 0)
        self._set(ARR_INGAME, 0, 0, P.ARR_OBJECT, 0)
        self._set(ARR_INGAME, 0, 1, P.ARR_INT, 1)
        self._set(ARR_INGAME, 0, 2, P.ARR_INT, 0)
        self._set(ARR_CP, 0, 0, P.ARR_PARAMETER, mat44([0, 2, CHECKPOINT_Z]))
        self._set(ARR_CP, 0, 1, P.ARR_PARAMETER, 0)
        self._set(ARR_RESET, 0, 0, P.ARR_PARAMETER, mat44(START))
        self._set(ARR_RESET, 1, 0, P.ARR_PARAMETER, mat44([0, 2, CHECKPOINT_Z - 3]))
        for r in range(13):
            self._set(ARR_ALL, r, 6, P.ARR_INT, 100 * (r + 1))

    def _array_defs(self):
        defs = {
            ARR_ENERGY: ("Energy", [("Points", 1, None), ("Lifes", 1, None), ("StartPoints", 1, None),
                                    ("StartLifes", 1, None), ("Timefactor", 5, "Time"), ("LifeBonus", 1, None)]),
            ARR_CUR: ("CurrentLevel", [("Level ID", 1, None), ("ActiveBall", 4, None)]),
            ARR_INGAME: ("IngameParameter", [("Activetrafo", 4, None), ("Activate Sector", 1, None),
                                             ("Deactivate Sector", 1, None)]),
            ARR_CP: ("Checkpoints", [("Matrix", 5, "Matrix"), ("Object", 5, "3D Entity")]),
            ARR_RESET: ("ResetPoints", [("Resetpoint", 5, "Matrix")]),
            ARR_ALL: ("AllLevel", [(f"c{i}", 1, None) for i in range(8)]),
        }
        out = []
        for ck, (name, cols) in defs.items():
            nrows = 1 + max(r for (a, r, _c) in self.arrays if a == ck)
            cells = [[self.arrays.get((ck, r, c), (0, b""))[1] for c in range(len(cols))] for r in range(nrows)]
            out.append(P.ArrayDef(ck, name, [P.ArrayColumn(n, t, pt) for n, t, pt in cols], cells))
        return out

    def _changed_cells(self, full=False):
        out = []
        for key, val in self.arrays.items():
            if full or self.sent_arrays.get(key) != val:
                out.append(P.ArrayCell(key[0], key[1], key[2], val[0], val[1]))
                self.sent_arrays[key] = val
        return out

    # level message ----------------------------------------------------------------------------
    def level_message(self) -> P.LevelMessage:
        L = ROAD_LENGTH
        floorV = np.array([[-ROAD_HALF_WIDTH, 0, 0], [ROAD_HALF_WIDTH, 0, 0], [ROAD_HALF_WIDTH, 0, -L],
                           [-ROAD_HALF_WIDTH, 0, -L]], np.float32)
        floorF = np.array([[0, 2, 1], [0, 3, 2]], np.uint32)
        boxV, boxF = box_mesh((1.0, 1.0, 1.0))
        ents = [
            P.Entity(ENT_FLOOR, "A01_Floor_01", 41, 0, True, IDENT43.copy(), 2, 2, b""),
            P.Entity(ENT_BOX, "P_Box_MF", 41, 0, True, mat43(self.box_pos), 0, 1, b""),
            P.Entity(ENT_BALL, "Ball_Wood", 41, 0, True, mat43(START), -1, -1, b""),
            P.Entity(ENT_END, "PE_Balloon_Platform", 41, 0, True, mat43([0, 2, -L]), 0, -1, b""),
            P.Entity(ENT_START, "PS_FourFlames_MF", 33, 0, True, mat43(START), -1, -1, b""),
            P.Entity(ENT_DEPTH, "Quader01", 41, 0, False, mat43([0, -25, -100]), -1, -1, b""),
            # not a real level: no reference asset applies, whatever the level number (ballrain.path)
            P.Entity(ENT_SYNTHETIC, SYNTHETIC_LEVEL_MARKER, 41, 0, False, IDENT43.copy(), -1, -1, b""),
        ]
        score_ids = []
        for src in self.sources:
            for k in range(7):
                ck = src["base"] + k
                score_ids.append(ck)
                ents.append(P.Entity(ck, f"P_Extra_Point_Ball{k}{src['suffix']}", 41, 0, True, mat43(src["pos"]),
                                     -1, -1, b""))
        grps = [P.Group(9001, "Phys_Floors", np.array([ENT_FLOOR], np.uint32)),
                P.Group(9002, "Sound_RollID_02", np.array([ENT_FLOOR], np.uint32)),
                P.Group(9003, "DepthTestCubes", np.array([ENT_DEPTH], np.uint32)),
                P.Group(9004, "All_P_Extra_Point", np.array(score_ids, np.uint32))]
        phys = [P.PhysRecord(0, LOAD_FRAMES - 5, ENT_FLOOR, False, True, 0.7, 0.3, 0, "Floor", False, True,
                             False, 0, 0, "A01_Floor_01_Mesh", (0, 0, 0), [], [], [8001]),
                P.PhysRecord(1, LOAD_FRAMES - 5, ENT_BOX, False, False, 0.7, 0.4, 4, "", False, True,
                             False, 0.1, 0.1, "P_Box_Mesh", (0, 0, 0), [8002], [], []),
                P.PhysRecord(2, LOAD_FRAMES - 1, ENT_BALL, False, False, 0.8, 0.2, 1.9, "Ball", False, True,
                             False, 0.9, 0.9, "", (0, 0, 0), [], [(np.zeros(3, np.float32), 2.0)], [])]
        meshes = [P.Mesh(ENT_FLOOR, 8001, 1, np.eye(4, dtype=np.float32), floorV, floorF),
                  P.Mesh(ENT_BOX, 8002, 0, mat44(self.box_pos), boxV, boxF)]
        return P.LevelMessage(
            frames_done=self.frames_done, entities=ents, groups=grps, arrays=self._array_defs(), phys=phys,
            meshes=meshes, ivp=[P.IvpObject(ENT_FLOOR, 2, True, 0.0, b""), P.IvpObject(ENT_BALL, 3, False, 2.0, b"")],
            depth=[(ENT_DEPTH, np.array([-50, -30, -250], np.float32), np.array([50, -20, 50], np.float32))],
            watch=[(ENT_BALL, "All_Balls")] + [(ck, "All_P_Extra_Point") for ck in score_ids])

    # simulation -------------------------------------------------------------------------------
    def _ev(self, code, param=0):
        self.events.append((self.frames_done + 1, code, param))  # number of the frame being executed

    def on_road(self, p) -> bool:
        return abs(p[0]) <= ROAD_HALF_WIDTH and -ROAD_LENGTH <= p[2] <= 0.0

    def _sim_frame(self, delta: np.float32, keys: int) -> None:
        """Execute one frame (the frame with index self.frames_done)."""
        n = self.frames_done + 1
        dt = float(delta) / 1000.0
        if n == LOAD_FRAMES:
            self.loaded = True
            self._ev(Ev.StartLevel)
            self.sr = 0.0
            self.points = 1000
            self._set(ARR_ENERGY, 0, 1, P.ARR_INT, self.lives)
            self._set(ARR_CUR, 0, 1, P.ARR_OBJECT, ENT_BALL)
            if self.levelmsg is not None:
                self.levelmsg.write_bytes(P.build_level_message(self.level_message()))
        if n == CAMNAV_AT:
            self._ev(Ev.CamNavActive)
        if n == BALLNAV_AT:
            self._start_control()
        self._pause_input(n, keys)
        if self.loaded and not self.frozen:
            self.box_t += dt
            self.box_pos[0] = 3.0 + np.sin(self.box_t)
        if self.ball_nav and self.dead_at is None and not self.finished and not self.frozen:
            f, r = self.cam_rot[2].astype(np.float64), self.cam_rot[0].astype(np.float64)
            a = np.zeros(3)
            a += f * bool(keys & tas.KEY_UP) - f * bool(keys & tas.KEY_DOWN)
            a += r * bool(keys & tas.KEY_RIGHT) - r * bool(keys & tas.KEY_LEFT)
            self.vel[[0, 2]] += ACCEL * dt * a[[0, 2]]
            self.vel[[0, 2]] *= 0.995
            if self.on_road(self.pos) and self.pos[1] >= 2.0 - 1e-6:
                self.vel[1] = 0.0
                self.pos[1] = 2.0
            else:
                self.vel[1] -= GRAVITY * dt
            self.pos += self.vel * dt
            if self.on_road(self.pos) and 1.0 < self.pos[1] < 2.0:
                self.pos[1], self.vel[1] = 2.0, 0.0
            for src in self.sources:
                if src["hit"] is None and np.linalg.norm(self.pos - src["pos"]) < 2.5:
                    src["hit"] = n
                    self.score_taken = True
                    self.points += 100
                    self._ev(Ev.ExtraPoint)
            if self.sector == 1 and self.pos[2] < CHECKPOINT_Z:
                self.sector = 2
                self._set(ARR_INGAME, 0, 1, P.ARR_INT, 2)
                self._ev(Ev.PreCheckpoint)
                self._ev(Ev.PostCheckpoint)
                self.cp_loss_at = n + 2
            if self.pos[1] < -10.0:
                self._ev(Ev.BallOff)
                self._ev(Ev.CounterInactive)
                self._ev(Ev.BallNavInactive)
                self.counter = False
                self.ball_nav = False
                self.dead_at = n
                if self.lives == 0:
                    self.game_over_at = n + 2
            elif self.pos[2] < -ROAD_LENGTH + 5 and self.on_road(self.pos):
                self.finished = True
                self._ev(Ev.LevelFinish)
        if self.dead_at is not None and self.game_over_at is None:
            k = n - self.dead_at
            if k == DEATH_SUBLIFE:
                self.lives -= 1
                self._ev(Ev.PreSubLife)
                self._ev(Ev.PostSubLife)
            elif k == DEATH_RESPAWN:
                rp = START if self.sector == 1 else np.array([0, 2, CHECKPOINT_Z - 3])
                self.pos, self.vel = np.array(rp, float), np.zeros(3)
                self.dead_at = None
                self._start_control()
        if self.game_over_at is not None and n == self.game_over_at:
            self._ev(Ev.GameOver)
        self._followers_frame(n)
        self._sr_frame(delta)
        if self.counter and not self.frozen:
            self.counter_ms += float(delta)
            while self.counter_ms >= 500.0:
                self.counter_ms -= 500.0
                self.points = max(0, self.points - 1)
        if self.loaded:
            self._set(ARR_ENERGY, 0, 0, P.ARR_INT, self.points)
            self._set(ARR_ENERGY, 0, 1, P.ARR_INT, self.lives)
        self.last_delta = delta

    def _pause_input(self, n, keys):
        """Esc edge: pause / unpause (see the module docstring)."""
        for when, what in [x for x in self.pending if x[0] == n]:
            self.pending.remove((when, what))
            if what == "pause":
                self.paused = True
                self.pause_step_at = n + PAUSE_STEP
                self._ev(Ev.PauseLevel)
                if self.ball_nav:
                    self._ev(Ev.BallNavInactive)
            else:
                self.paused = False
                self._ev(Ev.UnpauseLevel)
                if self.dead_at is None and not self.finished and self.loaded:
                    self._ev(Ev.BallNavActive)
        down = bool(keys & tas.KEY_ESC) and not (self.prev_keys & tas.KEY_ESC)
        self.prev_keys = keys
        if down and self.loaded and not self.finished and not self.pending:
            if not self.frozen:
                self._freeze_next = True
                self.pending.append((n + 2, "pause"))
            else:
                self.pending.append((n + 2, "unpause"))
                self._thaw_next = True

    def followers(self) -> int:
        return sum(1 for src in self.sources if src["hit"] is not None for g in src["gone"] if g is None)

    def _followers_frame(self, n):
        """Arrivals (+20, hidden), losses 2 frames after a PreCheckpoint."""
        lose = self.cp_loss_at is not None and n == self.cp_loss_at
        for src in self.sources:
            if src["hit"] is None:
                continue
            for i in range(6):
                if src["gone"][i] is not None:
                    continue
                if lose:
                    src["gone"][i] = ("lost", n)
                    continue
                due = (self.paused and n >= self.pause_step_at) or (
                    not self.frozen and n >= src["hit"] + FOLLOW_FIRST + FOLLOW_STEP * i)
                if due:
                    src["gone"][i] = ("arrived", n)
                    self.points += 20
                    self._ev(Ev.ExtraPoint)
                    if self.paused:
                        self.pause_step_at = n + PAUSE_STEP
                    break  # one arrival per frame
        if lose:
            self.cp_loss_at = None
        if getattr(self, "_freeze_next", False):
            self._freeze_next, self.frozen = False, True
        if getattr(self, "_thaw_next", False):
            self._thaw_next, self.frozen = False, False  # physics resumes on the next frame

    def _watch_cells(self, full=False):
        out = []
        for src in self.sources:
            for k in range(7):
                if k == 0:
                    vis, pos = int(src["hit"] is None), src["pos"]
                else:
                    following = src["hit"] is not None and src["gone"][k - 1] is None
                    vis = int(src["hit"] is None or following)
                    ang = k * np.pi / 3
                    pos = (self.pos + [2.5 * np.cos(ang), 0.5, 2.5 * np.sin(ang)]) if following else src["pos"]
                ck = src["base"] + k
                val = (vis, tuple(np.round(np.asarray(pos, np.float32), 4)))
                if full or self.watch_sent.get(ck) != val:
                    self.watch_sent[ck] = val
                    out.append((ck, vis, np.asarray(pos, np.float32)))
        return out

    def _start_control(self):
        self.ball_nav = True
        self.counter = True
        self._ev(Ev.BallNavActive)
        self._ev(Ev.CounterActive)

    def _sr_frame(self, delta):
        # BML+ 0.3.12: events of this frame first (StartLevel reset / start / pause), then add if running
        n = self.frames_done + 1
        for num, code, _ in self.events:
            if num != n:
                continue
            if code == Ev.StartLevel:
                self.sr, self._sr_run = 0.0, False
            elif code in (Ev.CounterActive, Ev.UnpauseLevel):
                self._sr_run = True
            elif code in (Ev.CounterInactive, Ev.PauseLevel):
                self._sr_run = False
        if getattr(self, "_sr_run", False):
            self.sr += float(delta)

    # observation ------------------------------------------------------------------------------
    def _hs(self):
        return self.points + self.lives * 200 + 100 * self.level

    def observation(self, ev_from: int, full_arrays=False, covered=1, final=False) -> bytes:
        secs = []
        t = P.pack_fixed(P.TIME_DT, env_time=self.frames_done * 0.0075757, next_psi=0, last_psi=0,
                         mgr_delta_ms=float(self.last_delta), mgr_step_s=float(self.last_delta) * 0.002 / 1000,
                         time_factor=0.002, delta_ms=float(self.last_delta), sr_double=self.sr,
                         sr_float=np.float32(self.sr), sr_running=int(getattr(self, "_sr_run", False)),
                         frames_covered=covered)
        t += P.pack_fixed(P.TIME_EXT_DT, sr_f32sum=np.float32(self.sr), frame_counter=self.frames_done,
                          tail_frames=self.tail_frames)  # appended fields, as the real mod writes them
        secs.append(("TIME", t))
        evs = [e for e in self.events[ev_from:]]
        if self.loaded:
            alive = self.dead_at is None or (self.frames_done - self.dead_at) < 2
            m = mat43(self.pos)
            secs.append(("BALL", P.pack_fixed(
                P.BALL_DT, ckid=ENT_BALL, type=0, found=int(alive), ck_matrix=m, pos=self.pos,
                q_last=(0, 0, 0, 1), q_next=(0, 0, 0, 1), tpsi=0.0, speed=self.vel.astype(np.float32),
                current_speed=float(np.linalg.norm(self.vel)), extra_radius=2.0, core_state=1)))
            gap = self.pos[1] - 2.0
            fric, mind = [], []
            if self.on_road(self.pos) and -1.0 < gap < 1.2:
                mind.append((ENT_FLOOR, 3, 2, 0, max(gap, 0.0), (0, 1, 0, 0), 2.0, 1, 0, -1, 1,
                             np.zeros((3, 3)), np.zeros((3, 3))))
                if gap < 0.05:
                    fric.append((ENT_FLOOR, 3, 2, 1, 0, 0.013, 0.56, (0, -1, 0), self.pos - [0, 2, 0], 1))
            secs.append(("FRIC", P.pack_list("FRIC", np.array(fric, P.FRIC_DT))))
            secs.append(("MIND", P.pack_list("MIND", np.array(mind, P.MIND_DT))))
            bodies = []
            if np.linalg.norm(self.box_pos - self.pos) < float(self.env.get("BALLRAIN_BODY_RADIUS") or 60):
                bodies.append((ENT_BOX, 1, self.box_pos, (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0),
                               (np.cos(self.box_t), 0, 0), (0, 0, 0), mat43(self.box_pos)))
            secs.append(("BODY", P.pack_list("BODY", np.array(bodies, P.BODY_DT))))
            cam = np.zeros(1, P.CAMS_DT)
            for i, ck in enumerate(CAM_IDS):
                cam["cams"][0, i]["ckid"] = ck
                cam["cams"][0, i]["matrix"] = mat43(self.pos, self.cam_rot)
            cam["attached"] = CAM_IDS[4]
            secs.append(("CAMS", cam.tobytes()))
            g = np.zeros(1, P.GLCT_DT)
            g["groups"][0, 1] = (int(bool(fric)), len(fric))
            secs.append(("GLCT", g.tobytes()))
        if self.loaded:
            w = self._watch_cells()
            secs.append(("WTCH", P.pack_list("WTCH", np.array(w, P.WTCH_DT))))
            fol = self.followers()
            secs.append(("SCOR", P.pack_fixed(P.SCOR_DT, hs_valid=1, types=(1, 1, 1, 1, 5, 1), bonus_type=1,
                                               energy=(self.points, self.lives, 1000, 3, 0, 200),
                                               level_bonus=100 * self.level, hs=self._hs())
                         + P.pack_fixed(P.SCOR_EXT_DT, followers=fol,
                                        sources=sum(1 for s_ in self.sources if s_["hit"] is not None
                                                    and any(g is None for g in s_["gone"])))))
        if final or self.finished:
            hs = self.hs_at_finish if self.hs_at_finish is not None else self._hs()
            secs.append(("SREN", P.pack_fixed(
                P.SREN_DT, sr_double=self.sr, sr_f32sum=np.float32(self.sr), sr_bml=np.float32(self.sr),
                tail_frames=self.tail_frames, level_finished=int(self.finished),
                counter_inactive_seen=int(self.finished and not getattr(self, "_sr_run", False)),
                sr_double_at_finish=self.sr_at_finish or 0.0, hs_valid=1, types=(1, 1, 1, 1, 5, 1), bonus_type=1,
                energy=self.energy_at_finish or (self.points, self.lives, 1000, 3, 0, 200),
                level_bonus=100 * self.level, hs=hs)
                + P.pack_fixed(P.SREN_EXT_DT, followers_at_finish=self.followers_at_finish,
                               followers_end=self.followers(), post_frames=0)))
        cells = self._changed_cells(full_arrays)
        if cells:
            secs.append(("ARRS", P.pack_arrs(cells)))
        if evs:
            secs.append(("EVTS", P.pack_list("EVTS", np.array(evs, P.EVT_DT))))
        if final and "headless" in self.hello_text:  # the headless backend's run fingerprint (toy: over the frame log)
            import hashlib
            h = int.from_bytes(hashlib.blake2b(np.asarray(self.log, np.float64).tobytes(), digest_size=8).digest(), "little")
            secs.append(("HASH", P.pack_fixed(P.HASH_DT, trace_hash=h, frames=self.frames_done)))
        secs.append(("ZZZZ", b"future section"))  # unknown tags must be skipped
        return P.build_message(P.OBS_MAGIC, self.frames_done, secs, flags=P.OBS_FINAL if final else 0)

    # main loop --------------------------------------------------------------------------------
    def run(self) -> None:
        ch = self.chan
        ch.hello(self.hello_text)
        self._ev(Ev.PreLoadLevel)
        ch.set_state(P.ModState.LOADING)
        ev_from = 0
        if self.prefix is not None:
            for i, fr in enumerate(self.prefix):
                self._exec(fr["delta"], int(fr["keys"]))
                if self.prefix_events and i + 1 < len(self.prefix):
                    # like the real mod: one OBS_PREFIX observation per prefix frame, then wait for the
                    # acknowledgement (an empty action) before the next prefix frame
                    msg = bytearray(self.observation(ev_from, full_arrays=ev_from == 0))
                    struct.pack_into("<H", msg, 6, P.OBS_PREFIX)
                    ch.write_obs(bytes(msg), self.frames_done)
                    ev_from = len(self.events)
                    a = ch.wait_action(self.timeout_s, cancel=self.kill)
                    if a is None or self.kill.is_set():
                        return
                    if len(a.frames) or a.flags:
                        raise RuntimeError("fake game: prefix observation answered with a non-empty action")
                    self.prefix_acks += 1
        ch.write_obs(self.observation(ev_from, full_arrays=ev_from == 0), self.frames_done)
        ev_from = len(self.events)
        while not self.kill.is_set():
            ch.set_state(P.ModState.WAITING)
            a = ch.wait_action(self.timeout_s, cancel=self.kill)
            if self.kill.is_set():
                return
            if a is None:
                self._end(P.EndReason.TIMEOUT, ev_from)
                return
            ch.set_state(P.ModState.RUNNING)
            if a.flags & P.ACT_STOP:
                self._end(P.EndReason.PY_STOP, ev_from)
                return
            frames = a.expanded()
            for fr in frames:
                ev0 = len(self.events)
                self._exec(fr["delta"], int(fr["keys"]))
                new = [c for _, c, _ in self.events[ev0:]]
                if any(c == Ev.GameOver for c in new):
                    self._end(P.EndReason.GAME_OVER, ev_from)
                    return
                if any(c == Ev.LevelFinish for c in new):
                    self._tail()
                    self._end(P.EndReason.LEVEL_FINISH, ev_from)
                    return
                if self.max_frames is not None and self.frames_done >= self.max_frames:
                    self._end(P.EndReason.FRAME_LIMIT, ev_from)
                    return
                if any(a.stop_mask >> c & 1 for c in new):
                    break
            ch.write_obs(self.observation(ev_from, covered=len(frames)), self.frames_done)
            ev_from = len(self.events)

    def _exec(self, delta, keys):
        self._sim_frame(np.float32(delta), keys)
        self.log.append((np.float32(delta), keys))
        self.frames_done += 1

    def _tail(self):
        """agent-protocol.md 5: frames after LevelFinish until CounterInactive, not logged."""
        self.sr_at_finish = self.sr
        self.hs_at_finish = self._hs()
        self.energy_at_finish = (self.points, self.lives, 1000, 3, 0, 200)
        self.followers_at_finish = self.followers()  # tail frames do not move the toy small balls
        self.tail_frames = SR_TAIL
        for i in range(SR_TAIL):
            if i == SR_TAIL - 1:
                self.events.append((self.frames_done + 1 + i, Ev.CounterInactive, 0))
            # tail frames run the timer; they do not advance the exported frame count
            if i < SR_TAIL - 1:
                self.sr += float(self.last_delta)
        self._sr_run = False

    def _end(self, reason, ev_from):
        self.end_reason = reason
        self.events.append((self.frames_done, Ev.RunEnd, reason))
        if self.framelog is not None:
            tas.save_raw(self.framelog, tas.make_frames([d for d, _ in self.log], [k for _, k in self.log])
                         if self.log else np.zeros(0, tas.FRAME_DTYPE))
        self.chan.set_state(P.ModState.ENDED, reason, self.frames_done)
        if self.final_delay_s:
            import time
            time.sleep(self.final_delay_s)  # the real mod sets ENDED before publishing the final obs
        self.chan.write_obs(self.observation(ev_from, final=True), self.frames_done)

    def close(self):
        self.chan.close()


class FakeHandle:
    def __init__(self, game: FakeGame):
        self.game = game
        self.pid = os.getpid()
        self.wait_handle = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.game.run()
        except Exception as e:  # surfaced through exit_code / error
            self.error = e
            try:
                self.game.chan.set_state(P.ModState.ERROR)
                self.game.chan.signal()
            except Exception:
                pass
        finally:
            self.game.close()

    def wait(self, timeout_s: float) -> bool:
        self.thread.join(timeout_s)
        return not self.thread.is_alive()

    def terminate(self, code: int = 1):
        self.game.kill.set()

    def exit_code(self):
        return None if self.thread.is_alive() else (1 if self.error else 0)

    def close(self):
        pass


class FakeLauncher:
    """Drop-in for client.GameLauncher. transport="socket" (with backend="headless", the default for it) speaks the
    headless backend's socket transport (ballrain.transport.SocketChannel) and hello."""

    def __init__(self, transport: str = "shm", backend: str | None = None, socket_kind: str | None = None,
                 **game_kwargs):
        if transport not in ("shm", "socket"):
            raise ValueError(transport)
        self.transport = transport
        self.backend = backend or ("headless" if transport == "socket" else "retail")
        self.socket_kind = socket_kind
        if self.backend == "headless":
            game_kwargs.setdefault("hello", "FakeGame 0.1 headless")
        self.game_kwargs = game_kwargs
        self.handles: list = []
        self.envs: list = []

    def make_channel(self, cfg):
        if self.transport == "socket":
            return SocketChannel(self.socket_kind)
        from .transport import PyChannel, make_name
        return PyChannel(make_name(cfg.copy))

    def start(self, env: dict):
        self.envs.append(dict(env))
        h = FakeHandle(FakeGame(env, **self.game_kwargs))
        self.handles.append(h)
        return h

    @staticmethod
    def stop(handle, wait_s: float = 10.0):
        if not handle.wait(wait_s):
            handle.terminate()
            handle.wait(5)
        if handle.error:
            raise handle.error
        return handle.exit_code()


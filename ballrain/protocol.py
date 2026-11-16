"""Wire formats of the BallrainIO agent mode (docs/agent-protocol.md, contract v1).

Everything here is pure byte layout: no Windows calls, no game knowledge beyond field names.
The transport (shared memory + events) lives in ballrain.transport; the meaning of the fields in
docs/state-spec.md; the features derived from them in ballrain.features.

Layout rules this module assumes. agent-protocol.md gives field ORDER; the items below are where it
is silent, and the C++ side must match them (they are also listed in docs/observations.md, appendix):

  A1  All multi-byte values are little-endian and PACKED: no padding between fields, none between
      array items (the C++ side writes with #pragma pack(1) structs or field-by-field memcpy).
  A2  Message header (OBS1 / LVL1): u32 magic, u16 version, u16 flags, u32 byte_len, u32 frames_done,
      u32 section_count, u32 reserved = 24 bytes. byte_len is the WHOLE message including the header.
  A3  Section header: u32 tag, u16 sec_version, u16 sec_flags, u32 payload_len (12 bytes).
      payload_len is the unpadded payload length; the next section starts at align4(payload_len).
      Readers accept trailing payload bytes (fields are only ever appended) and skip unknown tags.
  A4  Per-frame list sections (FRIC, MIND, BODY, ARRS, EVTS, WTCH, PHYS) start with u16 n, items follow
      directly. TRAC starts with u32 n (a block can cover more than 65535 frames). PHYS items are u32
      physicalize-record sequence numbers.
  A5  CK world matrices "12 x f32" are rows 0..3 of the 4x4 VxMatrix, columns 0..2 (row 3 = position);
      Virtools row-vector convention: world = [x y z 1] * M. Quaternions are x, y, z, w.
  A6  MIND items always carry both triangles (18 f32); n_tris (0, 1 or 2) says how many are valid.
  A7  ACT1 header: u32 magic, u16 version, u16 flags, u32 py_seq, u32 n_frames, u32 repeat,
      u32 stop_lo, u32 stop_hi, u32 obs_mask, u32 reserved = 36 bytes; event code c is stop bit c.
      obs_mask bits DISABLE optional sections (OBSMASK_*); 0 = everything.
  A8  Level messages: every section starts with u32 n. Strings are indices into the message's own STRS
      table (u32 n, then n x {u16 len, bytes}, ANSI code page, decoded here as latin-1, lossless);
      0xFFFFFFFF = no string. Each LVL1 message (the full file and every message appended to .inc)
      has its own STRS. Item layouts are given by the _LVL_* readers below.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------------------------ constants

def fourcc(s: str) -> int:
    b = s.encode("ascii")
    if len(b) != 4:
        raise ValueError(s)
    return struct.unpack("<I", b)[0]


def tag_str(tag: int) -> str:
    return struct.pack("<I", tag).decode("latin-1")


SHM_MAGIC = fourcc("BRSM")  # 0x4D535242
assert SHM_MAGIC == 0x4D535242
OBS_MAGIC = fourcc("OBS1")
ACT_MAGIC = fourcc("ACT1")
LVL_MAGIC = fourcc("LVL1")
PROTOCOL_VERSION = 1
OBS_VERSION = 1
ACT_VERSION = 1

SHM_DEFAULT_SIZE = 4 << 20
SHM_OBS_OFFSET = 0x1000
SHM_OBS_CAPACITY = 3 << 20
SHM_ACT_OFFSET = SHM_OBS_OFFSET + SHM_OBS_CAPACITY
SHM_ACT_CAPACITY = 512 << 10
SHM_VERSION_LEN = 180

# shared-memory header offsets
H_MAGIC, H_VERSION, H_SIZE = 0x00, 0x04, 0x08
H_OBS_OFF, H_OBS_CAP, H_ACT_OFF, H_ACT_CAP = 0x0C, 0x10, 0x14, 0x18
H_MOD_SEQ, H_PY_SEQ = 0x20, 0x28
H_MOD_STATE, H_END_REASON, H_MOD_PID, H_PY_PID = 0x30, 0x34, 0x38, 0x3C
H_FRAMES_DONE, H_LAST_ERROR, H_OBS_VERSIONS, H_MOD_VERSION = 0x40, 0x44, 0x48, 0x4C
SHM_HEADER_SIZE = H_MOD_VERSION + SHM_VERSION_LEN  # 0x100


class ModState:
    INIT, HELLO, MENU, LOADING, RUNNING, WAITING, ENDED, ERROR = range(8)
    NAMES = ("init", "hello", "menu", "loading", "running", "waiting", "ended", "error")


class EndReason:
    NONE, LEVEL_FINISH, GAME_OVER, FRAME_LIMIT, PY_STOP, RESET_EXIT, TIMEOUT, ERROR = range(8)
    NAMES = ("none", "LevelFinish", "GameOver", "frame_limit", "python_stop", "reset_or_exit", "timeout", "error")


EVENT_NAMES = {
    1: "PreLoadLevel", 2: "PostLoadLevel", 3: "StartLevel", 4: "CamNavActive", 5: "CamNavInactive",
    6: "BallNavActive", 7: "BallNavInactive", 8: "CounterActive", 9: "CounterInactive", 10: "PauseLevel",
    11: "UnpauseLevel", 12: "BallOff", 13: "Dead", 14: "PreSubLife", 15: "PostSubLife",
    16: "PreCheckpoint", 17: "PostCheckpoint", 18: "ExtraPoint", 19: "PreLifeUp", 20: "PostLifeUp",
    21: "LevelFinish", 22: "GameOver", 23: "PreEndLevel", 24: "PostEndLevel", 25: "Physicalize",
    26: "Unphysicalize", 27: "ActiveBallChanged", 28: "PreResetLevel", 29: "PreExitLevel", 30: "RunEnd",
}
EVENT_CODES = {v: k for k, v in EVENT_NAMES.items()}


class Ev:
    """Event codes as attributes: Ev.BallOff == 12."""


for _c, _n in EVENT_NAMES.items():
    setattr(Ev, _n, _c)


def event_mask(*events) -> int:
    """64-bit stop mask from event codes or names."""
    m = 0
    for e in events:
        c = EVENT_CODES[e] if isinstance(e, str) else int(e)
        if not 0 <= c < 64:
            raise ValueError(f"event code {c} outside 0..63")
        m |= 1 << c
    return m


# ACT1 flags
ACT_STOP = 1 << 0
ACT_WANT_TRAC = 1 << 1
ACT_REWRITE_LEVEL = 1 << 2

# obs_mask bits (set = section disabled); assumption A7
OBSMASK_BODY = 1 << 0
OBSMASK_MIND = 1 << 1
OBSMASK_FRIC = 1 << 2
OBSMASK_CAMS = 1 << 3
OBSMASK_GLCT = 1 << 4
OBSMASK_WTCH = 1 << 5
OBSMASK_BALL = 1 << 6
OBSMASK_ARRS = 1 << 7

BALL_TYPES = {0: "wood", 1: "paper", 2: "stone", 254: "unknown", 255: "none"}

# CK_ARRAYTYPE
ARR_INT, ARR_FLOAT, ARR_STRING, ARR_OBJECT, ARR_PARAMETER = 1, 2, 3, 4, 5

NO_STRING = 0xFFFFFFFF


class ProtocolError(ValueError):
    pass


def align4(n: int) -> int:
    return (n + 3) & ~3


# ------------------------------------------------------------------------- per-frame item dtypes

M12 = ("<f4", (4, 3))

TIME_DT = np.dtype([
    ("env_time", "<f8"), ("next_psi", "<f8"), ("last_psi", "<f8"),
    ("mgr_delta_ms", "<f4"), ("mgr_step_s", "<f4"), ("time_factor", "<f4"),
    ("delta_ms", "<f4"), ("sr_double", "<f8"), ("sr_float", "<f4"), ("sr_running", "u1"),
    ("frames_covered", "<u4"),
])
BALL_DT = np.dtype([
    ("ckid", "<u4"), ("type", "u1"), ("found", "u1"), ("ck_matrix",) + M12,
    ("pos", "<f8", 3), ("q_last", "<f8", 4), ("q_next", "<f8", 4), ("tpsi", "<f8"),
    ("speed", "<f4", 3), ("speed_change", "<f4", 3), ("rot_speed", "<f4", 3), ("rot_speed_change", "<f4", 3),
    ("rot_axis", "<f4", 3), ("current_speed", "<f4"), ("abs_omega", "<f4"),
    ("core_flags", "<u4"), ("core_state", "<u4"), ("extra_radius", "<f4"),
])
FRIC_DT = np.dtype([
    ("partner", "<u4"), ("self_status", "u1"), ("partner_status", "u1"), ("cp_status", "u1"), ("pad", "u1"),
    ("gap", "<f4"), ("friction", "<f4"), ("lt_normal", "<f4", 3), ("lt_point", "<f8", 3), ("lt_valid", "u1"),
])
MIND_DT = np.dtype([
    ("partner", "<u4"), ("self_status", "u1"), ("partner_status", "u1"), ("pad", "<u2"),
    ("gap", "<f4"), ("plane", "<f4", 4), ("sum_extra_radius", "<f4"), ("ledge", "<u4"),
    ("tri_index", "<u2"), ("edge_index", "i1"), ("n_tris", "u1"),
    ("tri", "<f4", (3, 3)), ("tri_adj", "<f4", (3, 3)),
])
BODY_DT = np.dtype([
    ("ckid", "<u4"), ("core_state", "<u4"), ("pos", "<f8", 3), ("q", "<f8", 4), ("q_cfo", "<f8", 4),
    ("shift", "<f4", 3), ("speed", "<f4", 3), ("rot_speed", "<f4", 3), ("ck_matrix",) + M12,
])
CAM_DT = np.dtype([("ckid", "<u4"), ("matrix",) + M12])
CAMS_DT = np.dtype([("cams", CAM_DT, 5), ("attached", "<u4")])
CAM_NAMES = ("Cam_OrientRef", "Cam_Orient", "Cam_Pos", "Cam_Target", "InGameCam")
GLCT_DT = np.dtype([("bb", "<u4"), ("pulses", "<u2"), ("groups", "<i4", (3, 2))])  # groups[k] = (active, number)
# Fields the mod appends to TIME v1 (BallrainIO Agent.cpp): f32 SR sum, raw frame counter (SR tail
# frames included), tail frames so far.
TIME_EXT_DT = np.dtype([("sr_f32sum", "<f4"), ("frame_counter", "<u4"), ("tail_frames", "<u4")])
# Extension sections written by BallrainIO (not in agent-protocol.md v1; parsed when present).
_HS_FIELDS = [("hs_valid", "u1"), ("types", "u1", 6), ("bonus_type", "u1"), ("energy", "<u4", 6),
              ("level_bonus", "<u4"), ("bml_hs", "<i4"), ("hs", "<i4")]
SCOR_DT = np.dtype(_HS_FIELDS + [("ball_offs", "<u4"), ("last_balloff", "<u4"), ("last_ballnav", "<u4")])
SREN_DT = np.dtype([("sr_double", "<f8"), ("sr_f32sum", "<f4"), ("sr_bml", "<f4"), ("tail_frames", "<u4"),
                    ("level_finished", "u1"), ("counter_inactive_seen", "u1"), ("sr_running", "u1"), ("pad", "u1"),
                    ("sr_double_at_finish", "<f8"), ("sr_f32sum_at_finish", "<f4")] + _HS_FIELDS)
# Fields BallrainIO 0.3.1 appends to SCOR (at 56) and SREN (at 80): small score balls still following
# the player ball (hidden P_Extra_Point_Ball0 of an instance with visible P_Extra_Point_Ball1..6), touched
# score balls with a following small ball; SREN: followers at the LevelFinish hook, at the time of the
# observation (end of the SR tail), frames of the diagnostic post-finish window (BALLRAIN_POST_FINISH).
SCOR_EXT_DT = np.dtype([("followers", "<u2"), ("sources", "<u2")])
SREN_EXT_DT = np.dtype([("followers_at_finish", "<u2"), ("followers_end", "<u2"), ("post_frames", "<u4")])
EVT_DT = np.dtype([("frame", "<u4"), ("code", "<u2"), ("param", "<u2")])
WTCH_DT = np.dtype([("ckid", "<u4"), ("visible", "u1"), ("pos", "<f4", 3)])
TRAC_DT = np.dtype([("frame", "<u4"), ("matrix",) + M12])
ARRS_ITEM_HDR = struct.Struct("<IHHBB")  # array ckid, row, col, col type, len

for _dt, _size in ((TIME_DT, 57), (FRIC_DT, 53), (MIND_DT, 112), (BODY_DT, 180), (CAMS_DT, 264),
                   (GLCT_DT, 30), (EVT_DT, 8), (WTCH_DT, 17), (TRAC_DT, 52), (TIME_EXT_DT, 12), (SCOR_DT, 56),
                   (SREN_DT, 80), (SCOR_EXT_DT, 4), (SREN_EXT_DT, 8)):
    assert _dt.itemsize == _size, (_dt, _dt.itemsize)
assert BALL_DT.itemsize == 4 + 2 + 48 + 24 + 32 + 32 + 8 + 60 + 8 + 8 + 4

FRAME_ACT_DT = np.dtype([("delta", "<f4"), ("keys", "<u4")])  # == ballrain.tas.FRAME_DTYPE

LIST_SECTIONS = {  # tag -> (count format, item dtype)
    "FRIC": ("<H", FRIC_DT), "MIND": ("<H", MIND_DT), "BODY": ("<H", BODY_DT),
    "EVTS": ("<H", EVT_DT), "WTCH": ("<H", WTCH_DT), "TRAC": ("<I", TRAC_DT),
}
FIXED_SECTIONS = {"TIME": TIME_DT, "BALL": BALL_DT, "CAMS": CAMS_DT, "GLCT": GLCT_DT, "SCOR": SCOR_DT, "SREN": SREN_DT}
_EXTENSIONS = {"SCOR": (SCOR_EXT_DT, "scor_ext"), "SREN": (SREN_EXT_DT, "sren_ext")}  # appended fields

# OBS1 header flags (BallrainIO Agent.cpp)
OBS_FINAL = 1  # last observation of the run
OBS_PREFIX = 2  # written during the prefix replay (BALLRAIN_PREFIX_EVENTS)
OBS_DROPPED = 4  # a section did not fit into the obs area and was left out
OBS_MINIMAL = 8  # only TIME / EVTS (e.g. the first observation at PreLoadLevel)

MSG_HDR = struct.Struct("<IHHIIII")  # magic, version, flags, byte_len, frames_done, section_count, reserved
SEC_HDR = struct.Struct("<IHHI")  # tag, sec_version, sec_flags, payload_len
ACT_HDR = struct.Struct("<IHHIIIIIII")
assert MSG_HDR.size == 24 and SEC_HDR.size == 12 and ACT_HDR.size == 36


# ------------------------------------------------------------------------------ generic sections

@dataclass
class Section:
    tag: str
    version: int
    flags: int
    payload: bytes


@dataclass
class Message:
    magic: int
    version: int
    flags: int
    frames_done: int
    sections: list[Section]
    byte_len: int

    def get(self, tag: str) -> Section | None:
        for s in self.sections:
            if s.tag == tag:
                return s
        return None


def split_message(buf, offset: int = 0, expect_magic: int | None = None) -> Message:
    """Split one OBS1/LVL1 message into sections (no section parsing)."""
    mv = memoryview(buf)
    if len(mv) - offset < MSG_HDR.size:
        raise ProtocolError("message shorter than its header")
    magic, version, flags, byte_len, frames_done, count, _ = MSG_HDR.unpack_from(mv, offset)
    if expect_magic is not None and magic != expect_magic:
        raise ProtocolError(f"bad magic {tag_str(magic)!r}, expected {tag_str(expect_magic)!r}")
    if byte_len < MSG_HDR.size or offset + byte_len > len(mv):
        raise ProtocolError(f"byte_len {byte_len} outside buffer of {len(mv) - offset}")
    end = offset + byte_len
    pos = offset + MSG_HDR.size
    sections = []
    for _ in range(count):
        if pos + SEC_HDR.size > end:
            raise ProtocolError("section header beyond message end")
        tag, sv, sf, plen = SEC_HDR.unpack_from(mv, pos)
        pos += SEC_HDR.size
        if pos + plen > end:
            raise ProtocolError(f"section {tag_str(tag)} payload beyond message end")
        sections.append(Section(tag_str(tag), sv, sf, bytes(mv[pos:pos + plen])))
        pos += align4(plen)
    return Message(magic, version, flags, frames_done, sections, byte_len)


def build_message(magic: int, frames_done: int, sections, version: int = 1, flags: int = 0) -> bytes:
    """sections: iterable of (tag, payload bytes) or (tag, payload, sec_version)."""
    body = bytearray()
    n = 0
    for s in sections:
        tag, payload = s[0], bytes(s[1])
        sv = s[2] if len(s) > 2 else 1
        body += SEC_HDR.pack(fourcc(tag), sv, 0, len(payload)) + payload
        body += b"\0" * (align4(len(payload)) - len(payload))
        n += 1
    return MSG_HDR.pack(magic, version, flags, MSG_HDR.size + len(body), frames_done, n, 0) + bytes(body)


# ---------------------------------------------------------------------------------- observations

@dataclass
class ArrayCell:
    array: int
    row: int
    col: int
    col_type: int
    data: bytes


@dataclass
class Observation:
    """One parsed OBS1 message. Missing optional sections are None (lists: empty arrays)."""
    frames_done: int
    flags: int = 0
    time: np.void | None = None
    ball: np.void | None = None
    fric: np.ndarray = field(default_factory=lambda: np.zeros(0, FRIC_DT))
    mind: np.ndarray = field(default_factory=lambda: np.zeros(0, MIND_DT))
    body: np.ndarray = field(default_factory=lambda: np.zeros(0, BODY_DT))
    cams: np.void | None = None
    glct: np.void | None = None
    time_ext: np.void | None = None
    scor: np.void | None = None
    sren: np.void | None = None
    scor_ext: np.void | None = None  # BallrainIO >= 0.3.1
    sren_ext: np.void | None = None  # BallrainIO >= 0.3.1
    arrs: list = field(default_factory=list)
    events: np.ndarray = field(default_factory=lambda: np.zeros(0, EVT_DT))
    wtch: np.ndarray = field(default_factory=lambda: np.zeros(0, WTCH_DT))
    phys: np.ndarray = field(default_factory=lambda: np.zeros(0, "<u4"))
    trac: np.ndarray | None = None
    unknown: list = field(default_factory=list)  # tags skipped
    raw: bytes = b""

    @property
    def final(self) -> bool:
        return bool(self.flags & OBS_FINAL)

    def event_codes(self) -> list[int]:
        return [int(c) for c in self.events["code"]]

    def has_event(self, *names_or_codes) -> bool:
        want = {EVENT_CODES[e] if isinstance(e, str) else int(e) for e in names_or_codes}
        return any(int(c) in want for c in self.events["code"])


def _parse_fixed(sec: Section, dt: np.dtype):
    if len(sec.payload) < dt.itemsize:
        raise ProtocolError(f"{sec.tag}: payload {len(sec.payload)} < {dt.itemsize}")
    return np.frombuffer(sec.payload, dt, count=1)[0].copy()


def _parse_list(sec: Section, cfmt: str, dt: np.dtype) -> np.ndarray:
    csz = struct.calcsize(cfmt)
    if len(sec.payload) < csz:
        raise ProtocolError(f"{sec.tag}: no count")
    (n,) = struct.unpack_from(cfmt, sec.payload, 0)
    body = len(sec.payload) - csz
    stride = dt.itemsize
    if sec.version > 1 and n:
        stride = body // n  # newer version with appended item fields: keep the known prefix
        if stride < dt.itemsize:
            raise ProtocolError(f"{sec.tag} v{sec.version}: item stride {stride} < {dt.itemsize}")
    if n * stride > body:
        raise ProtocolError(f"{sec.tag}: {n} items of {stride} bytes exceed payload {body}")
    if stride == dt.itemsize:
        return np.frombuffer(sec.payload, dt, count=n, offset=csz).copy()
    raw = np.frombuffer(sec.payload, np.uint8, count=n * stride, offset=csz).reshape(n, stride)
    return np.ascontiguousarray(raw[:, :dt.itemsize]).view(dt).reshape(n).copy()


def parse_arrs(payload: bytes) -> list[ArrayCell]:
    (n,) = struct.unpack_from("<H", payload, 0)
    pos = 2
    out = []
    for _ in range(n):
        arr, row, col, ctype, ln = ARRS_ITEM_HDR.unpack_from(payload, pos)
        pos += ARRS_ITEM_HDR.size
        if pos + ln > len(payload):
            raise ProtocolError("ARRS cell beyond payload")
        out.append(ArrayCell(arr, row, col, ctype, bytes(payload[pos:pos + ln])))
        pos += ln
    return out


def parse_observation(buf, offset: int = 0) -> Observation:
    msg = split_message(buf, offset, OBS_MAGIC)
    if msg.version != OBS_VERSION:
        raise ProtocolError(f"OBS version {msg.version} unsupported")
    o = Observation(frames_done=msg.frames_done, flags=msg.flags,
                    raw=bytes(memoryview(buf)[offset:offset + msg.byte_len]))
    for sec in msg.sections:
        t = sec.tag
        if t in FIXED_SECTIONS:
            setattr(o, t.lower(), _parse_fixed(sec, FIXED_SECTIONS[t]))
            if t == "TIME" and len(sec.payload) >= TIME_DT.itemsize + TIME_EXT_DT.itemsize:
                o.time_ext = np.frombuffer(sec.payload, TIME_EXT_DT, count=1, offset=TIME_DT.itemsize)[0].copy()
            elif t in _EXTENSIONS:
                ext_dt, attr = _EXTENSIONS[t]
                base = FIXED_SECTIONS[t].itemsize
                if len(sec.payload) >= base + ext_dt.itemsize:
                    setattr(o, attr, np.frombuffer(sec.payload, ext_dt, count=1, offset=base)[0].copy())
        elif t in LIST_SECTIONS:
            arr = _parse_list(sec, *LIST_SECTIONS[t])
            setattr(o, {"EVTS": "events"}.get(t, t.lower()), arr)
        elif t == "ARRS":
            o.arrs = parse_arrs(sec.payload)
        elif t == "PHYS":
            (n,) = struct.unpack_from("<H", sec.payload, 0)
            o.phys = np.frombuffer(sec.payload, "<u4", count=n, offset=2).copy()
        else:
            o.unknown.append(t)
    return o


# encoders (used by the fake mod and tests; the real writer is the C++ mod)

def pack_fixed(dt: np.dtype, **values) -> bytes:
    a = np.zeros(1, dt)
    for k, v in values.items():
        a[k] = v
    return a.tobytes()


def pack_list(tag: str, items) -> bytes:
    cfmt, dt = LIST_SECTIONS[tag]
    items = np.asarray(items, dt) if not isinstance(items, np.ndarray) else items.astype(dt, copy=False)
    return struct.pack(cfmt, len(items)) + items.tobytes()


def pack_arrs(cells) -> bytes:
    out = bytearray(struct.pack("<H", len(cells)))
    for c in cells:
        if len(c.data) > 255:
            raise ProtocolError("ARRS cell longer than 255 bytes")
        out += ARRS_ITEM_HDR.pack(c.array, c.row, c.col, c.col_type, len(c.data)) + c.data
    return bytes(out)


def pack_phys_list(seqs) -> bytes:
    seqs = np.asarray(seqs, "<u4")
    return struct.pack("<H", len(seqs)) + seqs.tobytes()


# ---------------------------------------------------------------------------------------- actions

@dataclass
class Action:
    frames: np.ndarray  # FRAME_ACT_DT / tas.FRAME_DTYPE
    repeat: int = 1
    stop_mask: int = 0
    flags: int = 0
    obs_mask: int = 0

    @property
    def total_frames(self) -> int:
        return len(self.frames) * self.repeat

    def expanded(self) -> np.ndarray:
        return np.repeat(self.frames, self.repeat) if self.repeat != 1 else self.frames


def encode_action(a: Action, py_seq: int) -> bytes:
    frames = np.asarray(a.frames)
    if frames.dtype != FRAME_ACT_DT:
        raise TypeError(f"frames must have dtype {FRAME_ACT_DT}")
    if a.repeat < 1:
        raise ValueError("repeat must be >= 1")
    hdr = ACT_HDR.pack(ACT_MAGIC, ACT_VERSION, a.flags, py_seq & 0xFFFFFFFF, len(frames), a.repeat,
                       a.stop_mask & 0xFFFFFFFF, (a.stop_mask >> 32) & 0xFFFFFFFF, a.obs_mask, 0)
    return hdr + frames.tobytes()


def decode_action(buf, offset: int = 0) -> tuple[Action, int]:
    magic, ver, flags, seq, n, rep, lo, hi, om, _ = ACT_HDR.unpack_from(buf, offset)
    if magic != ACT_MAGIC or ver != ACT_VERSION:
        raise ProtocolError("bad ACT header")
    frames = np.frombuffer(buf, FRAME_ACT_DT, count=n, offset=offset + ACT_HDR.size).copy()
    return Action(frames, rep, lo | (hi << 32), flags, om), seq


# --------------------------------------------------------------------------------- level messages

class _Reader:
    def __init__(self, payload: bytes, strings=None):
        self.b = payload
        self.p = 0
        self.strings = strings or []

    def u(self, fmt):
        v = struct.unpack_from(fmt, self.b, self.p)
        self.p += struct.calcsize(fmt)
        return v if len(v) > 1 else v[0]

    def arr(self, dt, n):
        dt = np.dtype(dt)
        a = np.frombuffer(self.b, dt, count=n, offset=self.p).copy()
        self.p += dt.itemsize * n
        return a

    def raw(self, n):
        v = bytes(self.b[self.p:self.p + n])
        if len(v) != n:
            raise ProtocolError("short read")
        self.p += n
        return v

    def s(self):
        i = self.u("<I")
        if i == NO_STRING:
            return None
        return self.lookup(i)

    def lookup(self, i):
        if i >= len(self.strings):
            raise ProtocolError(f"string index {i} outside table of {len(self.strings)}")
        return self.strings[i]


class _Writer:
    def __init__(self, strings: list):
        self.b = bytearray()
        self.strings = strings
        self._idx = {s: i for i, s in enumerate(strings)}

    def p(self, fmt, *v):
        self.b += struct.pack(fmt, *v)

    def arr(self, a, dt):
        self.b += np.asarray(a, dt).tobytes()

    def s(self, text):
        if text is None:
            self.p("<I", NO_STRING)
            return
        if text not in self._idx:
            self._idx[text] = len(self.strings)
            self.strings.append(text)
        self.p("<I", self._idx[text])


def _dec(b: bytes) -> str:
    return b.decode("latin-1")


def _enc(s: str) -> bytes:
    return s.encode("latin-1")


@dataclass
class Entity:
    ckid: int
    name: str | None
    class_id: int
    parent: int
    visible: bool
    matrix: np.ndarray  # (4,3)
    contact_id: int  # Continuous Contact ID, -1 = attribute absent
    coll_id: int  # Coll Detection ID, -1 = absent
    floor: bytes  # raw bytes of the Floor attribute (empty = absent)


@dataclass
class Group:
    ckid: int
    name: str | None
    members: np.ndarray  # u4


@dataclass
class ArrayColumn:
    name: str | None
    type: int
    param_type: str | None  # parameter type name for PARAMETER columns ("Matrix", "Boolean", ...)


@dataclass
class ArrayDef:
    ckid: int
    name: str | None
    columns: list[ArrayColumn]
    cells: list[list[bytes]]  # rows x cols raw cell bytes


@dataclass
class PhysRecord:
    seq: int
    frame: int
    ckid: int
    unphysicalize: bool
    fixed: bool = False
    friction: float = 0.0
    elasticity: float = 0.0
    mass: float = 0.0
    collision_group: str | None = None
    frozen: bool = False
    collision: bool = True
    calc_mass_center: bool = False
    linear_damping: float = 0.0
    rot_damping: float = 0.0
    surface: str | None = None
    mass_center: tuple = (0.0, 0.0, 0.0)
    convex: list = field(default_factory=list)
    balls: list = field(default_factory=list)  # [(center(3), radius)]
    concave: list = field(default_factory=list)


@dataclass
class Mesh:
    ckid: int  # entity
    mesh: int
    kind: int  # 0 convex, 1 concave
    matrix: np.ndarray  # (4,4) world matrix at export
    vertices: np.ndarray  # (nv,3) f4 local
    faces: np.ndarray  # (nf,3) u4


@dataclass
class IvpObject:
    ckid: int
    object_type: int
    unmoveable: bool
    extra_radius: float
    material: bytes


@dataclass
class LevelMessage:
    frames_done: int
    strings: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    arrays: list = field(default_factory=list)
    phys: list = field(default_factory=list)
    meshes: list = field(default_factory=list)
    ivp: list = field(default_factory=list)
    depth: list = field(default_factory=list)  # [(ckid, min(3), max(3))]
    watch: list = field(default_factory=list)  # [(ckid, group name)]
    entity_meshes: list = field(default_factory=list)  # [(ckid, mesh ckid)]
    tables: list = field(default_factory=list)  # [{"cams": [5 ckids], "contact_bbs": [...]}]
    unknown: list = field(default_factory=list)


def _read_ents(r: _Reader):
    out = []
    for _ in range(r.u("<I")):
        ckid, name_i = r.u("<II")
        # the mod writes an all-zero record (name index 0) for a missing entity: ckid 0 is skipped later
        name = None if ckid == 0 or name_i == NO_STRING else r.lookup(name_i)
        cls, parent, vis = r.u("<IIB")
        m = r.arr("<f4", 12).reshape(4, 3)
        cid, coll = r.u("<ii")
        floor = r.raw(r.u("<H"))
        out.append(Entity(ckid, name, cls, parent, bool(vis), m, cid, coll, floor))
    return out


def _write_ents(w: _Writer, ents):
    w.p("<I", len(ents))
    for e in ents:
        w.p("<I", e.ckid)
        w.s(e.name)
        w.p("<IIB", e.class_id, e.parent, int(e.visible))
        w.arr(np.asarray(e.matrix).reshape(12), "<f4")
        w.p("<iiH", e.contact_id, e.coll_id, len(e.floor))
        w.b += e.floor


def _read_grps(r):
    out = []
    for _ in range(r.u("<I")):
        ckid = r.u("<I")
        name = r.s()
        n = r.u("<I")
        out.append(Group(ckid, name, r.arr("<u4", n)))
    return out


def _write_grps(w, grps):
    w.p("<I", len(grps))
    for g in grps:
        w.p("<I", g.ckid)
        w.s(g.name)
        w.p("<I", len(g.members))
        w.arr(g.members, "<u4")


def _read_arrs(r):
    out = []
    for _ in range(r.u("<I")):
        ckid = r.u("<I")
        name = r.s()
        rows, ncols = r.u("<HH")
        cols = []
        for _c in range(ncols):
            cname = r.s()
            ctype = r.u("<B")
            cols.append(ArrayColumn(cname, ctype, r.s()))
        cells = [[r.raw(r.u("<B")) for _c in range(ncols)] for _r in range(rows)]
        out.append(ArrayDef(ckid, name, cols, cells))
    return out


def _write_arrs(w, arrs):
    w.p("<I", len(arrs))
    for a in arrs:
        w.p("<I", a.ckid)
        w.s(a.name)
        w.p("<HH", len(a.cells), len(a.columns))
        for c in a.columns:
            w.s(c.name)
            w.p("<B", c.type)
            w.s(c.param_type)
        for row in a.cells:
            for cell in row:
                w.p("<B", len(cell))
                w.b += cell


def _read_phys(r):
    out = []
    for _ in range(r.u("<I")):
        seq, frame, ckid, unphys, fixed = r.u("<IIIBB")
        fr, el, mass = r.u("<fff")
        grp = r.s()
        frozen, coll, calcmc = r.u("<BBB")
        ld, rd = r.u("<ff")
        surf = r.s()
        mc = tuple(r.arr("<f4", 3).tolist())
        conv = r.arr("<u4", r.u("<H")).tolist()
        balls = []
        for _b in range(r.u("<H")):
            c = r.arr("<f4", 3)
            balls.append((c, r.u("<f")))
        conc = r.arr("<u4", r.u("<H")).tolist()
        out.append(PhysRecord(seq, frame, ckid, bool(unphys), bool(fixed), fr, el, mass, grp, bool(frozen),
                              bool(coll), bool(calcmc), ld, rd, surf, mc, conv, balls, conc))
    return out


def _write_phys(w, recs):
    w.p("<I", len(recs))
    for p in recs:
        w.p("<IIIBB", p.seq, p.frame, p.ckid, int(p.unphysicalize), int(p.fixed))
        w.p("<fff", p.friction, p.elasticity, p.mass)
        w.s(p.collision_group)
        w.p("<BBB", int(p.frozen), int(p.collision), int(p.calc_mass_center))
        w.p("<ff", p.linear_damping, p.rot_damping)
        w.s(p.surface)
        w.arr(p.mass_center, "<f4")
        w.p("<H", len(p.convex))
        w.arr(p.convex, "<u4")
        w.p("<H", len(p.balls))
        for c, rad in p.balls:
            w.arr(c, "<f4")
            w.p("<f", rad)
        w.p("<H", len(p.concave))
        w.arr(p.concave, "<u4")


def _read_mesh(r):
    out = []
    for _ in range(r.u("<I")):
        ckid, mesh, kind = r.u("<IIB")
        m = r.arr("<f4", 16).reshape(4, 4)
        v = r.arr("<f4", 3 * r.u("<I")).reshape(-1, 3)
        f = r.arr("<u4", 3 * r.u("<I")).reshape(-1, 3)
        out.append(Mesh(ckid, mesh, kind, m, v, f))
    return out


def _write_mesh(w, meshes):
    w.p("<I", len(meshes))
    for m in meshes:
        w.p("<IIB", m.ckid, m.mesh, m.kind)
        w.arr(np.asarray(m.matrix).reshape(16), "<f4")
        w.p("<I", len(m.vertices))
        w.arr(m.vertices, "<f4")
        w.p("<I", len(m.faces))
        w.arr(m.faces, "<u4")


def _read_ivpo(r):
    out = []
    for _ in range(r.u("<I")):
        ckid, ot, um, xr = r.u("<IiBf")
        out.append(IvpObject(ckid, ot, bool(um), xr, r.raw(32)))
    return out


def _write_ivpo(w, objs):
    w.p("<I", len(objs))
    for o in objs:
        w.p("<IiBf", o.ckid, o.object_type, int(o.unmoveable), o.extra_radius)
        w.b += o.material.ljust(32, b"\0")[:32]


def _read_dept(r):
    out = []
    for _ in range(r.u("<I")):
        ckid = r.u("<I")
        box = r.arr("<f4", 6)
        out.append((ckid, box[:3], box[3:]))
    return out


def _write_dept(w, boxes):
    w.p("<I", len(boxes))
    for ckid, lo, hi in boxes:
        w.p("<I", ckid)
        w.arr(np.concatenate([lo, hi]), "<f4")


def _read_emsh(r):
    return [tuple(r.u("<II")) for _ in range(r.u("<I"))]  # (entity, current mesh)


def _write_emsh(w, items):
    w.p("<I", len(items))
    for e, m in items:
        w.p("<II", e, m)


def _read_ltab(r):
    cams = list(r.u("<5I"))
    n = r.u("<I")
    return [{"cams": cams, "contact_bbs": r.arr("<u4", n).tolist()}]


def _write_ltab(w, items):
    t = items[0]
    w.p("<5I", *t["cams"])
    w.p("<I", len(t["contact_bbs"]))
    w.arr(t["contact_bbs"], "<u4")


def _read_wlst(r):
    out = []
    for _ in range(r.u("<I")):
        ckid = r.u("<I")
        out.append((ckid, r.s()))
    return out


def _write_wlst(w, items):
    w.p("<I", len(items))
    for ckid, grp in items:
        w.p("<I", ckid)
        w.s(grp)


_LVL_SECTIONS = {  # tag -> (attribute, reader, writer)
    "ENTS": ("entities", _read_ents, _write_ents),
    "GRPS": ("groups", _read_grps, _write_grps),
    "ARRS": ("arrays", _read_arrs, _write_arrs),
    "PHYS": ("phys", _read_phys, _write_phys),
    "MESH": ("meshes", _read_mesh, _write_mesh),
    "IVPO": ("ivp", _read_ivpo, _write_ivpo),
    "DEPT": ("depth", _read_dept, _write_dept),
    "WLST": ("watch", _read_wlst, _write_wlst),
    "EMSH": ("entity_meshes", _read_emsh, _write_emsh),  # BallrainIO extension
    "LTAB": ("tables", _read_ltab, _write_ltab),  # BallrainIO extension: camera ids, contact blocks
}


def parse_level_message(buf, offset: int = 0) -> tuple[LevelMessage, int]:
    """Parse one LVL1 message at offset; returns (message, byte_len)."""
    msg = split_message(buf, offset, LVL_MAGIC)
    lm = LevelMessage(msg.frames_done)
    strs = msg.get("STRS")
    if strs is not None:
        r = _Reader(strs.payload)
        lm.strings = [_dec(r.raw(r.u("<H"))) for _ in range(r.u("<I"))]
    for sec in msg.sections:
        if sec.tag == "STRS":
            continue
        spec = _LVL_SECTIONS.get(sec.tag)
        if spec is None:
            lm.unknown.append(sec.tag)
            continue
        try:
            setattr(lm, spec[0], spec[1](_Reader(sec.payload, lm.strings)))
        except struct.error as e:
            raise ProtocolError(f"LVL1 {sec.tag}: {e}") from e
    return lm, msg.byte_len


def parse_level_stream(data: bytes) -> tuple[list[LevelMessage], int]:
    """Parse concatenated LVL1 messages (the .inc file). A trailing incomplete message is left
    unread (the writer may be mid-append); returns (messages, bytes consumed)."""
    out, pos = [], 0
    while len(data) - pos >= MSG_HDR.size:
        (byte_len,) = struct.unpack_from("<I", data, pos + 8)
        if pos + byte_len > len(data):
            break
        lm, n = parse_level_message(data, pos)
        out.append(lm)
        pos += n
    return out, pos


def build_level_message(lm: LevelMessage) -> bytes:
    strings: list = []
    payloads = []
    for tag, (attr, _r, wfn) in _LVL_SECTIONS.items():
        items = getattr(lm, attr)
        if not items:
            continue
        w = _Writer(strings)
        wfn(w, items)
        payloads.append((tag, bytes(w.b)))
    sw = bytearray(struct.pack("<I", len(strings)))
    for s in strings:
        b = _enc(s)
        sw += struct.pack("<H", len(b)) + b
    return build_message(LVL_MAGIC, lm.frames_done, [("STRS", bytes(sw))] + payloads)


# ------------------------------------------------------------------------------- cell decoding

def decode_cell(data: bytes, col_type: int, param_type: str | None = None):
    """Typed value of a raw array cell: int, float, str, ckid (int), matrix (4,4) or raw bytes."""
    if col_type == ARR_INT or col_type == ARR_OBJECT:
        return struct.unpack("<i" if col_type == ARR_INT else "<I", data[:4])[0] if len(data) >= 4 else 0
    if col_type == ARR_FLOAT:
        return struct.unpack("<f", data[:4])[0] if len(data) >= 4 else 0.0
    if col_type == ARR_STRING:
        return _dec(data.rstrip(b"\0"))
    if col_type == ARR_PARAMETER:
        pt = (param_type or "").lower()
        if len(data) == 64 or pt == "matrix":
            return np.frombuffer(data[:64], "<f4").reshape(4, 4).copy()
        if pt in ("boolean", "bool"):
            return bool(struct.unpack("<i", data[:4])[0])
        if pt in ("float", "time", "percentage") and len(data) >= 4:
            return struct.unpack("<f", data[:4])[0]
        if len(data) == 4:
            return struct.unpack("<i", data)[0]
        return data
    return data


def encode_cell(value, col_type: int) -> bytes:
    if col_type == ARR_INT:
        return struct.pack("<i", int(value))
    if col_type == ARR_OBJECT:
        return struct.pack("<I", int(value))
    if col_type == ARR_FLOAT:
        return struct.pack("<f", float(value))
    if col_type == ARR_STRING:
        return _enc(value)
    if isinstance(value, np.ndarray):
        return np.asarray(value, "<f4").tobytes()
    if isinstance(value, bool):
        return struct.pack("<i", int(value))
    if isinstance(value, (int, np.integer)):
        return struct.pack("<i", int(value))
    if isinstance(value, float):
        return struct.pack("<f", value)
    return bytes(value)

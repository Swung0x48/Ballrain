"""Pool wire protocol (docs/pool-prototype.md): strict JSON control messages and the binary chunk codec.

Nothing in here executes code from its input: JSON is parsed with the standard parser (duplicate keys, NaN /
Infinity, oversize input and deep nesting refused before or while parsing) and then validated against a fixed
schema (required fields, exact types -- a bool is never an int --, ranges, string lengths, no unknown fields);
chunks are a fixed binary layout whose arrays are read with numpy.frombuffer at a dtype fixed by this module (never
the sender's), after their lengths and sha256 were checked. No pickle, no numpy.load, no eval.

Chunk layout (all integers little-endian):

    magic  b"BRPC"            4 bytes
    format u16 (CHUNK_FORMAT) 2
    flags  u16 (0)            2
    hlen   u32                4
    header JSON (UTF-8)       hlen bytes   (CHUNK_HEADER schema; its "arrays" list describes the body sections)
    body                      sum of the sections' stored sizes, in the order of "arrays"
    digest sha256             32 bytes over everything before it

Each section: {"name", "codec": "raw" | "zstd", "nbytes" (stored), "sha256" (of the decoded bytes)}; its dtype and
shape are not sent: they follow from the name (CHUNK_ARRAYS) and the header's steps / n_obs, so a sender cannot
choose them. A zstd section is decompressed with an output bound equal to the expected decoded size and a window of
at most 8 MiB; its stored size may not exceed zstd_bound(decoded size) (no padded frames), a raw section's stored size
is exactly its decoded size.

Engine-state hash chain (chain v3, per task, across its chunks; ballrain.pool.engine_chain): h(-1) =
chain_seed(task_id), h(i) = sha256(h(i-1) || d_i), d_i = the digest of the canonical engine messages (OBS1 and LVL1
bytes, address-carrying fields left out) received after row i-1 up to row i, over the task's observation rows (the
observation before each step, plus the one after the last step). Chunk seq s carries rows i = step0 .. step0 + steps
(steps + 1 rows) and chain_prev = h(step0 - 1); so the last row of chunk s is the first row of chunk s + 1. The chain
commits to what the engine sent, not to the features computed from it: the verifier's replay must reproduce the
engine bytes exactly (the engine is deterministic), while the features, policy outputs and rewards it recomputes
are compared with tolerances (another CPU / compiler rounds slightly differently). The float16 copy of the
observations ("obs") is what the learner trains on; it must lie within obs_tolerance of the verifier's own features.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass

import numpy as np

from .reasons import Reason, payload

PROTOCOL_VERSION = 1
CHUNK_FORMAT = 3  # 2: the obs_hash chain over the engine state (chain v2); 3: chain v3 (machine-dependent fields out)
CHUNK_MAGIC = b"BRPC"
_PREAMBLE = struct.Struct("<4sHHI")

MAX_JSON_BYTES = 64 * 1024  # control messages
MAX_JSON_DEPTH = 12
MAX_CHUNK_BYTES = 4 * 1024 * 1024  # one encoded chunk (the gateway's request-body limit)
MAX_CHUNK_HEADER_BYTES = 16 * 1024
MAX_CHUNK_STEPS = 4096
MAX_N_OBS = 1024
MAX_N_ACT = 64
MAX_TASK_STEPS = 1 << 20
MAX_LOGP = 1e-5  # rounding slack above 0
MIN_LOGP = -60.0

ENGINE_KINDS = ("2002", "x87soft")
PLATFORMS = ("windows", "macos", "linux")
TASK_KINDS = ("sample", "eval")
# why a task's last chunk ended (the "end" of a final chunk)
END_REASONS = ("LevelFinish", "GameOver", "truncated", "cap", "stall", "max_steps", "max_frames", "deadline",
               "stopped", "error", "respawn_skipped")  # respawn_skipped: format 3 (counters.skip_respawn)
# ends the client declares (the replay cannot reproduce them: the player stopped, the deadline passed, the local
# engine failed); the final step is then marked truncated by the client. Every other end is checked by the replay.
CLIENT_ENDS = ("deadline", "stopped", "error")
FLAG_TERMINATED = 1
FLAG_TRUNCATED = 2

HEX64 = re.compile(r"[0-9a-f]{64}")
HEX32 = re.compile(r"[0-9a-f]{32}")
ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
BRANCH_ID_RE = re.compile(r"[a-z][a-z0-9-]*")  # a multi-branch manifest member id (ballrain.multi_branch)
VERSION_RE = re.compile(r"[0-9A-Za-z.+_-]{1,32}")
USER_CODE_RE = re.compile(r"[A-Z2-9]{4}-[A-Z2-9]{4}")

PHYSICS_RT_2002_SHA256 = "e72e4afcfa5c33a7d3d27776137f8c997b3c52d89d8a8a4745f1ca21e45893ec"
# files of the player's Ballance folder the client hashes (docs/volunteer-compute.md 1); directories end in "/"
# and are hashed as a tree (tree_digest). The x87-soft edition loads only these four (measured: the engine runs
# bit-identically with nothing else of the folder); the 2002 engine also loads BuildingBlocks/physics_RT.dll and the
# retail Bin/VxMath.dll (phys2002's VxQuaternion::FromMatrix import)
GAME_FILE_LIST = ("base.cmo", "Database.tdb", "3D Entities/", "Text/")
GAME_FILE_LIST_2002 = GAME_FILE_LIST + ("Bin/VxMath.dll", "BuildingBlocks/physics_RT.dll")


class ProtocolError(ValueError):
    """A message or chunk that does not follow the protocol. .reason / .params build the refusal payload."""

    def __init__(self, reason: Reason, detail: str = "", **params):
        self.reason = Reason(reason)
        self.params = dict(params)
        if detail:
            self.params.setdefault("detail", detail)
        super().__init__(f"{self.reason.value}: {detail or params}")

    def payload(self) -> dict:
        return payload(self.reason, **self.params)


# ============================================================================================== strict JSON

def _no_constant(name):
    raise ValueError(f"{name} is not allowed")


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k[:40]!r}")
        out[k] = v
    return out


def _depth(o, d=0) -> int:
    if d > MAX_JSON_DEPTH:
        raise ValueError(f"nesting deeper than {MAX_JSON_DEPTH}")
    if isinstance(o, dict):
        return max([_depth(v, d + 1) for v in o.values()], default=d)
    if isinstance(o, list):
        return max([_depth(v, d + 1) for v in o], default=d)
    return d


def loads_strict(data: bytes | str, max_bytes: int = MAX_JSON_BYTES, reason: Reason = Reason.BAD_MESSAGE):
    """Parse JSON refusing oversize input (checked before parsing), duplicate keys, NaN / Infinity, invalid UTF-8
    and nesting deeper than MAX_JSON_DEPTH. Raises ProtocolError(reason)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ProtocolError(reason, "not bytes")
    if len(data) > max_bytes:
        raise ProtocolError(reason, f"{len(data)} bytes > {max_bytes}")
    # nesting is checked on the raw text first (a deeply nested array would hit the parser's recursion)
    depth = 0
    in_str = esc = False
    for b in bytes(data):
        if in_str:
            if esc:
                esc = False
            elif b == 0x5C:
                esc = True
            elif b == 0x22:
                in_str = False
        elif b == 0x22:
            in_str = True
        elif b in (0x5B, 0x7B):
            depth += 1
            if depth > MAX_JSON_DEPTH + 1:
                raise ProtocolError(reason, f"nesting deeper than {MAX_JSON_DEPTH}")
        elif b in (0x5D, 0x7D):
            depth -= 1
    try:
        text = bytes(data).decode("utf-8")
        o = json.loads(text, object_pairs_hook=_pairs, parse_constant=_no_constant)
        _depth(o)
    except (ValueError, RecursionError) as e:
        raise ProtocolError(reason, f"invalid JSON: {str(e)[:120]}") from None
    return o


def dumps(obj) -> bytes:
    """Canonical JSON (sorted keys, no spaces, UTF-8, NaN / Infinity refused): what every sender writes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


# ============================================================================================== schema

class Spec:
    nullable = False

    def check(self, v, path: str):  # pragma: no cover - interface
        raise NotImplementedError


def _fail(path, msg):
    raise ProtocolError(Reason.BAD_MESSAGE, f"{path}: {msg}")


class Str(Spec):
    def __init__(self, max_len: int = 200, pattern: re.Pattern | None = None, enum=None, min_len: int = 0,
                 nullable: bool = False):
        self.max_len, self.pattern, self.enum, self.min_len, self.nullable = max_len, pattern, enum, min_len, nullable

    def check(self, v, path):
        if not isinstance(v, str):
            _fail(path, "not a string")
        if not self.min_len <= len(v) <= self.max_len:
            _fail(path, f"length {len(v)} not in [{self.min_len}, {self.max_len}]")
        if self.enum is not None and v not in self.enum:
            _fail(path, f"{v[:40]!r} not one of {list(self.enum)}")
        if self.pattern is not None and not self.pattern.fullmatch(v):
            _fail(path, "bad format")
        if any(not (ch.isprintable() or ch in "\n\t") for ch in v):
            _fail(path, "control characters")
        return v


class Int(Spec):
    def __init__(self, lo: int = 0, hi: int = 2 ** 53, nullable: bool = False):
        self.lo, self.hi, self.nullable = lo, hi, nullable

    def check(self, v, path):
        if isinstance(v, bool) or not isinstance(v, int):
            _fail(path, "not an integer")
        if not self.lo <= v <= self.hi:
            _fail(path, f"{v} not in [{self.lo}, {self.hi}]")
        return v


class Float(Spec):
    def __init__(self, lo: float = -1e300, hi: float = 1e300, nullable: bool = False):
        self.lo, self.hi, self.nullable = lo, hi, nullable

    def check(self, v, path):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _fail(path, "not a number")
        if not math.isfinite(v) or not self.lo <= v <= self.hi:
            _fail(path, f"{v} not in [{self.lo}, {self.hi}]")
        return float(v)


class Bool(Spec):
    def __init__(self, nullable: bool = False):
        self.nullable = nullable

    def check(self, v, path):
        if not isinstance(v, bool):
            _fail(path, "not a boolean")
        return v


class Const(Spec):
    def __init__(self, value):
        self.value = value

    def check(self, v, path):
        if type(v) is not type(self.value) or v != self.value:
            _fail(path, f"must be {self.value!r}")
        return v


class List(Spec):
    def __init__(self, item: Spec, max_items: int = 64, min_items: int = 0, nullable: bool = False):
        self.item, self.max_items, self.min_items, self.nullable = item, max_items, min_items, nullable

    def check(self, v, path):
        if not isinstance(v, list):
            _fail(path, "not a list")
        if not self.min_items <= len(v) <= self.max_items:
            _fail(path, f"{len(v)} items not in [{self.min_items}, {self.max_items}]")
        return [check(self.item, x, f"{path}[{i}]") for i, x in enumerate(v)]


class Map(Spec):
    """An object with free keys (key pattern, at most max_items) and values of one spec."""

    def __init__(self, value: Spec, key: Str | None = None, max_items: int = 64, nullable: bool = False):
        self.value, self.key, self.max_items, self.nullable = value, key or Str(64, ID_RE), max_items, nullable

    def check(self, v, path):
        if not isinstance(v, dict):
            _fail(path, "not an object")
        if len(v) > self.max_items:
            _fail(path, f"{len(v)} keys > {self.max_items}")
        return {self.key.check(k, f"{path} key"): check(self.value, x, f"{path}.{k}") for k, x in v.items()}


class Obj(Spec):
    """An object with known fields: {name: spec} required, {name: spec} optional; unknown fields refused."""

    def __init__(self, required: dict, optional: dict | None = None, nullable: bool = False):
        self.required, self.optional, self.nullable = required, optional or {}, nullable

    def check(self, v, path):
        if not isinstance(v, dict):
            _fail(path, "not an object")
        unknown = set(v) - set(self.required) - set(self.optional)
        if unknown:
            _fail(path, f"unknown fields {sorted(unknown)[:5]}")
        out = {}
        for k, s in self.required.items():
            if k not in v:
                _fail(path, f"missing field {k!r}")
            out[k] = check(s, v[k], f"{path}.{k}")
        for k, s in self.optional.items():
            if k in v:
                out[k] = check(s, v[k], f"{path}.{k}")
        return out


def check(spec: Spec, v, path: str = "$"):
    if v is None:
        if spec.nullable:
            return None
        _fail(path, "null")
    return spec.check(v, path)


def nullable(spec: Spec) -> Spec:
    spec.nullable = True
    return spec


# ---------------------------------------------------------------------------------------------- shared parts

HEX64_S = Str(64, HEX64, min_len=64)
SHA = HEX64_S
TASK_ID = Str(64, ID_RE, min_len=1)
NICK = Str(64, min_len=0)
class _Scalar(Spec):
    """A reason parameter: a string (<= 200 characters), an integer, a finite float or a boolean."""

    def check(self, v, path):
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return Int(-2 ** 53, 2 ** 53).check(v, path)
        if isinstance(v, float):
            return Float().check(v, path)
        return Str(200).check(v, path)


def _reason(nullable_: bool = False) -> Obj:
    return Obj({"code": Str(40, enum=[r.value for r in Reason]), "message": Str(1000),
                "params": Map(_Scalar(), max_items=16)}, nullable=nullable_)


REASON = _reason()

# hello.game_files: exactly the GAME_FILE_LIST(_2002) names as keys (the server compares each with its sha256)
GAME_FILES = Map(HEX64_S, key=Str(64, enum=GAME_FILE_LIST_2002, min_len=1), max_items=len(GAME_FILE_LIST_2002))
ENGINE = Obj({"kind": Str(16, enum=ENGINE_KINDS), "sha256": SHA, "version": Str(64, VERSION_RE, min_len=1)},
             {"hello": Str(200)})
USER = Obj({"forum_id": Int(1, 2 ** 53), "username": Str(64, min_len=1), "nickname": NICK},
           {"avatar_url": Str(500, nullable=True)})
STATS = Obj({"steps_accepted": Int(0), "steps_pending": Int(0), "steps_verified": Int(0), "chunks_accepted": Int(0),
             "chunks_rejected": Int(0)}, {"last_rejection": _reason(True)})

# The task (ballrain.pool.task.Task.to_dict); the client never chooses any of it
REACH_ASSET = Obj({"level": Int(1, 13), "key_sha256": SHA, "sha256": SHA, "size": Int(1, 128 << 20), "zsha256": SHA,
                   "zsize": Int(1, 32 << 20)})  # served at /pool/assets/<zsha256>.zst (no url field)
FEATURE_MODULE_ASSET = Obj({"sha256": SHA, "size": Int(8, 64 << 20), "sig": Str(128, re.compile(r"[0-9a-f]{128}"), min_len=128),
                            "abi": Int(1, 1000)})
TASK = Obj({
    "task_id": TASK_ID,
    "kind": Str(16, enum=TASK_KINDS),
    "run": Str(64, ID_RE, min_len=1),
    "policy_sha": SHA,
    "level": Int(1, 13),
    "record": Str(64, ID_RE, min_len=1),  # the start record (dataset name)
    "record_sha256": SHA,
    "start_frame": Int(0, 10 ** 7),  # k: the record's first k frames are the prefix; 0 = from level load
    "ball_order_seed": Int(0, 2 ** 32 - 1),  # headless retail:<seed>
    "engine": Str(16, enum=ENGINE_KINDS),
    "chunk_steps": Int(1, MAX_CHUNK_STEPS),
    "max_steps": Int(1, MAX_TASK_STEPS),  # decision steps over the whole task (all chunks)
    "max_frames": Int(1, 10 ** 8),  # game frames from the hand-over (cap; truncation "max_frames")
    "deadline_s": Int(1, 7 * 86400),  # wall seconds from issue until the last chunk must have arrived
    "issued_at": Int(0, 2 ** 40),  # unix seconds (server clock)
    "greedy": Bool(),  # eval tasks: argmax actions (behaviour log-probs still sent)
    "sample_seed": Int(0, 2 ** 53 - 1),  # sampling tasks: action draws u_t = sampling.sample_u(seed, t)
    # sha256 of the start-state pack and of the reference line asset; branch: the route-branch pack when the
    # policy follows one (its manifest's features.branch_sha256 pins the pack's files); reach (f7): one entry per
    # fastreach fields file the level can need (docs/f7-native-port.md 8), served zstd-compressed at
    # /pool/assets/<zsha256>.zst; the client picks the entry whose key_sha256 is sha256 of its geometry's fields key
    # feature_module (feature_module_v1 sessions only, ballrain/pool/modules.py): the signed wasm feature module,
    # served at /pool/assets/<sha256>.wasm; sig: ed25519 over its bytes (hex), checked by the client's built-in key
    "assets": Obj({"starts": SHA, "path": SHA}, {"branch": SHA, "reach": List(REACH_ASSET, 8),
                                                 "feature_module": FEATURE_MODULE_ASSET}),
}, {
    # format 3 (docs/pool-prototype.md 5): an at-rest start after the hand-over at start_frame (the env's rest_starts):
    # drive to target (x, z) within max_frames, then brake; target null = brake only. Drawn by the server
    # (max_frames absent = the env's rest_max_frames)
    # branch_s (target null): the stone branch's at-rest start, the measured drive.npz replay to this arc length of
    # the branch line (env._branch_rest; rest_starts lo-hi@branch:s); branch_id (multi-branch policies only, wire v2,
    # docs/pool-prototype.md 15): the manifest member whose drive it is (lo-hi@branch:<id>:s), absent = the unique
    # legacy member
    # reward_inputs_v1 sessions (the gateway recomputes the reward, server_rewards.py): the chunks carry rin / episodes
    "reward_inputs": Bool(),
    "rest": Obj({"target": List(Float(-1e6, 1e6), 2, 2, nullable=True)},
                {"max_frames": Int(1, 10 ** 6), "branch_s": Float(0, 1e6), "branch_id": Str(32, BRANCH_ID_RE, min_len=1)},
                nullable=True),
    # the per-life draws (ballrain.pool.counters): counter randomisation (the manifest's rollout.counter_rand) and the
    # early-respawn draw (rollout.p_early_respawn)
    "life_seed": Int(0, 2 ** 53 - 1),
    # the curriculum frontier when the task was issued: a death whose respawn sector the record enters before it
    # skips the respawn life (rollout.p_early_respawn; absent = every respawn is played)
    "respawn_frontier": Int(0, 10 ** 7),
})


# ---------------------------------------------------------------------------------------------- messages

def _msg(type_: str, required: dict, optional: dict | None = None) -> Obj:
    return Obj(dict({"protocol": Const(PROTOCOL_VERSION), "type": Const(type_)}, **required), optional)


INSTANCE = Obj({"slot": Int(0, 63), "state": Str(16, enum=("idle", "starting", "running", "paused", "error")),
                "steps": Int(0)}, {"task_id": nullable(Str(64, ID_RE, min_len=1)), "error": Str(300)})

MESSAGES: dict[str, Obj] = {
    # ---- login (POST /pool/login): the controller starts a login, the player authorises in the browser, the
    # controller polls with the secret it got
    "login_start": _msg("login_start", {"client_version": Str(32, VERSION_RE, min_len=1)}),
    "login_started": _msg("login_started", {"login_id": Str(64, ID_RE, min_len=16), "poll_secret": Str(64, ID_RE, min_len=32),
                                            "browser_url": Str(500, min_len=1), "user_code": Str(9, USER_CODE_RE, min_len=9),
                                            "expires_in_s": Int(1, 3600),
                                            "poll_s": Int(1, 60)}),
    "login_poll": _msg("login_poll", {"login_id": Str(64, ID_RE, min_len=16), "poll_secret": Str(64, ID_RE, min_len=32)}),
    "login_status": _msg("login_status", {"status": Str(16, enum=("pending", "ok", "refused", "expired"))},
                         {"token": Str(128, ID_RE, min_len=32), "user": USER, "reason": REASON}),
    # ---- session
    # capabilities: what the client can run (ballrain.pool.capabilities; absent = none); the gateway hands a policy's
    # tasks only to clients that have its manifest's "requires". Unknown names are allowed (and ignored)
    "hello": _msg("hello", {"client_version": Str(32, VERSION_RE, min_len=1), "platform": Str(16, enum=PLATFORMS),
                            "engine": ENGINE, "game_files": GAME_FILES, "instances": Int(1, 64)},
                  {"capabilities": List(Str(32, ID_RE, min_len=1), 32)}),
    "hello_ok": _msg("hello_ok", {"session_id": Str(64, ID_RE, min_len=16), "max_instances": Int(1, 64),
                                  "heartbeat_s": Int(5, 3600), "task_poll_s": Int(1, 3600), "user": USER,
                                  "leaderboard": Bool(), "server_time": Int(0, 2 ** 40)},
                     {"notice": Str(500)}),
    "task": _msg("task", {"task": TASK}),
    # POST /pool/task: the client closes a task that ended without a final chunk (its engine failed at the start, or
    # right after a chunk was cut) once every chunk it has was posted: chunks = how many (next seq). The gateway
    # closes it as at its deadline (the verifier takes what arrived) and the instance slot is free at once
    "task_end": _msg("task_end", {"task_id": TASK_ID, "chunks": Int(0, 10 ** 6), "reason": Str(300)}),
    "task_end_ok": _msg("task_end_ok", {"task_id": TASK_ID, "state": Str(16)}),
    "no_task": _msg("no_task", {"reason": REASON, "retry_s": Int(1, 3600)}),
    "chunk_result": _msg("chunk_result", {"task_id": TASK_ID, "seq": Int(0, 10 ** 6),
                                          "status": Str(16, enum=("accepted", "held", "rejected")),
                                          "verify": Str(16, enum=("pending", "passed", "failed", "skipped")),
                                          "steps": Int(0, MAX_CHUNK_STEPS)},
                         {"reason": REASON, "token_revoked": Bool()}),
    "heartbeat": _msg("heartbeat", {"session_id": Str(64, ID_RE, min_len=16), "running": Bool(),
                                    "instances": List(INSTANCE, 64), "steps_per_s": Float(0, 1e7)}),
    "heartbeat_ok": _msg("heartbeat_ok", {"server_time": Int(0, 2 ** 40), "max_instances": Int(0, 64)},
                         {"notice": Str(500), "revoked": REASON}),
    "me": _msg("me", {"user": USER, "leaderboard": Bool(), "token_state": Str(16, enum=("active", "revoked")),
                      "stats": STATS}, {"revoke_reason": REASON}),
    "me_update": _msg("me_update", {"leaderboard": Bool()}),
    "leaderboard": _msg("leaderboard", {"updated": Int(0, 2 ** 40), "entries": List(
        Obj({"rank": Int(1, 10 ** 6), "nickname": NICK, "steps_verified": Int(0)}), 200)}),
    "error": _msg("error", {"reason": REASON}),
}


def validate(kind: str, obj) -> dict:
    """obj checked against MESSAGES[kind] (ProtocolError on any deviation); returns the validated copy."""
    if kind not in MESSAGES:
        raise KeyError(kind)
    if isinstance(obj, dict) and "protocol" in obj and obj["protocol"] != PROTOCOL_VERSION \
            and isinstance(obj["protocol"], int) and not isinstance(obj["protocol"], bool):
        raise ProtocolError(Reason.PROTOCOL_UNSUPPORTED, got=str(obj["protocol"])[:20], supported=PROTOCOL_VERSION)
    return check(MESSAGES[kind], obj)


def parse_message(kind: str, data: bytes | str) -> dict:
    """Bytes of a control message -> validated dict (strict JSON + schema)."""
    return validate(kind, loads_strict(data))


def make_message(kind: str, **fields) -> dict:
    """A message of `kind` with protocol / type filled in, validated (a sender's own bug fails here)."""
    return validate(kind, dict({"protocol": PROTOCOL_VERSION, "type": kind}, **fields))


def encode_message(kind: str, **fields) -> bytes:
    return dumps(make_message(kind, **fields))


# ============================================================================================== chunk codec

# name -> (dtype, shape template, codecs allowed, required); "T" = steps, "T1" = steps + 1, "O" = n_obs
CHUNK_ARRAYS: dict[str, tuple] = {
    "actions": ("|u1", ("T",), ("raw", "zstd"), True),
    "rewards": ("<f4", ("T",), ("raw", "zstd"), True),  # the training reward (worker semantics, docs/pool-prototype.md)
    "logp": ("<f4", ("T",), ("raw", "zstd"), True),  # behaviour log-prob of the action taken
    "values": ("<f4", ("T1",), ("raw", "zstd"), True),  # value of every observation row (row T = bootstrap)
    "flags": ("|u1", ("T",), ("raw", "zstd"), True),  # FLAG_TERMINATED | FLAG_TRUNCATED
    "obs": ("<f2", ("T1", "O"), ("zstd",), True),  # raw (unnormalised) observations as float16
    "obs_hash": ("|u1", ("T1", 32), ("raw", "zstd"), True),  # the hash chain value after each row
    "env_rewards": ("<f4", ("T",), ("raw", "zstd"), False),  # BallanceEnv.step's reward alone
    "sector": ("|u1", ("T",), ("raw", "zstd"), False),  # sector after the step
    "frames": ("<u2", ("T",), ("raw", "zstd"), False),  # game frames the step advanced
    "sr_ms": ("<f8", ("T",), ("raw", "zstd"), False),  # IGT (SR double sum, ms) after the step
    "arc": ("<f4", ("T",), ("raw", "zstd"), False),  # reference-line arc length after the step (nan: none)
    # reward_inputs_v1 (the task's reward_inputs; server_rewards.RIN_*): what the reward reads per step, NaN = none
    "rin": ("<f8", ("T", 17), ("raw", "zstd"), False),
}
_FINITE = ("rewards", "logp", "values", "obs", "env_rewards", "sr_ms")
MAX_CHUNK_RAW_BYTES = 32 * 1024 * 1024

CHUNK_SECTION = Obj({"name": Str(32, enum=tuple(CHUNK_ARRAYS)), "codec": Str(8, enum=("raw", "zstd")),
                     "nbytes": Int(0, MAX_CHUNK_BYTES), "sha256": SHA})
CHUNK_HEADER = Obj({
    "protocol": Const(PROTOCOL_VERSION), "type": Const("chunk"),
    "task_id": TASK_ID, "seq": Int(0, 10 ** 6), "step0": Int(0, MAX_TASK_STEPS),
    "steps": Int(1, MAX_CHUNK_STEPS), "n_obs": Int(1, MAX_N_OBS), "n_act": Int(1, MAX_N_ACT),
    "policy_sha": SHA, "chain_prev": SHA, "final": Bool(),
    "end": Str(24, enum=END_REASONS, nullable=True),
    "arrays": List(CHUNK_SECTION, len(CHUNK_ARRAYS), 1),
}, {"stats": Obj({}, {"frames": Int(0, 10 ** 9), "wall_s": Float(0, 1e7), "episodes": Int(0, MAX_CHUNK_STEPS)}),
    # reward_inputs_v1: the episodes starting in this chunk (step: the task's step index of their first step)
    "episodes": List(Obj({"step": Int(0, MAX_TASK_STEPS), "kind": Str(16, enum=("tas", "tas_rest", "respawn")),
                          "frame": Int(0, 10 ** 7), "sector": Int(0, 255), "checkpoints": Int(0, 255),
                          "sr_ms": Float(-1e9, 1e10), "s": Float(-1e7, 1e7, nullable=True),
                          "ref_elapsed_ms": Float(-1e9, 1e10, nullable=True)}), MAX_CHUNK_STEPS)})


@dataclass
class Chunk:
    header: dict
    arrays: dict  # name -> read-only numpy array (dtype / shape from CHUNK_ARRAYS)

    @property
    def steps(self) -> int:
        return int(self.header["steps"])


def chain_seed(task_id: str) -> bytes:
    from .engine_chain import chain_seed as _seed
    return _seed(task_id)


def chain_from_digests(digests, prev: bytes) -> np.ndarray:
    """Chain values after each row from the rows' engine digests (R x 32 bytes), starting from prev: (R, 32) uint8."""
    out = np.zeros((len(digests), 32), np.uint8)
    h = bytes(prev)
    for i, d in enumerate(digests):
        h = hashlib.sha256(h + bytes(d)).digest()
        out[i] = np.frombuffer(h, np.uint8)
    return out


# Tolerances of the verifier's comparisons (docs/pool-prototype.md 10.6): the features, rewards and arc lengths the
# verifier recomputes may differ from the client's by float rounding of another CPU / compiler / BLAS
OBS_TOL = 1e-4  # beyond the float16 rounding of the submitted copy: |obs16 - own| <= half a float16 ulp + OBS_TOL x max(1, |own|)
REWARD_TOL = 1e-4  # |r - own r| <= REWARD_TOL x max(1, |own r|) (rewards and env_rewards)
ARC_TOL = 1e-3  # |arc - own arc| <= ARC_TOL x max(1, |own arc|); both NaN = equal


def obs_tolerance(own32) -> np.ndarray:
    """Per-entry tolerance of a submitted float16 observation against the verifier's own float32 one."""
    own = np.abs(np.asarray(own32, np.float64))
    ulp = np.spacing(np.abs(np.asarray(own32, np.float32)).astype(np.float16)).astype(np.float64)
    return ulp + OBS_TOL * np.maximum(1.0, own)  # one float16 ulp: the client's value may round to either neighbour


def obs_close(sub16, own32) -> bool:
    sub = np.asarray(sub16, np.float16).astype(np.float64)
    own = np.asarray(own32, np.float32).astype(np.float64)
    return bool((np.abs(sub - own) <= obs_tolerance(own32)).all())


def value_close(sub, own, tol: float) -> bool:
    """|sub - own| <= tol x max(1, |own|); NaN only equals NaN."""
    sub, own = float(sub), float(own)
    if np.isnan(sub) or np.isnan(own):
        return bool(np.isnan(sub) and np.isnan(own))
    return abs(sub - own) <= tol * max(1.0, abs(own))


def first_mismatch(a, b) -> int | None:
    """First row index where two (R, ...) arrays differ (bytewise; a length difference counts at the shorter
    length), or None."""
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    if n:
        ne = (a[:n].reshape(n, -1) != b[:n].reshape(n, -1)).any(1) if a.ndim > 1 or b.ndim > 1 else a[:n] != b[:n]
        idx = np.flatnonzero(ne)
        if len(idx):
            return int(idx[0])
    return None if len(a) == len(b) else n


def _shape(tmpl, steps, n_obs):
    return tuple({"T": steps, "T1": steps + 1, "O": n_obs}.get(d, d) for d in tmpl)


# ---------------------------------------------------------------------------------------------- zstd shim

def _zstd():
    try:
        from compression import zstd  # Python >= 3.14
        return "stdlib", zstd
    except ImportError:
        pass
    try:
        import zstandard  # PyPI fallback for older interpreters
        return "zstandard", zstandard
    except ImportError:
        return None, None


def zstd_compress(data: bytes, level: int = 3) -> bytes:
    kind, z = _zstd()
    if kind == "stdlib":
        return z.compress(data, level=level)
    if kind == "zstandard":
        return z.ZstdCompressor(level=level).compress(data)
    raise RuntimeError("no zstd: Python >= 3.14 (compression.zstd) or the zstandard package is needed")


ZSTD_WINDOW_LOG_MAX = 23


def zstd_bound(size: int) -> int:
    """The most a zstd frame of `size` decoded bytes may take (above libzstd's ZSTD_compressBound): a padded frame
    (empty blocks before the data) is refused, so a chunk cannot be inflated to fill the store."""
    return size + (size >> 7) + 64


def zstd_decompress_exact(data: bytes, size: int) -> bytes:
    """Decompress one zstd frame whose output must be exactly `size` bytes; the output is bounded to size + 1
    while decompressing, so a decompression bomb is refused without allocating it."""
    kind, z = _zstd()
    if kind is None:
        raise RuntimeError("no zstd: Python >= 3.14 (compression.zstd) or the zstandard package is needed")
    try:
        if kind == "stdlib":
            # window at most 8 MiB (the largest section, a 4097 x 1024 float16 obs, is 8 MiB): a frame asking for a
            # bigger window is refused before its buffers are committed
            d = z.ZstdDecompressor(options={z.DecompressionParameter.window_log_max: ZSTD_WINDOW_LOG_MAX})
            out = d.decompress(data, max_length=size + 1)
            if len(out) != size or not d.eof or d.unused_data:
                raise ValueError(f"decoded size differs from {size} or trailing data")
            return out
        # zstandard (untested fallback, the venv runs 3.14): a stream reader read with an explicit bound, one frame
        with z.ZstdDecompressor(max_window_size=1 << ZSTD_WINDOW_LOG_MAX).stream_reader(
                bytes(data), read_across_frames=False) as r:
            out = r.read(size + 1)
            if len(out) != size or r.read(1):
                raise ValueError(f"decoded size differs from {size}")
        return out
    except ProtocolError:
        raise
    except Exception as e:  # noqa: BLE001  (any decoder error is a bad chunk)
        raise ProtocolError(Reason.BAD_CHUNK, f"zstd: {str(e)[:120]}") from None


# ---------------------------------------------------------------------------------------------- encode / decode

def encode_chunk(header: dict, arrays: dict, level: int = 3) -> bytes:
    """header: CHUNK_HEADER fields except "arrays" (protocol / type filled in); arrays: name -> array convertible
    to the CHUNK_ARRAYS dtype and shape (obs may be float32: it is converted to float16 here). The result is
    decoded once before returning (a sender's own bug fails here, not at the gateway)."""
    h = dict(header)
    h.setdefault("protocol", PROTOCOL_VERSION)
    h.setdefault("type", "chunk")
    steps, n_obs = int(h["steps"]), int(h["n_obs"])
    sections, body = [], []
    for name, (dt, tmpl, codecs, required) in CHUNK_ARRAYS.items():
        if name not in arrays:
            if required:
                raise ValueError(f"chunk array {name!r} missing")
            continue
        a = np.ascontiguousarray(np.asarray(arrays[name]).astype(dt, copy=False))
        want = _shape(tmpl, steps, n_obs)
        if a.shape != want:
            raise ValueError(f"chunk array {name}: shape {a.shape} != {want}")
        raw = a.tobytes()
        codec = "raw" if "raw" in codecs else "zstd"
        stored = zstd_compress(raw, level) if codec == "zstd" else raw
        sections.append({"name": name, "codec": codec, "nbytes": len(stored), "sha256": hashlib.sha256(raw).hexdigest()})
        body.append(stored)
    unknown = set(arrays) - set(CHUNK_ARRAYS)
    if unknown:
        raise ValueError(f"unknown chunk arrays {sorted(unknown)}")
    h["arrays"] = sections
    hj = dumps(h)
    pre = _PREAMBLE.pack(CHUNK_MAGIC, CHUNK_FORMAT, 0, len(hj))
    out = pre + hj + b"".join(body)
    out += hashlib.sha256(out).digest()
    decode_chunk(out)
    return out


def decode_chunk(data: bytes, max_bytes: int = MAX_CHUNK_BYTES, n_act: int | None = None,
                 n_obs: int | None = None) -> Chunk:
    """Encoded chunk -> Chunk, checking (in this order) total size, magic / format, the trailing sha256, header
    size, strict JSON, the header schema, the section list (known names, each once, all required ones, allowed
    codec), section sizes against the body, each section's decoded size and sha256, then the content: finite
    floats, actions < n_act, log-probs <= 0, flag bits, the final chunk's last flag. n_act / n_obs, when given,
    must equal the header's (the task's policy). Raises ProtocolError (BAD_CHUNK / CHUNK_TOO_LARGE)."""
    B = Reason.BAD_CHUNK
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ProtocolError(B, "not bytes")
    data = bytes(data)
    if len(data) > max_bytes:
        raise ProtocolError(Reason.CHUNK_TOO_LARGE, size=len(data), limit=max_bytes)
    if len(data) < _PREAMBLE.size + 2 + 32:
        raise ProtocolError(B, "too short")
    magic, fmt, flags, hlen = _PREAMBLE.unpack_from(data, 0)
    if magic != CHUNK_MAGIC:
        raise ProtocolError(B, "bad magic")
    if fmt != CHUNK_FORMAT or flags != 0:
        raise ProtocolError(B, f"format {fmt} / flags {flags} unsupported")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ProtocolError(B, "chunk sha256 mismatch")
    if hlen > MAX_CHUNK_HEADER_BYTES or _PREAMBLE.size + hlen > len(data) - 32:
        raise ProtocolError(B, f"header length {hlen}")
    hb = data[_PREAMBLE.size:_PREAMBLE.size + hlen]
    try:
        h = check(CHUNK_HEADER, loads_strict(hb, MAX_CHUNK_HEADER_BYTES, B), "header")
    except ProtocolError as e:
        if e.reason == Reason.BAD_MESSAGE:
            raise ProtocolError(B, e.params.get("detail", "")) from None
        raise
    steps, no, na = h["steps"], h["n_obs"], h["n_act"]
    if n_act is not None and na != n_act:
        raise ProtocolError(B, f"n_act {na} != {n_act}")
    if n_obs is not None and no != n_obs:
        raise ProtocolError(B, f"n_obs {no} != {n_obs}")
    if h["final"] != (h["end"] is not None):
        raise ProtocolError(B, "end is set exactly on the final chunk")
    names = [s["name"] for s in h["arrays"]]
    if len(set(names)) != len(names):
        raise ProtocolError(B, "a section appears twice")
    missing = [n for n, spec in CHUNK_ARRAYS.items() if spec[3] and n not in names]
    if missing:
        raise ProtocolError(B, f"missing sections {missing}")
    body = memoryview(data)[_PREAMBLE.size + hlen:len(data) - 32]
    if sum(s["nbytes"] for s in h["arrays"]) != len(body):
        raise ProtocolError(B, "section sizes do not add up to the body")
    total_raw = 0
    arrays, off = {}, 0
    for s in h["arrays"]:
        dt, tmpl, codecs, _ = CHUNK_ARRAYS[s["name"]]
        if s["codec"] not in codecs:
            raise ProtocolError(B, f"{s['name']}: codec {s['codec']} not allowed")
        shape = _shape(tmpl, steps, no)
        size = int(np.dtype(dt).itemsize * np.prod(shape))
        total_raw += size
        if total_raw > MAX_CHUNK_RAW_BYTES:
            raise ProtocolError(B, "decoded size too large")
        if s["codec"] == "raw" and s["nbytes"] != size:
            raise ProtocolError(B, f"{s['name']}: {s['nbytes']} raw bytes, expected {size}")
        if s["codec"] == "zstd" and s["nbytes"] > zstd_bound(size):
            raise ProtocolError(B, f"{s['name']}: {s['nbytes']} zstd bytes for {size} decoded (padded frame)")
        stored = bytes(body[off:off + s["nbytes"]])
        off += s["nbytes"]
        raw = zstd_decompress_exact(stored, size) if s["codec"] == "zstd" else stored
        if len(raw) != size:
            raise ProtocolError(B, f"{s['name']}: {len(raw)} bytes, expected {size}")
        if hashlib.sha256(raw).hexdigest() != s["sha256"]:
            raise ProtocolError(B, f"{s['name']}: sha256 mismatch")
        a = np.frombuffer(raw, dtype=np.dtype(dt)).reshape(shape)
        a.flags.writeable = False
        arrays[s["name"]] = a
    for n in _FINITE:
        if n in arrays and not np.isfinite(arrays[n]).all():
            raise ProtocolError(B, f"{n}: non-finite values")
    if "arc" in arrays and np.isinf(arrays["arc"]).any():
        raise ProtocolError(B, "arc: infinite values")
    if int(arrays["actions"].max(initial=0)) >= na:
        raise ProtocolError(B, f"action >= n_act {na}")
    lp = arrays["logp"]
    if (lp > MAX_LOGP).any() or (lp < MIN_LOGP).any():
        i = int(np.flatnonzero((lp > MAX_LOGP) | (lp < MIN_LOGP))[0])
        raise ProtocolError(Reason.LOGP_INVALID, task_id=h["task_id"], chunk=h["seq"], step=h["step0"] + i)
    fl = arrays["flags"]
    if (fl & ~np.uint8(FLAG_TERMINATED | FLAG_TRUNCATED)).any():
        raise ProtocolError(B, "unknown flag bits")
    if ((fl & FLAG_TERMINATED) & ((fl & FLAG_TRUNCATED) >> 1)).any():
        raise ProtocolError(B, "a step both terminated and truncated")
    if (fl[:-1] & FLAG_TRUNCATED).any():
        raise ProtocolError(B, "truncation before the last step of a chunk")
    if (fl[-1] & FLAG_TRUNCATED) and not h["final"]:
        raise ProtocolError(B, "truncated step in a non-final chunk")
    if h["final"] and not fl[-1]:
        raise ProtocolError(B, "the final chunk's last step is neither terminated nor truncated")
    if h["step0"] + steps > MAX_TASK_STEPS:
        raise ProtocolError(B, "step0 + steps too large")
    return Chunk(h, arrays)


def max_chunk_bytes(steps: int, n_obs: int, header_slack: int = MAX_CHUNK_HEADER_BYTES) -> int:
    """The largest encoding of a chunk with these dimensions: every section at its zstd bound (raw sections are
    smaller), the preamble, a header of at most header_slack bytes and the digest. The gateway refuses bigger bodies."""
    total = 0
    for dt, tmpl, _, _ in CHUNK_ARRAYS.values():
        size = int(np.dtype(dt).itemsize * np.prod(_shape(tmpl, steps, n_obs)))
        total += zstd_bound(size)
    return _PREAMBLE.size + header_slack + total + 32


def check_chain_link(prev: Chunk, nxt: Chunk) -> str | None:
    """Continuity of two consecutive chunks of one task; None when nxt follows prev, else what is wrong."""
    hp, hn = prev.header, nxt.header
    if hn["task_id"] != hp["task_id"] or hn["seq"] != hp["seq"] + 1:
        return "not consecutive"
    if hp["final"]:
        return "chunk after the final one"
    if hn["step0"] != hp["step0"] + hp["steps"]:
        return "step0 does not continue"
    if hn["policy_sha"] != hp["policy_sha"]:
        return "policy changed inside a task"
    if hn["chain_prev"] != bytes(prev.arrays["obs_hash"][-2]).hex():
        return "chain_prev is not the previous chunk's second-to-last chain value"
    if bytes(nxt.arrays["obs_hash"][0]) != bytes(prev.arrays["obs_hash"][-1]):
        return "first row's chain value differs from the previous chunk's last"
    if not np.array_equal(nxt.arrays["obs"][0].view(np.uint16), prev.arrays["obs"][-1].view(np.uint16)):
        return "first observation differs from the previous chunk's last"
    return None


# ============================================================================================== game files

def canonical_relpath(p: str) -> str:
    """Relative path as the digests use it: forward slashes, lower case (Windows folders are case-insensitive)."""
    return str(p).replace("\\", "/").lower()


def tree_digest(entries: dict) -> str:
    """Digest of a directory: entries = {path relative to the directory: file sha256 hex}; lines
    "<sha256> <canonical path>\\n" sorted by canonical path, sha256 of their UTF-8."""
    lines = sorted((canonical_relpath(k), v) for k, v in entries.items())
    return hashlib.sha256("".join(f"{v} {k}\n" for k, v in lines).encode("utf-8")).hexdigest()

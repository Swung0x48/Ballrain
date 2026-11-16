"""Shared memory + named events transport of the agent mode (docs/agent-protocol.md section 1-2).

Python creates the mapping and both auto-reset events (`PyChannel`); the mod opens them. `ModChannel`
is the mod's side in Python, used by the fake game (ballrain.fakegame) and by tests, so the real
transport code is exercised without the game.

Strict lock-step: the mod writes an observation, bumps mod_seq, sets <name>.obs; Python reads it,
writes an action, bumps py_seq, sets <name>.act. Payload first, then seq, then SetEvent (SetEvent is
a full barrier on Windows).

The socket transport (`SocketChannel` / `SocketModChannel`, docs/headless.md) carries the same OBS1 / ACT1
messages for the headless backend, which runs on any OS: Python listens (unix socket or TCP, the legacy
prototype's convention: Python is the server, port base 27787), the game connects, Python pings, the game
pongs with its hello. Frames are u32 type + u32 length + payload, little-endian; the header fields the
shared memory carries (mod_state, end_reason, frames_done, last_error, mod_seq) travel with every
observation and in STATE messages.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import secrets
import select
import socket
import struct
import sys
import tempfile
import threading
import time

if sys.platform == "win32":
    from ctypes import wintypes

from . import protocol as P

if sys.platform == "win32":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.CreateEventW.restype = wintypes.HANDLE
    _k32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.OpenEventW.restype = wintypes.HANDLE
    _k32.SetEvent.argtypes = [wintypes.HANDLE]
    _k32.ResetEvent.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.WaitForMultipleObjects.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL,
                                            wintypes.DWORD]
    _k32.WaitForMultipleObjects.restype = wintypes.DWORD

EVENT_MODIFY_STATE = 0x0002
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102
WAIT_FAILED = 0xFFFFFFFF


class TransportError(RuntimeError):
    pass


class PeerGone(TransportError):
    """The other side (game process / Python) went away while we waited."""


class Timeout(TransportError):
    pass


def make_name(copy: str, pid: int | None = None) -> str:
    """Local\\ballrain-<copy>-<pid>-<nonce>; contains no user information."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in copy)
    return f"Local\\ballrain-{safe}-{pid or os.getpid()}-{secrets.token_hex(4)}"


def _check_win():
    if sys.platform != "win32":
        raise TransportError("the shared-memory transport needs Windows")


def _event(name: str, create: bool):
    _check_win()
    if create:
        ctypes.set_last_error(0)
        h = _k32.CreateEventW(None, False, False, name)  # auto-reset, initially unset
        if h and ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
            _k32.CloseHandle(h)
            raise TransportError(f"event {name} already exists")
    else:
        h = _k32.OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, False, name)
    if not h:
        raise TransportError(f"cannot {'create' if create else 'open'} event {name}: error {ctypes.get_last_error()}")
    return h


def wait_handles(handles, timeout_s: float | None) -> int | None:
    """Index of the first signalled handle, or None on timeout."""
    _check_win()
    arr = (wintypes.HANDLE * len(handles))(*handles)
    ms = 0xFFFFFFFF if timeout_s is None else max(0, int(timeout_s * 1000))
    r = _k32.WaitForMultipleObjects(len(handles), arr, False, ms)
    if r == WAIT_TIMEOUT:
        return None
    if r == WAIT_FAILED:
        raise TransportError(f"WaitForMultipleObjects failed: {ctypes.get_last_error()}")
    if WAIT_OBJECT_0 <= r < WAIT_OBJECT_0 + len(handles):
        return r - WAIT_OBJECT_0
    raise TransportError(f"WaitForMultipleObjects returned {r:#x}")


class _Mapping:
    def __init__(self, name: str, size: int):
        _check_win()
        self.name = name
        self.size = size
        self.mm = mmap.mmap(-1, size, tagname=name)

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self.mm, off)[0]

    def u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.mm, off)[0]

    def set_u32(self, off: int, v: int) -> None:
        struct.pack_into("<I", self.mm, off, v & 0xFFFFFFFF)

    def set_u64(self, off: int, v: int) -> None:
        struct.pack_into("<Q", self.mm, off, v)

    def header(self) -> dict:
        raw = bytes(self.mm[P.H_MOD_VERSION:P.H_MOD_VERSION + P.SHM_VERSION_LEN])
        return {
            "magic": self.u32(P.H_MAGIC), "version": self.u32(P.H_VERSION), "size": self.u32(P.H_SIZE),
            "obs_off": self.u32(P.H_OBS_OFF), "obs_cap": self.u32(P.H_OBS_CAP),
            "act_off": self.u32(P.H_ACT_OFF), "act_cap": self.u32(P.H_ACT_CAP),
            "mod_seq": self.u64(P.H_MOD_SEQ), "py_seq": self.u64(P.H_PY_SEQ),
            "mod_state": self.u32(P.H_MOD_STATE), "end_reason": self.u32(P.H_END_REASON),
            "mod_pid": self.u32(P.H_MOD_PID), "py_pid": self.u32(P.H_PY_PID),
            "frames_done": self.u32(P.H_FRAMES_DONE), "last_error": self.u32(P.H_LAST_ERROR),
            "obs_versions": self.u32(P.H_OBS_VERSIONS),
            "mod_version": raw.split(b"\0", 1)[0].decode("latin-1"),
        }

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None


class PyChannel:
    """Python's end: creates the mapping and events and writes the header."""

    def __init__(self, name: str, size: int = P.SHM_DEFAULT_SIZE, obs_cap: int = P.SHM_OBS_CAPACITY,
                 act_cap: int = P.SHM_ACT_CAPACITY):
        if P.SHM_OBS_OFFSET + obs_cap + act_cap > size:
            raise ValueError("obs + act areas exceed the mapping")
        self.name = name
        self.ev_obs = _event(name + ".obs", True)
        try:
            self.ev_act = _event(name + ".act", True)
        except Exception:
            _k32.CloseHandle(self.ev_obs)
            raise
        self.m = _Mapping(name, size)
        mm = self.m.mm
        mm[:P.SHM_HEADER_SIZE] = b"\0" * P.SHM_HEADER_SIZE
        for off, v in ((P.H_MAGIC, P.SHM_MAGIC), (P.H_VERSION, P.PROTOCOL_VERSION), (P.H_SIZE, size),
                       (P.H_OBS_OFF, P.SHM_OBS_OFFSET), (P.H_OBS_CAP, obs_cap),
                       (P.H_ACT_OFF, P.SHM_OBS_OFFSET + obs_cap), (P.H_ACT_CAP, act_cap),
                       (P.H_PY_PID, os.getpid())):
            self.m.set_u32(off, v)
        self.obs_off, self.obs_cap = P.SHM_OBS_OFFSET, obs_cap
        self.act_off, self.act_cap = P.SHM_OBS_OFFSET + obs_cap, act_cap
        self.last_mod_seq = 0
        self.py_seq = 0

    def header(self) -> dict:
        return self.m.header()

    @property
    def max_action_frames(self) -> int:
        return (self.act_cap - P.ACT_HDR.size) // 8

    def wait_obs(self, timeout_s: float | None, process_handle=None, poll_s: float = 0.25,
                 final_grace_s: float = 2.0) -> bytes | None:
        """Wait for the next observation; returns its bytes, or None when the mod reached the ENDED or
        ERROR state without a new observation. Raises PeerGone if the process exits, Timeout on timeout.

        The mod sets ENDED / ERROR before it publishes the final observation (BallrainIO AgentFinish),
        so on those states we wait up to final_grace_s more for the seq to advance."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        handles = [self.ev_obs] + ([process_handle] if process_handle else [])
        grace_until = None
        while True:
            seq = self.m.u64(P.H_MOD_SEQ)
            if seq > self.last_mod_seq:
                # gaps are tolerated (BALLRAIN_PREFIX_EVENTS writes without waiting): only the latest counts
                self.last_mod_seq = seq
                return self._read_obs()
            st = self.m.u32(P.H_MOD_STATE)
            if st in (P.ModState.ENDED, P.ModState.ERROR):
                now = time.monotonic()
                if grace_until is None:
                    grace_until = now + final_grace_s
                if now >= grace_until:
                    return None
                wait_handles([self.ev_obs], min(0.05, grace_until - now))
                continue
            left = poll_s if deadline is None else min(poll_s, deadline - time.monotonic())
            if left <= 0:
                raise Timeout(f"no observation within {timeout_s} s (mod_state {P.ModState.NAMES[st] if st < 8 else st})")
            r = wait_handles(handles, left)
            if r == 1:
                if self.m.u64(P.H_MOD_SEQ) > self.last_mod_seq:
                    continue
                raise PeerGone("game process exited")

    def _read_obs(self) -> bytes:
        mm = self.m.mm
        o = self.obs_off
        (byte_len,) = struct.unpack_from("<I", mm, o + 8)
        if not P.MSG_HDR.size <= byte_len <= self.obs_cap:
            raise TransportError(f"observation byte_len {byte_len} outside obs area")
        return bytes(mm[o:o + byte_len])

    def send_action(self, action: P.Action) -> int:
        if len(action.frames) > self.max_action_frames:
            raise ValueError(f"{len(action.frames)} frames exceed the act area ({self.max_action_frames})")
        self.py_seq += 1
        data = P.encode_action(action, self.py_seq)
        self.m.mm[self.act_off:self.act_off + len(data)] = data
        self.m.set_u64(P.H_PY_SEQ, self.py_seq)
        _k32.SetEvent(self.ev_act)
        return self.py_seq

    def wait_hello(self, timeout_s: float, process_handle=None) -> dict:
        deadline = time.monotonic() + timeout_s
        handles = [self.ev_obs] + ([process_handle] if process_handle else [])
        while True:
            h = self.header()
            if h["mod_state"] == P.ModState.ERROR:
                raise TransportError(f"mod refused to run (last_error {h['last_error']}, see ModLoader.log)")
            if h["mod_state"] >= P.ModState.HELLO or h["mod_seq"] > 0:
                return h
            left = deadline - time.monotonic()
            if left <= 0:
                raise Timeout(f"no hello within {timeout_s} s")
            r = wait_handles(handles, min(left, 0.25))
            if r == 1:
                raise PeerGone("game process exited before hello")
            if r == 0 and self.m.u64(P.H_MOD_SEQ) == 0:
                pass  # hello signalled through the obs event: re-read the header

    def close(self):
        self.m.close()
        for h in (self.ev_obs, self.ev_act):
            if h:
                _k32.CloseHandle(h)
        self.ev_obs = self.ev_act = None


class ModChannel:
    """The mod's end (fake game / tests): opens the mapping and events created by PyChannel.
    (The socket transport's mod end is SocketModChannel.)"""

    def __init__(self, name: str):
        self.name = name
        self.ev_obs = _event(name + ".obs", False)
        self.ev_act = _event(name + ".act", False)
        probe = _Mapping(name, P.SHM_HEADER_SIZE)
        size = probe.u32(P.H_SIZE)
        if probe.u32(P.H_MAGIC) != P.SHM_MAGIC or probe.u32(P.H_VERSION) != P.PROTOCOL_VERSION:
            probe.close()
            raise TransportError("shared memory magic/version mismatch")
        probe.close()
        self.m = _Mapping(name, size)
        h = self.m.header()
        self.obs_off, self.obs_cap = h["obs_off"], h["obs_cap"]
        self.act_off, self.act_cap = h["act_off"], h["act_cap"]
        self.mod_seq = h["mod_seq"]
        self.last_py_seq = h["py_seq"]

    def hello(self, version: str, pid: int | None = None, obs_versions: int = 1 << 1) -> None:
        b = version.encode("latin-1")[:P.SHM_VERSION_LEN - 1]
        self.m.mm[P.H_MOD_VERSION:P.H_MOD_VERSION + P.SHM_VERSION_LEN] = b.ljust(P.SHM_VERSION_LEN, b"\0")
        self.m.set_u32(P.H_MOD_PID, pid or os.getpid())
        self.m.set_u32(P.H_OBS_VERSIONS, obs_versions)
        self.set_state(P.ModState.HELLO)
        _k32.SetEvent(self.ev_obs)

    def set_state(self, state: int, end_reason: int | None = None, frames_done: int | None = None):
        if end_reason is not None:
            self.m.set_u32(P.H_END_REASON, end_reason)
        if frames_done is not None:
            self.m.set_u32(P.H_FRAMES_DONE, frames_done)
        self.m.set_u32(P.H_MOD_STATE, state)

    def write_obs(self, data: bytes, frames_done: int, signal: bool = True) -> None:
        if len(data) > self.obs_cap:
            raise TransportError("observation exceeds obs area")
        self.m.mm[self.obs_off:self.obs_off + len(data)] = data
        self.m.set_u32(P.H_FRAMES_DONE, frames_done)
        self.mod_seq += 1
        self.m.set_u64(P.H_MOD_SEQ, self.mod_seq)
        if signal:
            _k32.SetEvent(self.ev_obs)

    def signal(self):
        _k32.SetEvent(self.ev_obs)

    def wait_action(self, timeout_s: float | None, cancel=None) -> P.Action | None:
        """Next action, or None on timeout (or when the threading.Event `cancel` is set)."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if cancel is not None and cancel.is_set():
                return None
            seq = self.m.u64(P.H_PY_SEQ)
            if seq > self.last_py_seq:
                self.last_py_seq = seq
                a, aseq = P.decode_action(self.m.mm, self.act_off)
                if aseq != seq & 0xFFFFFFFF:
                    raise TransportError(f"action py_seq {aseq} != header {seq}")
                return a
            left = 0.25 if deadline is None else min(0.25, deadline - time.monotonic())
            if left <= 0:
                return None
            wait_handles([self.ev_act], left)

    def close(self):
        self.m.close()
        for h in (self.ev_obs, self.ev_act):
            if h:
                _k32.CloseHandle(h)
        self.ev_obs = self.ev_act = None


# ------------------------------------------------------------------------------------ socket transport

BR_PING, BR_PONG, BR_STATE, BR_OBS, BR_ACT = 0x100, 0x101, 0x102, 0x103, 0x104
FRAME_HDR = struct.Struct("<II")  # type, payload length
OBS_HDR = struct.Struct("<IIIIQ")  # mod_state, end_reason, frames_done, last_error, mod_seq
STATE_HDR = struct.Struct("<IIII")  # mod_state, end_reason, frames_done, last_error
PONG = struct.Struct("<III180s")  # pid, obs version bitmap, mod_state, version string
DEFAULT_TCP_PORT = 27787  # the legacy TCP prototype's port (legacy/comm_protocol.md); 0 = any free port
SOCKET_MAX_ACTION_FRAMES = 1 << 20
_SOCK_COUNTER = iter(range(1 << 62))


def default_socket_kind() -> str:
    return "unix" if hasattr(socket, "AF_UNIX") and sys.platform != "win32" else "tcp"


def _recv_exact(sock, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class SocketChannel:
    """Python's end of the socket transport: listens, accepts the game's connection, then the same calls as
    PyChannel (header, wait_hello, wait_obs, send_action, close). EOF on the socket is PeerGone (the game
    process is gone) unless the game had ended; process_handle arguments are accepted and ignored."""

    env_var = "BALLRAIN_SOCK"

    def __init__(self, kind: str | None = None, host: str = "127.0.0.1", port: int = 0, path: str | None = None):
        self.kind = kind or default_socket_kind()
        self._path = None
        if self.kind == "unix":
            # macOS limits sun_path to 104 bytes: a short name in the temp directory
            self._path = path or os.path.join(tempfile.gettempdir(),
                                              f"brl-{os.getpid()}-{next(_SOCK_COUNTER)}-{secrets.token_hex(2)}.sock")
            if os.path.exists(self._path):
                os.unlink(self._path)
            self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.listener.bind(self._path)
            self.name = f"unix:{self._path}"
        elif self.kind == "tcp":
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener.bind((host, port))
            self.name = f"tcp:{host}:{self.listener.getsockname()[1]}"
        else:
            raise ValueError(f"unknown socket kind {kind!r}")
        self.listener.listen(1)
        self.conn = None
        self._hdr = {"magic": 0, "version": P.PROTOCOL_VERSION, "size": 0, "obs_off": 0, "obs_cap": 0, "act_off": 0,
                     "act_cap": 0, "mod_seq": 0, "py_seq": 0, "mod_state": P.ModState.INIT, "end_reason": 0,
                     "mod_pid": 0, "py_pid": os.getpid(), "frames_done": 0, "last_error": 0, "obs_versions": 0,
                     "mod_version": ""}
        self.last_mod_seq = 0
        self.py_seq = 0
        self._closed_peer = False

    @property
    def max_action_frames(self) -> int:
        return SOCKET_MAX_ACTION_FRAMES

    def header(self) -> dict:
        return dict(self._hdr)

    def _accept(self, deadline: float | None) -> None:
        if self.conn is not None:
            return
        left = None if deadline is None else max(0.0, deadline - time.monotonic())
        self.listener.settimeout(left)
        try:
            conn, _ = self.listener.accept()
        except (socket.timeout, TimeoutError):
            raise Timeout("the game did not connect") from None
        conn.settimeout(None)
        if self.kind == "tcp":
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.conn = conn

    def _recv(self, deadline: float | None):
        """(type, payload), or None on EOF. Raises Timeout."""
        if self._closed_peer:
            return None
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            raise Timeout("no message from the game")
        r, _, _ = select.select([self.conn], [], [], left)
        if not r:
            raise Timeout("no message from the game")
        try:
            hdr = _recv_exact(self.conn, FRAME_HDR.size)
            if hdr is None:
                self._closed_peer = True
                return None
            typ, n = FRAME_HDR.unpack(hdr)
            payload = _recv_exact(self.conn, n) if n else b""
        except OSError:
            self._closed_peer = True
            return None
        if payload is None:
            self._closed_peer = True
            return None
        return typ, payload

    def _apply_state(self, payload: bytes) -> None:
        st, end, frames, err = STATE_HDR.unpack_from(payload, 0)
        self._hdr.update(mod_state=st, end_reason=end, frames_done=frames, last_error=err)

    def wait_hello(self, timeout_s: float, process_handle=None) -> dict:
        deadline = time.monotonic() + timeout_s
        self._accept(deadline)
        self._send(BR_PING, b"")
        while True:
            m = self._recv(deadline)
            if m is None:
                raise PeerGone("game process exited before hello")
            typ, payload = m
            if typ == BR_STATE:
                self._apply_state(payload)
                if self._hdr["mod_state"] == P.ModState.ERROR:
                    raise TransportError(f"game refused to run (last_error {self._hdr['last_error']}, see its log)")
            elif typ == BR_PONG:
                pid, vers, st, ver = PONG.unpack_from(payload, 0)
                self._hdr.update(mod_pid=pid, obs_versions=vers, mod_state=st,
                                 mod_version=ver.split(b"\0", 1)[0].decode("latin-1"))
                return self.header()

    def wait_obs(self, timeout_s: float | None, process_handle=None, poll_s: float = 0.25,
                 final_grace_s: float = 2.0) -> bytes | None:
        """Next observation's bytes; None when the game ended (or failed) without a new one."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        if self.conn is None:
            self._accept(deadline)
        while True:
            m = self._recv(deadline)
            if m is None:
                if self._hdr["mod_state"] in (P.ModState.ENDED, P.ModState.ERROR):
                    return None
                raise PeerGone("game process closed the connection")
            typ, payload = m
            if typ == BR_STATE:
                self._apply_state(payload)
            elif typ == BR_OBS:
                st, end, frames, err, seq = OBS_HDR.unpack_from(payload, 0)
                self._hdr.update(mod_state=st, end_reason=end, frames_done=frames, last_error=err, mod_seq=seq)
                self.last_mod_seq = seq
                return payload[OBS_HDR.size:]

    def _send(self, typ: int, payload: bytes) -> None:
        try:
            self.conn.sendall(FRAME_HDR.pack(typ, len(payload)) + payload)
        except OSError as e:
            raise PeerGone(f"send failed: {e}") from None

    def send_action(self, action: P.Action) -> int:
        if len(action.frames) > self.max_action_frames:
            raise ValueError(f"{len(action.frames)} frames exceed {self.max_action_frames}")
        self.py_seq += 1
        self._send(BR_ACT, P.encode_action(action, self.py_seq))
        self._hdr["py_seq"] = self.py_seq
        return self.py_seq

    def close(self):
        for s in (self.conn, self.listener):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self.conn = None
        if self._path and os.path.exists(self._path):
            try:
                os.unlink(self._path)
            except OSError:
                pass
        self._path = None


class SocketModChannel:
    """The game's end of the socket transport, for the fake game and tests (the real one is
    headless/src/ballrain_agent.cpp). Same calls as ModChannel."""

    def __init__(self, name: str, connect_timeout_s: float = 30.0):
        self.name = name
        if name.startswith("unix:"):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            addr = name[5:]
        elif name.startswith("tcp:"):
            host, port = name[4:].rsplit(":", 1)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            addr = (host, int(port))
        else:
            raise TransportError(f"bad socket address {name!r}")
        self.sock.settimeout(connect_timeout_s)
        self.sock.connect(addr)
        self.sock.settimeout(None)
        self.state, self.end_reason, self.frames_done, self.last_error = 0, 0, 0, 0
        self.mod_seq = 0
        self.last_py_seq = 0
        self._lock = threading.Lock()
        self._pinged = False

    def _send(self, typ, payload):
        with self._lock:
            self.sock.sendall(FRAME_HDR.pack(typ, len(payload)) + payload)

    def _recv(self, timeout_s):
        r, _, _ = select.select([self.sock], [], [], timeout_s)
        if not r:
            return "timeout"
        try:
            hdr = _recv_exact(self.sock, FRAME_HDR.size)
            if hdr is None:
                return None
            typ, n = FRAME_HDR.unpack(hdr)
            payload = _recv_exact(self.sock, n) if n else b""
        except OSError:
            return None
        return None if payload is None else (typ, payload)

    def hello(self, version: str, pid: int | None = None, obs_versions: int = 1 << 1) -> None:
        if not self._pinged:
            m = self._recv(30.0)
            if not isinstance(m, tuple) or m[0] != BR_PING:
                raise TransportError("no ping from the client")
            self._pinged = True
        self.state = P.ModState.HELLO
        self._send(BR_PONG, PONG.pack(pid or os.getpid(), obs_versions, self.state,
                                      version.encode("latin-1")[:179]))

    def set_state(self, state: int, end_reason: int | None = None, frames_done: int | None = None):
        if end_reason is not None:
            self.end_reason = end_reason
        if frames_done is not None:
            self.frames_done = frames_done
        self.state = state
        try:
            self._send(BR_STATE, STATE_HDR.pack(self.state, self.end_reason, self.frames_done, self.last_error))
        except OSError:
            pass

    def write_obs(self, data: bytes, frames_done: int, signal: bool = True) -> None:
        self.frames_done = frames_done
        self.mod_seq += 1
        self._send(BR_OBS, OBS_HDR.pack(self.state, self.end_reason, frames_done, self.last_error, self.mod_seq)
                   + bytes(data))

    def signal(self):
        pass

    def wait_action(self, timeout_s: float | None, cancel=None) -> P.Action | None:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            if cancel is not None and cancel.is_set():
                return None
            left = 0.25 if deadline is None else min(0.25, deadline - time.monotonic())
            if left <= 0:
                return None
            m = self._recv(left)
            if m == "timeout":
                continue
            if m is None:
                return None  # client gone: the real game ends as on a timeout
            typ, payload = m
            if typ != BR_ACT:
                continue
            a, seq = P.decode_action(payload, 0)
            if seq != (self.last_py_seq + 1) & 0xFFFFFFFF:
                raise TransportError(f"action py_seq {seq} != {self.last_py_seq + 1}")
            self.last_py_seq = seq
            return a

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

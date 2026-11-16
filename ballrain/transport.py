"""Shared memory + named events transport of the agent mode (docs/agent-protocol.md section 1-2).

Python creates the mapping and both auto-reset events (`PyChannel`); the mod opens them. `ModChannel`
is the mod's side in Python, used by the fake game (ballrain.fakegame) and by tests, so the real
transport code is exercised without the game.

Strict lock-step: the mod writes an observation, bumps mod_seq, sets <name>.obs; Python reads it,
writes an action, bumps py_seq, sets <name>.act. Payload first, then seq, then SetEvent (SetEvent is
a full barrier on Windows).
"""
from __future__ import annotations

import ctypes
import mmap
import os
import secrets
import struct
import sys
import time
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
    """The mod's end (fake game / tests): opens the mapping and events created by PyChannel."""

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

"""Per-process muting of launched Player.exe instances through Windows Core Audio (WASAPI sessions).

Only the audio sessions of processes this launcher started are touched: every render endpoint's
IAudioSessionManager2 is enumerated, each session's IAudioSessionControl2::GetProcessId is compared
with the tracked PIDs, and ISimpleAudioVolume::SetMute(TRUE) is applied to the matching ones. The game
is not changed in any way (its DirectSound buffers keep playing; the session is muted in the audio
engine), so traces are unaffected. No system-wide setting, endpoint volume or other application's
session is changed.

A session only exists once the game opens its audio device (~0.8 s after start), and a process can
get a new session later (new stream / device change), so a daemon watcher thread re-scans the
tracked PIDs: every POLL_FAST_S until each new process has a session (at most FAST_WINDOW_S), then
every POLL_SLOW_S until the process exits (one scan ~7 ms). Before muting, the PID's image path is
checked once against the copy's Player.exe; the watcher holds the subprocess.Popen object (an open
process handle), so a tracked PID cannot be reused by another process while it is tracked.

Windows remembers a session's mute per executable path: once muted, later launches of the same
<copy>/Bin/Player.exe start muted even without this module, so the opt-out (BALLRAIN_AUDIO=1) actively
unmutes instead of leaving the session alone. The session peak meter is measured before the mute
(it still moves while muted); the render endpoint's meter shows what is actually played.

Plain ctypes COM (vtable calls), no extra dependency. Windows only; elsewhere everything is a no-op.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

AUDIO_VAR = "BALLRAIN_AUDIO"  # "1": leave game audio on (launcher-only, never passed to the game)

POLL_FAST_S = 0.1
FAST_WINDOW_S = 30.0
POLL_SLOW_S = 2.0


def audio_enabled(audio: bool | None = None) -> bool:
    """Whether launched games keep their sound: `audio` if given, else BALLRAIN_AUDIO (unset / "0" = muted)."""
    if audio is not None:
        return bool(audio)
    return os.environ.get(AUDIO_VAR, "").strip() not in ("", "0")


@dataclass(frozen=True)
class SessionInfo:
    pid: int
    muted: bool
    state: int  # AudioSessionState: 0 inactive, 1 active, 2 expired
    peak: float | None = None
    endpoint: int = 0  # index of the render endpoint in the enumeration


def sessions_to_change(sessions, tracked: dict) -> list[int]:
    """Indices of sessions whose mute state must be changed: the session belongs to a tracked PID
    (pid 0 = system sounds, never) and its mute flag differs from tracked[pid] (True = want muted)."""
    out = []
    for i, s in enumerate(sessions):
        if s.pid and s.pid in tracked and bool(s.muted) != bool(tracked[s.pid]):
            out.append(i)
    return out


# ------------------------------------------------------------------------------------------ COM

if sys.platform == "win32":
    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]

    def _guid(s: str) -> GUID:
        g = GUID()
        if ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p("{" + s + "}"), ctypes.byref(g)):
            raise OSError(f"bad GUID {s}")
        return g

    _ole32 = ctypes.OleDLL("ole32")
    _ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    _ole32.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    _ole32w = ctypes.WinDLL("ole32")  # CoInitializeEx returns S_FALSE (1) when already initialised
    _ole32w.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    _ole32w.CoInitializeEx.restype = ctypes.c_long
    _ole32w.CoUninitialize.argtypes = []

    CLSID_MMDeviceEnumerator = _guid("BCDE0395-E52F-467C-8E3D-C4579291692E")
    IID_IMMDeviceEnumerator = _guid("A95664D2-9614-4F35-A746-DE8DB63617E6")
    IID_IAudioSessionManager2 = _guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F")
    IID_IAudioSessionControl2 = _guid("BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D")
    IID_ISimpleAudioVolume = _guid("87CE5498-68D6-44E5-9215-6DA47EF883D8")
    IID_IAudioMeterInformation = _guid("C02216F6-8C67-4B5B-9D00-D008E73E0064")

CLSCTX_ALL = 0x17
COINIT_MULTITHREADED = 0x0
eRender = 0
DEVICE_STATE_ACTIVE = 0x1
RPC_E_CHANGED_MODE = -2147417850


class _Com:
    """A raw COM interface pointer with vtable calls; released on release() / garbage collection."""

    def __init__(self, ptr: int):
        self.ptr = ptr

    def call(self, index: int, restype, argtypes, *args):
        vtbl = ctypes.cast(ctypes.c_void_p(self.ptr), ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])
        return fn(self.ptr, *args)

    def hr(self, index: int, argtypes, *args) -> None:
        r = self.call(index, ctypes.c_long, argtypes, *args)
        if r < 0:
            raise OSError(r & 0xFFFFFFFF, f"COM call (vtable {index}) failed: 0x{r & 0xFFFFFFFF:08X}")

    def out_ptr(self, index: int, argtypes, *args) -> "_Com":
        p = ctypes.c_void_p()
        self.hr(index, list(argtypes) + [ctypes.POINTER(ctypes.c_void_p)], *args, ctypes.byref(p))
        return _Com(p.value)

    def qi(self, iid) -> "_Com":
        return self.out_ptr(0, [ctypes.POINTER(GUID)], ctypes.byref(iid))

    def release(self) -> None:
        if self.ptr:
            self.call(2, ctypes.c_ulong, [])
            self.ptr = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def com_init() -> bool:
    """CoInitializeEx(MTA) for the calling thread; True if this call must be paired with CoUninitialize."""
    r = _ole32w.CoInitializeEx(None, COINIT_MULTITHREADED)
    if r == RPC_E_CHANGED_MODE:  # thread already in an STA: COM is usable, nothing to undo
        return False
    if r < 0:
        raise OSError(r & 0xFFFFFFFF, "CoInitializeEx failed")
    return True


def _session_managers():
    """IAudioSessionManager2 of every active render endpoint."""
    enum_p = ctypes.c_void_p()
    _ole32.CoCreateInstance(ctypes.byref(CLSID_MMDeviceEnumerator), None, CLSCTX_ALL,
                            ctypes.byref(IID_IMMDeviceEnumerator), ctypes.byref(enum_p))
    enum = _Com(enum_p.value)
    coll = enum.out_ptr(3, [ctypes.c_int, wintypes.DWORD], eRender, DEVICE_STATE_ACTIVE)  # EnumAudioEndpoints
    n = wintypes.UINT()
    coll.hr(3, [ctypes.POINTER(wintypes.UINT)], ctypes.byref(n))  # GetCount
    out = []
    for i in range(n.value):
        dev = coll.out_ptr(4, [wintypes.UINT], i)  # Item
        out.append(dev.out_ptr(3, [ctypes.POINTER(GUID), wintypes.DWORD, ctypes.c_void_p],  # Activate
                               ctypes.byref(IID_IAudioSessionManager2), CLSCTX_ALL, None))
    return out


def _iter_sessions(pids=None):
    """(SessionInfo, IAudioSessionControl2) for every session on every active render endpoint
    (only those of `pids` when given)."""
    for e, mgr in enumerate(_session_managers()):
        senum = mgr.out_ptr(5, [])  # GetSessionEnumerator
        cnt = ctypes.c_int()
        senum.hr(3, [ctypes.POINTER(ctypes.c_int)], ctypes.byref(cnt))
        for i in range(cnt.value):
            ctl = senum.out_ptr(4, [ctypes.c_int], i).qi(IID_IAudioSessionControl2)
            pid = wintypes.DWORD()
            r = ctl.call(14, ctypes.c_long, [ctypes.POINTER(wintypes.DWORD)], ctypes.byref(pid))  # GetProcessId
            if r < 0 and r != 0x889000D - 0x100000000:  # AUDCLNT_S_NO_SINGLE_PROCESS is a success code
                pid.value = 0
            if pids is not None and pid.value not in pids:
                continue
            state = ctypes.c_int()
            ctl.hr(3, [ctypes.POINTER(ctypes.c_int)], ctypes.byref(state))  # GetState
            vol = ctl.qi(IID_ISimpleAudioVolume)
            m = wintypes.BOOL()
            vol.hr(6, [ctypes.POINTER(wintypes.BOOL)], ctypes.byref(m))  # GetMute
            yield SessionInfo(int(pid.value), bool(m.value), state.value, None, e), ctl


def _snapshot(pids, peak: bool) -> list[SessionInfo]:
    out = []
    for info, ctl in _iter_sessions(None if pids is None else set(pids)):
        if peak:
            meter = ctl.qi(IID_IAudioMeterInformation)
            v = ctypes.c_float()
            meter.hr(3, [ctypes.POINTER(ctypes.c_float)], ctypes.byref(v))  # GetPeakValue
            meter.release()
            info = SessionInfo(info.pid, info.muted, info.state, float(v.value), info.endpoint)
        ctl.release()
        out.append(info)
    return out


def list_sessions(pids=None, peak: bool = False) -> list[SessionInfo]:
    """Read-only snapshot of the audio sessions (of `pids` when given), optionally with the peak meter."""
    owned = com_init()
    try:
        return _snapshot(pids, peak)  # every interface pointer is released when this returns
    finally:
        if owned:
            _ole32w.CoUninitialize()


def apply_mute(tracked: dict) -> tuple[list[SessionInfo], list[SessionInfo]]:
    """Set the mute flag of every session of a tracked PID to tracked[pid] (True = muted) where it differs.
    COM must be initialised on the calling thread. Returns (sessions found, sessions changed)."""
    if not tracked:
        return [], []
    items = list(_iter_sessions(set(tracked)))
    changed = []
    for i in sessions_to_change([s for s, _ in items], tracked):
        info, ctl = items[i]
        ctl.qi(IID_ISimpleAudioVolume).hr(5, [wintypes.BOOL, ctypes.c_void_p], bool(tracked[info.pid]), None)  # SetMute
        changed.append(info)
    return [info for info, _ in items], changed


# -------------------------------------------------------------------------------------- watcher

def _image_matches(pid: int, exe) -> bool:
    from .instances import _norm, process_path  # late import: instances imports this module
    p = process_path(pid)
    return bool(p) and _norm(p) == _norm(exe)


class _Watcher:
    """Daemon thread that keeps the sessions of the tracked processes at their wanted mute state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.procs: dict = {}  # pid -> _Tracked
        self.wake = threading.Event()
        self.thread = None
        self.errors = 0
        self.last_error = None
        self.changes = 0

    def add(self, proc, exe, muted: bool) -> None:
        with self.lock:
            self.procs[proc.pid] = _Tracked(proc, exe, bool(muted), time.monotonic())
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="ballrain-audio-mute", daemon=True)
                self.thread.start()
        self.wake.set()

    def _round(self) -> float | None:
        """One scan; returns the wait before the next one (None: nothing tracked any more, stop)."""
        with self.lock:
            for pid in [p for p, t in self.procs.items() if t.proc.poll() is not None]:
                del self.procs[pid]  # exited: drop the handle (the PID may be reused from now on)
            if not self.procs:
                self.thread = None
                return None
            items = list(self.procs.items())
        tracked = {}
        for pid, t in items:
            if t.verified is None:  # once: the PID cannot be reused while we hold its handle
                t.verified = _image_matches(pid, t.exe)
            if t.verified:
                tracked[pid] = t.muted
        try:
            found, changed = apply_mute(tracked)
            self.changes += len(changed)
            seen = {s.pid for s in found}
            for pid, t in items:
                if pid in seen:
                    t.has_session = True
        except Exception as e:  # audio service hiccup, device change: try again next round
            self.errors += 1
            self.last_error = repr(e)
        now = time.monotonic()
        fast = any(t.verified and not t.has_session and now - t.t_added < FAST_WINDOW_S for _, t in items)
        return POLL_FAST_S if fast else POLL_SLOW_S

    def _run(self) -> None:
        owned = False
        try:
            owned = com_init()
            while True:
                wait = self._round()
                if wait is None:
                    return
                self.wake.wait(wait)
                self.wake.clear()
        finally:
            if owned:
                _ole32w.CoUninitialize()


@dataclass
class _Tracked:
    proc: object
    exe: object
    muted: bool
    t_added: float
    verified: bool | None = None
    has_session: bool = False


_WATCHER = _Watcher()


def watch(proc, exe, muted: bool = True) -> None:
    """Keep the audio sessions of the launched process `proc` (a subprocess.Popen whose image must be
    `exe`) muted (or, with muted=False, unmuted) until it exits. No-op off Windows."""
    if sys.platform != "win32":
        return
    _WATCHER.add(proc, exe, muted)

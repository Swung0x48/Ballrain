"""Whole-process snapshot of a headless engine this process started (Windows, 32-bit engine under WOW64), from outside:
save = suspend every thread of the engine and read every committed writable private / image page; load = suspend,
free what was allocated since, write the pages back, resume. No engine change: the engine (with its in-process 2002
physics DLL) does not know it happened (docs/segsearch.md "Rollback").

Only for a process this Python process launched (its own Popen handle and pid): never look a process up by name.
Kernel state is not in the snapshot (socket buffers, files the engine appended to, handles): save and load only while
the engine waits for the next action, and never load into a process that ended.

    snap = ProcessSnapshot(popen)          # popen = the engine's subprocess.Popen
    img = snap.save()                      # engine idle in its recv
    ... run ...
    snap.load(img)                         # back to the saved state
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import time
from dataclasses import dataclass, field

k32 = ctypes.WinDLL("kernel32", use_last_error=True)

MEM_COMMIT, MEM_RESERVE, MEM_FREE = 0x1000, 0x2000, 0x10000
MEM_DECOMMIT, MEM_RELEASE = 0x4000, 0x8000
MEM_PRIVATE, MEM_MAPPED, MEM_IMAGE = 0x20000, 0x40000, 0x1000000
PAGE_NOACCESS, PAGE_GUARD = 0x01, 0x100
WRITABLE = 0x04 | 0x08 | 0x40 | 0x80  # READWRITE, WRITECOPY, EXECUTE_READWRITE, EXECUTE_WRITECOPY
TH32CS_SNAPTHREAD = 0x4
THREAD_ALL_ACCESS = 0x1FFFFF
WOW64_CONTEXT_ALL = 0x0001003F
WOW64_CONTEXT_SIZE = 716
# WOW64_CONTEXT field offsets (winnt.h): the integer / control registers compared at load
CTX_FIELDS = {"Dr0": 4, "Edi": 156, "Esi": 160, "Ebx": 164, "Edx": 168, "Ecx": 172, "Eax": 176, "Ebp": 180, "Eip": 184,
              "SegCs": 188, "EFlags": 192, "Esp": 196, "SegSs": 200}
# FloatSave starts at 28 (ControlWord first); ExtendedRegisters (FXSAVE) at 204: FCW +0, MXCSR +24
FP_CONTROL = {"FloatSave.ControlWord": (28, 2), "FXSAVE.FCW": (204, 2), "FXSAVE.MXCSR": (228, 4)}


class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_uint64), ("AllocationBase", ctypes.c_uint64), ("AllocationProtect", wt.DWORD),
                ("PartitionId", wt.WORD), ("RegionSize", ctypes.c_uint64), ("State", wt.DWORD), ("Protect", wt.DWORD),
                ("Type", wt.DWORD)]


class THREADENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ThreadID", wt.DWORD), ("th32OwnerProcessID", wt.DWORD),
                ("tpBasePri", wt.LONG), ("tpDeltaPri", wt.LONG), ("dwFlags", wt.DWORD)]


k32.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
k32.VirtualQueryEx.restype = ctypes.c_size_t
k32.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k32.WriteProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
k32.VirtualAllocEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualAllocEx.restype = ctypes.c_void_p
k32.VirtualFreeEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
k32.VirtualProtectEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, ctypes.POINTER(wt.DWORD)]
k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
k32.Thread32First.argtypes = [wt.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.Thread32Next.argtypes = [wt.HANDLE, ctypes.POINTER(THREADENTRY32)]
k32.OpenThread.restype = wt.HANDLE
k32.Wow64SuspendThread.argtypes = [wt.HANDLE]
k32.ResumeThread.argtypes = [wt.HANDLE]
k32.Wow64GetThreadContext.argtypes = [wt.HANDLE, ctypes.c_void_p]
k32.QueryThreadCycleTime.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_uint64)]
k32.GetProcessHandleCount.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
k32.CloseHandle.argtypes = [wt.HANDLE]


class SnapshotError(RuntimeError):
    pass


def _check(ok, what):
    if not ok:
        raise SnapshotError(f"{what} failed: {ctypes.WinError(ctypes.get_last_error())}")


@dataclass
class Image:
    regions: list            # (base, size, protect, type, allocation_base, bytes)
    private_allocs: set      # allocation bases of committed private regions at the save
    contexts: dict           # tid -> raw WOW64_CONTEXT bytes
    nbytes: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)  # caller state (python side) kept with the image


class ProcessSnapshot:
    def __init__(self, popen):
        self.pid = int(popen.pid)
        self.hproc = int(popen._handle)  # the handle CreateProcess gave us: full access to our own child
        self.popen = popen

    # ------------------------------------------------------------------ threads
    def thread_ids(self) -> list[int]:
        snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if snap in (None, wt.HANDLE(-1).value):
            raise SnapshotError("CreateToolhelp32Snapshot failed")
        out = []
        try:
            te = THREADENTRY32()
            te.dwSize = ctypes.sizeof(te)
            ok = k32.Thread32First(snap, ctypes.byref(te))
            while ok:
                if te.th32OwnerProcessID == self.pid:
                    out.append(int(te.th32ThreadID))
                ok = k32.Thread32Next(snap, ctypes.byref(te))
        finally:
            k32.CloseHandle(snap)
        return sorted(out)

    def _open_threads(self):
        hs = {}
        for tid in self.thread_ids():
            h = k32.OpenThread(THREAD_ALL_ACCESS, False, tid)
            _check(h, f"OpenThread {tid}")
            hs[tid] = h
        return hs

    @staticmethod
    def _context(h) -> bytes:
        buf = ctypes.create_string_buffer(WOW64_CONTEXT_SIZE)
        ctypes.c_uint32.from_buffer(buf, 0).value = WOW64_CONTEXT_ALL
        _check(k32.Wow64GetThreadContext(h, buf), "Wow64GetThreadContext")
        return buf.raw

    @staticmethod
    def _cycles(h) -> int:
        c = ctypes.c_uint64()
        _check(k32.QueryThreadCycleTime(h, ctypes.byref(c)), "QueryThreadCycleTime")
        return c.value

    def wait_idle(self, hs: dict, quiet_s: float = 0.003, timeout_s: float = 5.0) -> None:
        """Until no thread of the engine runs for quiet_s (the main thread blocked in its recv)."""
        t_end = time.monotonic() + timeout_s
        last = {t: self._cycles(h) for t, h in hs.items()}
        quiet_since = time.monotonic()
        while time.monotonic() < t_end:
            time.sleep(0.0005)
            now = {t: self._cycles(h) for t, h in hs.items()}
            if now != last:
                last = now
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= quiet_s:
                return
        raise SnapshotError("engine threads did not go idle")

    def _suspend(self, hs):
        for t, h in hs.items():
            if k32.Wow64SuspendThread(h) == 0xFFFFFFFF:
                raise SnapshotError(f"suspend {t}: {ctypes.WinError(ctypes.get_last_error())}")

    def _resume(self, hs):
        for h in hs.values():
            k32.ResumeThread(h)

    @staticmethod
    def _close(hs):
        for h in hs.values():
            k32.CloseHandle(h)

    # ------------------------------------------------------------------ memory
    def regions(self):
        """Committed regions: (base, size, state, protect, type, allocation_base)."""
        out = []
        addr = 0
        mbi = MBI()
        while k32.VirtualQueryEx(self.hproc, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            base, size = int(mbi.BaseAddress), int(mbi.RegionSize)
            out.append((base, size, int(mbi.State), int(mbi.Protect), int(mbi.Type), int(mbi.AllocationBase)))
            addr = base + size
            if size == 0:
                break
        return out

    @staticmethod
    def _saved_kind(state, protect, typ) -> bool:
        return (state == MEM_COMMIT and typ in (MEM_PRIVATE, MEM_IMAGE) and protect & WRITABLE
                and not protect & (PAGE_GUARD | PAGE_NOACCESS))

    def _read(self, base, size) -> bytes:
        buf = ctypes.create_string_buffer(size)
        n = ctypes.c_size_t()
        _check(k32.ReadProcessMemory(self.hproc, ctypes.c_void_p(base), buf, size, ctypes.byref(n)), f"read {base:#x}+{size:#x}")
        return buf.raw

    def _write(self, base, data):
        n = ctypes.c_size_t()
        _check(k32.WriteProcessMemory(self.hproc, ctypes.c_void_p(base), data, len(data), ctypes.byref(n)),
               f"write {base:#x}+{len(data):#x}")

    def handle_count(self) -> int:
        c = wt.DWORD()
        _check(k32.GetProcessHandleCount(self.hproc, ctypes.byref(c)), "GetProcessHandleCount")
        return int(c.value)

    # ------------------------------------------------------------------ save / load
    def save(self) -> Image:
        if self.popen.poll() is not None:
            raise SnapshotError("engine process has ended")
        t0 = time.perf_counter()
        hs = self._open_threads()
        try:
            self.wait_idle(hs)
            self._suspend(hs)
            try:
                ctx = {t: self._context(h) for t, h in hs.items()}
                regs, allocs, nb = [], set(), 0
                for base, size, state, prot, typ, ab in self.regions():
                    if state == MEM_COMMIT and typ == MEM_PRIVATE:
                        allocs.add(ab)
                    if self._saved_kind(state, prot, typ):
                        regs.append((base, size, prot, typ, ab, self._read(base, size)))
                        nb += size
            finally:
                self._resume(hs)
        finally:
            self._close(hs)
        return Image(regs, allocs, ctx, nb, time.perf_counter() - t0)

    def load(self, img: Image) -> dict:
        """Write the image back. Refuses (SnapshotError, process left as it was) when the thread set or any thread's
        register context differs from the save's."""
        if self.popen.poll() is not None:
            raise SnapshotError("engine process has ended")
        t0 = time.perf_counter()
        hs = self._open_threads()
        stats = {"released": 0, "decommitted": 0, "recommitted": 0, "written": 0}
        try:
            if sorted(hs) != sorted(img.contexts):
                raise SnapshotError(f"thread set changed: {sorted(img.contexts)} -> {sorted(hs)}")
            self.wait_idle(hs)
            self._suspend(hs)
            try:
                for t, h in hs.items():
                    now, was = self._context(h), img.contexts[t]
                    # integer / control registers and the x87 control word / MXCSR must match; the x87 / XMM data
                    # registers of a thread blocked in a call are dead values (volatile across calls on x86-32)
                    diff = {k: (int.from_bytes(was[o:o + 4], "little"), int.from_bytes(now[o:o + 4], "little"))
                            for k, o in CTX_FIELDS.items() if now[o:o + 4] != was[o:o + 4]}
                    for k, (o, n) in FP_CONTROL.items():
                        if now[o:o + n] != was[o:o + n]:
                            diff[k] = (was[o:o + n].hex(), now[o:o + n].hex())
                    if diff:
                        raise SnapshotError(f"thread {t} context differs from the save: {diff}")
                    if now != was:
                        stats["fp_data_differs"] = stats.get("fp_data_differs", 0) + 1
                saved = {r[0]: r for r in img.regions}
                cur = self.regions()
                # private allocations made since the save: release; committed private pages the save did not have: decommit
                released = set()
                for base, size, state, prot, typ, ab in cur:
                    if state == MEM_COMMIT and typ == MEM_PRIVATE and ab not in img.private_allocs and ab not in released:
                        _check(k32.VirtualFreeEx(self.hproc, ctypes.c_void_p(ab), 0, MEM_RELEASE), f"release {ab:#x}")
                        released.add(ab)
                stats["released"] = len(released)
                covered = [(r[0], r[0] + r[1]) for r in img.regions]
                for base, size, state, prot, typ, ab in self.regions():
                    if state == MEM_COMMIT and typ == MEM_PRIVATE and prot & WRITABLE and not prot & PAGE_GUARD:
                        if not any(lo <= base and base + size <= hi for lo, hi in covered):
                            # partly covered regions are split by page below
                            for p in range(base, base + size, 0x1000):
                                if not any(lo <= p < hi for lo, hi in covered):
                                    k32.VirtualFreeEx(self.hproc, ctypes.c_void_p(p), 0x1000, MEM_DECOMMIT)
                                    stats["decommitted"] += 1
                for base, size, prot, typ, ab, data in img.regions:
                    mbi = MBI()
                    k32.VirtualQueryEx(self.hproc, ctypes.c_void_p(base), ctypes.byref(mbi), ctypes.sizeof(mbi))
                    if int(mbi.State) != MEM_COMMIT or int(mbi.RegionSize) < size:
                        flags = MEM_COMMIT | (MEM_RESERVE if int(mbi.State) == MEM_FREE else 0)
                        _check(k32.VirtualAllocEx(self.hproc, ctypes.c_void_p(base), size, flags, prot), f"recommit {base:#x}")
                        stats["recommitted"] += 1
                    self._write(base, data)
                    stats["written"] += size
            finally:
                self._resume(hs)
        finally:
            self._close(hs)
        stats["seconds"] = time.perf_counter() - t0
        return stats

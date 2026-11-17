"""Controller-side process lifecycle (docs/pool-prototype.md section 9): every simulator and helper the controller starts
dies with it -- on a normal exit, Ctrl+C, a closed terminal window, a crash or a hard kill (docs/volunteer-compute.md 11.4).

- Windows: one Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (+ below-normal priority class, no error dialogs).
  Each child is created suspended, assigned to the job, then resumed, so it cannot start anything outside the job; the
  engines a worker starts later inherit the job. The job handle is never inheritable: when the controller process ends
  (however it ends) the kernel closes the handle and terminates every process in the job.
- Linux: each child is a new session (start_new_session: its own process group, the engines it starts join the group)
  and sets PR_SET_PDEATHSIG(SIGTERM) on itself (bind_to_parent); the worker's SIGTERM handler kills its whole group. A parent-watch thread is the backup.
- macOS (and the generic fallback): the child watches its parent (stdin EOF and getppid()) and kills its group.

Also: below-normal priority, "is Player.exe running" and "on battery" probes for the automatic pause (best effort, every
probe can be replaced -- tests and machines that run Player.exe copies for other reasons pass their own).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"

# Windows constants
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x00000020
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9


# ============================================================================================== Windows job object

if IS_WIN:
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                      "OtherOperationCount", "ReadTransferCount",
                                                      "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT), ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.TerminateJobObject.restype = wintypes.BOOL
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.GetExitCodeProcess.restype = wintypes.BOOL
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.SetPriorityClass.restype = wintypes.BOOL
    _k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.GetCurrentProcess.argtypes = []
    _k32.TerminateProcess.restype = wintypes.BOOL
    _k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _ntdll.NtResumeProcess.restype = ctypes.c_long
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

    def _winerr(what: str) -> OSError:
        e = ctypes.get_last_error()
        return OSError(e, f"{what}: {ctypes.FormatError(e).strip()}")

    class JobObject:
        """A kill-on-close job (not inheritable: lpJobAttributes NULL). Children assigned to it -- and everything they
        start -- end when the last handle closes, i.e. when this process ends."""

        def __init__(self, below_normal: bool = True):
            h = _k32.CreateJobObjectW(None, None)
            if not h:
                raise _winerr("CreateJobObjectW")
            self.handle = h
            info = _EXTENDED_LIMIT()
            flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            if below_normal:
                flags |= JOB_OBJECT_LIMIT_PRIORITY_CLASS
                info.BasicLimitInformation.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS
            info.BasicLimitInformation.LimitFlags = flags
            if not _k32.SetInformationJobObject(h, JobObjectExtendedLimitInformation, ctypes.byref(info),
                                                ctypes.sizeof(info)):
                err = _winerr("SetInformationJobObject")
                _k32.CloseHandle(h)
                raise err

        def assign(self, proc_handle: int) -> None:
            if not _k32.AssignProcessToJobObject(self.handle, int(proc_handle)):
                raise _winerr("AssignProcessToJobObject")

        def terminate(self, code: int = 1) -> None:
            if self.handle:
                _k32.TerminateJobObject(self.handle, code)

        def close(self) -> None:
            if self.handle:
                _k32.CloseHandle(self.handle)
                self.handle = None

    def _resume(proc_handle: int) -> None:
        st = _ntdll.NtResumeProcess(int(proc_handle))
        if st != 0:
            raise OSError(f"NtResumeProcess: NTSTATUS {st & 0xFFFFFFFF:#x}")

    def pid_alive(pid: int) -> bool:
        """True while a process with this id exists (SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION)."""
        h = _k32.OpenProcess(0x00100000 | 0x1000, False, int(pid))
        if not h:
            return False
        try:
            return _k32.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            _k32.CloseHandle(h)

    def kill_pid(pid: int) -> None:
        """TerminateProcess on one pid (tests: the hard kill of the controller)."""
        h = _k32.OpenProcess(0x0001, False, int(pid))  # PROCESS_TERMINATE
        if h:
            try:
                _k32.TerminateProcess(h, 9)
            finally:
                _k32.CloseHandle(h)
else:
    JobObject = None  # type: ignore

    def pid_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        try:  # a zombie child of ours counts as gone
            done, _ = os.waitpid(int(pid), os.WNOHANG)
            return done == 0
        except ChildProcessError:
            return True

    def kill_pid(pid: int) -> None:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


# ============================================================================================== priority

def lower_priority() -> None:
    """Below-normal priority for this process (children inherit it)."""
    try:
        if IS_WIN:
            _k32.SetPriorityClass(_k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(10)
    except OSError:
        pass


# ============================================================================================== child processes

def python_command() -> tuple[list, dict]:
    """The interpreter to start a child with, and extra environment. A Windows venv's python.exe is a redirector that
    starts the base interpreter as its own child; start the base interpreter directly (as multiprocessing does) so the
    pid we hold is the real worker."""
    exe = sys.executable
    extra = {}
    base = getattr(sys, "_base_executable", None)
    if IS_WIN and base and os.path.normcase(base) != os.path.normcase(exe) and os.path.isfile(base):
        extra["__PYVENV_LAUNCHER__"] = exe
        exe = base
    return [exe], extra


def bind_to_parent() -> None:
    """Child side, POSIX, first thing in the worker: Linux PR_SET_PDEATHSIG(SIGTERM), then check that the parent is
    still the controller (BALLRAIN_POOL_PARENT_PID) -- if it died before prctl ran, exit now (the race-free pattern).
    Done in the child itself rather than in a preexec_fn: nothing runs between fork and exec of a threaded controller
    (start_new_session=True does the setsid, the worker lowers its own priority)."""
    if IS_WIN:
        return
    if IS_LINUX:
        try:
            import ctypes as _c
            libc = _c.CDLL(None, use_errno=True)
            libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
        except (OSError, AttributeError):
            pass
    want = os.environ.get("BALLRAIN_POOL_PARENT_PID")
    if want and want.isdigit() and os.getppid() != int(want):
        os._exit(3)


class ChildGroup:
    """Starts and tracks the controller's child processes so that none of them outlives the controller."""

    def __init__(self):
        self.job = JobObject() if IS_WIN else None
        self.procs: list[subprocess.Popen] = []
        self._lock = threading.Lock()

    def spawn(self, args: list, env: dict | None = None, cwd=None, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
              stderr=None) -> subprocess.Popen:
        kw: dict = {"stdin": stdin, "stdout": stdout, "stderr": stderr, "env": env, "cwd": cwd, "close_fds": True}
        if IS_WIN:
            kw["creationflags"] = CREATE_SUSPENDED | CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS \
                | CREATE_NEW_PROCESS_GROUP  # Ctrl+C in the controller's console is the controller's, not the worker's
        else:
            kw["start_new_session"] = True  # its own session / process group: the engines it starts join the group
        p = subprocess.Popen(args, **kw)
        if IS_WIN:
            try:
                self.job.assign(p._handle)  # noqa: SLF001  (the process handle Popen holds)
                _resume(p._handle)  # noqa: SLF001
            except OSError:
                p.kill()
                p.wait(5)
                raise
        with self._lock:
            self.procs.append(p)
        return p

    def alive(self) -> list[subprocess.Popen]:
        with self._lock:
            self.procs = [p for p in self.procs if p.poll() is None]
            return list(self.procs)

    def kill_tree(self, p: subprocess.Popen) -> None:
        """Kill one child and everything it started (POSIX: its process group; Windows: the process -- its engines exit
        on the socket EOF, and the job catches the rest when the controller ends)."""
        if p.poll() is not None:
            return
        if IS_WIN:
            p.kill()
        else:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                p.kill()

    def shutdown(self, grace_s: float = 3.0) -> None:
        """Ask nicely (the caller has sent its stop messages), wait up to grace_s, then kill everything: on Windows the
        whole job (workers and engines alike)."""
        t = time.monotonic() + grace_s
        for p in self.alive():
            try:
                p.wait(max(0.0, t - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for p in self.alive():
            self.kill_tree(p)
        if IS_WIN and self.job is not None:
            self.job.terminate()
        for p in list(self.procs):
            try:
                p.wait(2.0)
            except subprocess.TimeoutExpired:
                pass


# ============================================================================================== child side

class ParentWatch:
    """Child side: exit (killing our process group on POSIX) when the controller is gone. Signals: stdin EOF (the
    pipe's write end closes with the controller), the parent pid changing (POSIX), the parent handle signalled
    (Windows; the job usually gets there first)."""

    def __init__(self, on_gone, poll_s: float = 0.5):
        self.on_gone = on_gone
        self.poll_s = poll_s
        self.ppid = os.getppid()
        self._parent_handle = None
        if IS_WIN:
            ppid = int(os.environ.get("BALLRAIN_POOL_PARENT_PID") or self.ppid)
            h = _k32.OpenProcess(0x00100000, False, ppid)  # SYNCHRONIZE
            self._parent_handle = h or None
        if not IS_WIN:
            try:
                signal.signal(signal.SIGTERM, lambda *_: self.on_gone("SIGTERM"))
            except ValueError:  # not the main thread
                pass
        threading.Thread(target=self._run, daemon=True, name="parent-watch").start()

    def _run(self):
        while True:
            if IS_WIN:
                if self._parent_handle is not None and _k32.WaitForSingleObject(self._parent_handle, 0) == 0:
                    self.on_gone("parent exited")
                    return
            elif os.getppid() != self.ppid:
                self.on_gone("parent exited")
                return
            time.sleep(self.poll_s)


def kill_own_group() -> None:
    """Child side, POSIX: SIGKILL our process group (ourselves and the engines we started)."""
    if not IS_WIN:
        try:
            os.killpg(os.getpgid(0), signal.SIGKILL)
        except OSError:
            pass
    os._exit(3)


# ============================================================================================== auto-pause probes

def _process_names_windows() -> list[str]:
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    out = []
    try:
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(e)
        ok = k.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out.append(e.szExeFile)
            ok = k.Process32NextW(snap, ctypes.byref(e))
    finally:
        k.CloseHandle(snap)
    return out


def player_running() -> bool:
    """True while a Ballance Player.exe runs on this machine (the player is playing: pause). Read-only probe: the
    controller never touches that process."""
    try:
        if IS_WIN:
            return any(n.lower() == "player.exe" for n in _process_names_windows())
        if IS_LINUX:  # Ballance under Wine shows up as Player.exe in /proc/<pid>/comm or cmdline
            for d in os.listdir("/proc"):
                if not d.isdigit():
                    continue
                try:
                    with open(f"/proc/{d}/cmdline", "rb") as f:
                        cmd = f.read(4096).lower()
                except OSError:
                    continue
                if b"player.exe" in cmd:
                    return True
            return False
        if IS_MAC:
            r = subprocess.run(["ps", "-axo", "comm"], capture_output=True, text=True, timeout=5)
            return any(line.strip().lower().endswith("player.exe") for line in r.stdout.splitlines())
    except Exception:  # noqa: BLE001  best effort
        return False
    return False


def on_battery() -> bool:
    """True when the machine runs on battery (best effort; False when unknown, e.g. a desktop)."""
    try:
        if IS_WIN:
            import ctypes
            from ctypes import wintypes

            class SYSTEM_POWER_STATUS(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                            ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                            ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]
            s = SYSTEM_POWER_STATUS()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
                return False
            return s.ACLineStatus == 0 and s.BatteryFlag != 128  # 128: no system battery
        if IS_LINUX:
            base = "/sys/class/power_supply"
            if not os.path.isdir(base):
                return False
            mains = []
            for d in os.listdir(base):
                try:
                    with open(os.path.join(base, d, "type")) as f:
                        kind = f.read().strip()
                    if kind == "Mains":
                        with open(os.path.join(base, d, "online")) as f:
                            mains.append(f.read().strip() == "1")
                except OSError:
                    continue
            return bool(mains) and not any(mains)
        if IS_MAC:
            r = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
            return "Battery Power" in r.stdout
    except Exception:  # noqa: BLE001
        return False
    return False

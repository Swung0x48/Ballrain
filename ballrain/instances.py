"""Management of the dedicated game copies under <repo>/runtime/<name>.

Safety model:
- Player.exe is only ever launched from <copy>/Bin (cwd = Bin), one process per copy.
- A process is only ever stopped through a handle to the PID this module launched, after
  re-checking that its image path is exactly <copy>/Bin/Player.exe.
- The launch recipe (how Player.exe is started) is a single replaceable function with the
  signature recipe(copy: GameCopy, env: dict[str, str | None]) -> pid. env maps variable
  names to values; None means "make sure it is unset". Set LAUNCH_RECIPE or pass recipe=...

Machine paths are never hard-coded: the runtime directory is <repo>/runtime, overridable by
the BALLRAIN_RUNTIME environment variable.

Game windows: the direct recipe starts Player.exe with STARTUPINFO wShowWindow = SW_HIDE, so its main
window is never shown (no window, no taskbar button). Set BALLRAIN_VISIBLE=1 (inherited by worker
processes) or pass visible=True to see them again. Traces are unchanged by this (README, "游戏窗口").

Foreground: Player.exe takes the foreground at start-up (before any mod loads); BallrainIO >= 0.3.2 hands it
straight back unless the launch sets BALLRAIN_KEEP_FOCUS=1, which launch_direct does for visible windows,
keep_focus=True or BALLRAIN_FOCUS=1 (README, "游戏窗口"). runtime/ref (no BallrainIO) keeps the foreground.

Game sound: launch_direct mutes the Windows audio sessions of the process it started (ballrain.audio,
per PID, Core Audio ISimpleAudioVolume::SetMute); BALLRAIN_AUDIO=1 or audio_on=True keeps the sound.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

from . import audio, tas

REPO_ROOT = Path(__file__).resolve().parents[1]

# sha256 of the 2002 physics_RT.dll (computed from runtime/ref/BuildingBlocks/physics_RT.dll).
PHYSICS_RT_SHA256 = "e72e4afcfa5c33a7d3d27776137f8c997b3c52d89d8a8a4745f1ca21e45893ec"

REFERENCE_COPY = "ref"
TRAINING_COPY = "train-00"

# Variables of the BallrainIO contract and of the patched TASSupport. Every launch sets or
# explicitly clears all of them so nothing leaks in from the parent / shell environment.
BALLRAIN_VARS = (
    "BALLRAIN_MODE",
    "BALLRAIN_FRAMES",
    "BALLRAIN_LEVEL",
    "BALLRAIN_TRACE",
    "BALLRAIN_FRAMELOG",
    "BALLRAIN_EVENTS",
    "BALLRAIN_SKIP_RENDER",
    "BALLRAIN_EXIT",
    "BALLRAIN_LOADS",
    "BALLRAIN_DEBUG",  # diagnostic logging in the mod; always cleared unless a launch sets it
    # agent mode (docs/agent-protocol.md section 6)
    "BALLRAIN_SHM",
    "BALLRAIN_PARENT_PID",
    "BALLRAIN_TIMEOUT_MS",
    "BALLRAIN_LEVELMSG",
    "BALLRAIN_PREFIX",
    "BALLRAIN_PREFIX_EVENTS",
    "BALLRAIN_MAX_FRAMES",
    "BALLRAIN_BODY_RADIUS",
    "BALLRAIN_SR_TAIL",
    "BALLRAIN_POST_FINISH",  # diagnostic: idle frames after the SR tail (BallrainIO >= 0.3.1)
    "BALLRAIN_POST_LOG",  # diagnostic: per-frame text log from LevelFinish to the end of the run
    "BALLRAIN_KEEP_FOCUS",  # "1": the mod does not hand the foreground back (BallrainIO >= 0.3.2; set by the launch)
)
BMMO_VARS = ("BMMO_TAS_TRACE", "BMMO_TAS_OFFSETS", "BMMO_TAS_CORE", "BMMO_TAS_PC")
MANAGED_VARS = BALLRAIN_VARS + BMMO_VARS
# Read by this module only (never passed to the game).
VISIBLE_VAR = "BALLRAIN_VISIBLE"  # "1": show the game windows (debugging); unset / "0": hidden (default)
AUDIO_VAR = audio.AUDIO_VAR  # "BALLRAIN_AUDIO" = "1": keep game sound; unset / "0": muted (default)
FOCUS_VAR = "BALLRAIN_FOCUS"  # "1": let the game keep the foreground; unset / "0": BallrainIO hands it back (default)
LAUNCHER_VARS = (VISIBLE_VAR, AUDIO_VAR, FOCUS_VAR)

# TASSupport.cfg keys this module may edit, with their cfg type letter.
TAS_CFG_KEYS = {
    "AutoLoadTAS": "S",
    "AutoLoadLevel": "I",
    "LegacyMode": "B",
    "ExitOnFinish": "B",
    "SkipRenderUntil": "I",
}

LAUNCH_SCRIPT_NAME = "ballrain-py-launch.cmd"
SCRIPT_SUFFIXES = (".cmd", ".bat", ".ps1")


class InstanceError(RuntimeError):
    pass


def runtime_dir() -> Path:
    return Path(os.environ.get("BALLRAIN_RUNTIME") or (REPO_ROOT / "runtime"))


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _is_under(path, root) -> bool:
    p, r = _norm(path), _norm(root).rstrip("\\/")
    return p == r or p.startswith(r + os.sep)


@dataclass(frozen=True)
class GameCopy:
    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def bin_dir(self) -> Path:
        return self.root / "Bin"

    @property
    def exe(self) -> Path:
        return self.bin_dir / "Player.exe"

    @property
    def physics_dll(self) -> Path:
        return self.root / "BuildingBlocks" / "physics_RT.dll"

    @property
    def modloader_dir(self) -> Path:
        return self.root / "ModLoader"

    @property
    def mods_dir(self) -> Path:
        return self.modloader_dir / "Mods"

    @property
    def records_dir(self) -> Path:
        return self.modloader_dir / "TASRecords"

    @property
    def tas_cfg(self) -> Path:
        return self.modloader_dir / "Configs" / "TASSupport.cfg"

    @property
    def modloader_log(self) -> Path:
        return self.modloader_dir / "ModLoader.log"

    def has_tassupport(self) -> bool:
        return (self.mods_dir / "TASSupport.bmodp").exists()

    def has_ballrainio(self) -> bool:
        return any(self.mods_dir.glob("BallrainIO*"))


def get_copy(name: str, runtime=None) -> GameCopy:
    base = Path(runtime) if runtime is not None else runtime_dir()
    root = (base / name).resolve()
    if not _is_under(root, base.resolve()) or root == base.resolve():
        raise InstanceError(f"copy name {name!r} escapes the runtime directory")
    if not root.is_dir():
        raise InstanceError(f"game copy not found: {root}")
    return GameCopy(root)


# ---------------------------------------------------------------------------------------- checks

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_physics(copy: GameCopy, expected: str | None = None) -> str:
    expected = expected or PHYSICS_RT_SHA256
    if not copy.physics_dll.is_file():
        raise InstanceError(f"{copy.physics_dll} missing")
    got = sha256_file(copy.physics_dll)
    if got != expected:
        raise InstanceError(f"{copy.physics_dll}: sha256 {got} != expected {expected}")
    return got


_ABS_PATH_RE = re.compile(rb'(?<![A-Za-z])[A-Za-z]:[\\/][^"\r\n%<>|]*')


def find_foreign_script_paths(copy: GameCopy) -> list[tuple[Path, str]]:
    """Absolute paths in launch scripts (copy root and Bin) that point outside this copy."""
    found = []
    for d in (copy.root, copy.bin_dir):
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if not (f.is_file() and f.suffix.lower() in SCRIPT_SUFFIXES):
                continue
            for m in _ABS_PATH_RE.finditer(f.read_bytes()):
                p = m.group(0).decode("mbcs" if sys.platform == "win32" else "latin-1", "replace").strip()
                if not _is_under(p, copy.root):
                    found.append((f, p))
    return found


def verify_no_foreign_scripts(copy: GameCopy) -> None:
    bad = find_foreign_script_paths(copy)
    if bad:
        lines = "\n".join(f"  {f}: {p}" for f, p in bad)
        raise InstanceError(f"launch scripts in {copy.root} reference paths outside the copy:\n{lines}")


# ------------------------------------------------------------------------------------ cfg editing

def _cfg_value(key: str, value) -> bytes:
    kind = TAS_CFG_KEYS[key]
    if kind == "B":
        if value not in (0, 1, True, False):
            raise ValueError(f"{key}: boolean expected, got {value!r}")
        return b"1" if value else b"0"
    if kind == "I":
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{key}: integer expected, got {value!r}")
        return str(int(value)).encode()
    s = str(value)
    if any(c in s for c in "\r\n") or s != s.strip():
        raise ValueError(f"{key}: invalid string value {value!r}")
    return s.encode("mbcs" if sys.platform == "win32" else "utf-8")


def edit_cfg_bytes(data: bytes, values: dict) -> bytes:
    """Replace the values of existing 'T Key value' lines; tabs, type letters and EOLs are preserved."""
    for key, value in values.items():
        if key not in TAS_CFG_KEYS:
            raise KeyError(f"{key} is not an editable TASSupport.cfg key")
        pat = re.compile(rb"^([ \t]*)([A-Z]) (" + re.escape(key.encode()) + rb") [^\r\n]*?(\r?)$", re.M)
        matches = list(pat.finditer(data))
        if len(matches) != 1:
            raise InstanceError(f"TASSupport.cfg: expected exactly one '{key}' line, found {len(matches)}")
        m = matches[0]
        if m.group(2).decode() != TAS_CFG_KEYS[key]:
            raise InstanceError(f"TASSupport.cfg: {key} has type {m.group(2).decode()}, expected {TAS_CFG_KEYS[key]}")
        new = m.group(1) + m.group(2) + b" " + m.group(3) + b" " + _cfg_value(key, value) + m.group(4)
        data = data[: m.start()] + new + data[m.end():]
    return data


def read_cfg_values(data: bytes) -> dict:
    out = {}
    for key in TAS_CFG_KEYS:
        m = re.search(rb"^[ \t]*[A-Z] " + re.escape(key.encode()) + rb" ([^\r\n]*?)\r?$", data, re.M)
        out[key] = m.group(1).decode("latin-1") if m else None
    return out


def set_tas_config(copy: GameCopy, **values) -> dict:
    data = copy.tas_cfg.read_bytes()
    new = edit_cfg_bytes(data, values)
    if new != data:
        copy.tas_cfg.write_bytes(new)
    return read_cfg_values(new)


# ------------------------------------------------------------------------------ process plumbing

PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102

if sys.platform == "win32":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.K32EnumProcesses.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]


def _image_path(handle) -> str | None:
    buf = ctypes.create_unicode_buffer(32768)
    size = wintypes.DWORD(len(buf))
    if not _k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
        return None
    return buf.value


def list_pids() -> list[int]:
    n = 4096
    while True:
        arr = (wintypes.DWORD * n)()
        got = wintypes.DWORD()
        if not _k32.K32EnumProcesses(arr, ctypes.sizeof(arr), ctypes.byref(got)):
            raise OSError(ctypes.get_last_error(), "EnumProcesses failed")
        count = got.value // ctypes.sizeof(wintypes.DWORD)
        if count < n:
            return [int(p) for p in arr[:count] if p]
        n *= 2


def process_path(pid: int) -> str | None:
    h = _k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        return _image_path(h)
    finally:
        _k32.CloseHandle(h)


def pids_with_image(exe) -> list[int]:
    want = _norm(exe)
    return [pid for pid in list_pids() if (p := process_path(pid)) and _norm(p) == want]


def running_in_copy(copy: GameCopy) -> list[int]:
    return pids_with_image(copy.exe)


class ProcessHandle:
    """Handle to one process, bound to an expected image path; terminate() re-verifies it."""

    def __init__(self, pid: int, expected_exe):
        self.pid = pid
        self.expected_exe = Path(expected_exe)
        self._h = _k32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, False, pid)
        if not self._h:
            raise InstanceError(f"cannot open pid {pid} (error {ctypes.get_last_error()})")
        if not (self.image() is None and self.wait(0)):  # an already-exited process may report no image
            self.verify()

    def image(self) -> str | None:
        return _image_path(self._h)

    def verify(self) -> None:
        img = self.image()
        if img is None or _norm(img) != _norm(self.expected_exe):
            raise InstanceError(f"pid {self.pid} is {img!r}, not {self.expected_exe}")

    def wait(self, timeout_s: float) -> bool:
        """True if the process exited within the timeout."""
        ms = max(0, int(timeout_s * 1000))
        r = _k32.WaitForSingleObject(self._h, ms)
        if r == WAIT_OBJECT_0:
            return True
        if r == WAIT_TIMEOUT:
            return False
        raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")

    def exit_code(self) -> int | None:
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(self._h, ctypes.byref(code)):
            return None
        return None if code.value == 259 else int(code.value)  # 259 = STILL_ACTIVE

    def terminate(self, code: int = 1) -> None:
        self.verify()  # never stop anything that is not the expected executable
        if not _k32.TerminateProcess(self._h, code):
            raise OSError(ctypes.get_last_error(), f"TerminateProcess({self.pid}) failed")

    def close(self) -> None:
        if self._h:
            _k32.CloseHandle(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------------- launch recipes

def full_env(env: dict) -> dict:
    """Complete environment for direct process creation: parent env minus managed / launcher vars, plus env."""
    out = {k: v for k, v in os.environ.items() if k.upper() not in MANAGED_VARS + LAUNCHER_VARS}
    for k, v in env.items():
        if v is not None:
            out[k] = str(v)
    return out


_CMD_UNSAFE = set('"%^&|<>!\r\n')


def launch_script_text(copy: GameCopy, env: dict) -> str:
    """The .cmd used by the explorer recipe. All managed vars are set or cleared explicitly."""
    names = list(MANAGED_VARS) + [k for k in env if k not in MANAGED_VARS]
    lines = ["@echo off"]
    for k in names:
        v = env.get(k)
        s = "" if v is None else str(v)
        if _CMD_UNSAFE & set(k + s):
            raise InstanceError(f"value for {k} contains characters unsafe in a .cmd: {s!r}")
        lines.append(f'set "{k}={s}"')
    lines.append(f'cd /d "{copy.bin_dir}"')
    lines.append(f'start "" {copy.exe.name}')
    return "\r\n".join(lines) + "\r\n"


def _wait_new_pid(copy: GameCopy, before: set, timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        new = [p for p in running_in_copy(copy) if p not in before]
        if len(new) == 1:
            return new[0]
        if len(new) > 1:
            raise InstanceError(f"several new Player.exe in {copy.root}: {new}")
        time.sleep(0.1)
    raise InstanceError(f"no Player.exe from {copy.root} appeared within {timeout_s} s")


def launch_explorer_cmd(copy: GameCopy, env: dict, start_timeout: float = 30.0) -> int:
    """Known-good recipe: write a .cmd into the copy and open it through explorer.exe.

    explorer hands the script to the shell, so the game inherits the shell's environment, not
    ours: every variable is therefore written into the script. The script is removed once the
    new Player.exe is found (or the start times out). The game window is always visible with
    this recipe (BALLRAIN_VISIBLE applies to the direct recipe only).
    """
    script = copy.root / LAUNCH_SCRIPT_NAME
    env = {**env, "BALLRAIN_KEEP_FOCUS": "1"}  # visible window: the user may want to click into it
    script.write_bytes(launch_script_text(copy, env).encode("mbcs"))
    before = set(running_in_copy(copy))
    try:
        subprocess.run(["explorer.exe", str(script)], check=False)  # explorer's exit code is meaningless
        return _wait_new_pid(copy, before, start_timeout)
    finally:
        script.unlink(missing_ok=True)


_LAUNCHED: dict = {}


SW_HIDE = 0


def windows_visible(visible: bool | None = None) -> bool:
    """Whether launched game windows are shown: `visible` if given, else BALLRAIN_VISIBLE (unset / "0" = hidden)."""
    if visible is not None:
        return bool(visible)
    return os.environ.get(VISIBLE_VAR, "").strip() not in ("", "0")


def game_keeps_focus(keep_focus: bool | None = None, visible: bool = False) -> bool:
    """Whether the launch tells BallrainIO to leave the foreground alone (BALLRAIN_KEEP_FOCUS=1): `keep_focus`
    if given, else BALLRAIN_FOCUS=1, else only for a visible window (one the user may click into)."""
    if keep_focus is not None:
        return bool(keep_focus)
    if os.environ.get(FOCUS_VAR, "").strip() not in ("", "0"):
        return True
    return bool(visible)


def hidden_startupinfo():
    """STARTUPINFO that makes the game's first ShowWindow of its main window SW_HIDE.

    Player.exe shows its main window with ShowWindow(SW_SHOW) (Windows substitutes wShowWindow for the first
    such call of an overlapped window) and never shows it again in windowed mode; the main loop does not depend
    on visibility or focus. Not covered: the start-up splash (a popup shown before the main window, from
    Bin/splash.bmp; see set_splash), and the SetFocus right after ShowWindow can still make the hidden window
    the foreground window when Windows lets a new process take the foreground. A separate desktop
    (CreateDesktop + lpDesktop) would hide everything but is not usable: off the input desktop D3D reports the
    HAL device as not available (D3DERR_NOTAVAILABLE) and Player.exe exits with "No render driver found"."""
    return subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=SW_HIDE)


NOFOCUS_DIR = Path(__file__).resolve().parents[1] / "runtime" / "nofocus"  # built by tools/nofocus/build.bat
NOFOCUS_VAR = "BALLRAIN_NOFOCUS"  # "0" = hidden launches go straight to Player.exe (no activation hook)
_nofocus_warned = False


def nofocus_available() -> bool:
    return sys.platform == "win32" and (NOFOCUS_DIR / "nofocus.dll").is_file() and (NOFOCUS_DIR / "nofocus_launch.exe").is_file()


class _LaunchedProc:
    """Popen-like stand-in for a process started by nofocus_launch.exe (holds a handle so the PID is not reused)."""

    def __init__(self, pid: int):
        self.pid = pid
        self._h = _k32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)

    def poll(self):
        code = wintypes.DWORD()
        if not self._h or not _k32.GetExitCodeProcess(self._h, ctypes.byref(code)):
            return -1
        return None if code.value == 259 else code.value  # 259 = STILL_ACTIVE

    def __del__(self):
        if getattr(self, "_h", None):
            _k32.CloseHandle(self._h)
            self._h = None


def launch_nofocus(copy: GameCopy, env: dict, visible: bool | None = None, audio_on: bool | None = None,
                   keep_focus: bool | None = None) -> int:
    """launch_direct, but Player.exe is created suspended by runtime/nofocus/nofocus_launch.exe, which loads
    nofocus.dll on its main thread before the game's own code runs. The DLL installs a thread WH_CBT hook that
    refuses every window activation, so no game window (splash, main window) ever becomes the active or
    foreground window. Same working directory (<copy>/Bin), environment and show flags as launch_direct;
    traces were checked byte-identical to direct launches (BallrainIO and TASSupport paths)."""
    if not nofocus_available():
        raise InstanceError(f"no-focus launcher not built ({NOFOCUS_DIR}); run tools/nofocus/build.bat")
    show = windows_visible(visible)
    if game_keeps_focus(keep_focus, show):
        env = {**env, "BALLRAIN_KEEP_FOCUS": "1"}
    full = full_env(env)
    if os.environ.get("NOFOCUS_LOG"):
        full["NOFOCUS_LOG"] = os.environ["NOFOCUS_LOG"]
    out = subprocess.run([str(NOFOCUS_DIR / "nofocus_launch.exe"), str(NOFOCUS_DIR / "nofocus.dll"), str(copy.exe)],
                         cwd=str(copy.bin_dir), env=full, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                         timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                         startupinfo=None if show else hidden_startupinfo())
    line = out.stdout.strip()
    if out.returncode != 0 or not line.startswith("pid "):
        raise InstanceError(f"nofocus_launch failed (exit {out.returncode}): {line!r}")
    p = _LaunchedProc(int(line.split()[1]))
    _LAUNCHED[p.pid] = p
    audio.watch(p, copy.exe, muted=not audio.audio_enabled(audio_on))
    return p.pid


def launch_direct(copy: GameCopy, env: dict, visible: bool | None = None, audio_on: bool | None = None,
                  keep_focus: bool | None = None) -> int:
    """Direct process creation (like Start-Process -WorkingDirectory <copy>/Bin [-WindowStyle Hidden]).
    Windows are hidden unless windows_visible(visible); the game's audio sessions are kept muted unless
    audio.audio_enabled(audio_on) (BALLRAIN_AUDIO=1), in which case they are kept unmuted (Windows remembers
    the mute per executable path); BALLRAIN_KEEP_FOCUS=1 is passed when game_keeps_focus(keep_focus, visible)."""
    show = windows_visible(visible)
    if not show and os.environ.get(NOFOCUS_VAR, "").strip() != "0":
        if nofocus_available():  # hidden game: start it through the no-focus launcher
            return launch_nofocus(copy, env, visible=False, audio_on=audio_on, keep_focus=keep_focus)
        global _nofocus_warned
        if not _nofocus_warned:
            print(f"warning: {NOFOCUS_DIR} not built (tools/nofocus/build.bat); hidden games may take the foreground",
                  file=sys.stderr, flush=True)
            _nofocus_warned = True
    if game_keeps_focus(keep_focus, show):
        env = {**env, "BALLRAIN_KEEP_FOCUS": "1"}
    p = subprocess.Popen(
        [str(copy.exe)],
        cwd=str(copy.bin_dir),
        env=full_env(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        close_fds=True,
        startupinfo=None if show or sys.platform != "win32" else hidden_startupinfo(),
    )
    _LAUNCHED[p.pid] = p  # keep the process handle open so the PID cannot be reused while we track it
    audio.watch(p, copy.exe, muted=not audio.audio_enabled(audio_on))
    return p.pid


LAUNCH_RECIPES = {"explorer_cmd": launch_explorer_cmd, "direct": launch_direct, "nofocus": launch_nofocus}
LAUNCH_RECIPE = launch_explorer_cmd


# -------------------------------------------------------------------------------------- running

@dataclass
class RunResult:
    copy: str
    pid: int | None
    exit_code: int | None
    timed_out: bool
    killed: bool
    wall_s: float
    outputs: dict = field(default_factory=dict)
    missing_outputs: list = field(default_factory=list)  # expected outputs not (freshly) written
    started_at: float = 0.0  # time.time() just before launch

    @property
    def ok(self) -> bool:
        return not self.timed_out and not self.missing_outputs


def _rotate(path: Path) -> None:
    """Move a previous output out of the way (to <name>.prev) so a failed run cannot reuse it."""
    if path.exists():
        prev = path.with_name(path.name + ".prev")
        os.replace(path, prev)


def preflight(copy: GameCopy) -> None:
    verify_physics(copy)
    verify_no_foreign_scripts(copy)
    if not copy.exe.is_file():
        raise InstanceError(f"{copy.exe} missing")
    running = running_in_copy(copy)
    if running:
        raise InstanceError(f"Player.exe from {copy.root} is already running (pid {running}); refusing to launch")


def run_copy(copy: GameCopy, env: dict, outputs: dict, timeout: float, recipe=None) -> RunResult:
    """Launch, wait up to `timeout` seconds, stop our own process on timeout, check outputs."""
    recipe = recipe or LAUNCH_RECIPE
    preflight(copy)
    for p in outputs.values():
        _rotate(Path(p))
    t_wall = time.time()
    t0 = time.monotonic()
    pid = recipe(copy, env)
    timed_out = killed = False
    with ProcessHandle(pid, copy.exe) as h:
        remaining = timeout - (time.monotonic() - t0)
        if not h.wait(remaining):
            timed_out = True
            h.terminate()
            killed = True
            h.wait(10)
        wall = time.monotonic() - t0
        code = h.exit_code()
    _LAUNCHED.pop(pid, None)
    missing = []
    for k, p in outputs.items():
        p = Path(p)
        if not p.exists() or p.stat().st_mtime < t_wall - 2:
            missing.append(k)
    return RunResult(copy.name, pid, code, timed_out, killed, wall, dict(outputs), missing, t_wall)


def start_copy(copy: GameCopy, env: dict, recipe=None) -> "ProcessHandle":
    """Launch without waiting (agent mode): preflight, start, return a verified handle to our own PID."""
    recipe = recipe or LAUNCH_RECIPE
    preflight(copy)
    pid = recipe(copy, env)
    return ProcessHandle(pid, copy.exe)


def stop_handle(h: "ProcessHandle", wait_s: float = 10.0) -> int | None:
    """Wait for our own process to exit; terminate it (after re-verifying its image) if it does not."""
    try:
        if not h.wait(wait_s):
            h.terminate()
            h.wait(10)
        return h.exit_code()
    finally:
        _LAUNCHED.pop(h.pid, None)
        h.close()


# ------------------------------------------------------------------------------- training copies

TRAINING_COPY_FMT = "train-{:02d}"
COPY_SKIP_NAMES = {"ModLoader.log", "CommandBar.history", "ImGui.ini", LAUNCH_SCRIPT_NAME}
COPY_SKIP_SUFFIXES = (".prev", ".cmd", ".bat", ".ps1", ".log")
_HASH_SUFFIXES = (".dll", ".exe", ".bmodp", ".nmo", ".cmo", ".tdb")  # .tdb: same-size option edits


def training_copy_names(n: int) -> list[str]:
    return [TRAINING_COPY_FMT.format(i) for i in range(n)]


def _copy_ignore(_dir, names):
    return [n for n in names if n in COPY_SKIP_NAMES or n.lower().endswith(COPY_SKIP_SUFFIXES)]


MOD_REL = "ModLoader/Mods/BallrainIO.bmodp"  # rebuilt often: synced by sync_mod, not compared


def _tree_files(root: Path) -> dict:
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root)
            if any(part in COPY_SKIP_NAMES for part in rel.parts) or p.name.lower().endswith(COPY_SKIP_SUFFIXES):
                continue
            if rel.as_posix() == MOD_REL:
                continue
            out[rel.as_posix()] = p
    return out


def mod_hashes(names, runtime=None) -> dict:
    """sha256 of the installed BallrainIO.bmodp per copy (None when missing)."""
    out = {}
    for n in names:
        p = get_copy(n, runtime).root / MOD_REL
        out[n] = sha256_file(p) if p.is_file() else None
    return out


def verify_copy_matches(src: GameCopy, dst: GameCopy, hash_all: bool = False) -> list[str]:
    """Differences between two copies: missing/extra files, sizes, sha256 of binaries (all with hash_all).
    The BallrainIO mod itself is not compared (see sync_mod / mod_hashes)."""
    a, b = _tree_files(src.root), _tree_files(dst.root)
    probs = [f"missing {k}" for k in sorted(a.keys() - b.keys())]
    probs += [f"extra {k}" for k in sorted(b.keys() - a.keys())]
    for k in sorted(a.keys() & b.keys()):
        if a[k].stat().st_size != b[k].stat().st_size:
            probs.append(f"size {k}")
        elif (hash_all or k.lower().endswith(_HASH_SUFFIXES)) and sha256_file(a[k]) != sha256_file(b[k]):
            probs.append(f"sha256 {k}")
    return probs


def create_training_copies(n: int = 8, source: str = TRAINING_COPY, runtime=None, hash_all: bool = False) -> dict:
    """Create train-01..train-(n-1) as copies of `source` (logs and launch scripts excluded), then
    verify physics_RT.dll against the 2002 hash and every binary against the source. Existing copies
    are only verified, never overwritten. Returns {name: "created" | "exists"} or raises."""
    base = Path(runtime) if runtime is not None else runtime_dir()
    src = get_copy(source, base)
    verify_physics(src)
    out = {}
    for name in training_copy_names(n):
        if name == source:
            continue
        root = base / name
        if not root.exists():
            shutil.copytree(src.root, root, ignore=_copy_ignore)
            out[name] = "created"
        else:
            out[name] = "exists"
        dst = get_copy(name, base)
        verify_physics(dst)
        verify_no_foreign_scripts(dst)
        if dst.has_tassupport():
            raise InstanceError(f"{dst.root} contains TASSupport.bmodp; training copies must not")
        probs = verify_copy_matches(src, dst, hash_all)
        if probs:
            raise InstanceError(f"{dst.root} differs from {src.root}: {probs[:10]}")
    return out


def sync_mod(build_path, names, runtime=None) -> dict:
    """Install a built BallrainIO.bmodp into each named copy; returns {name: sha256}. Copies that run
    a game process are refused (the mod file is locked and the change would not take effect)."""
    build_path = Path(build_path)
    want = sha256_file(build_path)
    out = {}
    for name in names:
        gc = get_copy(name, runtime)
        if running_in_copy(gc):
            raise InstanceError(f"{gc.root} has a running Player.exe; not replacing its mod")
        dst = gc.mods_dir / "BallrainIO.bmodp"
        if not dst.exists() or sha256_file(dst) != want:
            shutil.copyfile(build_path, dst)
        got = sha256_file(dst)
        if got != want:
            raise InstanceError(f"{dst}: sha256 {got} != {want}")
        out[name] = got
    return out


SPLASH_REL = "Bin/splash.bmp"
SPLASH_OFF_SUFFIX = ".off"


def splash_state(copy: GameCopy) -> str:
    """'on' (Bin/splash.bmp present), 'off' (renamed to splash.bmp.off), 'missing' or 'both' (inconsistent)."""
    on = (copy.root / SPLASH_REL).is_file()
    off = (copy.root / (SPLASH_REL + SPLASH_OFF_SUFFIX)).is_file()
    return {(True, False): "on", (False, True): "off", (False, False): "missing", (True, True): "both"}[(on, off)]


def set_splash(copy: GameCopy, enabled: bool) -> str:
    """Enable / disable Player.exe's start-up splash by renaming Bin/splash.bmp <-> Bin/splash.bmp.off.

    Player.exe loads <exe dir>/splash.bmp before anything else; when that fails it skips the splash
    window and carries on (the return value is ignored), so the only effect is no ~0.4 s popup (which
    takes the foreground) and a launch about 0.8 s shorter. Traces are unchanged (README, "游戏窗口").
    Refuses a copy with a running Player.exe. Returns the new state."""
    if running_in_copy(copy):
        raise InstanceError(f"{copy.root} has a running Player.exe; not changing its splash")
    on, off = copy.root / SPLASH_REL, copy.root / (SPLASH_REL + SPLASH_OFF_SUFFIX)
    state = splash_state(copy)
    if state in ("missing", "both"):
        raise InstanceError(f"{copy.root}: splash is {state} ({on}, {off})")
    if enabled and state == "off":
        os.replace(off, on)
    elif not enabled and state == "on":
        os.replace(on, off)
    return splash_state(copy)


# ------------------------------------------------------------------------------- Database.tdb volume

TDB_REL = "Database.tdb"
TDB_TYPES = {1: "int", 2: "float", 3: "string"}


def tdb_decode(data: bytes) -> bytes:
    """Database.tdb byte coding (Ballance): plain = -(rol3(b) ^ 0xAF) mod 256."""
    return bytes((-((((b << 3) | (b >> 5)) & 0xFF) ^ 0xAF)) & 0xFF for b in data)


def tdb_encode(plain: bytes) -> bytes:
    out = bytearray()
    for c in plain:
        x = ((-c) & 0xFF) ^ 0xAF
        out.append(((x >> 3) | (x << 5)) & 0xFF)
    return bytes(out)


def tdb_arrays(plain: bytes) -> dict:
    """Parse decoded Database.tdb: {array name: {"cols": [(name, type)], "rows": n, "cells": {(col, row): (offset, size)}}}.

    Layout per array: name NUL, u32 byte count of the rest, u32 columns, u32 rows, i32 key column, per column
    (name NUL, u32 type: 1 int, 2 float, 3 string), then the cells column by column (4 bytes, or a NUL-terminated
    string). Raises on anything that does not parse exactly to the end of the file."""
    import struct
    arrays, pos = {}, 0
    while pos < len(plain):
        end_name = plain.index(b"\0", pos)
        name = plain[pos:end_name].decode("latin-1")
        size, ncols, nrows, _key = struct.unpack_from("<IIIi", plain, end_name + 1)
        stop = end_name + 5 + size
        q = end_name + 17
        cols = []
        for _ in range(ncols):
            e = plain.index(b"\0", q)
            (t,) = struct.unpack_from("<I", plain, e + 1)
            if t not in TDB_TYPES:
                raise InstanceError(f"Database.tdb {name}: unknown column type {t}")
            cols.append((plain[q:e].decode("latin-1"), TDB_TYPES[t]))
            q = e + 5
        cells = {}
        for c, (_, t) in enumerate(cols):
            for r in range(nrows):
                n = plain.index(b"\0", q) + 1 - q if t == "string" else 4
                cells[(c, r)] = (q, n)
                q += n
        if q != stop or stop > len(plain) or name in arrays:
            raise InstanceError(f"Database.tdb {name}: parse ends at {q}, expected {stop}")
        arrays[name] = {"cols": cols, "rows": nrows, "cells": cells}
        pos = stop
    return arrays


def _tdb_volume_cell(plain: bytes) -> tuple[int, int]:
    opts = tdb_arrays(plain).get("DB_Options")
    if opts is None or opts["rows"] != 1:
        raise InstanceError("Database.tdb: DB_Options with one row expected")
    names = [n for n, _ in opts["cols"]]
    if names.count("Volume") != 1 or opts["cols"][names.index("Volume")][1] != "float":
        raise InstanceError("Database.tdb: one float column DB_Options.Volume expected")
    return opts["cells"][(names.index("Volume"), 0)]


def tdb_volume(copy: GameCopy) -> float:
    """The game's own volume option (DB_Options.Volume, 0..1; 1.0 in the stock file)."""
    import struct
    plain = tdb_decode((copy.root / TDB_REL).read_bytes())
    off, _ = _tdb_volume_cell(plain)
    return struct.unpack_from("<f", plain, off)[0]


def tdb_with_volume(data: bytes, volume: float) -> bytes:
    """Encoded Database.tdb `data` with only the 4 bytes of DB_Options.Volume replaced."""
    import struct
    if not 0.0 <= float(volume) <= 1.0:
        raise ValueError(f"volume must be in [0, 1], got {volume!r}")
    plain = tdb_decode(data)
    if tdb_encode(plain) != data:
        raise InstanceError("Database.tdb does not round-trip through decode / encode")
    off, n = _tdb_volume_cell(plain)
    new = tdb_encode(plain[:off] + struct.pack("<f", float(volume)) + plain[off + n:])
    if len(new) != len(data) or sum(a != b for a, b in zip(new, data)) > 4:
        raise InstanceError("Database.tdb volume edit touched more than the Volume cell")
    return new


def set_tdb_volume(copy: GameCopy, volume: float) -> float:
    """Set the game's own volume option in <copy>/Database.tdb (only those 4 bytes change). Refuses a copy
    with a running Player.exe. Returns the new value read back."""
    if running_in_copy(copy):
        raise InstanceError(f"{copy.root} has a running Player.exe; not changing its Database.tdb")
    path = copy.root / TDB_REL
    data = path.read_bytes()
    new = tdb_with_volume(data, volume)
    if new != data:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(new)
        os.replace(tmp, path)
    return tdb_volume(copy)


def install_record(copy: GameCopy, tas_path) -> str:
    """Make the record available as ModLoader/TASRecords/<stem>.tas; returns the AutoLoadTAS stem."""
    src = Path(tas_path).resolve()
    tas.load(src)  # strict legacy validation
    dst = copy.records_dir / src.name
    if src.suffix.lower() != ".tas":
        raise InstanceError(f"{src} is not a .tas file")
    if _norm(src) != _norm(dst):
        if dst.exists():
            if dst.read_bytes() != src.read_bytes():
                raise InstanceError(f"{dst} exists with different content; rename {src.name}")
        else:
            copy.records_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    return src.stem


def _check_level(level) -> int:
    if isinstance(level, bool) or int(level) != level or not 1 <= int(level) <= 13:
        raise ValueError(f"level must be 1..13, got {level!r}")
    return int(level)


def run_reference(tas_path, level, trace_path, timeout: float = 120, *, copy: str = REFERENCE_COPY,
                  skip_render_until: int = 999999, runtime=None, recipe=None) -> dict:
    """Replay a legacy .tas in the reference copy with the real TASSupport and record BMMO_TAS_TRACE."""
    gc = get_copy(copy, runtime)
    if not gc.has_tassupport():
        raise InstanceError(f"{gc.root} has no TASSupport.bmodp")
    level = _check_level(level)
    trace_path = Path(trace_path).resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    preflight(gc)  # before touching cfg / records
    stem = install_record(gc, tas_path)
    cfg = set_tas_config(gc, AutoLoadTAS=stem, AutoLoadLevel=level, LegacyMode=1, ExitOnFinish=1,
                         SkipRenderUntil=skip_render_until)
    env = {k: None for k in MANAGED_VARS}
    env["BMMO_TAS_TRACE"] = str(trace_path)
    res = run_copy(gc, env, {"trace": trace_path}, timeout, recipe)
    log_copy = trace_path.with_name(trace_path.name + ".modloader.log")
    _rotate(log_copy)
    fresh_log = gc.modloader_log.exists() and gc.modloader_log.stat().st_mtime >= res.started_at - 2
    if fresh_log:  # never pass off a previous run's log as this run's
        shutil.copyfile(gc.modloader_log, log_copy)
    return {
        "trace": trace_path,
        "modloader_log": log_copy if fresh_log else None,
        "wall_s": res.wall_s,
        "result": res,
        "cfg": cfg,
    }


def ballrain_output_paths(out_dir, loads: int = 1) -> list[dict]:
    """Output paths per load: load 1 uses the base names, load k > 1 appends '.k'."""
    out_dir = Path(out_dir).resolve()
    base = {"trace": out_dir / "trace.txt", "framelog": out_dir / "framelog.bin", "events": out_dir / "events.txt"}
    res = []
    for k in range(1, loads + 1):
        res.append({n: (p if k == 1 else p.with_name(f"{p.name}.{k}")) for n, p in base.items()})
    return res


def ballrain_env(frames_path, level, out_dir, loads: int = 1, skip_render: bool = True, exit_game: bool = True) -> dict:
    paths = ballrain_output_paths(out_dir, 1)[0]
    env = {k: None for k in MANAGED_VARS}
    env.update({
        "BALLRAIN_MODE": "replay",
        "BALLRAIN_FRAMES": str(Path(frames_path).resolve()),
        "BALLRAIN_LEVEL": str(_check_level(level)),
        "BALLRAIN_TRACE": str(paths["trace"]),
        "BALLRAIN_FRAMELOG": str(paths["framelog"]),
        "BALLRAIN_EVENTS": str(paths["events"]),
        "BALLRAIN_SKIP_RENDER": "1" if skip_render else None,
        "BALLRAIN_EXIT": "1" if exit_game else None,
        "BALLRAIN_LOADS": str(int(loads)),
    })
    return env


def run_ballrain(frames_or_tas_path, level, out_dir, loads: int = 1, skip_render: bool = True,
                 timeout: float = 120, *, copy: str = TRAINING_COPY, runtime=None, recipe=None) -> dict:
    """Replay raw frames (or a legacy .tas, decoded here) through BallrainIO in a training copy."""
    if int(loads) < 1:
        raise ValueError("loads must be >= 1")
    gc = get_copy(copy, runtime)
    if gc.has_tassupport():
        raise InstanceError(f"{gc.root} contains TASSupport.bmodp; BallrainIO copies must not")
    if not gc.has_ballrainio():
        raise InstanceError(f"{gc.mods_dir} has no BallrainIO mod")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = tas.load_any(frames_or_tas_path)
    frames_path = out_dir / "frames.bin"
    if _norm(frames_path) != _norm(frames_or_tas_path):
        tas.save_raw(frames_path, frames)
    env = ballrain_env(frames_path, level, out_dir, loads, skip_render, True)
    per_load = ballrain_output_paths(out_dir, loads)
    outputs = {f"{n}.{k}": p for k, d in enumerate(per_load, 1) for n, p in d.items()}
    res = run_copy(gc, env, outputs, timeout, recipe)
    return {"frames": frames_path, "loads": per_load, "wall_s": res.wall_s, "result": res}

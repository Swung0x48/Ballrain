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

from . import tas

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
)
BMMO_VARS = ("BMMO_TAS_TRACE", "BMMO_TAS_OFFSETS", "BMMO_TAS_CORE", "BMMO_TAS_PC")
MANAGED_VARS = BALLRAIN_VARS + BMMO_VARS

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
    """Complete environment for direct process creation: parent env minus managed vars, plus env."""
    out = {k: v for k, v in os.environ.items() if k.upper() not in MANAGED_VARS}
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
    new Player.exe is found (or the start times out).
    """
    script = copy.root / LAUNCH_SCRIPT_NAME
    script.write_bytes(launch_script_text(copy, env).encode("mbcs"))
    before = set(running_in_copy(copy))
    try:
        subprocess.run(["explorer.exe", str(script)], check=False)  # explorer's exit code is meaningless
        return _wait_new_pid(copy, before, start_timeout)
    finally:
        script.unlink(missing_ok=True)


_LAUNCHED: dict = {}


def launch_direct(copy: GameCopy, env: dict) -> int:
    """Direct process creation (like Start-Process -WorkingDirectory <copy>/Bin)."""
    p = subprocess.Popen(
        [str(copy.exe)],
        cwd=str(copy.bin_dir),
        env=full_env(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        close_fds=True,
    )
    _LAUNCHED[p.pid] = p  # keep the process handle open so the PID cannot be reused while we track it
    return p.pid


LAUNCH_RECIPES = {"explorer_cmd": launch_explorer_cmd, "direct": launch_direct}
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
_HASH_SUFFIXES = (".dll", ".exe", ".bmodp", ".nmo", ".cmo")


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

"""The player's side of the game: editions (which headless engine), the Ballance folder check with clear Chinese
messages, and the hello fields (docs/pool-prototype.md 2.2, section 9).

Editions (docs/volunteer-compute.md 11.3):
- "windows": the 2002 headless host (Windows x86 build of SimTool with the phys2002 loader); it loads the player's
  own BuildingBlocks/physics_RT.dll, which must be the 2002 file (sha256 e72e4afc...). Engine kind "2002".
- "all": the x87-soft headless (source physics compiled with soft x87 maths): Windows build bin-x86soft, macOS /
  Linux builds from headless/ (build_mac.sh; a Linux build script is future work). Engine kind "x87soft".

Only what can be judged here is judged here: a missing file or folder, and the physics DLL's well-known sha256. Whether
the other files are the original ones is the gateway's call (hello -> game_files_mismatch, shown verbatim).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import protocol as P
from .task import file_sha256, hash_game_files

EDITIONS = {
    "windows": {"engine": "2002", "label": "Windows 版（2002 原版物理）"},
    "all": {"engine": "x87soft", "label": "全平台版（x87-soft 物理）"},
}
ENGINE_NAMES = ("BallanceMMOSimTool.exe", "BallanceMMOSimTool")


def platform_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def default_edition() -> str:
    return "windows" if platform_name() == "windows" else "all"


def edition_engine(edition: str) -> str:
    if edition not in EDITIONS:
        raise ValueError(f"未知的版本 {edition!r}（windows 或 all）")
    if edition == "windows" and platform_name() != "windows":
        raise ValueError("Windows 版只能在 Windows 上运行，请选择全平台版")
    return EDITIONS[edition]["engine"]


def bundled_engine(edition: str, bundle_dir=None) -> Path | None:
    """The engine shipped next to the controller: <bundle>/engine/<edition>/<platform>/BallanceMMOSimTool[.exe].
    None when the bundle has none (a source checkout: give the path explicitly)."""
    base = Path(bundle_dir) if bundle_dir else Path(__file__).resolve().parents[2] / "engine"
    d = base / edition / platform_name()
    for n in ENGINE_NAMES:
        if (d / n).is_file():
            return d / n
    return None


@dataclass
class Problem:
    name: str  # the GAME_FILE_LIST entry, or "engine"
    kind: str  # "missing" | "empty" | "wrong" | "engine"
    message: str  # Chinese, for the player


def check_game_dir(root, edition: str) -> list[Problem]:
    """Problems of the player's Ballance folder for an edition (empty list = fine here; the gateway checks the rest)."""
    out: list[Problem] = []
    if not root:
        return [Problem("", "missing", "还没有选择 Ballance 安装目录。")]
    r = Path(root)
    if not r.is_dir():
        return [Problem("", "missing", f"找不到 Ballance 安装目录：{r}")]
    engine = edition_engine(edition)
    names = P.GAME_FILE_LIST_2002 if engine == "2002" else P.GAME_FILE_LIST
    for name in names:
        p = r / name.rstrip("/")
        if name.endswith("/"):
            if not p.is_dir():
                out.append(Problem(name, "missing", f"缺少文件夹：{name.rstrip('/')}（应在 Ballance 安装目录下）"))
            elif not any(f.is_file() for f in p.rglob("*")):
                out.append(Problem(name, "empty", f"文件夹 {name.rstrip('/')} 是空的，游戏文件不完整。"))
            continue
        if not p.is_file():
            hint = "Windows 版需要 2002 年原版的物理 DLL；没有的话可以改用全平台版。" \
                if name == "BuildingBlocks/physics_RT.dll" else "游戏文件不完整，请检查安装目录。"
            out.append(Problem(name, "missing", f"缺少文件：{name}。{hint}"))
            continue
        if name == "BuildingBlocks/physics_RT.dll":
            sha = file_sha256(p)
            if sha != P.PHYSICS_RT_2002_SHA256:
                out.append(Problem(name, "wrong",
                                   f"{name} 不是 2002 年原版（sha256 {sha[:12]}…，应为 "
                                   f"{P.PHYSICS_RT_2002_SHA256[:12]}…）。Windows 版只能加载原版物理 DLL，"
                                   "请换回原版文件，或者改用全平台版。"))
    return out


def check_engine(binary, edition: str) -> list[Problem]:
    if not binary:
        return [Problem("engine", "engine", "没有找到无头引擎，请在设置里指定引擎文件（BallanceMMOSimTool）。")]
    b = Path(binary)
    if not b.is_file():
        return [Problem("engine", "engine", f"找不到无头引擎文件：{b}")]
    if platform_name() != "windows" and not os.access(b, os.X_OK):
        return [Problem("engine", "engine", f"无头引擎文件没有执行权限：{b}")]
    return []


def engine_info(binary, edition: str, version: str | None = None) -> dict:
    """hello.engine: kind, the engine file's sha256 and a version string (the bundle's ENGINE_VERSION file next to the
    binary, else dev-<sha prefix>)."""
    b = Path(binary)
    sha = file_sha256(b)
    if version is None:
        vf = b.parent / "ENGINE_VERSION"
        version = vf.read_text(encoding="utf-8").strip() if vf.is_file() else f"dev-{sha[:12]}"
    if not P.VERSION_RE.fullmatch(version):
        version = f"dev-{sha[:12]}"
    return {"kind": edition_engine(edition), "sha256": sha, "version": version}


def game_hashes(root, edition: str) -> dict:
    return hash_game_files(root, edition_engine(edition))

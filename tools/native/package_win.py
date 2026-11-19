"""The Windows beta zip of the volunteer client (docs/native-client.md section 8).

    python tools/native/package_win.py --engine <bin-x86soft BallanceMMOSimTool.exe> --engine-version <string>
        --license BallanceMMO=<its LICENSE file> [--license NAME=FILE ...]
        [--gateway https://...] [--update-url https://.../release.json] [--pubkey-file release-signing.pub]
        [--sign-key release-signing.key --download-url https://.../BallrainCoordinator-<v>-windows-x64.zip]
        [--out dist/native]

Builds a release configuration (native/build_win.cmd with the gateway / update URL / public key as CMake cache
variables: they are build inputs, never committed) and assembles

    BallrainCoordinator-<version>-windows-x64/
        BallrainCoordinator.exe               the TUI client (static, no runtime needed)
        ballrain_coordinator.exe              the same controller without a UI (login / run / diag; servers, SSH)
        engine/x87soft/windows/BallanceMMOSimTool.exe, msvcp140.dll, vcruntime140.dll (x86, app-local), ENGINE_VERSION
        使用说明.txt                          the tester quick start (docs/native-client-testers.md is the full guide)
        LICENSES/                             Ballrain's vendored libraries + the engine's licences given by --license

into a zip (fixed file order and times: the same inputs give the same zip), and prints its sha256. With --sign-key and
--download-url it also writes release.json (nc_release manifest) and release.json.sig next to the zip: upload all
three to the GitHub release. No game file goes into the zip (the player uses their own Ballance).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VCREDIST = ["msvcp140.dll", "vcruntime140.dll"]  # what BallanceMMOSimTool.exe (x86, /MD) imports besides the UCRT
QUICKSTART = """Ballrain 协调器（Coordinator）{version}（Windows 测试版）

1. 把这个压缩包完整解压到一个文件夹（不要直接在压缩包里运行）。
2. 双击 BallrainCoordinator.exe。第一次运行时 Windows 可能提示“Windows 已保护你的电脑”（SmartScreen）：
   点“更多信息”，再点“仍要运行”。这个测试版还没有代码签名，所以会这样。
3. 首次设置：填写你自己电脑上的 Ballance 安装目录（含 base.cmo 的文件夹），可以点“查找常见位置”。
   客户端只读取 base.cmo、Database.tdb、3D Entities、Text，不会改动任何游戏文件。
4. 点“在浏览器中登录”，用 Ballance 论坛账号登录（论坛密码只在浏览器里输入，客户端看不到）。
5. 回到客户端点“开始”。玩 Ballance（Player.exe）或电脑用电池时会自动暂停。
6. 遇到问题：点“导出诊断”，把生成的 zip 发给测试负责人（里面的令牌、用户名、路径、IP 都已替换掉）。

数据、日志和登录状态都保存在 %USERPROFILE%\\.ballrain（程序文件夹不会被写入）；删除这个文件夹即可清除。
完整说明见项目文档 docs/native-client-testers.md。
"""


def log(msg):
    print(msg, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def client_version() -> str:
    m = re.search(r"project\(ballrain_native VERSION ([0-9.]+)", (REPO / "native" / "CMakeLists.txt").read_text("utf-8"))
    return m.group(1)


def vc_redist_x86() -> Path:
    roots = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
             Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio"]
    for r in roots:
        for crt in sorted(r.glob("*/*/VC/Redist/MSVC/*/x86/Microsoft.VC14*.CRT"), reverse=True):
            if all((crt / n).is_file() for n in VCREDIST):
                return crt
    raise SystemExit("the x86 VC++ runtime (VC/Redist/MSVC/<ver>/x86/Microsoft.VC14x.CRT) was not found")


def build(gateway: str, update_url: str, pubkey: str) -> Path:
    bdir = REPO / "build" / "native-release-pkg"
    env = dict(os.environ, BALLRAIN_BUILD_DIR=str(bdir),
               BALLRAIN_CMAKE_ARGS=f'-DBALLRAIN_DEFAULT_GATEWAY="{gateway}" -DBALLRAIN_UPDATE_URL="{update_url}" '
                                   f'-DBALLRAIN_RELEASE_PUBKEY="{pubkey}"')
    r = subprocess.run(["cmd", "/c", str(REPO / "native" / "build_win.cmd"), "Release"], env=env)
    if r.returncode != 0:
        raise SystemExit("build failed")
    st = subprocess.run([str(bdir / "nc_selftest.exe")], capture_output=True, text=True)
    if st.returncode != 0:
        raise SystemExit("nc_selftest failed: " + st.stdout[-500:])
    return bdir


def zip_dir(src: Path, dst: Path) -> None:
    files = sorted(p for p in src.rglob("*") if p.is_file())
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            zi = zipfile.ZipInfo(str(Path(src.name) / p.relative_to(src)).replace("\\", "/"), (2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="the x87-soft BallanceMMOSimTool.exe (bin-x86soft) to ship")
    ap.add_argument("--engine-version", required=True, help="ENGINE_VERSION (hello.engine.version), e.g. x87soft-2026.11")
    ap.add_argument("--license", action="append", default=[], help="NAME=FILE: a licence of the engine's sources")
    ap.add_argument("--gateway", default=os.environ.get("BALLRAIN_GATEWAY", ""))
    ap.add_argument("--update-url", default=os.environ.get("BALLRAIN_UPDATE_URL", ""))
    ap.add_argument("--pubkey-file", help="release-signing.pub (nc_release keygen)")
    ap.add_argument("--sign-key", help="release-signing.key: also write release.json + .sig")
    ap.add_argument("--download-url", help="the zip's https URL in the release (for release.json)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default=str(REPO / "dist" / "native"))
    ap.add_argument("--no-build", action="store_true", help="use build/native-release-pkg as it is")
    a = ap.parse_args(argv)
    version = client_version()
    pubkey = Path(a.pubkey_file).read_text("utf-8").strip() if a.pubkey_file else ""
    if a.update_url and not pubkey:
        raise SystemExit("--update-url needs --pubkey-file (the client verifies the manifest's signature)")
    bdir = REPO / "build" / "native-release-pkg" if a.no_build else build(a.gateway, a.update_url, pubkey)
    out = Path(a.out).resolve()
    name = f"BallrainCoordinator-{version}-windows-x64"
    stage = out / name
    if stage.exists():
        shutil.rmtree(stage)
    edir = stage / "engine" / "x87soft" / "windows"
    edir.mkdir(parents=True)
    shutil.copyfile(bdir / "BallrainCoordinator.exe", stage / "BallrainCoordinator.exe")
    shutil.copyfile(bdir / "ballrain_coordinator.exe", stage / "ballrain_coordinator.exe")  # the headless driver (hosts without a console UI)
    shutil.copyfile(a.engine, edir / "BallanceMMOSimTool.exe")
    crt = vc_redist_x86()
    for n in VCREDIST:
        shutil.copyfile(crt / n, edir / n)
    (edir / "ENGINE_VERSION").write_text(a.engine_version + "\n", encoding="utf-8")
    (stage / "使用说明.txt").write_text(QUICKSTART.format(version=version).replace("\n", "\r\n"), encoding="utf-8-sig")
    lic = stage / "LICENSES"
    lic.mkdir()
    tp = REPO / "native" / "third_party"
    for n, f in (("miniz.txt", tp / "miniz" / "LICENSE"), ("FTXUI.txt", tp / "ftxui" / "LICENSE"),
                 ("ed25519.txt", tp / "ed25519" / "license.txt")):
        shutil.copyfile(f, lic / n)
    zhead = (tp / "zstd" / "zstd.c").read_text("utf-8", errors="replace").split("*/")[0] + "*/\n"
    (lic / "zstd.txt").write_text(zhead, encoding="utf-8")
    (lic / "README.txt").write_text((tp / "README.md").read_text("utf-8"), encoding="utf-8")
    for spec in a.license:
        n, _, f = spec.partition("=")
        shutil.copyfile(f, lic / f"{n}.txt")
    zp = out / f"{name}.zip"
    zip_dir(stage, zp)
    digest = sha256(zp)
    log(f"{zp.name}: {zp.stat().st_size} bytes, sha256 {digest}")
    log(f"  engine sha256 {sha256(edir / 'BallanceMMOSimTool.exe')} (the gateway's [engines.x87soft] allowlist needs it)")
    if a.sign_key:
        if not a.download_url:
            raise SystemExit("--sign-key needs --download-url")
        rel = bdir / "nc_release.exe"
        subprocess.run([str(rel), "manifest", "--version", version, "--notes", a.notes, "--out", str(out / "release.json"),
                        "--asset", f"windows-x64={a.download_url}={zp}"], check=True)
        subprocess.run([str(rel), "sign", "--key", a.sign_key, "--manifest", str(out / "release.json")], check=True)
        log("release.json + release.json.sig written (upload them with the zip)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

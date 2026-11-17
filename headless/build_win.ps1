# Builds the Windows x86 headless host (BallanceMMOSimTool with --ballrain-agent) that runs the ORIGINAL 2002
# physics_RT.dll (headless/README.md "Windows x86: the 2002 DLL host").
#
#   powershell -ExecutionPolicy Bypass -File headless/build_win.ps1 SRC_TAR WORK_DIR
#
# SRC_TAR: the output of headless/mk_src.sh (src.tar, prefix src/). WORK_DIR gets src/ (pristine + patches 0002 and
# 0003 + headless/src + BallrainIO/src/Ivp2002.h) and build/; it must be a new or a scratch directory (src/ is
# replaced). Environment (no machine paths are kept in this file):
#   VCVARS           vcvarsamd64_x86.bat of Visual Studio (default: VS 18 Community's standard install path)
#   CMAKE / NINJA    default: from PATH
#   VCPKG_ROOT       a vcpkg tree (its scripts/buildsystems/vcpkg.cmake is the toolchain)
#   VCPKG_INSTALLED  the x86-windows packages the server CMake wants (Protobuf, GameNetworkingSockets, ...)
#   SDL3_DIR, GTEST_DIR, YAMLCPP_DIR   FetchContent source trees (the build is FETCHCONTENT_FULLY_DISCONNECTED)
#   PATCH            GNU patch (default: Git for Windows' usr/bin/patch.exe, else patch from PATH)
#   JOBS             default: logical processors - 1
# Flags: the sim-only x87 SOFT recipe of build_mac.sh (the source physics stays in the binary: BALLRAIN_HL_PHYS2002=0
# selects it), for x86 (the 2002 DLL is a 32-bit DLL), BMMO_BUILD_ID ballrain-hlwin-p2002.
param([Parameter(Mandatory = $true)][string]$SrcTar, [Parameter(Mandatory = $true)][string]$Work)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Here

function Need([string]$name) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if (-not $v) { throw "$name is not set (see the header of build_win.ps1)" }
    return $v.Replace('\', '/')
}
$vcvars = $env:VCVARS
if (-not $vcvars) { $vcvars = Join-Path $env:ProgramFiles "Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsamd64_x86.bat" }
if (-not (Test-Path $vcvars)) { throw "VCVARS: $vcvars not found" }
$cmake = if ($env:CMAKE) { $env:CMAKE } else { "cmake" }
$ninja = if ($env:NINJA) { $env:NINJA } else { (Get-Command ninja -ErrorAction Stop).Source }
$patch = $env:PATCH
if (-not $patch) {
    $git = Join-Path $env:ProgramFiles "Git\usr\bin\patch.exe"
    $patch = if (Test-Path $git) { $git } else { (Get-Command patch -ErrorAction Stop).Source }
}
$vcpkg = Need "VCPKG_ROOT"
$vcpkgInstalled = Need "VCPKG_INSTALLED"
$sdl3 = Need "SDL3_DIR"
$gtest = Need "GTEST_DIR"
$yaml = Need "YAMLCPP_DIR"
$jobs = if ($env:JOBS) { $env:JOBS } else { [Math]::Max(1, [Environment]::ProcessorCount - 1) }

New-Item -ItemType Directory -Force $Work | Out-Null
$Work = (Resolve-Path $Work).Path
$src = Join-Path $Work "src"
if (Test-Path $src) { Remove-Item -Recurse -Force $src }
# bsdtar cannot create the few symlinks of the third-party test trees on Windows: they are not built, the rest is checked
& tar -xf $SrcTar -C $Work 2>$null
if (-not (Test-Path (Join-Path $src "BallanceMMOServer\sim\sim_tool.cpp"))) { throw "extraction of $SrcTar failed" }
foreach ($p in @("0002-ballrain-agent-entry.diff", "0003-hlwin-phys2002.diff")) {
    Write-Host "applying $p"
    & $patch -d $src -p1 --forward --quiet -i (Join-Path $Here "patches\$p")
    if ($LASTEXITCODE -ne 0) { throw "patch $p failed" }
}
$sim = Join-Path $src "BallanceMMOServer\sim"
foreach ($f in @("ballrain_agent.cpp", "ballrain_agent.h", "ballrain_wire.h", "phys2002.cpp", "phys2002.hpp")) {
    Copy-Item (Join-Path $Here "src\$f") $sim -Force
}
Copy-Item (Join-Path $Repo "BallrainIO\src\Ivp2002.h") $sim -Force   # the 2002 DLL layout, shared with the retail mod

$build = Join-Path $Work "build"
$cfg = @(
    "-S", "`"$src`"", "-B", "`"$build`"", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
    "-DCMAKE_MAKE_PROGRAM=`"$($ninja.Replace('\', '/'))`"",
    "-DCMAKE_TOOLCHAIN_FILE=`"$vcpkg/scripts/buildsystems/vcpkg.cmake`"", "-DVCPKG_TARGET_TRIPLET=x86-windows",
    "-DVCPKG_INSTALLED_DIR=`"$vcpkgInstalled`"", "-DVCPKG_MANIFEST_INSTALL=OFF",
    "-DFETCHCONTENT_SOURCE_DIR_BMMO_SDL3=`"$sdl3`"", "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=`"$gtest`"",
    "-DFETCHCONTENT_SOURCE_DIR_YAML-CPP=`"$yaml`"", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
    "-DBUILD_CLIENT=OFF", "-DBUILD_SERVER=ON", "-DBMMO_BUILD_SIM=ON", "-DBMMO_BUILD_SIM_ONLY=ON", "-DBMMO_BUILD_PHYSICS_RT=OFF",
    "-DBMMO_PHYSICS_PORTABLE_MATH=ON", "-DBMMO_PHYSICS_FP_STRICT=ON", "-DBMMO_PHYSICS_X87=SOFT", "-DBMMO_BUILD_ID=ballrain-hlwin-p2002")
$bat = Join-Path $Work "build_steps.bat"
@(
    "@echo off",
    "call `"$vcvars`" >nul || exit /b 4",
    "set VSLANG=1033",
    "`"$cmake`" $($cfg -join ' ') > `"$Work\configure.log`" 2>&1 || exit /b 3",
    "`"$cmake`" --build `"$build`" --target BallanceMMOSimTool -- -j $jobs > `"$Work\build.log`" 2>&1 || exit /b 2"
) | Set-Content -Encoding ascii $bat
$t0 = Get-Date
& cmd /c "`"$bat`""
$rc = $LASTEXITCODE
if ($rc -eq 3) { Get-Content (Join-Path $Work "configure.log") -Tail 30; throw "CONFIGURE-FAILED" }
if ($rc -ne 0) { Select-String -Path (Join-Path $Work "build.log") -Pattern "error|FAILED" | Select-Object -First 40; throw "BUILD-FAILED ($rc)" }
$exe = Join-Path $build "BallanceMMOServer\BallanceMMOSimTool.exe"
Write-Host ("build seconds: {0:N0}" -f ((Get-Date) - $t0).TotalSeconds)
Get-Item $exe | Format-List FullName, Length, LastWriteTime
Write-Host "BUILD-OK"

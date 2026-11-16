#!/bin/sh
# Builds the headless agent backend (BallanceMMOSimTool with --ballrain-agent) from the pinned sources.
#
#   headless/build_mac.sh SRC_TAR WORK_DIR
#
# SRC_TAR: the output of headless/mk_src.sh (src.tar, prefix src/). WORK_DIR gets src/ (pristine + patch 0002 +
# headless/src) and build/. Environment: SDL3_DIR (an SDL3 source tree, required by the engine's CMake; on macOS
# with two local edits: CMakeLists.txt "if(MACOS AND SDL_VIDEO)" for the tray, src/tray/dummy/SDL_tray.c "#if 1"),
# CMAKE / NINJA (default: from PATH), JOBS (default: all cores - 1).
# Flags: the sim-only x87 SOFT recipe (BMMO_BUILD_SIM_ONLY, no physics_RT plugin, portable math, strict FP); the
# product's cmake/BallancedHeadless.cmake adds VX_X87_GAME (the game-side x87 math, the CK-id ball-pair key).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TAR=$1
WORK=$2
CMAKE=${CMAKE:-cmake}
NINJA=${NINJA:-ninja}
: "${SDL3_DIR:?SDL3_DIR = an SDL3 source tree}"
JOBS=${JOBS:-$(( $(sysctl -n hw.ncpu 2>/dev/null || nproc) - 1 ))}
mkdir -p "$WORK"
rm -rf "$WORK/src"
tar xf "$TAR" -C "$WORK"
for p in "$HERE"/patches/0*.diff; do
  echo "applying $(basename "$p")"
  patch -d "$WORK/src" -p1 --forward --quiet < "$p"
done
cp "$HERE/src/ballrain_agent.cpp" "$HERE/src/ballrain_agent.h" "$HERE/src/ballrain_wire.h" "$WORK/src/BallanceMMOServer/sim/"
"$CMAKE" -S "$WORK/src" -B "$WORK/build" -G Ninja -DCMAKE_MAKE_PROGRAM="$(command -v "$NINJA")" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_CLIENT=OFF -DBUILD_SERVER=ON -DBMMO_BUILD_SIM=ON -DBMMO_BUILD_SIM_ONLY=ON -DBMMO_BUILD_PHYSICS_RT=OFF \
  -DBMMO_PHYSICS_PORTABLE_MATH=ON -DBMMO_PHYSICS_FP_STRICT=ON -DBMMO_PHYSICS_X87=SOFT -DBMMO_BUILD_ID=ballrain-headless-x87s \
  -DFETCHCONTENT_SOURCE_DIR_BMMO_SDL3="$SDL3_DIR" -DFETCHCONTENT_FULLY_DISCONNECTED=ON > "$WORK/configure.log" 2>&1 \
  || { tail -30 "$WORK/configure.log"; echo CONFIGURE-FAILED; exit 3; }
start=$(date +%s)
"$NINJA" -C "$WORK/build" -j "$JOBS" BallanceMMOSimTool > "$WORK/build.log" 2>&1 \
  || { grep -E "error:|FAILED|Undefined|ld: " "$WORK/build.log" | head -40; echo BUILD-FAILED; exit 2; }
echo "build seconds: $(( $(date +%s) - start ))"
ls -la "$WORK/build/BallanceMMOServer/BallanceMMOSimTool"
echo BUILD-OK

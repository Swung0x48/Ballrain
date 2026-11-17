#!/bin/sh
# Mac: Python environment and the headless engine for training.
#   MAC_DIR=$HOME/ballrain headless/mac/setup_mac.sh [SRC_TAR]     uv sync; build the engine if SRC_TAR is given
#   MAC_DIR=$HOME/ballrain headless/mac/setup_mac.sh --activate    record the build in $MAC_DIR/hl/ENGINE (after the gate)
# - uv (PyPI: python3 -m pip install --user uv) syncs the repo's environment with CPython 3.12 (uv python install).
# - SRC_TAR (headless/mk_src.sh output, copied to the Mac) builds the engine into a NEW directory named after the
#   BallanceMMO commit mk_src.sh pins and the hashes of patch 0002 and the agent sources,
#   $MAC_DIR/hl/x87s-<commit 8>-<0002 sha8>-<agent sha8>: a binary that a running trainer uses is never overwritten.
# - --activate writes that path into $MAC_DIR/hl/ENGINE, which train_mac.sh reads when a run starts. Run it only after
#   the gate (docs/headless.md section 7) passed on the new build.
set -eu
D=${MAC_DIR:-$HOME/ballrain}
R=$D/repo
C=$(sed -n 's/^C=\${BMMO_COMMIT:-\([0-9a-f]*\)}$/\1/p' "$R/headless/mk_src.sh" | cut -c1-8)
P=$(cat "$R"/headless/patches/0002-*.diff | shasum -a 256 | cut -c1-8)   # 0003: Windows x86 only
A=$(cat "$R/headless/src/ballrain_agent.cpp" "$R/headless/src/ballrain_agent.h" "$R/headless/src/ballrain_wire.h" | shasum -a 256 | cut -c1-8)
W=$D/hl/x87s-$C-$P-$A
B=$W/build/BallanceMMOServer/BallanceMMOSimTool
if [ "${1:-}" = --activate ]; then
  [ -x "$B" ] || { echo "engine $B missing: build it first"; exit 2; }
  echo "$B" > "$D/hl/ENGINE"
  echo "engine: $B (recorded in $D/hl/ENGINE)"
  exit 0
fi
UV=${UV:-$(command -v uv || echo "$HOME/Library/Python/3.9/bin/uv")}
cd "$R"
"$UV" python install 3.12
"$UV" sync --python 3.12
if [ ! -x "$B" ] && [ -n "${1:-}" ]; then
  T=$HOME/Library/Python/3.9/bin
  CMAKE=${CMAKE:-$T/cmake} NINJA=${NINJA:-$T/ninja} SDL3_DIR=${SDL3_DIR:-$D/sdl3} \
    nice -n 10 /bin/sh "$R/headless/build_mac.sh" "$1" "$W"
fi
if [ -x "$B" ]; then
  echo "engine built: $B (current: $(cat "$D/hl/ENGINE" 2>/dev/null || echo none); gate it, then --activate)"
else
  echo "engine $B missing: pass SRC_TAR to build it"
fi
[ -f "$D/game/base.cmo" ] && echo "game: $D/game" || echo "game folder missing: copy a Ballance install to $D/game"

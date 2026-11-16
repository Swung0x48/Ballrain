#!/bin/sh
# Mac: Python environment and the headless engine for training (run once per push / engine update).
#   MAC_DIR=$HOME/ballrain headless/mac/setup_mac.sh [SRC_TAR]
# - uv (PyPI: python3 -m pip install --user uv) syncs the repo's environment with CPython 3.12 (uv python install).
# - SRC_TAR (headless/mk_src.sh output, copied to the Mac) builds the engine into a NEW directory named after the hashes
#   of patch 0001 and the agent source, $MAC_DIR/hl/hx1c-<0001 sha8>-<agent sha8>: a binary that a running trainer
#   uses is never overwritten. The path goes into $MAC_DIR/hl/ENGINE, which train_mac.sh reads when a run starts.
set -eu
D=${MAC_DIR:-$HOME/ballrain}
UV=${UV:-$(command -v uv || echo "$HOME/Library/Python/3.9/bin/uv")}
cd "$D/repo"
"$UV" python install 3.12
"$UV" sync --python 3.12
P=$(shasum -a 256 "$D"/repo/headless/patches/0001-*.diff | cut -c1-8)
A=$(shasum -a 256 "$D/repo/headless/src/ballrain_agent.cpp" | cut -c1-8)
W=$D/hl/hx1c-$P-$A
B=$W/build/BallanceMMOServer/BallanceMMOSimTool
if [ ! -x "$B" ] && [ -n "${1:-}" ]; then
  T=$HOME/Library/Python/3.9/bin
  CMAKE=${CMAKE:-$T/cmake} NINJA=${NINJA:-$T/ninja} SDL3_DIR=${SDL3_DIR:-$D/sdl3} \
    /bin/sh "$D/repo/headless/build_mac.sh" "$1" "$W"
fi
if [ -x "$B" ]; then
  echo "$B" > "$D/hl/ENGINE"
  echo "engine: $B (recorded in $D/hl/ENGINE)"
else
  echo "engine $B missing: pass SRC_TAR to build it"
fi
[ -f "$D/game/base.cmo" ] && echo "game: $D/game" || echo "game folder missing: copy a Ballance install to $D/game"

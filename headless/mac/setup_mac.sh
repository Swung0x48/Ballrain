#!/bin/sh
# Mac: Python environment and the headless engine for training (run once per push / engine update).
#   MAC_DIR=$HOME/ballrain headless/mac/setup_mac.sh [SRC_TAR]
# - uv (PyPI: python3 -m pip install --user uv) syncs the repo's environment with CPython 3.12 (uv python install).
# - SRC_TAR (headless/mk_src.sh output, copied to the Mac) builds the engine into $MAC_DIR/hl/work-exact unless built.
set -eu
D=${MAC_DIR:-$HOME/ballrain}
UV=${UV:-$(command -v uv || echo "$HOME/Library/Python/3.9/bin/uv")}
cd "$D/repo"
"$UV" python install 3.12
"$UV" sync --python 3.12
B=$D/hl/work-exact/build/BallanceMMOServer/BallanceMMOSimTool
if [ ! -x "$B" ] && [ -n "${1:-}" ]; then
  T=$HOME/Library/Python/3.9/bin
  CMAKE=${CMAKE:-$T/cmake} NINJA=${NINJA:-$T/ninja} SDL3_DIR=${SDL3_DIR:-$D/sdl3} \
    /bin/sh "$D/repo/headless/build_mac.sh" "$1" "$D/hl/work-exact"
fi
[ -x "$B" ] && echo "engine: $B" || echo "engine missing: build it (headless/build_mac.sh)"
[ -f "$D/game/base.cmo" ] && echo "game: $D/game" || echo "game folder missing: copy a Ballance install to $D/game"

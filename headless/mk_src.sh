#!/bin/bash
# Packs the headless engine's sources at the pinned commits (headless/README.md) with git archive, read-only:
# BallanceMMO (branch x87-soft, product commit) + its submodules at the commits that commit pins, recursively, from
# their local checkouts (BMMO_REPO/submodule/...; each must have its pinned commit), and SDL3. Nothing is fetched and
# nothing in those repositories changes. Line endings are the repositories' own (LF): patch 0002 applies to them.
#
#   headless/mk_src.sh OUT_DIR          (env: BMMO_REPO, BMMO_COMMIT, SDL3_REPO, SDL3_COMMIT)
#
# Output: OUT_DIR/src.tar (prefix src/), OUT_DIR/sdl3.tar (prefix sdl3/), OUT_DIR/commits.txt. With the default
# BMMO_COMMIT the submodule pins the physics depends on are checked against headless/README.md's table.
set -euo pipefail
R=${BMMO_REPO:?BMMO_REPO = a clone of BallanceMMO that has the x87-soft commits}
C=${BMMO_COMMIT:-ba5ba279374bfcad2bd5dfe81e129dda507a2881}
SDL=${SDL3_REPO:-}
SDLC=${SDL3_COMMIT:-96292a5b464258a2b926e0a3d72f8b98c2a81aa6}
OUT=$1
mkdir -p "$OUT"
G="git -c core.autocrlf=false"
echo "BallanceMMO $C" > "$OUT/commits.txt"
$G -C "$R" archive --format=tar --prefix=src/ -o "$OUT/src.tar" "$C" CMakeLists.txt cmake BallanceMMOCommon BallanceMMOServer scripts
n=0
add_repo() {   # dir commit prefix
  local dir=$1 pin=$2 prefix=$3
  n=$((n + 1))
  $G -C "$dir" archive --format=tar --prefix="src/$prefix/" -o "$OUT/part_$n.tar" "$pin"
  tar -Af "$OUT/src.tar" "$OUT/part_$n.tar"
  rm -f "$OUT/part_$n.tar"
  echo "$prefix $pin" >> "$OUT/commits.txt"
  $G -C "$dir" ls-tree -r "$pin" | awk '$2 == "commit" {print $3, $4}' | while read -r sub path; do
    if [ "$prefix" = submodule/GameNetworkingSockets ]; then echo "$prefix/$path $sub left out" >> "$OUT/commits.txt"
    elif [ -e "$dir/$path/.git" ]; then add_repo "$dir/$path" "$sub" "$prefix/$path"
    else echo "$prefix/$path $sub NOT-CHECKED-OUT (left out)" >> "$OUT/commits.txt"; fi
  done
}
for sm in Ballanced zstd asio replxx ya_getopt picojson parallel-hashmap GameNetworkingSockets; do
  add_repo "$R/submodule/$sm" "$($G -C "$R" ls-tree "$C" submodule/$sm | awk '{print $3}')" "submodule/$sm"
done
if [ -n "$SDL" ]; then
  $G -C "$SDL" archive --format=tar --prefix=sdl3/ -o "$OUT/sdl3.tar" "$SDLC"
  echo "sdl3 $SDLC" >> "$OUT/commits.txt"
fi
# the product pins (headless/README.md); a different BMMO_COMMIT is a deliberate experiment and is not checked
if [ "$C" = ba5ba279374bfcad2bd5dfe81e129dda507a2881 ]; then
  for want in "submodule/Ballanced b70ab01b750a8985655dc1353b6d99bbf676cbe2"               "submodule/Ballanced/Source/VxMath 1b07da5bcfa809db1f5bdad2267cff6026787dd9"               "submodule/Ballanced/Source/RenderEngine cd6f55a8855a80c50dc0f89accdb4a39f3807122"               "submodule/Ballanced/Source/BuildingBlocks 2f1cdbdb5901686b51fd9110392b2d68497b1218"               "submodule/Ballanced/Source/BuildingBlocks/physics_RT/ivp bbd137ac7678815535d424a61da5e13679c9892e"; do
    grep -qx "$want" "$OUT/commits.txt" || { echo "pin mismatch: want $want"; exit 3; }
  done
fi
cat "$OUT/commits.txt"

#!/bin/bash
# PC -> Mac: the working tree of this checkout (tracked + untracked, not ignored; runtime/ excluded, so uncommitted
# training code goes along; benchmarks/refpaths is tracked) into $MAC_DIR/repo, the tas-v1 records named with -r
# (TAS start states: raw.npz, meta.json, starts.npz, camera.npz, level messages) and checkpoints named with -c
# (a path under runtime/, copied to the same place, e.g. runtime/m2-train/r2/latest.pt).
#   BALLRAIN_MAC=user@host [BALLRAIN_MAC_SSH="-i <key>"] [MAC_DIR=ballrain] headless/mac/push_repo.sh [-r RECORD]... [-c FILE]...
# The Mac's .venv and runtime/ are kept (files are only added / overwritten).
set -euo pipefail
: "${BALLRAIN_MAC:?BALLRAIN_MAC=user@host}"
SSH=(ssh -o BatchMode=yes ${BALLRAIN_MAC_SSH:-} "$BALLRAIN_MAC")
D=${MAC_DIR:-ballrain}
cd "$(dirname "$0")/../.."
recs=() ckpts=()
while [ $# -gt 0 ]; do
  case "$1" in
    -r) recs+=("$2"); shift 2;;
    -c) ckpts+=("$2"); shift 2;;
    *) echo "usage: $0 [-r RECORD]... [-c runtime/FILE]..."; exit 2;;
  esac
done
git ls-files -z --cached --others --exclude-standard | grep -zv -e '^runtime/' -e '^\.claude/' | tar cf - --null -T - \
  | "${SSH[@]}" "mkdir -p $D/repo && rm -rf $D/repo/headless/patches && tar xf - -C $D/repo && find $D/repo -path $D/repo/.venv -prune -o -name __pycache__ -prune -exec rm -rf {} +"
files=()
for rec in "${recs[@]}"; do
  src=runtime/datasets/tas-v1/$rec
  [ -d "$src" ] || { echo "no dataset record $src"; exit 2; }
  for f in raw.npz meta.json starts.npz camera.npz level.msg level.msg.inc events.txt; do
    [ -f "$src/$f" ] && files+=("$src/$f")
  done
done
[ ${#recs[@]} -gt 0 ] && files+=(runtime/datasets/tas-v1/index.json)
for c in "${ckpts[@]}"; do
  case "$c" in runtime/*) [ -f "$c" ] && files+=("$c") || { echo "no file $c"; exit 2; };; *) echo "-c takes a path under runtime/"; exit 2;; esac
done
if [ ${#files[@]} -gt 0 ]; then
  tar cf - "${files[@]}" | "${SSH[@]}" "tar xf - -C $D/repo"
fi
git rev-parse HEAD | "${SSH[@]}" "cat > $D/repo/.push_base_commit && echo pushed to $D/repo, base commit \$(cat $D/repo/.push_base_commit), ${#files[@]} runtime files"

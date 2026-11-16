#!/bin/bash
# Mac -> PC: the traced run directories of a training run (train_m2.py --trace-sample: kept whatever the outcome) for
# the export spot-checks (scripts/sim2real_m2.py spotcheck): only .tas, .json, events.txt and trace.txt of finished runs
# (a .json present), every INTERVAL seconds, into runtime/m2-train-mac/RUN/episodes-traced/<copy>/<run_id>/.
#   BALLRAIN_MAC=user@host [BALLRAIN_MAC_SSH="-i <key>"] [MAC_DIR=ballrain] headless/mac/pull_traced.sh RUN [INTERVAL] [--once]
set -euo pipefail
: "${BALLRAIN_MAC:?BALLRAIN_MAC=user@host}"
RUN=$1; INTERVAL=${2:-600}; ONCE=${3:-}
SSH=(ssh -o BatchMode=yes ${BALLRAIN_MAC_SSH:-} "$BALLRAIN_MAC")
D=${MAC_DIR:-ballrain}
cd "$(dirname "$0")/../.."
OUT=runtime/m2-train-mac/$RUN/episodes-traced
mkdir -p "$OUT"
while true; do
  "${SSH[@]}" "cd $D/repo/runtime/m2-train/$RUN/episodes 2>/dev/null || exit 0; S=../.pull_traced_stamp && N=\$S.new && touch \$N &&
    for t in \$(if [ -f \$S ]; then find . -name trace.txt -newer \$S; else find . -name trace.txt; fi); do
      d=\$(dirname \$t); ls \$d/*.json >/dev/null 2>&1 || continue
      for f in \$d/*.json \$d/*.tas \$d/events.txt \$d/trace.txt; do [ -f \$f ] && printf '%s\0' \$f; done
    done | tar cf - --null -T - && mv \$N \$S" | tar xf - -C "$OUT" && echo "$(date '+%H:%M:%S') traced runs: $(find "$OUT" -name trace.txt | wc -l)"
  [ "$ONCE" = --once ] && break
  sleep "$INTERVAL"
done

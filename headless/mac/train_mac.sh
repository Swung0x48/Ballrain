#!/bin/sh
# Mac: start scripts/train_m2.py with the headless backend, detached and awake (caffeinate: a user-activity assertion
# first, then -s -i for the job: plain -i lets the idle Mac drop into Maintenance Sleep between ssh calls).
#   MAC_DIR=$HOME/ballrain headless/mac/train_mac.sh RUN [train_m2.py args ...]
# Outputs in $MAC_DIR/repo/runtime/m2-train/RUN (train.log there + nohup.log). Started from /bin/sh, not zsh, so the
# job keeps normal priority. Stop: touch $MAC_DIR/repo/runtime/m2-train/RUN/STOP (the learner's own stop file).
set -eu
D=${MAC_DIR:-$HOME/ballrain}
RUN=$1; shift
UV=${UV:-$(command -v uv || echo "$HOME/Library/Python/3.9/bin/uv")}
export BALLRAIN_RUNTIME=${BALLRAIN_RUNTIME:-$D/repo/runtime}
# the engine: BALLRAIN_HL_BIN, else the build setup_mac.sh recorded in $D/hl/ENGINE (fixed for the run at its start)
export BALLRAIN_HL_BIN=${BALLRAIN_HL_BIN:-$(cat "$D/hl/ENGINE")}
export BALLRAIN_HL_ROOT=${BALLRAIN_HL_ROOT:-$D/game}
export PYTHONPATH=$D/repo
mkdir -p "$BALLRAIN_RUNTIME/m2-train/$RUN"
cd "$D/repo"
caffeinate -u -t 2
nohup caffeinate -s -i "$UV" run --offline python scripts/train_m2.py --run "$RUN" --backend headless "$@" \
  > "$BALLRAIN_RUNTIME/m2-train/$RUN/nohup.log" 2>&1 < /dev/null &
echo "started pid $! (run $RUN, engine $BALLRAIN_HL_BIN)"

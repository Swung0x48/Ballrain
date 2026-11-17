#!/bin/bash
# Mac -> PC: copy a training run's directory (logs, tensorboard, checkpoints) every INTERVAL seconds into
# runtime/m2-train-mac/RUN of this checkout, so TensorBoard on the PC can show it (ssh + tar; no rsync needed).
#   BALLRAIN_MAC=user@host [BALLRAIN_MAC_SSH="-i <key>"] [MAC_DIR=ballrain] headless/mac/pull_run.sh RUN [INTERVAL] [--once]
# Episode directories are left on the Mac (the worker prunes them); only files changed since the last copy are sent.
set -euo pipefail
: "${BALLRAIN_MAC:?BALLRAIN_MAC=user@host}"
RUN=$1; INTERVAL=${2:-300}; ONCE=${3:-}
SSH=(ssh -o BatchMode=yes ${BALLRAIN_MAC_SSH:-} "$BALLRAIN_MAC")
D=${MAC_DIR:-ballrain}
cd "$(dirname "$0")/../.."
OUT=runtime/m2-train-mac
mkdir -p "$OUT/$RUN"
# expose the run as runtime/m2-train/RUN-mac so the TensorBoard logdir (and the public scalar mirror) picks it up
LINK=runtime/m2-train/$RUN-mac
if [ ! -e "$LINK" ]; then
  mkdir -p runtime/m2-train
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) powershell -NoProfile -Command "New-Item -ItemType Junction -Path '$(cygpath -w "$PWD/$LINK")' -Target '$(cygpath -w "$PWD/$OUT/$RUN")' | Out-Null" ;;
    *) ln -s "../m2-train-mac/$RUN" "$LINK" ;;
  esac
fi
while true; do
  # only files changed since the previous copy (a stamp file next to the run on the Mac; BSD find has no -newermt @N)
  "${SSH[@]}" "cd $D/repo/runtime/m2-train && S=.pull_stamp_$RUN && N=\$S.new && touch \$N &&       if [ -f \$S ]; then find $RUN -type f -newer \$S ! -path '*/episodes/*' -print0;       else find $RUN -type f ! -path '*/episodes/*' -print0; fi | tar cf - --null -T - && mv \$N \$S"     | tar xf - -C "$OUT" && echo "$(date '+%H:%M:%S') synced $OUT/$RUN"
  [ "$ONCE" = --once ] && break
  sleep "$INTERVAL"
done

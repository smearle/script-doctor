#!/usr/bin/env bash
# Launch run_vq_usage_ablation.sh in the background and record its PID.
#
# This wrapper is useful in managed environments where GPU access must be
# granted to the launcher process: the nohup child is spawned from inside that
# granted process instead of through an outer shell background operator.
set -euo pipefail

cd "$(dirname "$0")/../.."

GAMES=${GAMES:-scaling_14}
SEED=${SEED:-0}
SAVE_ROOT=${SAVE_ROOT:-nca_wm/logs/vq_usage_ablation_${GAMES}_s${SEED}}
mkdir -p "$SAVE_ROOT"

log="$SAVE_ROOT/launcher.log"
pid_file="$SAVE_ROOT/launcher.pid"
TRACE_LAUNCH=${TRACE_LAUNCH:-0}

if [ "$TRACE_LAUNCH" = "1" ]; then
    nohup bash -x nca_wm/scripts/run_vq_usage_ablation.sh > "$log" 2>&1 &
else
    nohup nca_wm/scripts/run_vq_usage_ablation.sh > "$log" 2>&1 &
fi
pid=$!
printf "%s\n" "$pid" > "$pid_file"

echo "launched pid=$pid"
echo "log=$log"
echo "pid_file=$pid_file"

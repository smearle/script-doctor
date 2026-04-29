#!/usr/bin/env bash
# GPU 1 parallel queue: after clearing retry finishes, run phase 2 (hid=256
# scaling_6 + small) and phase 3 (scaling_large at hid=128 then hid=256).
# Waits on clearing via pgrep. Uses RUNNING.pid lock file (written by
# train.py) so the GPU 0 overnight_growth.sh skip-check avoids duplicate
# runs if it reaches the same save_dir while we're still training here.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

GPU=1

# Wait for clearing retry to finish
while pgrep -f 'nca_wm/train.py --games global_clearing --conditional --balanced_sampling --axis_pool --axis_cummax --global_pool' >/dev/null; do
    sleep 60
done
echo "[gpu1-phase23] clearing finished, starting parallel phase 2 + 3 on GPU 1"

run_one() {
    local games=$1 hid=$2 n_updates=$3 tag=$4
    local log="$LOGDIR/growth_${games}_h${hid}_${tag}_gpu${GPU}.log"

    local pool_tag="_ap_ac_gp"
    local save_dir="$REPO/nca_wm/logs/multi_${games}_cond_bal${pool_tag}_level-None_nca-4_hid-${hid}_lr-0.001_pat-80_s-0"

    # Skip if already done
    if [ -f "$save_dir/train_meta.json" ]; then
        local done_status=$("$PY" -c "
import json, sys
m = json.load(open(sys.argv[1]))
steps = m.get('total_steps', 0)
early = m.get('early_stopped', False)
req = m.get('n_updates_requested', 0)
target = int(sys.argv[2])
if steps >= target: print('done')
elif early and req >= target: print('done')
else: print('not_done')
" "$save_dir/train_meta.json" "$n_updates" 2>/dev/null || echo "not_done")
        if [ "$done_status" = "done" ]; then
            echo "[gpu1-phase23] SKIP $games (hid=$hid) — already finished"
            return
        fi
    fi

    # Skip if another process holds the lock
    if [ -f "$save_dir/RUNNING.pid" ]; then
        local lock_pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            echo "[gpu1-phase23] SKIP $games (hid=$hid) — already running (pid $lock_pid)"
            return
        fi
    fi

    echo "[gpu1-phase23] starting $games (hid=$hid, n_updates=$n_updates) -> $log"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games "$games" \
        --conditional \
        --balanced_sampling \
        --axis_pool --axis_cummax --global_pool \
        --n_hid "$hid" \
        --n_nca_steps 4 \
        --grad_clip 1.0 \
        --max_transitions_per_game 200000 \
        --n_updates "$n_updates" \
        --patience 80 --min_delta 1e-6 \
        --wandb \
        --sweep_name "overnight_growth" \
        >"$log" 2>&1
    echo "[gpu1-phase23] finished $games (hid=$hid), exit=$?"
}

# Phase 2: higher capacity. scaling_6@256 first (smaller dataset, faster
# feedback on whether hid=256 unlocks sub-1% on the 6-game set). Then
# small@256 to see if 9 games also benefit.
run_one scaling_6  256  80000  phase2
run_one small      256  80000  phase2

# Phase 3: 20-game gallery at both hid=128 and hid=256.
run_one scaling_large 128 100000 phase3
run_one scaling_large 256 100000 phase3

echo "[gpu1-phase23] All GPU 1 phase 2+3 runs finished."

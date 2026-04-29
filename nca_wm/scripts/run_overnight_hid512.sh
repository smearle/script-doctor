#!/usr/bin/env bash
# Overnight part 2: the hid scaling experiment.
#
# Observed trend: small@h128 → 4.2e-4 best loss; small@h256 → 9.2e-5 (4.5× better).
# Conjecture: multi-game joint training is capacity-bound. Does hid=512 continue
# the trend? Scaling_large + scaling_6 both show gains from capacity, so this
# gives us a third data point.
#
# Order matters: GPU 0 loads first; GPU 1 waits ~5 min to stagger dataset load
# and avoid the OOM collateral damage we saw earlier.
#
# Uses RUNNING.pid lock + the usual skip-check.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

run_one_gpu() {
    local gpu=$1 games=$2 hid=$3 n_updates=$4 tag=$5
    local log="$LOGDIR/growth_${games}_h${hid}_${tag}_gpu${gpu}.log"

    local pool_tag="_ap_ac_gp"
    local save_dir="$REPO/nca_wm/logs/multi_${games}_cond_bal${pool_tag}_level-None_nca-4_hid-${hid}_lr-0.001_pat-80_s-0"

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
            echo "[$tag] SKIP $games (hid=$hid) — already finished"
            return
        fi
    fi

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local lock_pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            echo "[$tag] SKIP $games (hid=$hid) — already running (pid $lock_pid)"
            return
        fi
    fi

    echo "[$tag] starting $games (hid=$hid, n_updates=$n_updates, GPU $gpu) -> $log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
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
        --sweep_name "overnight_hid512" \
        >"$log" 2>&1 &
    local child=$!
    echo "[$tag] pid=$child log=$log"
    wait "$child"
    echo "[$tag] finished $games (hid=$hid), exit=$?"
}

# GPU 0: small @ hid=512 (the big experiment; extends the scaling trend)
run_one_gpu 0 small 512 100000 phase4 &
gpu0_pid=$!
echo "[master] launched GPU 0 phase4 small@h512 via pid $gpu0_pid"

# Wait 5 minutes so GPU 1's dataset load doesn't collide with GPU 0's
echo "[master] sleeping 300s to stagger GPU 1"
sleep 300

# GPU 1: scaling_6 @ hid=512 (sanity check: does 6-game cleanly fit at h=512?)
run_one_gpu 1 scaling_6 512 80000 phase4 &
gpu1_pid=$!
echo "[master] launched GPU 1 phase4 scaling_6@h512 via pid $gpu1_pid"

wait $gpu0_pid
echo "[master] GPU 0 done"
wait $gpu1_pid
echo "[master] GPU 1 done"

echo "[master] All hid=512 runs finished."

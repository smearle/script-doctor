#!/usr/bin/env bash
# After h512 runs finish on GPUs 0 and 1, launch scaling_large on each.
# Uses the new v7 per-game native-shape dataset storage, which should
# reduce RAM from ~250GB (v6 padded) to ~30GB (per-game max).
#
# GPU 0 (freed by small@h512)  → scaling_large @ hid=128, 80K
# GPU 1 (freed by scaling_6@h512) → scaling_large @ hid=256, 100K

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

run_on_gpu() {
    local gpu=$1 hid=$2 n_updates=$3 tag=$4 wait_game=$5
    local log="$LOGDIR/growth_scaling_large_h${hid}_${tag}_gpu${gpu}.log"

    # Wait for the relevant h512 training on this GPU to exit
    echo "[$tag] GPU $gpu: waiting for h512 $wait_game to finish"
    while pgrep -f "train\.py --games ${wait_game} .* --n_hid 512 " >/dev/null; do
        sleep 60
    done
    echo "[$tag] GPU $gpu: h512 done, starting scaling_large"

    local pool_tag="_ap_ac_gp"
    local save_dir="$REPO/nca_wm/logs/multi_scaling_large_cond_bal${pool_tag}_level-None_nca-4_hid-${hid}_lr-0.001_pat-80_s-0"

    if [ -f "$save_dir/train_meta.json" ]; then
        local done_status=$("$PY" -c "
import json, sys
m = json.load(open(sys.argv[1]))
steps = m.get('total_steps', 0)
early = m.get('early_stopped', False)
req = m.get('n_updates_requested', 0)
target = int(sys.argv[2])
if steps >= target or (early and req >= target): print('done')
else: print('not_done')
" "$save_dir/train_meta.json" "$n_updates" 2>/dev/null || echo "not_done")
        if [ "$done_status" = "done" ]; then
            echo "[$tag] GPU $gpu: SKIP (already done)"
            return
        fi
    fi

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local lock_pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            echo "[$tag] GPU $gpu: SKIP (locked by pid $lock_pid)"
            return
        fi
    fi

    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games scaling_large \
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
        --sweep_name "overnight_scaling_large_v7" \
        >"$log" 2>&1
    echo "[$tag] GPU $gpu: scaling_large(hid=$hid) exit=$?"
}

# Launch both waits in parallel — each waits for its own GPU's h512 to exit
run_on_gpu 0 128 80000  phase5a small     &
pid_0=$!
run_on_gpu 1 256 100000 phase5b scaling_6 &
pid_1=$!

wait $pid_0
wait $pid_1
echo "[master] scaling_large v7 runs finished."

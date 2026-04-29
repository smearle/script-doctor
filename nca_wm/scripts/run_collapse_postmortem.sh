#!/usr/bin/env bash
# Collapse post-mortem experiments.
#
# GPU 0: scaling_14 @ hid=256 (binary-search midpoint between 9 and 19 games)
# GPU 1: scaling_large @ hid=512 (capacity ceiling test — does more width
#        escape the 19-game collapse?)
#
# Both run in parallel. Staggered by 30s to avoid dataset-load collision.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

run_one() {
    local gpu=$1 games=$2 hid=$3 n_updates=$4 tag=$5
    local log="$LOGDIR/collapse_postmortem_${games}_h${hid}_gpu${gpu}.log"
    local pool_tag="_ap_ac_gp"
    local save_dir="$REPO/nca_wm/logs/multi_${games}_cond_bal${pool_tag}_level-None_nca-4_hid-${hid}_lr-0.001_pat-80_s-0"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP $games@h$hid — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting $games @ hid=$hid on GPU $gpu"
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
        --sweep_name "collapse_postmortem" \
        >"$log" 2>&1 &
    local pid=$!
    echo "[$tag] pid=$pid -> $log"
    wait "$pid"
    echo "[$tag] finished $games@h$hid exit=$?"
}

# GPU 0: binary-search at 14 games
run_one 0 scaling_14 256 80000 gpu0_scaling_14 &
gpu0=$!

# Stagger GPU 1 by 30s so dataset loads don't collide on JAX compile
sleep 30

# GPU 1: capacity ceiling — does h=512 escape the 19-game collapse?
run_one 1 scaling_large 512 100000 gpu1_scaling_large_h512 &
gpu1=$!

wait $gpu0
wait $gpu1
echo "[master] post-mortem runs finished."

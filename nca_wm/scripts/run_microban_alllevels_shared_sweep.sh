#!/usr/bin/env bash
# Microban all-10-levels shared-weights depth sweep.
#
# Microban L0 turned out to be too easy — shared n=2 hit 0 wrong tiles on
# every L0 rollout (random/random_tf/bfs/astar), so depth was invisible
# there. This sweep trains on all 10 levels jointly so the model has to
# generalize across longer push-chain configurations and varied geometry,
# which is the real depth-helps-looping test for Microban.
set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_microban_arch
mkdir -p "$LOGDIR"

run_one() {
    local n_steps=$1
    local tag=$2
    local save_dir="$LOGDIR/microban_alllevels_n${n_steps}_shared_seed0"
    local log="$LOGDIR/microban_alllevels_n${n_steps}_shared_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting all-levels n=$n_steps shared"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Microban \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps "$n_steps" --n_nca_repeats "$n_steps" \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 100000 --search_timeout_ms 60000 \
        --max_transitions_per_game 100000 \
        --n_updates 15000 --patience 200 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 200 --ckpt_interval 1000 \
        --save_dir "$save_dir" \
        --seed 0 \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

run_one 2  "shared_n2"
run_one 4  "shared_n4"
run_one 8  "shared_n8"
run_one 16 "shared_n16"

echo "[master] microban all-levels shared sweep done."

#!/usr/bin/env bash
# E3 — multi-game shared-weights validation on scaling_large.
#
# Tests Q3 in ARCHITECTURE_REPORT: does the shared-weights win on Collapse
# port to the 19-game multi-game regime, where the body has to fit
# multiple distinct rule sets? Two runs at the canonical depth (n=4),
# shared vs per-step, same recipe and budget so the comparison is clean.
#
# This is much longer than the single-game sweeps — multi-game runs
# typically take several hours. Designed to launch overnight or after
# the single-game sweeps finish.
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_scaling_arch
mkdir -p "$LOGDIR"

run_one() {
    local n_repeats=$1   # 1 = per-step, 4 = fully shared at n_steps=4
    local tag=$2
    local save_dir="$LOGDIR/scaling_large_n4_${tag}_seed0"
    local log="$LOGDIR/scaling_large_n4_${tag}_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting scaling_large n_nca_repeats=$n_repeats"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games scaling_large \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps 4 --n_nca_repeats "$n_repeats" --n_slots 16 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling --kernel_sep \
        --token_decoder_loss_weight 0.1 \
        --n_search_steps 100000 --search_timeout_ms 60000 \
        --max_transitions_per_game 200000 \
        --n_updates 30000 --patience 300 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 500 --ckpt_interval 2000 \
        --save_dir "$save_dir" \
        --seed 0 \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

run_one 4 "shared"   # n_nca_steps=4, n_nca_repeats=4 → fully shared
run_one 1 "perstep"  # n_nca_steps=4, n_nca_repeats=1 → original per-step

echo "[master] scaling_large shared validation done."

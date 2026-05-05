#!/usr/bin/env bash
# v5: push the 5-sizes recipe further. Best v4 5-sizes config gave
# 0.31% BFS / 0.10% TF / 0.86% AR / 0.58% holdout / 0.39% OOD. Try:
#   5sizes_n512  : double the data (n=512), same compute
#   5sizes_30k   : same data (n=256), 2x compute (30k updates)
#
# Best arch from v1: d=16/per-step/pool=ON, h=256, batch=16, lr=3e-4.
# Multi-grid: {5x5, 6x6, 7x7, 8x7, 8x8} (incl authored 8x7 size).

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_neko_arch
mkdir -p "$LOGDIR"

run_one() {
    local gpu=$1
    local n_levels=$2
    local n_updates=$3
    local tag=$4
    local save_dir="$LOGDIR/${tag}"
    local log="$LOGDIR/${tag}.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running pid=$pid"; return
        fi
    fi
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/train_meta.json" ]; then
        echo "[$tag] SKIP — already complete"; return
    fi

    echo "[$tag] gpu=$gpu starting (n=$n_levels, updates=$n_updates)"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games nekopuzzle \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps 16 --n_nca_repeats 1 \
        --n_slots 16 --n_app_slots 1 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels "$n_levels" --synthetic_multi_grid \
        --synthetic_grid_sizes "5x5,6x6,7x7,8x7,8x8" \
        --synthetic_min_states 5 \
        --synthetic_max_iters_search 5000 \
        --synthetic_timeout_ms_search 2000 \
        --synthetic_mode tile_pattern_empirical \
        --max_transitions_per_game 200000 \
        --n_updates "$n_updates" --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 500 --ckpt_interval 5000 \
        --save_dir "$save_dir" \
        --seed 0 \
        > "$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

(
    run_one 0 512 15000 "neko_d16_pool_perstep_tpe_n512_5sizes_s0"
    echo "[gpu0] done"
) &
PA=$!

(
    run_one 1 256 30000 "neko_d16_pool_perstep_tpe_n256_5sizes_30k_s0"
    echo "[gpu1] done"
) &
PB=$!

wait "$PA"; wait "$PB"
echo "[master] v5 done."

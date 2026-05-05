#!/usr/bin/env bash
# v3: push synth volume further now that v2 showed n=64 → n=256 cuts BFS
# authored 3.15% → 1.92%. Test:
#   - tpe n=512 (more data, same compute)
#   - tpe n=256 + 30k updates (same data, 2x compute)
#
# Best arch from v1: d=16 / per-step / pool=ON, h=256, batch=16, lr=3e-4.

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

    echo "[$tag] gpu=$gpu starting (tpe, n=$n_levels, updates=$n_updates)"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games nekopuzzle \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps 16 --n_nca_repeats 1 \
        --n_slots 16 --n_app_slots 1 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels "$n_levels" --synthetic_multi_grid \
        --synthetic_grid_sizes "5x5,6x6,7x7,8x8" \
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
    run_one 1 512 15000 "neko_d16_pool_perstep_tpe_n512_s0"
    echo "[gpu1] queue done"
) &
PID_A=$!

(
    # Wait for GPU 0 eval to finish first
    while pgrep -f reeval_neko_arch_sweep > /dev/null 2>&1; do sleep 30; done
    run_one 0 256 30000 "neko_d16_pool_perstep_tpe_n256_30k_s0"
    echo "[gpu0] queue done"
) &
PID_B=$!

wait "$PID_A"; wait "$PID_B"
echo "[master] v3 done."

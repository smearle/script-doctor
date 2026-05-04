#!/usr/bin/env bash
# Just n=24 and n=32 (n=8, n=16, n=20 already done).
set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch

run_one() {
    local n=$1
    local save_dir="$LOGDIR/varislide_synth_pool_n${n}_seed0"
    local log="$LOGDIR/varislide_synth_pool_n${n}_seed0.out"
    rm -f "$save_dir/RUNNING.pid"
    echo "[$n] starting"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games varislide \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps "$n" --n_nca_repeats "$n" --n_slots 16 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels 64 --synthetic_multi_grid \
        --synthetic_grid_sizes "6x3,8x3,10x3,12x3,16x3" \
        --no-synthetic_require_solvable \
        --synthetic_min_states 5 \
        --n_search_steps 50000 --search_timeout_ms 60000 \
        --max_transitions_per_game 200000 \
        --n_updates 50000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 500 --ckpt_interval 5000 \
        --save_dir "$save_dir" \
        --seed 0 \
        >"$log" 2>&1
    echo "[$n] finished exit=$? -> $save_dir"
}

run_one 24
run_one 32
echo "[master] done."

#!/usr/bin/env bash
# Synthetic varislide sweep: do diverse levels force the model to learn the
# slide rule (vs memorizing per-state)?
#
# Three phases:
#   1. Generate ~64 synthetic varislide levels at each of widths {6, 8, 10, 12, 16}
#      (heights fixed at 3). Same engine, same rule, varied wall layouts.
#   2. Train with synthetic + authored. Also disable adaptive halt for v1
#      (use `none` mode — single readout) so we get the cleanest dynamics fit.
#   3. Two trains: pool-on (does it still memorize?) and pool-off (does it
#      finally need iterative computation?).
#
# After training, eval at held-out widths (20, 24) tests size generalization.
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch
mkdir -p "$LOGDIR"

run_one() {
    local pool_flags=$1
    local tag=$2
    local save_dir="$LOGDIR/varislide_synth_${tag}_seed0"
    local log="$LOGDIR/varislide_synth_${tag}_seed0.out"
    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running"; return
        fi
    fi
    echo "[$tag] starting ($pool_flags)"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games varislide \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
        $pool_flags \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels 64 --synthetic_multi_grid \
        --synthetic_grid_sizes "6x3,8x3,10x3,12x3,16x3" \
        --no-synthetic_require_solvable \
        --synthetic_min_states 5 \
        --n_search_steps 50000 --search_timeout_ms 60000 \
        --max_transitions_per_game 100000 \
        --n_updates 10000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 200 --ckpt_interval 1000 \
        --save_dir "$save_dir" \
        --seed 0 \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

run_one "--axis_pool --axis_cummax --global_pool"            "pool"
run_one "--no-axis_pool --no-axis_cummax --no-global_pool"   "nopool"

echo "[master] varislide synth sweep done."

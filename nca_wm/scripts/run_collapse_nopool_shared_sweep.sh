#!/usr/bin/env bash
# Collapse no-pool ablation under shared weights.
#
# The original no-pool runs (run_collapse_depth_sweep.sh) used per-step
# weights and showed identity-collapse: even at n=16+stab the un-pooled
# model couldn't escape ~30-40% change_err. The "Open question" in
# ARCHITECTURE_REPORT was: does shared-weights change that? Two
# hypotheses worth distinguishing:
#
#   (a) Pool features carry information no NCA depth can recover, full stop.
#       In which case no-pool shared also fails.
#   (b) The per-step body's identity-collapse was an *optimization* problem
#       caused by per-step over-parameterization, and the shared inductive
#       bias makes the no-pool case trainable.
#
# This sweep settles it. Two runs only — n=4 (matched to the per-step
# no-pool baseline) and n=16 (matched to the per-step no-pool-stab
# point), both with `--shared_weights` and no LN/input_skip patch (per
# the Collapse shared sweep finding that the patch is anti-helpful for
# shared bodies).
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_collapse_arch
mkdir -p "$LOGDIR"

run_one() {
    local n_steps=$1
    local tag=$2
    local save_dir="$LOGDIR/collapse_L0_n${n_steps}_nopool_shared_seed0"
    local log="$LOGDIR/collapse_L0_n${n_steps}_nopool_shared_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting NO-POOL SHARED n=$n_steps"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Collapse --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps "$n_steps" --n_nca_repeats "$n_steps" \
        --no-axis_pool --no-axis_cummax --no-global_pool \
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

run_one 4  "nopool_shared_n4"
run_one 16 "nopool_shared_n16"

echo "[master] collapse no-pool shared sweep done."

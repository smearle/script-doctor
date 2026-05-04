#!/usr/bin/env bash
# Collapse depth + skip-connection ablation.
#
# Tests two questions documented in nca_wm/ARCHITECTURE_REPORT.md:
#   1. Does increasing n_nca_steps help on a game with looping dynamics
#      (Collapse: gravity + hover-until-wall via `again` rules)?
#   2. Does the LayerNorm + input-skip stability patch unlock deeper unrolls
#      (n_steps=16) that fail under bare residuals?
#
# Single-GPU serial sweep (each run is small — Collapse level 0 is 10x48).
# Reuses the cached dataset across runs.
set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_collapse_arch
mkdir -p "$LOGDIR"

run_one() {
    local n_steps=$1
    local stab=$2     # "" or "stab"
    local extra=$3    # extra flags string
    local tag=$4
    local save_dir="$LOGDIR/collapse_L0_n${n_steps}${stab:+_$stab}_seed0"
    local log="$LOGDIR/collapse_L0_n${n_steps}${stab:+_$stab}_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting n_nca_steps=$n_steps $stab"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Collapse --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps "$n_steps" \
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
        $extra \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

# Depth sweep with stock rule_attn (no stability patch). This is the
# "depth-vs-instability" curve. n=4 is covered by collapse_L0_n4_baseline_seed0
# and not repeated here.
run_one 2  ""    ""                              "stock_n2"
run_one 8  ""    ""                              "stock_n8"
run_one 16 ""    ""                              "stock_n16"

# Re-run at the deepest depth with the stability patch on (LayerNorm +
# input-skip). If the bare-residual stack diverged at n=16, this should
# stabilize it. Compare end-of-train loss vs stock_n16.
run_one 16 "stab" "--use_layernorm --input_skip"  "stab_n16"

# Pool-ablation control: does removing the global-pool stack expose a
# "depth helps" effect on Collapse? If n=4-no-pool fails but n=16-no-pool
# (or n=16-no-pool-stab) succeeds, that's evidence that depth is doing
# real work for the looping dynamics — just hidden by pooling at default
# settings.
run_one_nopool() {
    local n_steps=$1
    local stab=$2
    local extra=$3
    local tag=$4
    local save_dir="$LOGDIR/collapse_L0_n${n_steps}_nopool${stab:+_$stab}_seed0"
    local log="$LOGDIR/collapse_L0_n${n_steps}_nopool${stab:+_$stab}_seed0.out"
    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi
    echo "[$tag] starting NO-POOL n=$n_steps $stab"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Collapse --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps "$n_steps" \
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
        $extra \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

run_one_nopool 4  ""    ""                              "nopool_n4"
run_one_nopool 16 "stab" "--use_layernorm --input_skip"  "nopool_n16_stab"

echo "[master] sweep done."

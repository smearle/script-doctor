#!/usr/bin/env bash
# Uniform-halt training on Collapse-L0 + comparison to learned-halt baseline.
#
# Tests the convergence-based halting alternative (Q2 / E6c). Three runs:
#   - learned: existing PonderNet-style with halt head (baseline)
#   - uniform: same per-step machinery but mean-over-k loss, halt_kl_weight=0
#   - none:    no per-step machinery (final-step loss only — back-compat)
#
# All three use the same architecture (n_steps=8, n_repeats=8 fully shared,
# h=128). The uniform variant should make the body's readout good at
# *every* step, which is the prerequisite for inference-time convergence
# stopping (||y_k - y_{k-1}|| < eps).
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch
mkdir -p "$LOGDIR"

run_one() {
    local mode=$1   # "learned", "uniform", "none"
    local extra=$2
    local tag=$3
    local save_dir="$LOGDIR/halt_collapse_${tag}_seed0"
    local log="$LOGDIR/halt_collapse_${tag}_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting halt mode=$mode"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Collapse --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps 8 --n_nca_repeats 8 --n_slots 16 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 100000 --search_timeout_ms 60000 \
        --max_transitions_per_game 100000 \
        --n_updates 5000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 200 --ckpt_interval 1000 \
        --save_dir "$save_dir" \
        --seed 0 \
        $extra \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

# Ponder/learned baseline (re-run at h=128 for fair comparison).
run_one learned "--adaptive_halt --halt_prior_p 0.2 --halt_kl_weight 0.01 --halt_mode ponder" "learned"

# Uniform-halt mode — every step's readout matters.
run_one uniform "--adaptive_halt --halt_kl_weight 0.0 --halt_mode uniform" "uniform"

# Plain final-step loss for back-compat (no adaptive halt).
run_one none "" "none"

echo "[master] uniform halt comparison sweep done."

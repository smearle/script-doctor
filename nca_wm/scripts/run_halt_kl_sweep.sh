#!/usr/bin/env bash
# Adaptive-halt KL-weight sweep on Collapse-L0 — fast Q2 probe.
#
# Tests how sensitive the learned halt distribution is to the
# KL-to-geometric-prior weight. Three runs varying halt_kl_weight over
# {0, 1e-2, 1e-1} (rest of the recipe matched to the F2 shared sweep
# at smaller scale). Goals:
#
# 1. With kl_weight=0, the prior has no effect — does the halt
#    distribution still concentrate, or stay flat / collapse to step 1?
# 2. With kl_weight=1e-2 (default), how close to the prior does it land?
#    Smoke-test (300 updates, h=64) already saw E[k]=2.56 vs prior 4.16.
# 3. With kl_weight=1e-1, the prior dominates — distribution should be
#    near-geometric.
#
# halt_prior_p=0.2 fixed (smoke value). n_steps=8, n_repeats=8 (fully
# shared, the all-shared regime adaptive halt requires).
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch
mkdir -p "$LOGDIR"

run_one() {
    local kl=$1
    local tag=$2
    local save_dir="$LOGDIR/halt_kl${kl}_seed0"
    local log="$LOGDIR/halt_kl${kl}_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting halt_kl_weight=$kl"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Collapse --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps 8 --n_nca_repeats 8 --n_slots 16 \
        --adaptive_halt --halt_prior_p 0.2 --halt_kl_weight "$kl" \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 100000 --search_timeout_ms 60000 \
        --max_transitions_per_game 100000 \
        --n_updates 3000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 200 --ckpt_interval 1000 \
        --save_dir "$save_dir" \
        --seed 0 \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

run_one 0     "kl0"
run_one 0.01  "kl1e-2"
run_one 0.1   "kl1e-1"

echo "[master] halt-kl sweep done."

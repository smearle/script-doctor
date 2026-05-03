#!/usr/bin/env bash
# Companion to run_bouncers_lr_sweep.sh — same six (L, R) configs on
# Bouncers but with all three global-context flags OFF (--no-axis_pool
# --no-axis_cummax --no-global_pool). Tests whether the apparent ease
# of the pool-on sweep was because axis_cummax / pool short-circuits
# `again`-rule propagation that would otherwise need iteration.
#
# Same recipe (h=256, batch=16, 15k steps, lr=3e-4) as the pool-on
# sweep so configs compare line-for-line. Logdirs use a `nopool`
# suffix; the plot script auto-discovers both variants.
set -e
cd /home/jupyter-smearle/script-doctor
GAME=Bouncers
GPU=${GPU:-1}

run() {
    local L=$1 R=$2
    local NS=$((L * R))
    local tag="${GAME}_nopool_L${L}_R${R}"
    local logdir="nca_wm/logs/single_${tag}"
    local logfile="/tmp/${tag}.log"
    if [[ -f "$logdir/eval_multigame.npz" ]]; then
        echo "[skip] $tag — already done ($logdir)"
        return
    fi
    echo "=== $tag : n_steps=$NS, n_repeats=$R, n_layers=$L (NO POOL) ==="
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python3 -u nca_wm/train.py \
        --games $GAME \
        --architecture rule_attn \
        --conditional \
        --no-axis_pool \
        --no-axis_cummax \
        --no-global_pool \
        --n_nca_steps $NS \
        --n_nca_repeats $R \
        --n_hid 256 \
        --n_slots 16 \
        --batch_size 16 \
        --n_updates 15000 \
        --lr 3e-4 \
        --grad_clip 0.5 \
        --change_loss_weight 5.0 \
        --patience 2000 \
        --search_timeout_ms 60000 \
        --log_interval 250 \
        --ckpt_interval 2500 \
        --save_dir "$logdir" \
        > "$logfile" 2>&1
    echo "  done — $logfile"
}

run 8 1
run 4 2
run 2 4
run 1 8
run 4 1
run 1 4

# Deep: same depth ladder as the pool variant. Without pool the baseline
# train loss is 3 orders of magnitude worse, so depth matters most here.
run 2 8    # total=16, engine-faithful
run 1 16   # total=16, max-shared
run 4 4    # total=16, balanced
run 2 16   # total=32, engine-faithful
run 1 32   # total=32, max-shared
run 1 64   # total=64, max-shared deep

echo "All Bouncers nopool (L,R) configs done."

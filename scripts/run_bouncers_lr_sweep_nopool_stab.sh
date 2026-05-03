#!/usr/bin/env bash
# Targeted ablation: add LN + input_skip to the no-pool deep configs that
# degraded on the bare sweep. Tests whether the architecture report's
# "stab patch off for shared weights" finding (from Collapse-with-pool)
# generalizes to Bouncers-without-pool, or was Collapse-specific.
#
# We test three deep no-pool configs that regressed below total=8:
#   - (L=2, R=8)   — total=16, was 1.90% BFS (vs (L=2,R=4) total=8 = 0.98%)
#   - (L=1, R=16)  — total=16, was 3.59%
#   - (L=2, R=16)  — total=32, expected to degrade further
#   - (L=1, R=32)  — total=32, expected to degrade further
# Plus one per-step deep config where the report says stab SHOULD help:
#   - (L=16, R=1)  — total=16, per-step deep, expected: stab helps a lot
#
# Runs at h=256, batch=16, 15k steps (matches the bare nopool sweep).
set -e
cd /home/jupyter-smearle/script-doctor
GAME=Bouncers
GPU=${GPU:-1}

run() {
    local L=$1 R=$2
    local NS=$((L * R))
    local tag="${GAME}_nopool_stab_L${L}_R${R}"
    local logdir="nca_wm/logs/single_${tag}"
    local logfile="/tmp/${tag}.log"
    if [[ -f "$logdir/eval_multigame.npz" ]]; then
        echo "[skip] $tag — already done ($logdir)"
        return
    fi
    echo "=== $tag : n_steps=$NS, n_repeats=$R, n_layers=$L (NO POOL + STAB) ==="
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python3 -u nca_wm/train.py \
        --games $GAME \
        --architecture rule_attn \
        --conditional \
        --no-axis_pool \
        --no-axis_cummax \
        --no-global_pool \
        --use_layernorm \
        --input_skip \
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

run 2 8     # total=16, shared block — was 1.90% bare
run 1 16    # total=16, max-shared — was 3.59% bare
run 2 16    # total=32, shared block
run 1 32    # total=32, max-shared
run 16 1    # total=16, per-step deep — report says stab should help

echo "All Bouncers nopool+stab (L,R) configs done."

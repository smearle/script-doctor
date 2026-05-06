#!/usr/bin/env bash
# Companion to run_scaling_14_mask_v1.sh — same recipe + mask_hidden but
# n_nca_repeats=1 (per-step weights, matching scaling_14_v3recipe's body).
# 3-way comparison:
#   v3recipe (existing): per-step,    no  mask, 150k → 2/5 heldout > id
#   mask_v1     (running): fully-shared, mask, 80k  → ?
#   mask_v2     (this):   per-step,    mask, 80k  → ?
# v2 vs v3recipe at matched 80k cell shows mask_hidden effect alone.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
SAVE_DIR=$REPO/nca_wm/logs/multi_scaling_14_mask_v2_perstep
LOG=$REPO/nca_wm/logs/multi_scaling_14_mask_v2_perstep.log
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

GPU=${1:-1}
SEED=${2:-0}

if [ -f "$SAVE_DIR/train_meta.json" ]; then
    echo "[skip] $SAVE_DIR already has train_meta.json"
else
    echo "[start] scaling_14_mask_v2_perstep on GPU $GPU seed=$SEED"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games scaling_14 \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps 8 --n_nca_repeats 1 \
        --n_slots 16 --n_app_slots 1 \
        --axis_pool --axis_cummax --global_pool \
        \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 100000 --search_timeout_ms 60000 \
        --max_transitions_per_game 200000 \
        --n_updates 80000 --patience 0 --min_delta 1e-7 \
        --batch_size 32 --lr 3e-4 \
        --log_interval 2000 --ckpt_interval 10000 \
        --save_dir "$SAVE_DIR" \
        --seed "$SEED" \
        > "$LOG" 2>&1
    echo "[done] scaling_14_mask_v2_perstep (exit=$?)"
fi

HELDOUT_LOG=$REPO/nca_wm/logs/multi_scaling_14_mask_v2_perstep.heldout.log
if [ -f "$SAVE_DIR/heldout_transfer_v1/results.json" ]; then
    echo "[skip] heldout already done"
else
    echo "[heldout-eval] scaling_14_mask_v2_perstep"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/heldout_eval.py" \
        --load "$SAVE_DIR" \
        --n_random_episodes 5 \
        --max_steps 30 \
        --include_train_sample 1 \
        --out_subdir heldout_transfer_v1 \
        > "$HELDOUT_LOG" 2>&1
    echo "[heldout-done] (exit=$?)"
fi

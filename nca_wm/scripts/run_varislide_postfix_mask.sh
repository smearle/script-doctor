#!/usr/bin/env bash
# Same setup as run_varislide_postfix.sh but with --mask_hidden and
# --mask_padded_loss to compare the effect of masking padding (both in the
# hidden-state propagation and in the loss/metric reduction).
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_canary
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

SEED=${1:-0}
GPU=${2:-1}
TAG="postfix_mask_h128_d16_s${SEED}"
SAVE_DIR="$LOGDIR/varislide_$TAG"
LOG="$LOGDIR/varislide_$TAG.out"

echo "[start] $TAG (gpu=$GPU)"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games varislide \
    --conditional --architecture rule_attn \
    --n_hid 128 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
    --axis_pool --axis_cummax --global_pool \
    --mask_hidden --mask_padded_loss \
    --change_loss_weight 5.0 --grad_clip 0.5 \
    --balanced_sampling \
    --synthetic_levels 64 --synthetic_multi_grid \
    --synthetic_grid_sizes "6x3,8x3,10x3,12x3,16x3" \
    --no-synthetic_require_solvable \
    --synthetic_min_states 5 \
    --synthetic_max_iters_search 5000 --synthetic_timeout_ms_search 2000 \
    --max_transitions_per_game 200000 \
    --n_updates 50000 --patience 0 --min_delta 1e-6 \
    --batch_size 16 --lr 3e-4 \
    --log_interval 1000 --ckpt_interval 10000 \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED" \
    > "$LOG" 2>&1
echo "[done] $TAG (exit=$?)"

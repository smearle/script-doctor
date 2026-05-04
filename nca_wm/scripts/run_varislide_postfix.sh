#!/usr/bin/env bash
# Re-test multi-grid varislide synth after the May 4 bit-pack fix (1fa557d).
# Same setup as run_varislide_long_seedsweep.sh canary baseline (h=128,
# n_steps=16, n_repeats=16, 50k updates) so results compare directly to the
# pre-fix runs in nca_wm/logs_canary/varislide_long50k_d16_s*.
#
# gdrtodd is on GPU 1 — pin to GPU 0.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_canary
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

SEED=${1:-0}
TAG="postfix_h128_d16_s${SEED}"
SAVE_DIR="$LOGDIR/varislide_$TAG"
LOG="$LOGDIR/varislide_$TAG.out"

echo "[start] $TAG (gpu=0)"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games varislide \
    --conditional --architecture rule_attn \
    --n_hid 128 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
    --axis_pool --axis_cummax --global_pool \
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

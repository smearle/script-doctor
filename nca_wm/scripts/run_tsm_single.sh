#!/usr/bin/env bash
# Single-game training on Travelling_salesman with the v3 recipe.
# Tests whether TSM is learnable in isolation. If it converges to low
# change_err here but stays high in multi-game training, the bottleneck is
# data dilution / batch sampling, not mechanics complexity.
#
# Recipe matches scaling_gallery_v3: rule_attn, n_hid=256, n_slots=16,
# d_slot=64, n_nca_steps=8, batch=32, lr=3e-4, mask_hidden=True,
# change_loss_weight=5.0. Reduced n_updates to 30k since we expect this to
# converge fast on a single game.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nca_wm/scripts/run_tsm_single.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG=/tmp/tsm_single.log
SAVE_DIR=nca_wm/logs/tsm_single_v3recipe

echo "Launching TSM single-game training → $SAVE_DIR (log $LOG)"

exec .venv/bin/python3 -m nca_wm.train \
    --game Travelling_salesman \
    --architecture rule_attn \
    --n_hid 256 \
    --n_slots 16 \
    --d_slot 64 \
    --n_app_slots 1 \
    --n_enc_layers 2 \
    --n_heads 4 \
    --d_model 64 \
    --d_z 64 \
    --n_nca_steps 8 \
    --batch_size 32 \
    --lr 3e-4 \
    --lr_schedule cosine \
    --lr_min 1e-7 \
    --n_updates 30000 \
    --patience 4000 \
    --change_loss_weight 5.0 \
    --balanced_sampling \
    --max_transitions_per_game 200000 \
    --search_timeout_ms 60000 \
    --n_search_steps 100000 \
    --mask_hidden \
    --conditional \
    --grad_clip 0.5 \
    --ckpt_interval 2500 \
    --log_interval 250 \
    --save_dir "$SAVE_DIR" \
    --seed 0 \
    >> "$LOG" 2>&1

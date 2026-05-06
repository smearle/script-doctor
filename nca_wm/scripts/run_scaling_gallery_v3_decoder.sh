#!/usr/bin/env bash
# scaling_gallery_v3 + joint token decoder. Same 94-game preset and
# recipe as run_scaling_gallery_v3.sh; the only added flag is
# --token_decoder_loss_weight 1.0 (joint encoder / NCA / SlotTokenDecoder
# training). A/B against multi_scaling_gallery_v3 to measure:
#   1. Does the decoder loss hurt prediction quality (state_loss, change_err)?
#   2. Does it tighten heldout latent geometry (1-NN cos dist)?
#   3. Does it improve unseen-game AR rollout vs identity?
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nca_wm/scripts/run_scaling_gallery_v3_decoder.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG=/tmp/scaling_gallery_v3_decoder.log
SAVE_DIR=nca_wm/logs/multi_scaling_gallery_v3_decoder

echo "Launching scaling_gallery_v3 + decoder → $SAVE_DIR (log $LOG)"

exec .venv/bin/python3 -m nca_wm.train \
    --games scaling_gallery_v3 \
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
    --n_updates 150000 \
    --patience 4000 \
    --change_loss_weight 5.0 \
    --balanced_sampling \
    --max_transitions_per_game 200000 \
    --search_timeout_ms 60000 \
    --n_search_steps 100000 \
    \
    --conditional \
    --grad_clip 0.5 \
    --ckpt_interval 2500 \
    --log_interval 250 \
    --token_decoder_loss_weight 1.0 \
    --decoder_d_model 128 \
    --decoder_n_layers 4 \
    --decoder_n_heads 4 \
    --save_dir "$SAVE_DIR" \
    --seed 0 \
    >> "$LOG" 2>&1

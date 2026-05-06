#!/usr/bin/env bash
# Launch scaling_gallery_v3 training (97 games = deduped gallery_v2 + 40
# stratified additions from scraped/increpare). Recipe matches v3_combined
# (rule_attn, n_hid=256, n_slots=16, d_slot=64, n_app_slots=1, n_nca=8,
# batch=32, lr=3e-4, 150k steps, change_loss_weight=5.0) so the only varied
# axis vs v3_combined is the dataset (97g vs 59g). One difference:
# mask_hidden=True (current default; v3_combined predates the default flip).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 nca_wm/scripts/run_scaling_gallery_v3.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG=/tmp/scaling_gallery_v3.log
SAVE_DIR=nca_wm/logs/multi_scaling_gallery_v3

echo "Launching scaling_gallery_v3 → $SAVE_DIR  (log: $LOG)"

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
    --save_dir "$SAVE_DIR" \
    --seed 0 \
    >> "$LOG" 2>&1

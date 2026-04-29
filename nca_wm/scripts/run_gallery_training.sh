#!/usr/bin/env bash
# Launch full-gallery NCA world model training.
#
# Recipe replicates cosine_v2 (the proven 19-game scaling_large run that hit
# 0.27% smoothed change_err and was still descending at end of training):
#   rule_attn architecture, K=16, d_slot=64, h=256, nca_steps=4
#   lr=3e-4 cosine -> 1e-7 over n_updates
#   grad_clip=0.5, change_loss_weight=5
#   batch_size=256 (gallery has 122 games, want >=2 samples/game on avg)
#   max_transitions_per_game=200000 (now affordable thanks to compressed cache)
#   patience=400 (loose — gallery convergence will be slower)
#   n_updates=400000 (cosine_v2 was still actively learning at 200K)
#
# Sweep-name knob lets us run multiple variants without colliding save_dirs.
#
# Usage:
#   GPU=0 nca_wm/scripts/run_gallery_training.sh [--sweep_name <tag>]

set -eu

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

GPU=${GPU:-0}
SWEEP_NAME=${SWEEP_NAME:-gallery_full}
LOG="$LOGDIR/gallery_rule_attn_k16_h256_${SWEEP_NAME}_gpu${GPU}.log"

cd "$REPO"
echo "[$(date)] Launching gallery training on GPU $GPU. Log: $LOG"

CUDA_VISIBLE_DEVICES=$GPU \
nohup "$PY" -u -m nca_wm.train \
    --games gallery \
    --conditional \
    --balanced_sampling \
    --axis_pool --axis_cummax --global_pool \
    --architecture rule_attn \
    --n_slots 16 \
    --d_slot 64 \
    --n_hid 256 \
    --n_nca_steps 4 \
    --lr 3e-4 \
    --lr_schedule cosine \
    --lr_min 1e-7 \
    --grad_clip 0.5 \
    --change_loss_weight 5.0 \
    --batch_size 256 \
    --max_transitions_per_game 200000 \
    --patience 400 \
    --min_delta 1e-8 \
    --n_updates 400000 \
    --ckpt_interval 1000 \
    --log_interval 100 \
    --wandb \
    --wandb_project nca-world-model \
    --sweep_name "$SWEEP_NAME" \
    > "$LOG" 2>&1 &

PID=$!
echo "[$(date)] Launched (PID=$PID)"
echo "Tail with: tail -f $LOG"

#!/usr/bin/env bash
# GPU 0 overnight queue.
#
# Refocused: the headline OOD claim (Train-199 conditional reaches
# 2.91 / 2.11 % TF mean / median, beats identity on 11/29 heldout
# games) currently rests on a single seed. A second seed there
# tightens that claim much more than a param-matched uncond run at
# Train-14. Param-matching is deferred — the n_hid that gives ~16M
# total params under the current pool topology is not yet pinned
# down, and the value of resolving it is smaller than confirming
# the headline scaling result is reproducible.
#
# Schedule:
#   1. Train-199 cond s1 (~5h)         — variance on the headline OOD claim
#   2. Train-199 cond s1 heldout       — fills Table 6 row
#   3. (slack) Train-14 uncond s3 OR a param-count sweep, depending
#      on remaining budget; this is queued but the script will skip
#      gracefully if eval data already exists.
#
# Usage:
#   nohup nca_wm/scripts/run_overnight_paper_variance_gpu0.sh \
#     > /tmp/overnight_paper_variance_gpu0.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=0
HELDOUT_FILE="data/heldout_v4_n30.json"

train() {
    local games=$1
    local cond=$2
    local nhid=$3
    local seed=$4
    local save_dir=$5

    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has eval_multigame.npz"
        return 0
    fi

    local cond_flag
    if [ "$cond" = "cond" ]; then cond_flag="--conditional"; else cond_flag="--no-conditional"; fi

    local extra_args=()
    if [ "$cond" = "cond" ]; then
        extra_args=(
            --n_slots 16 --d_slot 64 --n_app_slots 1
            --n_enc_layers 2 --n_heads 4
            --d_model 64 --d_z 64
            --encode_sprites --use_eos
            --token_decoder_loss_weight 1.0
            --decoder_d_model 128 --decoder_n_layers 4 --decoder_n_heads 4
        )
    fi

    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
        --games "$games" $cond_flag \
        --architecture rule_attn \
        --n_hid "$nhid" \
        --n_nca_steps 8 \
        --batch_size 32 \
        --n_updates 150000 \
        --lr 3e-4 --lr_schedule cosine --lr_min 1e-7 \
        --grad_clip 0.5 \
        --change_loss_weight 5.0 \
        --balanced_sampling \
        --max_transitions_per_game 200000 \
        --search_timeout_ms 60000 --n_search_steps 100000 \
        --input_skip --axis_pool --axis_cummax --global_pool \
        --patience 4000 --ckpt_interval 2500 --log_interval 250 \
        --seed "$seed" --save_dir "$save_dir" \
        "${extra_args[@]}"
}

heldout() {
    local save_dir=$1
    if [ -f "$save_dir/heldout_v4_n30/results.json" ]; then
        echo "[skip] $save_dir already has heldout_v4_n30/results.json"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python3 -m nca_wm.scripts.heldout_rollout_per_game \
        --load "$save_dir" \
        --heldout_file "$HELDOUT_FILE" \
        --max_levels_per_game 2 \
        --n_random_episodes 2 \
        --max_steps 20 \
        --skip_done 2>&1 | tail -30
}

stage() {
    local label=$1
    shift
    echo
    echo "=== [$(date '+%F %T')] $label ==="
    "$@"
    echo "=== [$(date '+%F %T')] $label done ==="
}

#------------------------------------------------------------------
# Run 1: Train-199 cond s1 — variance on the headline OOD claim
stage "Train-199 cond s1" \
    train scaling_gallery_v4 cond 256 1 nca_wm/logs/multi_scaling_gallery_v4_cond_match_s1
stage "Train-199 cond s1 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v4_cond_match_s1

#------------------------------------------------------------------
# Run 2 (slack): Train-14 uncond s3 — extra variance seed if time permits.
# This will silently skip if a previous run already populated the dir.
stage "Train-14 uncond s3 (slack)" \
    train scaling_14 uncond 256 3 nca_wm/logs/multi_scaling_14_uncond_match_s3
stage "Train-14 uncond s3 heldout (slack)" \
    heldout nca_wm/logs/multi_scaling_14_uncond_match_s3

echo
echo "=== gpu0 queue done at $(date '+%F %T') ==="

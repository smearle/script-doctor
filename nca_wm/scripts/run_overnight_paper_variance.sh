#!/usr/bin/env bash
# Sequential queue for paper-strengthening experiments on local GPU 1.
# Each run trains for 150k updates, runs in-distribution eval at end
# (eval_multigame.npz auto-saved), then runs the heldout AR rollout
# eval that populates heldout_v4_n30/results.json.
#
# Focus is variance estimation across seeds for the Train-14 and
# Train-59 cond / uncond cells. Box 210 is queued to handle Train-14
# cond s1 once its current Train-199 uncond run finishes; this script
# fills in the seeds local-side.
#
# Schedule (~8 hours total):
#   1. Train-14 uncond s1 @ n_hid=256       — variance (~1.7h)
#   2. Train-14 uncond s2 @ n_hid=256       — variance (~1.7h)
#   3. Train-14 cond   s2 @ n_hid=256       — variance (~1.7h)
#   4. Train-59 uncond s1 @ n_hid=256       — variance (~3h, partial OK)
#
# Param-matched (n_hid≈288 in current architecture) runs are deferred:
# the current pool topology makes the n_hid=448 "param-matched" recipe
# from the legacy paper text actually 2.5x oversized, so a fresh
# parameter-matched comparison needs an n_hid sweep first.
#
# Usage:
#   nohup nca_wm/scripts/run_overnight_paper_variance.sh \
#     > /tmp/overnight_paper_variance.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=1
HELDOUT_FILE="data/heldout_v4_n30.json"

train() {
    local games=$1
    local cond=$2     # "cond" or "uncond"
    local nhid=$3
    local seed=$4
    local save_dir=$5

    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has eval_multigame.npz; skipping training"
        return 0
    fi

    local cond_flag
    if [ "$cond" = "cond" ]; then
        cond_flag="--conditional"
    else
        cond_flag="--no-conditional"
    fi

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
# Run 1: Train-14 uncond s1 (variance)
stage "Train-14 uncond s1" \
    train scaling_14 uncond 256 1 nca_wm/logs/multi_scaling_14_uncond_match_s1
stage "Train-14 uncond s1 heldout" \
    heldout nca_wm/logs/multi_scaling_14_uncond_match_s1

#------------------------------------------------------------------
# Run 2: Train-14 uncond s2 (variance)
stage "Train-14 uncond s2" \
    train scaling_14 uncond 256 2 nca_wm/logs/multi_scaling_14_uncond_match_s2
stage "Train-14 uncond s2 heldout" \
    heldout nca_wm/logs/multi_scaling_14_uncond_match_s2

#------------------------------------------------------------------
# Run 3: Train-14 cond s2 (variance)
stage "Train-14 cond s2" \
    train scaling_14 cond 256 2 nca_wm/logs/multi_scaling_14_cond_match_s2
stage "Train-14 cond s2 heldout" \
    heldout nca_wm/logs/multi_scaling_14_cond_match_s2

#------------------------------------------------------------------
# Run 4: Train-59 uncond s1 (variance, ~3h — may not finish in remaining budget)
stage "Train-59 uncond s1" \
    train scaling_gallery_v2 uncond 256 1 nca_wm/logs/multi_scaling_gallery_v2_uncond_match_s1
stage "Train-59 uncond s1 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v2_uncond_match_s1

echo
echo "=== queue done at $(date '+%F %T') ==="

#!/usr/bin/env bash
# Direct Train-500 launcher (yalda GPU 1). Replaces the post-deadline
# queue's stages 3+4, which never ran because that queue's bash had
# already loaded an earlier version of the script and exited at the
# old end-of-script before reaching the Train-500 stages.
#
# Polls for GPU 1 free, then runs:
#   1. Train-500 cond s0  (~12-15 h on yalda after data collection)
#   2. Train-500 cond heldout
#   3. Train-500 uncond s0 (~9-12 h)
#   4. Train-500 uncond heldout
#
# Usage:
#   nohup nca_wm/scripts/run_train500.sh \
#     > /tmp/run_train500.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=${GPU:-1}
HELDOUT_FILE="data/heldout_v4_n30.json"
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-3000}

wait_for_gpu() {
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
        if [ -z "$used" ]; then
            echo "[wait] failed to query GPU $GPU; retrying in 60s"; sleep 60; continue
        fi
        if [ "$used" -lt "$GPU_FREE_THRESHOLD_MB" ]; then
            echo "[wait] GPU $GPU has ${used} MiB used (< $GPU_FREE_THRESHOLD_MB); proceeding"
            return 0
        fi
        echo "[wait] GPU $GPU still busy: ${used} MiB used; sleeping 120s"
        sleep 120
    done
}

stage() {
    local label=$1; shift
    echo
    echo "=== [$(date '+%F %T')] $label ==="
    "$@"
    echo "=== [$(date '+%F %T')] $label done ==="
}

train_v5() {
    local cond=$1
    local seed=$2
    local save_dir=$3

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
            --encode_sprites
            --token_decoder_loss_weight 1.0
            --decoder_d_model 128 --decoder_n_layers 4 --decoder_n_heads 4
        )
    fi

    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
        --games scaling_gallery_v5 $cond_flag \
        --architecture rule_attn \
        --n_hid 256 \
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
    local names
    names=$(.venv/bin/python3 -c "
import json
hd = json.load(open('$HELDOUT_FILE'))
print(';'.join(h['name'] for h in hd['heldout']))
")
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python3 -m nca_wm.heldout_eval \
        --load "$save_dir" \
        --heldout_games "$names" \
        --out_subdir heldout_v4_n30 \
        --max_levels_per_game 2 \
        --n_random_episodes 5 \
        --max_steps 30 \
        --include_train_sample 0 2>&1 | tail -40
}

#------------------------------------------------------------------
echo "=== [$(date '+%F %T')] Train-500 launcher start (GPU $GPU) ==="
wait_for_gpu

stage "Train-500 cond s0" \
    train_v5 cond 0 nca_wm/logs/multi_scaling_gallery_v5_cond_match_s0
stage "Train-500 cond s0 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v5_cond_match_s0

stage "Train-500 uncond s0" \
    train_v5 uncond 0 nca_wm/logs/multi_scaling_gallery_v5_uncond_match_s0
stage "Train-500 uncond s0 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v5_uncond_match_s0

echo
echo "=== [$(date '+%F %T')] Train-500 launcher done ==="

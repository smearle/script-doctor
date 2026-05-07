#!/usr/bin/env bash
# Post-deadline variance fill-in queue for GPU 1.
#
# After the deadline run_overnight_paper_variance.sh queue finishes
# (Train-59 uncond s1 heldout currently in flight), this queue fills
# the two highest-leverage variance gaps in the cond/uncond × scale
# grid for Table~\ref{tab:cond-vs-uncond-match} and
# Figure~\ref{fig:intersection-slope}:
#
#   1. Train-59  cond  s1   (~3h)   — companion seed for v2_cond
#   2. Train-199 uncond s1  (~8h)   — companion seed for v4_uncond
#                                     (the headline OOD claim)
#
# It polls until the GPU has < ${GPU_FREE_THRESHOLD_MB} MiB used so it
# does not collide with the still-running v2_uncond_s1 heldout. After
# that it runs strictly sequentially, calling the queue's heldout step
# per run as in run_overnight_paper_variance.sh.
#
# Usage:
#   nohup nca_wm/scripts/run_post_deadline_variance.sh \
#     > /tmp/post_deadline_variance.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=1
HELDOUT_FILE="data/heldout_v4_n30.json"
GPU_FREE_THRESHOLD_MB=3000

wait_for_gpu() {
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
        if [ -z "$used" ]; then
            echo "[wait] failed to query GPU $GPU; retrying in 60s"
            sleep 60
            continue
        fi
        if [ "$used" -lt "$GPU_FREE_THRESHOLD_MB" ]; then
            echo "[wait] GPU $GPU has ${used} MiB used (< $GPU_FREE_THRESHOLD_MB); proceeding"
            return 0
        fi
        echo "[wait] GPU $GPU still busy: ${used} MiB used (need < $GPU_FREE_THRESHOLD_MB); sleeping 120s"
        sleep 120
    done
}

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
            --encode_sprites
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

stage() {
    local label=$1
    shift
    echo
    echo "=== [$(date '+%F %T')] $label ==="
    "$@"
    echo "=== [$(date '+%F %T')] $label done ==="
}

#------------------------------------------------------------------
echo "=== [$(date '+%F %T')] post-deadline queue starting on GPU $GPU ==="
wait_for_gpu

#------------------------------------------------------------------
# Run 1: Train-59 cond s1 — variance on the Train-59 cond cell.
stage "Train-59 cond s1" \
    train scaling_gallery_v2 cond 256 1 nca_wm/logs/multi_scaling_gallery_v2_cond_match_s1
stage "Train-59 cond s1 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v2_cond_match_s1

#------------------------------------------------------------------
# Run 2: Train-199 uncond s1 — variance on the headline OOD claim
# (companion seed to v4_cond_match_s1, which is finishing now on GPU 0).
stage "Train-199 uncond s1" \
    train scaling_gallery_v4 uncond 256 1 nca_wm/logs/multi_scaling_gallery_v4_uncond_match_s1
stage "Train-199 uncond s1 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v4_uncond_match_s1

#------------------------------------------------------------------
# Run 3: Train-199 cond, n_hid=384 — larger-model probe for the
# discussion's "phase transition" hypothesis (does the encoder become
# load-bearing OOD at greater capacity?). All other recipe knobs
# match the canonical n_hid=256 run.
stage "Train-199 cond s0 n_hid=384" \
    train scaling_gallery_v4 cond 384 0 nca_wm/logs/multi_scaling_gallery_v4_cond_match_s0_h384
stage "Train-199 cond s0 n_hid=384 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v4_cond_match_s0_h384

echo
echo "=== [$(date '+%F %T')] post-deadline queue done ==="

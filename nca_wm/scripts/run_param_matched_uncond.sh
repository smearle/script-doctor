#!/usr/bin/env bash
# Yalda GPU 0 queue: parameter-matched uncond sweep, smallest dataset first.
#
# Cond model at n_hid=256 has 16.03M total params; uncond at n_hid=256 only
# has 13.12M because the encoder/decoder/slot stack adds ~2.9M. To run a
# fair "is cond's benefit just extra capacity?" test at each scale, we
# train uncond at n_hid=288 (body params ~n_hid² → ~16.6M total, slightly
# above cond's 16.03M).
#
# Box 210 is already running the Train-199 parameter-matched uncond. This
# queue covers the two smaller scales:
#
#   1. Train-14  uncond s0 n_hid=288  (~4 h)
#   2. Train-59  uncond s0 n_hid=288  (~7 h)
#
# Both pair with the existing Train-X cond_match_s0 runs in the table and
# the existing Train-X uncond_match runs at n_hid=256 (the smaller-uncond
# baseline). Together they let us quote, at each of three scales, a
# triplet: cond@256, uncond@256, uncond@288, and answer "did matching
# uncond's params to cond's close the cond-vs-uncond gap?" cleanly.
#
# Polls GPU 0 free before launching to avoid colliding with whatever the
# overnight queue lands on.
#
# Usage:
#   nohup nca_wm/scripts/run_param_matched_uncond.sh \
#     > /tmp/run_param_matched_uncond.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=0
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
        echo "[wait] GPU $GPU still busy: ${used} MiB used; sleeping 120s"
        sleep 120
    done
}

train_uncond() {
    local games=$1
    local nhid=$2
    local save_dir=$3

    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has eval_multigame.npz"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
        --games "$games" --no-conditional \
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
        --seed 0 --save_dir "$save_dir"
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
echo "=== [$(date '+%F %T')] param-matched uncond queue start (yalda GPU $GPU) ==="
wait_for_gpu

# Run 1: smallest first per request — Train-14 uncond at n_hid=288.
stage "Train-14 uncond s0 n_hid=288 (param-matched)" \
    train_uncond scaling_14 288 nca_wm/logs/multi_scaling_14_uncond_match_s0_h288
stage "Train-14 uncond s0 n_hid=288 heldout" \
    heldout nca_wm/logs/multi_scaling_14_uncond_match_s0_h288

# Run 2: Train-59 uncond at n_hid=288.
stage "Train-59 uncond s0 n_hid=288 (param-matched)" \
    train_uncond scaling_gallery_v2 288 nca_wm/logs/multi_scaling_gallery_v2_uncond_match_s0_h288
stage "Train-59 uncond s0 n_hid=288 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v2_uncond_match_s0_h288

echo
echo "=== [$(date '+%F %T')] param-matched uncond queue done ==="

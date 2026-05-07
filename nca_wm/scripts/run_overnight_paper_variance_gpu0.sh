#!/usr/bin/env bash
# Second overnight queue, runs on GPU 0. Two pieces:
#
# 1. Quick param-count sweep at several n_hid values to identify the
#    n_hid that gives uncond ~16M total params (matching the cond
#    Train-14 / Train-59 / Train-199 cond_match recipe). The n_hid=448
#    "match" cited in the legacy paper was 15M under the old pool
#    topology; under the current pool topology it lands at ~40M, so
#    we need to refind the match.
#
# 2. Once n_hid_match is known, train Train-14 uncond at that width
#    for a clean parameter-matched comparison, then fill in two
#    Train-59 cond variance seeds (s1, s2) for the rest of the budget.
#
# GPU 0 is shared with gdrtodd (488 MiB idle), but the user has
# confirmed the GPUs are ours for tonight.
#
# Usage:
#   nohup nca_wm/scripts/run_overnight_paper_variance_gpu0.sh \
#     > /tmp/overnight_paper_variance_gpu0.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=0
HELDOUT_FILE="data/heldout_v4_n30.json"

# Target total param count. cond_match's NCA body + slot encoder + decoder
# at n_hid=256 lands at ~16M (verified via Train-14, Train-59, Train-199
# cond_match s0 checkpoints).
TARGET_PARAMS=16000000

stage() {
    local label=$1
    shift
    echo
    echo "=== [$(date '+%F %T')] $label ==="
    "$@"
    echo "=== [$(date '+%F %T')] $label done ==="
}

# Quick param count: train --n_updates 1 with smallest data, grep "Model
# params:" line. Returns the param count via stdout.
count_params() {
    local nhid=$1
    local tmp=/tmp/_pcount_n${nhid}
    local log=/tmp/_pcount_n${nhid}.log
    rm -rf "$tmp"
    timeout 240 bash -c "
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
            --games scaling_14 --no-conditional \
            --architecture rule_attn --n_hid $nhid --n_nca_steps 8 --batch_size 32 \
            --n_updates 1 --lr 3e-4 --lr_schedule cosine --lr_min 1e-7 \
            --grad_clip 0.5 --change_loss_weight 5.0 --balanced_sampling \
            --max_transitions_per_game 50000 --search_timeout_ms 60000 --n_search_steps 100000 \
            --input_skip --axis_pool --axis_cummax --global_pool \
            --patience 4000 --ckpt_interval 100 --log_interval 100 \
            --seed 0 --save_dir $tmp
    " > "$log" 2>&1 || true
    local n
    n=$(grep -oE "Model params: [0-9,]+" "$log" | tail -1 | tr -d ',' | awk '{print $3}')
    rm -rf "$tmp" "$log"
    echo "$n"
}

#------------------------------------------------------------------
# Stage 1: sweep n_hid to find the param-matched value
echo "[gpu0] starting param-count sweep at $(date '+%F %T')"
declare -A NPARAMS
for nhid in 288 320 352 384; do
    n=$(count_params "$nhid")
    NPARAMS[$nhid]=$n
    echo "[gpu0] n_hid=$nhid → params=$n"
done

# Pick the n_hid whose param count is closest to TARGET_PARAMS.
best_nhid=288
best_diff=999999999
for nhid in "${!NPARAMS[@]}"; do
    n=${NPARAMS[$nhid]}
    if [ -z "$n" ]; then continue; fi
    diff=$((n - TARGET_PARAMS))
    diff=${diff#-}
    if [ "$diff" -lt "$best_diff" ]; then
        best_diff=$diff
        best_nhid=$nhid
    fi
done

echo "[gpu0] best param-matched n_hid = $best_nhid (params=${NPARAMS[$best_nhid]})"
echo "[gpu0] all candidates:"
for nhid in 288 320 352 384; do
    echo "  n_hid=$nhid params=${NPARAMS[$nhid]:-?}"
done

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

#------------------------------------------------------------------
# Stage 2: param-matched Train-14 uncond
match_dir="nca_wm/logs/multi_scaling_14_uncond_match_n${best_nhid}_s0"
stage "Train-14 uncond n_hid=${best_nhid} s0 (param-matched)" \
    train scaling_14 uncond "$best_nhid" 0 "$match_dir"
stage "Train-14 uncond n_hid=${best_nhid} s0 heldout" \
    heldout "$match_dir"

#------------------------------------------------------------------
# Stage 3: Train-59 cond variance (s1, then s2 if budget allows)
stage "Train-59 cond s1" \
    train scaling_gallery_v2 cond 256 1 nca_wm/logs/multi_scaling_gallery_v2_cond_match_s1
stage "Train-59 cond s1 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v2_cond_match_s1

stage "Train-59 cond s2" \
    train scaling_gallery_v2 cond 256 2 nca_wm/logs/multi_scaling_gallery_v2_cond_match_s2
stage "Train-59 cond s2 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v2_cond_match_s2

echo
echo "=== gpu0 queue done at $(date '+%F %T') ==="

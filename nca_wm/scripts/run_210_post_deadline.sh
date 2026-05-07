#!/usr/bin/env bash
# Box 210 post-deadline queue. Box 210's local logs already contain
# trained-but-not-yet-evaluated checkpoints from a recent variance pass
# (Train-14 cond s2 in particular). Recover those via heldout eval, then
# launch a brand new experiment that complements yalda's queue:
# parameter-matched uncond Train-199 at n_hid=288. The n_hid=288 picks
# the closest body size to cond's total parameter count
# (cond@n_hid=256 = 16.03M; uncond@n_hid=288 ≈ 16.6M, scaling body ∝ n_hid²).
#
# Schedule:
#   1. heldout on multi_scaling_14_cond_match_s2 (~3 min)  -- Train-14 cond N=2
#   2. train  multi_scaling_gallery_v4_uncond_match_s0_h288 (~8 h)
#       — directly tests whether cond's ID benefit is just extra capacity
#   3. heldout on multi_scaling_gallery_v4_uncond_match_s0_h288
#
# Usage (from yalda, via ssh):
#   ssh 210 'cd ~/script-doctor && nohup nca_wm/scripts/run_210_post_deadline.sh \
#     > /tmp/run_210_post_deadline.log 2>&1 &'
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=0
HELDOUT_FILE="data/heldout_v4_n30.json"

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

train_uncond() {
    local nhid=$1
    local save_dir=$2

    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has eval_multigame.npz"
        return 0
    fi

    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
        --games scaling_gallery_v4 --no-conditional \
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

stage() {
    local label=$1
    shift
    echo
    echo "=== [$(date '+%F %T')] $label ==="
    "$@"
    echo "=== [$(date '+%F %T')] $label done ==="
}

#------------------------------------------------------------------
echo "=== [$(date '+%F %T')] 210 post-deadline queue start (GPU $GPU) ==="

# Run 1: recover heldout for the Train-14 cond s2 already trained on 210.
stage "Train-14 cond s2 heldout (recovered)" \
    heldout nca_wm/logs/multi_scaling_14_cond_match_s2

# Run 2 (slack, only if we don't already have results from yalda):
# heldout on the Train-59 cond s1 trained on 210. Yalda's queue is also
# evaluating its own copy; whichever finishes first populates the row.
stage "Train-59 cond s1 heldout (210 copy, slack)" \
    heldout nca_wm/logs/multi_scaling_gallery_v2_cond_match_s1

#------------------------------------------------------------------
# Run 3: parameter-matched uncond Train-199. Train-199 cond at n_hid=256
# is 16.03M params; uncond at n_hid=256 is 13.13M (encoder+decoder add
# ~2.9M). With body params ~n_hid², n_hid=288 lands the uncond at
# ~16.6M, slightly above cond — the more-rigorous "give uncond an unfair
# capacity advantage" version of the param-matched test.
stage "Train-199 uncond s0 n_hid=288 (param-matched)" \
    train_uncond 288 nca_wm/logs/multi_scaling_gallery_v4_uncond_match_s0_h288
stage "Train-199 uncond s0 n_hid=288 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v4_uncond_match_s0_h288

echo
echo "=== [$(date '+%F %T')] 210 post-deadline queue done ==="

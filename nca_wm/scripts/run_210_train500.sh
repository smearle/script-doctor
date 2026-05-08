#!/usr/bin/env bash
# Box 210: Train-500 uncond s0. Pairs with yalda's Train-500 cond s0
# (scheduled as stage 4 of run_post_deadline_variance.sh, ~24h out).
# Running uncond on 210 in parallel halves the wall-clock to a
# Train-500 cond+uncond pair.
#
# This is the FIRST run that exercises v17 cache format end-to-end,
# so it will collect transitions for all 500 games from scratch
# (~hours of CPU work) before training starts. The resulting cache
# is reusable: yalda's later Train-500 cond run will find it and
# skip the collection step IF we rsync the cache over after this
# finishes the data step.
#
# Usage (from yalda):
#   ssh 210 'cd ~/script-doctor && nohup nca_wm/scripts/run_210_train500.sh \
#     > /tmp/run_210_train500.log 2>&1 &'
set -uo pipefail

if [ -s "$HOME/.nvm/nvm.sh" ]; then
    export NVM_DIR="$HOME/.nvm"
    . "$NVM_DIR/nvm.sh" >/dev/null 2>&1 || true
fi
if ! command -v node >/dev/null 2>&1; then
    NODE_BIN=$(ls -d "$HOME"/.nvm/versions/node/*/bin 2>/dev/null | sort -V | tail -1)
    if [ -n "${NODE_BIN:-}" ]; then export PATH="$NODE_BIN:$PATH"; fi
fi
echo "node: $(command -v node 2>/dev/null) ($(node --version 2>/dev/null || echo 'MISSING'))"

cd "$(dirname "$0")/../.."

GPU=0
HELDOUT_FILE="data/heldout_v4_n30.json"

stage() {
    local label=$1; shift
    echo
    echo "=== [$(date '+%F %T')] $label ==="
    "$@"
    echo "=== [$(date '+%F %T')] $label done ==="
}

train_uncond_v500() {
    local save_dir=$1
    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has eval_multigame.npz"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
        --games scaling_gallery_v5 --no-conditional \
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

#------------------------------------------------------------------
echo "=== [$(date '+%F %T')] 210 Train-500 uncond start (GPU $GPU) ==="

stage "Train-500 uncond s0 (collect+train)" \
    train_uncond_v500 nca_wm/logs/multi_scaling_gallery_v5_uncond_match_s0
stage "Train-500 uncond s0 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v5_uncond_match_s0

echo
echo "=== [$(date '+%F %T')] 210 Train-500 uncond done ==="

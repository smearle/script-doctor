#!/usr/bin/env bash
# Box 210: Train-199 cond s2 — third seed for the headline scaling claim.
#
# Pre-requisites:
# - rollout_data/ contains all 199 v4 games' per-game caches (rsync'd
#   from yalda before launch)
# - 210 has 30 GB RAM and the v4 merged cache is ~13 GB, so this fits.
# - First run on 210 to use v17 train.py: per-game caches will be re-read
#   if the v17 transition format differs, but worst case we re-collect.
#
# Usage:
#   ssh 210 'cd ~/script-doctor && nohup nca_wm/scripts/run_210_v4_cond_s2.sh \
#     > /tmp/run_210_v4_cond_s2.log 2>&1 &'
set -uo pipefail

if [ -s "$HOME/.nvm/nvm.sh" ]; then
    export NVM_DIR="$HOME/.nvm"
    . "$NVM_DIR/nvm.sh" >/dev/null 2>&1 || true
fi
if ! command -v node >/dev/null 2>&1; then
    NODE_BIN=$(ls -d "$HOME"/.nvm/versions/node/*/bin 2>/dev/null | sort -V | tail -1)
    if [ -n "${NODE_BIN:-}" ]; then export PATH="$NODE_BIN:$PATH"; fi
fi
echo "node: $(command -v node 2>/dev/null) ($(node --version 2>/dev/null || echo MISSING))"

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

train_v4_cond() {
    local seed=$1
    local save_dir=$2
    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has eval_multigame.npz"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
        --games scaling_gallery_v4 --conditional \
        --architecture rule_attn \
        --n_hid 256 \
        --n_slots 16 --d_slot 64 --n_app_slots 1 \
        --n_enc_layers 2 --n_heads 4 \
        --d_model 64 --d_z 64 \
        --encode_sprites \
        --token_decoder_loss_weight 1.0 \
        --decoder_d_model 128 --decoder_n_layers 4 --decoder_n_heads 4 \
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
        --seed "$seed" --save_dir "$save_dir"
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
echo "=== [$(date '+%F %T')] 210 Train-199 cond s2 start (GPU $GPU) ==="

stage "Train-199 cond s2" \
    train_v4_cond 2 nca_wm/logs/multi_scaling_gallery_v4_cond_match_s2
stage "Train-199 cond s2 heldout" \
    heldout nca_wm/logs/multi_scaling_gallery_v4_cond_match_s2

echo
echo "=== [$(date '+%F %T')] 210 Train-199 cond s2 done ==="

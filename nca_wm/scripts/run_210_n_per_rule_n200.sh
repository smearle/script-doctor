#!/usr/bin/env bash
# Box 210: extend the dedup_pool n_per_rule sweep with n=200 (one notch
# beyond yalda's n=100 stage). Both cond and uncond, sequential. Same
# val_frac=0.10, same per-bucket batch_size scaling, same MEM_FRACTION
# safety net.
#
# Save dirs use the _dp suffix to live alongside yalda's dedup_pool runs:
#   nca_wm/logs/n_per_rule_200_cond_dp_val0.10_s0
#   nca_wm/logs/n_per_rule_200_uncond_h288_dp_val0.10_s0
#
# Usage (from yalda):
#   ssh 210 'cd ~/script-doctor && nohup nca_wm/scripts/run_210_n_per_rule_n200.sh \
#     > /tmp/run_210_n200.log 2>&1 &'
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
VAL_FRAC=${VAL_FRAC:-0.10}
N_UPDATES=${N_UPDATES:-150000}
N_VAL=${N_VAL:-200}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

stage() {
    local label=$1; shift
    echo
    echo "=== [$(date '+%F %T')] $label ==="
    "$@"
    echo "=== [$(date '+%F %T')] $label done ==="
}

train_n_per_rule_dp() {
    local n=$1
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
    local extra=()
    if [ "$cond" = "cond" ]; then
        extra=(
            --n_slots 16 --d_slot 64 --n_app_slots 1
            --n_enc_layers 2 --n_heads 4
            --d_model 64 --d_z 64
            --encode_sprites
            --token_decoder_loss_weight 1.0
            --decoder_d_model 128 --decoder_n_layers 4 --decoder_n_heads 4
        )
    fi
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION \
        .venv/bin/python3 -u -m nca_wm.train \
        --n_per_rule_games "$n" \
        --n_per_rule_universe dedup_pool \
        --n_per_rule_max_area 30 \
        $cond_flag \
        --architecture rule_attn \
        --n_hid "$nhid" \
        --n_nca_steps 8 \
        --batch_size 32 \
        --n_updates "$N_UPDATES" \
        --lr 3e-4 --lr_schedule cosine --lr_min 1e-7 \
        --grad_clip 0.5 \
        --change_loss_weight 5.0 \
        --balanced_sampling \
        --max_transitions_per_game 200000 \
        --search_timeout_ms 60000 --n_search_steps 100000 \
        --input_skip --axis_pool --axis_cummax --global_pool \
        --patience 0 --ckpt_interval 5000 --log_interval 250 \
        --val_frac "$VAL_FRAC" \
        --seed "$seed" --save_dir "$save_dir" \
        "${extra[@]}"
}

#------------------------------------------------------------------
echo "=== [$(date '+%F %T')] 210 n_per_rule=$N_VAL sweep start (mem_frac=$XLA_PYTHON_CLIENT_MEM_FRACTION) ==="

stage "Train n_per_rule_games=$N_VAL cond s0 (dedup_pool)" \
    train_n_per_rule_dp "$N_VAL" cond 256 0 \
        "nca_wm/logs/n_per_rule_${N_VAL}_cond_dp_val${VAL_FRAC}_s0"

stage "Train n_per_rule_games=$N_VAL uncond_h288 s0 (dedup_pool)" \
    train_n_per_rule_dp "$N_VAL" uncond 288 0 \
        "nca_wm/logs/n_per_rule_${N_VAL}_uncond_h288_dp_val${VAL_FRAC}_s0"

echo
echo "=== [$(date '+%F %T')] 210 n_per_rule=$N_VAL sweep done ==="

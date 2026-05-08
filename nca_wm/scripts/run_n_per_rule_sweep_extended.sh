#!/usr/bin/env bash
# Extended n_per_rule sweep: n in {5, 10, 20, 50} at 150k updates each
# with val_frac=0.10 and --patience 0 (no early stop). The point of
# this sweep is to find the breakpoint at which the model can no
# longer reach val change_err = 0 within the training budget — so we
# err on the side of giving each n MORE time, not less, to avoid
# calling "failed to converge" when the curves haven't settled yet.
#
# n=1,2,3 already finished at 30k updates with val change_err ~0
# (see /tmp/n_per_rule_sweep.log).
#
# Polls for GPU free before each stage (default GPU=0; override with
# GPU=1 to run on the other side).
#
# Usage:
#   nohup nca_wm/scripts/run_n_per_rule_sweep_extended.sh \
#     > /tmp/n_per_rule_sweep_extended.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=${GPU:-0}
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-3000}
VAL_FRAC=${VAL_FRAC:-0.10}
N_UPDATES=${N_UPDATES:-150000}
N_VALUES=${N_VALUES:-"5 10 20 50"}

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

train_n_per_rule_cond() {
    local n=$1
    local seed=$2
    local save_dir=$3
    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has eval_multigame.npz"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
        --n_per_rule_games "$n" --conditional \
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
        --seed "$seed" --save_dir "$save_dir"
}

#------------------------------------------------------------------
echo "=== [$(date '+%F %T')] extended n_per_rule sweep start (GPU $GPU, val_frac=$VAL_FRAC, n_updates=$N_UPDATES) ==="

for n in $N_VALUES; do
    wait_for_gpu
    stage "Train n_per_rule_games=$n cond s0 (val_frac=$VAL_FRAC, n_updates=$N_UPDATES)" \
        train_n_per_rule_cond "$n" 0 "nca_wm/logs/n_per_rule_${n}_cond_val${VAL_FRAC}_s0"
done

echo
echo "=== [$(date '+%F %T')] extended n_per_rule sweep done ==="

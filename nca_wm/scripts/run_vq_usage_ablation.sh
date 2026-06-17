#!/usr/bin/env bash
# VQ codebook usage-regularization ablation for the rule-attention NCA.
#
# The motivating failure mode is codebook collapse: prior VQ+decoder runs used
# game conditioning better than continuous slots, but only activated a tiny
# fraction of the 1024-entry codebook. This sweep keeps the successful
# VQ+decoder recipe and varies only the differentiable soft-assignment entropy
# penalty added by --vq_usage_loss_weight.
#
# Usage:
#   GPU=0 nca_wm/scripts/run_vq_usage_ablation.sh
#
# Useful overrides:
#   GAMES=scaling_14
#   WEIGHTS="0 0.001 0.01 0.05"
#   N_UPDATES=200000
#   SAVE_ROOT=nca_wm/logs/vq_usage_ablation_scaling14
#   EXTRA_ARGS="--val_frac 0.10"
#   WAIT_FOR_GPU=1 GPU_FREE_THRESHOLD_MB=4000
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU=${GPU:-0}
GAMES=${GAMES:-scaling_14}
WEIGHTS=${WEIGHTS:-"0 0.001 0.01 0.05"}
N_UPDATES=${N_UPDATES:-200000}
SEED=${SEED:-0}
SAVE_ROOT=${SAVE_ROOT:-nca_wm/logs/vq_usage_ablation_${GAMES}_s${SEED}}
TOKEN_DECODER_WEIGHT=${TOKEN_DECODER_WEIGHT:-1.0}
VQ_CODEBOOK_SIZE=${VQ_CODEBOOK_SIZE:-1024}
VQ_ENTROPY_TEMP=${VQ_ENTROPY_TEMP:-1.0}
EXTRA_ARGS=${EXTRA_ARGS:-}
WAIT_FOR_GPU=${WAIT_FOR_GPU:-1}
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-4000}

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

mkdir -p "$SAVE_ROOT"

wait_for_gpu() {
    if [ "$WAIT_FOR_GPU" != "1" ]; then
        return 0
    fi
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ' || true)
        total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ' || true)
        if [ -z "$used" ] || [ -z "$total" ]; then
            echo "[wait] failed to query GPU $GPU; retrying in 60s"
            sleep 60
            continue
        fi
        free=$((total - used))
        if [ "$free" -ge "$GPU_FREE_THRESHOLD_MB" ]; then
            echo "[wait] GPU $GPU free ${free} MiB >= ${GPU_FREE_THRESHOLD_MB}; proceeding"
            return 0
        fi
        echo "[wait] GPU $GPU free ${free} MiB < ${GPU_FREE_THRESHOLD_MB}; sleeping 120s"
        sleep 120
    done
}

for weight in $WEIGHTS; do
    tag=${weight//./p}
    tag=${tag//-/_}
    save_dir="$SAVE_ROOT/w${tag}"
    log="$SAVE_ROOT/w${tag}.log"

    if [ -f "$save_dir/curves.npz" ] || [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has outputs"
        continue
    fi

    echo "=== [$(date '+%F %T')] vq usage weight=$weight save_dir=$save_dir ==="
    wait_for_gpu
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION \
        XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
        .venv/bin/python3 -u -m nca_wm.train \
        --games "$GAMES" \
        --conditional \
        --architecture rule_attn \
        --n_hid 256 \
        --n_slots 16 --d_slot 64 --n_app_slots 1 \
        --n_enc_layers 2 --n_heads 4 \
        --d_model 64 --d_z 64 \
        --encode_sprites \
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
        --vq_codebook \
        --vq_codebook_size "$VQ_CODEBOOK_SIZE" \
        --vq_commitment_weight 0.25 \
        --vq_usage_loss_weight "$weight" \
        --vq_entropy_temp "$VQ_ENTROPY_TEMP" \
        --token_decoder_loss_weight "$TOKEN_DECODER_WEIGHT" \
        --decoder_d_model 128 --decoder_n_layers 4 --decoder_n_heads 4 \
        --patience 0 --ckpt_interval 5000 --log_interval 250 \
        --seed "$SEED" \
        --save_dir "$save_dir" \
        $EXTRA_ARGS \
        > "$log" 2>&1
done

echo "=== [$(date '+%F %T')] VQ usage ablation complete: $SAVE_ROOT ==="

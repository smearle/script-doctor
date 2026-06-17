#!/usr/bin/env bash
# Compare deterministic slot AE, Gaussian VAE, and VQ-VAE token reconstruction.
#
# Defaults use the best rollout-sensitive VQ-regularized checkpoint from the
# scaling_14 sweep and freeze its rule-slot encoder. Override N_UPDATES and
# decoder size for longer/full comparisons.
set -euo pipefail

cd "$(dirname "$0")/../.."

INIT_FROM=${INIT_FROM:-nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01}
SAVE_ROOT=${SAVE_ROOT:-nca_wm/refine-logs/slot_latent_compare_w0p01}
N_UPDATES=${N_UPDATES:-2000}
SEED=${SEED:-0}
DEC_D_MODEL=${DEC_D_MODEL:-64}
DEC_N_LAYERS=${DEC_N_LAYERS:-2}
DEC_N_HEADS=${DEC_N_HEADS:-4}
VAE_KL_WEIGHT=${VAE_KL_WEIGHT:-1e-4}
VQ_CODEBOOK_SIZE=${VQ_CODEBOOK_SIZE:-1024}
VQ_USAGE_LOSS_WEIGHT=${VQ_USAGE_LOSS_WEIGHT:-0.01}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$SAVE_ROOT"

run_one() {
    mode=$1
    out="$SAVE_ROOT/$mode"
    log="$SAVE_ROOT/$mode.log"
    echo "=== $(date '+%F %T') latent_model=$mode out=$out ==="
    args=(
        .venv/bin/python -m nca_wm.train_slot_ae
        --init_from "$INIT_FROM"
        --freeze_encoder
        --latent_model "$mode"
        --n_updates "$N_UPDATES"
        --log_interval 250
        --dec_d_model "$DEC_D_MODEL"
        --dec_n_layers "$DEC_N_LAYERS"
        --dec_n_heads "$DEC_N_HEADS"
        --seed "$SEED"
        --save_dir "$out"
    )
    if [ "$mode" = "vae" ]; then
        args+=(--vae_kl_weight "$VAE_KL_WEIGHT")
    elif [ "$mode" = "vqvae" ]; then
        args+=(
            --vq_codebook_size "$VQ_CODEBOOK_SIZE"
            --vq_usage_loss_weight "$VQ_USAGE_LOSS_WEIGHT"
        )
    fi
    MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl} JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} \
        "${args[@]}" $EXTRA_ARGS > "$log" 2>&1
}

run_one ae
run_one vae
run_one vqvae

echo "=== $(date '+%F %T') done: $SAVE_ROOT ==="

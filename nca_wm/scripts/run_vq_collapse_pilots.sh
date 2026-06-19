#!/usr/bin/env bash
# Fast diagnostic pilots for slot VQ-VAE hard-code collapse.
set -euo pipefail

cd "$(dirname "$0")/../.."

INIT_FROM=${INIT_FROM:-nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01}
SAVE_ROOT=${SAVE_ROOT:-nca_wm/refine-logs/vq_collapse_pilots_w0p01_500}
N_UPDATES=${N_UPDATES:-500}
SEED=${SEED:-0}
DEC_D_MODEL=${DEC_D_MODEL:-64}
DEC_N_LAYERS=${DEC_N_LAYERS:-2}
DEC_N_HEADS=${DEC_N_HEADS:-4}
VQ_CODEBOOK_SIZE=${VQ_CODEBOOK_SIZE:-1024}
VQ_USAGE_LOSS_WEIGHT=${VQ_USAGE_LOSS_WEIGHT:-0.01}
LOG_INTERVAL=${LOG_INTERVAL:-100}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$SAVE_ROOT"

run_one() {
    name=$1
    freeze=$2
    vq_init=$3
    slot_pre_norm=$4
    out="$SAVE_ROOT/$name"
    log="$SAVE_ROOT/$name.log"
    echo "=== $(date '+%F %T') pilot=$name out=$out ==="
    args=(
        .venv/bin/python -m nca_wm.train_slot_ae
        --init_from "$INIT_FROM"
        --latent_model vqvae
        --n_updates "$N_UPDATES"
        --log_interval "$LOG_INTERVAL"
        --dec_d_model "$DEC_D_MODEL"
        --dec_n_layers "$DEC_N_LAYERS"
        --dec_n_heads "$DEC_N_HEADS"
        --seed "$SEED"
        --save_dir "$out"
        --vq_codebook_size "$VQ_CODEBOOK_SIZE"
        --vq_usage_loss_weight "$VQ_USAGE_LOSS_WEIGHT"
        --vq_init "$vq_init"
        --slot_pre_norm "$slot_pre_norm"
    )
    if [ "$freeze" = "freeze" ]; then
        args+=(--freeze_encoder)
    fi

    env_args=(MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}")
    if [ -n "${JAX_PLATFORMS:-}" ]; then
        env_args+=(JAX_PLATFORMS="$JAX_PLATFORMS")
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        env_args+=(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
    fi
    env "${env_args[@]}" "${args[@]}" $EXTRA_ARGS > "$log" 2>&1
    env "${env_args[@]}" .venv/bin/python -m nca_wm.scripts.diagnose_slot_vq \
        "$out/slot_ae.pkl" >> "$log" 2>&1
}

run_one vq_random freeze random none
run_one vq_kmeans freeze kmeans none
run_one vq_slot_sample freeze slot_sample none
run_one vq_layernorm freeze random layernorm
run_one vq_kmeans_layernorm freeze kmeans layernorm
run_one vq_unfrozen unfreeze random none

env_args=(MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}")
if [ -n "${JAX_PLATFORMS:-}" ]; then
    env_args+=(JAX_PLATFORMS="$JAX_PLATFORMS")
fi
env "${env_args[@]}" .venv/bin/python -m nca_wm.scripts.eval_slot_latent_ablation \
    "$SAVE_ROOT/vq_random" \
    "$SAVE_ROOT/vq_kmeans" \
    "$SAVE_ROOT/vq_slot_sample" \
    "$SAVE_ROOT/vq_layernorm" \
    "$SAVE_ROOT/vq_kmeans_layernorm" \
    "$SAVE_ROOT/vq_unfrozen" \
    --out_dir "$SAVE_ROOT" \
    > "$SAVE_ROOT/latent_ablation_summary.log" 2>&1

echo "=== $(date '+%F %T') done: $SAVE_ROOT ==="

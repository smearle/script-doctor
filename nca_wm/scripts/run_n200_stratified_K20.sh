#!/usr/bin/env bash
# n=200 stratified K=20 (10 games per n_rules bucket, n_rules ∈ [1..20]) +
# objperm. Higher rule-complexity coverage than K=10 (where n_rules ∈ [1..10]).
# K=10 pilot won decisively (TF 2.27% vs sorted+objperm 2.47%); this checks
# whether covering n_rules ∈ [11..20] adds further OOD generalization.
#
# Wait for GPU 0 to vacate (n=200 K=10 eval_multigame still in progress).
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=${GPU:-0}
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-3000}
VAL_FRAC=${VAL_FRAC:-0.10}
SEED=${SEED:-0}
N_UPDATES=${N_UPDATES:-150000}
SAVE_DIR=${SAVE_DIR:-nca_wm/logs/n_per_rule_200_cond_dp_strat20_objperm_val${VAL_FRAC}_s${SEED}}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

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

echo "=== [$(date '+%F %T')] queueing n=200 stratified K=20 + objperm on GPU $GPU (save_dir=$SAVE_DIR) ==="

if [ -d "$SAVE_DIR" ] && [ -f "$SAVE_DIR/eval_multigame.npz" ]; then
    echo "[skip] $SAVE_DIR already has eval_multigame.npz"
    exit 0
fi

wait_for_gpu

echo "=== [$(date '+%F %T')] launching n=200 stratified K=20 + objperm training ==="

CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION \
    XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
    .venv/bin/python3 -u -m nca_wm.train \
    --n_per_rule_games 200 \
    --n_per_rule_universe dedup_pool \
    --n_per_rule_max_area 30 \
    --n_per_rule_strategy stratified \
    --n_per_rule_max_bucket 20 \
    --conditional \
    --obj_permute_aug \
    --n_slots 16 --d_slot 64 --n_app_slots 1 \
    --n_enc_layers 2 --n_heads 4 \
    --d_model 64 --d_z 64 \
    --encode_sprites \
    --token_decoder_loss_weight 1.0 \
    --decoder_d_model 128 --decoder_n_layers 4 --decoder_n_heads 4 \
    --architecture rule_attn \
    --n_hid 256 \
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
    --seed "$SEED" --save_dir "$SAVE_DIR"

echo "=== [$(date '+%F %T')] n=200 stratified K=20 + objperm done ==="

#!/usr/bin/env bash
# Additional-seed runs for the dedup_pool n_per_rule sweep, so the
# cond-vs-uncond scaling figure gets real across-seed error bars (seed-0 is
# the only seed today). Reuses the EXACT training invocation of
# run_n_per_rule_dp_sweep.sh; only --seed and the save_dir suffix change.
#
# Budgets match seed-0: n<=5 -> 30k, n in {10..200} -> 150k, n=400 -> 200k.
#
# Usage (two parallel streams, one per model type):
#   SEED=1 GPU=1 STAGES=cond   nohup nca_wm/scripts/run_n_per_rule_seed.sh \
#       > /tmp/n_per_rule_s1_cond.log 2>&1 &
#   SEED=1 GPU=0 STAGES=uncond MEM_FRACTION=0.6 \
#       nohup nca_wm/scripts/run_n_per_rule_seed.sh \
#       > /tmp/n_per_rule_s1_uncond.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

SEED=${SEED:-1}
GPU=${GPU:-1}
STAGES=${STAGES:-all}
VAL_FRAC=${VAL_FRAC:-0.10}
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-6000}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.95}

# n list and per-n budgets (match seed-0).
NS=${NS:-"1 2 3 5 10 20 50 100 200 400"}
declare -A BUDGETS=( [1]=30000 [2]=30000 [3]=30000 [5]=30000 \
    [10]=150000 [20]=150000 [50]=150000 [100]=150000 [200]=150000 [400]=200000 )

wait_for_gpu() {
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
        if [ -z "$used" ]; then echo "[wait] GPU $GPU query failed; 60s"; sleep 60; continue; fi
        if [ "$used" -lt "$GPU_FREE_THRESHOLD_MB" ]; then
            echo "[wait] GPU $GPU ${used} MiB used (< $GPU_FREE_THRESHOLD_MB); go"; return 0
        fi
        echo "[wait] GPU $GPU busy: ${used} MiB; sleep 120s"; sleep 120
    done
}

train_n_per_rule_dp() {
    local n=$1 cond=$2 nhid=$3 seed=$4 n_updates=$5 save_dir=$6
    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already complete"; return 0
    fi
    local cond_flag; [ "$cond" = "cond" ] && cond_flag="--conditional" || cond_flag="--no-conditional"
    local extra=()
    if [ "$cond" = "cond" ]; then
        extra=( --n_slots 16 --d_slot 64 --n_app_slots 1 --n_enc_layers 2 --n_heads 4
            --d_model 64 --d_z 64 --encode_sprites --token_decoder_loss_weight 1.0
            --decoder_d_model 128 --decoder_n_layers 4 --decoder_n_heads 4 )
    fi
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION \
        XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
        .venv/bin/python3 -u -m nca_wm.train \
        --n_per_rule_games "$n" --n_per_rule_universe dedup_pool --n_per_rule_max_area 30 \
        $cond_flag --architecture rule_attn --n_hid "$nhid" --n_nca_steps 8 \
        --batch_size 32 --n_updates "$n_updates" \
        --lr 3e-4 --lr_schedule cosine --lr_min 1e-7 --grad_clip 0.5 \
        --change_loss_weight 5.0 --balanced_sampling --max_transitions_per_game 200000 \
        --search_timeout_ms 60000 --n_search_steps 100000 \
        --input_skip --axis_pool --axis_cummax --global_pool \
        --patience 0 --ckpt_interval 5000 --log_interval 250 \
        --val_frac "$VAL_FRAC" --seed "$seed" --save_dir "$save_dir" "${extra[@]}"
}

echo "=== [$(date '+%F %T')] seed=$SEED sweep (GPU $GPU, STAGES=$STAGES, mem=$XLA_PYTHON_CLIENT_MEM_FRACTION) ==="
run_stage() {
    local kind=$1 nhid=$2
    for n in $NS; do
        wait_for_gpu
        local save_dir="nca_wm/logs/n_per_rule_${n}_${kind}_dp_val${VAL_FRAC}_s${SEED}"
        [ "$kind" = "uncond" ] && save_dir="nca_wm/logs/n_per_rule_${n}_uncond_h288_dp_val${VAL_FRAC}_s${SEED}"
        echo "=== [$(date '+%F %T')] n=$n $kind s$SEED (budget=${BUDGETS[$n]}) ==="
        train_n_per_rule_dp "$n" "$kind" "$nhid" "$SEED" "${BUDGETS[$n]}" "$save_dir"
    done
}

if [ "$STAGES" = "cond" ] || [ "$STAGES" = "all" ]; then run_stage cond 256; fi
if [ "$STAGES" = "uncond" ] || [ "$STAGES" = "all" ]; then run_stage uncond 288; fi
echo "=== [$(date '+%F %T')] seed=$SEED sweep done ==="

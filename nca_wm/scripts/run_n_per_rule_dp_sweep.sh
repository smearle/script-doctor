#!/usr/bin/env bash
# n_per_rule sweep on the dedup_pool universe (3,474 games, 100x more
# coverage at low n_rules than the legacy gallery universe). Both cond
# and uncond at every n; STAGES env var splits the work across GPUs.
#
# Each cell saved at nca_wm/logs/n_per_rule_${n}_${kind}_dp_val0.10_s0
# so the new dedup_pool runs don't clobber the existing gallery runs
# (n_per_rule_${n}_${kind}_val0.10_s0).
#
# OOM avoidance: max_level_area<=30 filter is applied by train.py (now
# the default for --n_per_rule_games). MEM_FRACTION=0.95 as belt-and-
# suspenders.
#
# Usage:
#   GPU=0 STAGES=cond   nohup nca_wm/scripts/run_n_per_rule_dp_sweep.sh \
#       > /tmp/n_per_rule_dp_cond.log 2>&1 &
#   GPU=1 STAGES=uncond nohup nca_wm/scripts/run_n_per_rule_dp_sweep.sh \
#       > /tmp/n_per_rule_dp_uncond.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=${GPU:-0}
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-3000}
VAL_FRAC=${VAL_FRAC:-0.10}
STAGES=${STAGES:-all}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}

# Per-n training budget: small n converges in <10k steps; harder n
# need more. 30k for n<=5; 150k beyond.
declare -A BUDGETS
BUDGETS[1]=30000
BUDGETS[2]=30000
BUDGETS[3]=30000
BUDGETS[5]=30000
BUDGETS[10]=150000
BUDGETS[20]=150000
BUDGETS[50]=150000
BUDGETS[100]=150000

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

train_n_per_rule_dp() {
    local n=$1
    local cond=$2     # "cond" or "uncond"
    local nhid=$3
    local seed=$4
    local n_updates=$5
    local save_dir=$6
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
    # Disable CUDA command-buffer caching: large per-game eval loops at
    # n>=100 accumulate hundreds of compiled graphs and OOM on
    # instantiation (553 alive graphs in the n=100 cond crash). Disabling
    # command buffers costs a small launch-overhead penalty but is the
    # only knob that prevents the per-game-eval JIT explosion.
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION \
        XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
        .venv/bin/python3 -u -m nca_wm.train \
        --n_per_rule_games "$n" \
        --n_per_rule_universe dedup_pool \
        --n_per_rule_max_area 30 \
        $cond_flag \
        --architecture rule_attn \
        --n_hid "$nhid" \
        --n_nca_steps 8 \
        --batch_size 32 \
        --n_updates "$n_updates" \
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
echo "=== [$(date '+%F %T')] dedup_pool n_per_rule sweep start (GPU $GPU, STAGES=$STAGES, mem_frac=$XLA_PYTHON_CLIENT_MEM_FRACTION) ==="

if [ "$STAGES" = "cond" ] || [ "$STAGES" = "all" ]; then
    for n in 1 2 3 5 10 20 50 100; do
        wait_for_gpu
        stage "Train n_per_rule_games=$n cond s0 (dedup_pool, n_updates=${BUDGETS[$n]})" \
            train_n_per_rule_dp "$n" cond 256 0 "${BUDGETS[$n]}" \
                "nca_wm/logs/n_per_rule_${n}_cond_dp_val${VAL_FRAC}_s0"
    done
fi

if [ "$STAGES" = "uncond" ] || [ "$STAGES" = "all" ]; then
    for n in 1 2 3 5 10 20 50 100; do
        wait_for_gpu
        stage "Train n_per_rule_games=$n uncond_h288 s0 (dedup_pool, n_updates=${BUDGETS[$n]})" \
            train_n_per_rule_dp "$n" uncond 288 0 "${BUDGETS[$n]}" \
                "nca_wm/logs/n_per_rule_${n}_uncond_h288_dp_val${VAL_FRAC}_s0"
    done
fi

echo
echo "=== [$(date '+%F %T')] dedup_pool n_per_rule sweep done ==="

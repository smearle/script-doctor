#!/usr/bin/env bash
# Parameter-matched uncond sweep along the n_per_rule_games axis.
# Companion to the cond sweep (run_n_per_rule_sweep.sh + extended). For
# each n in {1, 2, 3, 5, 10, 20, 50} we train an uncond model at
# n_hid=288 — body params ~n_hid² gives ~16.6M total, slightly above
# cond@n_hid=256's 16.03M. Same val_frac=0.10, same per-n n_updates as
# the cond sweep so the curves drop in directly next to the cond ones.
#
# Polls for GPU free before each stage. Default GPU=0.
#
# Usage:
#   nohup nca_wm/scripts/run_n_per_rule_uncond_sweep.sh \
#     > /tmp/n_per_rule_uncond_sweep.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=${GPU:-0}
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-3000}
VAL_FRAC=${VAL_FRAC:-0.10}
NHID=${NHID:-288}

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

train_n_per_rule_uncond() {
    local n=$1
    local n_updates=$2
    local seed=$3
    local save_dir=$4
    if [ -d "$save_dir" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "[skip] $save_dir already has eval_multigame.npz"
        return 0
    fi
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python3 -u -m nca_wm.train \
        --n_per_rule_games "$n" --no-conditional \
        --architecture rule_attn \
        --n_hid "$NHID" \
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
        --seed "$seed" --save_dir "$save_dir"
}

#------------------------------------------------------------------
echo "=== [$(date '+%F %T')] n_per_rule param-matched uncond sweep start (GPU $GPU, n_hid=$NHID) ==="

# Match the cond sweep's per-n budget: 30k for n in {1,2,3}, 150k for the rest.
# (Cond at 30k already reached val_cerr ~1e-9 for n=1,2,3; the harder
# n's needed 150k+ to converge.)
declare -A BUDGETS
BUDGETS[1]=30000
BUDGETS[2]=30000
BUDGETS[3]=30000
BUDGETS[5]=150000
BUDGETS[10]=150000
BUDGETS[20]=150000
BUDGETS[50]=150000

for n in 1 2 3 5 10 20 50; do
    n_updates=${BUDGETS[$n]}
    wait_for_gpu
    stage "uncond n_per_rule=$n n_hid=$NHID s0 (val_frac=$VAL_FRAC, n_updates=$n_updates)" \
        train_n_per_rule_uncond "$n" "$n_updates" 0 \
            "nca_wm/logs/n_per_rule_${n}_uncond_h${NHID}_val${VAL_FRAC}_s0"
done

echo
echo "=== [$(date '+%F %T')] n_per_rule param-matched uncond sweep done ==="

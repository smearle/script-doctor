#!/usr/bin/env bash
# Retry the runs that OOM'd at the default JAX preallocation
# (~75% of GPU memory). With XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAX
# can use nearly the entire 24 GB card, which is enough for the
# n_per_rule={20,50} cond/uncond runs and Train-500 cond/uncond that
# crashed with 17-22 GiB allocation requests.
#
# Polls for GPU free before each stage. Default GPU=0; pass GPU=1
# (or another) to put one set of runs on each card.
#
# Stages:
#   1. n_per_rule_games=20 cond  s0  (val_frac=0.10, 150k updates)
#   2. n_per_rule_games=50 cond  s0
#   3. n_per_rule_games=20 uncond s0 (param-matched n_hid=288)
#   4. n_per_rule_games=50 uncond s0
#
# (Train-500 cond/uncond retries are NOT in this script; they'd
# block the queue for >24h. Run them in a separate launcher.)
#
# Usage:
#   nohup nca_wm/scripts/run_oom_retries.sh \
#     > /tmp/oom_retries.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

GPU=${GPU:-0}
GPU_FREE_THRESHOLD_MB=${GPU_FREE_THRESHOLD_MB:-3000}
VAL_FRAC=${VAL_FRAC:-0.10}
N_UPDATES=${N_UPDATES:-150000}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
# STAGES selects which subset to run: "cond" | "uncond" | "all" (default).
# Lets two parallel queues split the work cleanly across GPUs without
# colliding on shared save_dirs.
STAGES=${STAGES:-all}

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

train_n_per_rule() {
    local n=$1
    local cond=$2     # "cond" or "uncond"
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
        --n_per_rule_games "$n" $cond_flag \
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
echo "=== [$(date '+%F %T')] OOM-retry queue start (GPU $GPU, mem_frac=$XLA_PYTHON_CLIENT_MEM_FRACTION) ==="

if [ "$STAGES" = "cond" ] || [ "$STAGES" = "all" ]; then
    for n in 20 50; do
        wait_for_gpu
        stage "Train n_per_rule_games=$n cond s0 (retry, mem_frac=$XLA_PYTHON_CLIENT_MEM_FRACTION)" \
            train_n_per_rule "$n" cond 256 0 "nca_wm/logs/n_per_rule_${n}_cond_val${VAL_FRAC}_s0"
    done
fi

if [ "$STAGES" = "uncond" ] || [ "$STAGES" = "all" ]; then
    for n in 20 50; do
        wait_for_gpu
        stage "Train n_per_rule_games=$n uncond_h288 s0 (retry, mem_frac=$XLA_PYTHON_CLIENT_MEM_FRACTION)" \
            train_n_per_rule "$n" uncond 288 0 "nca_wm/logs/n_per_rule_${n}_uncond_h288_val${VAL_FRAC}_s0"
    done
fi

echo
echo "=== [$(date '+%F %T')] OOM-retry queue done ==="

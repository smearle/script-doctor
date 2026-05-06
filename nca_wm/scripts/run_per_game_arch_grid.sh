#!/usr/bin/env bash
# Per-game × per-architecture grid: one rule_attn model trained per (game, arch)
# cell. Buckets mirror the Heroes sweep (ARCHITECTURE_REPORT F1, F2, F4) so the
# new cells are directly comparable to existing per-game arch sweeps.
#
# Buckets at fixed n_nca_steps=8 (rule_attn defaults: h=256, K=16, batch=16):
#   A: pool ON,            shared (n_repeats=8)
#   B: pool ON,            per-step (n_repeats=1)
#   C: pool OFF + skip,    shared
#   D: pool OFF + skip,    per-step
#
# Default game list covers distinct mechanic classes:
#   sokoban_basic        — axis-aligned chain push (small)
#   Microban             — same dynamics, more authored levels (transfer probe)
#   Heroes_of_Sokoban    — multi-bracket char swap + projectile travel
#   Bouncers             — non-axis-aligned looping projectile
#   nekopuzzle           — restricted-orientation movement
#   Travelling_salesman  — multi-bracket all-pairs
#
# Usage:
#   bash nca_wm/scripts/run_per_game_arch_grid.sh
#   GAMES="sokoban_basic Microban" BUCKETS="A C" bash ... # subset
#   N_UPDATES=10000 bash ...                              # tighter budget
set -u
cd /home/jupyter-earle/script-doctor

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_per_game_arch
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

GAMES="${GAMES:-sokoban_basic Microban Heroes_of_Sokoban Bouncers nekopuzzle Travelling_salesman}"
BUCKETS="${BUCKETS:-A B C D}"
DEPTH="${DEPTH:-8}"
N_UPDATES="${N_UPDATES:-15000}"
SEED="${SEED:-0}"

COMMON=(
    --level 0
    --conditional --architecture rule_attn
    --n_hid 256 --n_slots 16 --n_app_slots 1
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --n_search_steps 200000 --search_timeout_ms 120000
    --max_transitions_per_game 100000
    --n_updates "$N_UPDATES" --patience 0 --min_delta 1e-6
    --batch_size 16 --lr 3e-4
    --log_interval 500 --ckpt_interval 5000
    --seed "$SEED"
)

flags_for_bucket() {
    case "$1" in
        A) echo "--n_nca_steps $DEPTH --n_nca_repeats $DEPTH" ;;
        B) echo "--n_nca_steps $DEPTH --n_nca_repeats 1" ;;
        C) echo "--n_nca_steps $DEPTH --n_nca_repeats $DEPTH --no-axis_pool --no-axis_cummax --no-global_pool --input_skip" ;;
        D) echo "--n_nca_steps $DEPTH --n_nca_repeats 1       --no-axis_pool --no-axis_cummax --no-global_pool --input_skip" ;;
        *) echo "UNKNOWN_$1" ;;
    esac
}

run_one() {
    local gpu=$1; local game=$2; local bucket=$3
    local tag="${game}__${bucket}_d${DEPTH}"
    local save_dir="$LOGDIR/$tag"
    local log="$LOGDIR/$tag.out"
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/train_meta.json" ]; then
        echo "  [GPU $gpu] skip $tag (already done)"
        return
    fi
    local extra
    extra=$(flags_for_bucket "$bucket")
    echo "  [GPU $gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games "$game" \
        "${COMMON[@]}" $extra \
        --save_dir "$save_dir" \
        > "$log" 2>&1
    if [ -f "$save_dir/params.pkl" ]; then
        echo "  [GPU $gpu] done $tag"
    else
        echo "  [GPU $gpu] FAILED $tag (see $log)"
    fi
}

# Build queue of (game, bucket) pairs.
QUEUE=()
for g in $GAMES; do
    for b in $BUCKETS; do
        QUEUE+=("${g}|${b}")
    done
done

# GPUS env var: space-separated list (default just GPU 0 — this host has one).
GPUS_STR="${GPUS:-0}"
read -r -a GPUS <<< "$GPUS_STR"
N_GPUS=${#GPUS[@]}

echo "queue: ${#QUEUE[@]} cells (depth=$DEPTH, n_updates=$N_UPDATES, seed=$SEED, gpus=${GPUS_STR})"

i=0
while [ $i -lt ${#QUEUE[@]} ]; do
    pids=()
    for ((g_idx=0; g_idx<N_GPUS && i<${#QUEUE[@]}; g_idx++, i++)); do
        spec="${QUEUE[$i]}"
        IFS='|' read -r game bucket <<< "$spec"
        gpu="${GPUS[$g_idx]}"
        run_one "$gpu" "$game" "$bucket" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done
echo "ALL DONE"

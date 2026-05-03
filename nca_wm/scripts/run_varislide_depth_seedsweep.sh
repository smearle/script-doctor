#!/bin/bash
# Multi-seed depth sweep on multi-grid varislide.
# Tests "depth = iteration capacity" hypothesis under shared weights:
#   n_repeats = n_steps (fully shared), fixed n_hid=128.
#   depth ∈ {8, 16, 32, 64}, seed ∈ {0, 1, 2} → 12 runs.
# Two runs in parallel (one per GPU). All other recipe knobs match the
# E15/E17 multi-grid varislide setup so results are directly comparable.

set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_canary
mkdir -p $LOGDIR

# Share JAX compile cache across runs to amortize the compile cost
# (different n_steps still re-compile, but same depth + different seed reuses).
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

run_one() {
    local gpu=$1; local depth=$2; local seed=$3
    local tag="depth${depth}_s${seed}"
    local save_dir="$LOGDIR/varislide_$tag"
    local log="$LOGDIR/varislide_$tag.out"
    if [ -f "$save_dir/train_meta.json" ]; then
        echo "  skip $tag (already done)"
        return
    fi
    echo "  [GPU $gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY $REPO/nca_wm/train.py \
        --games varislide \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps $depth --n_nca_repeats $depth --n_slots 16 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels 64 --synthetic_multi_grid \
        --synthetic_grid_sizes "6x3,8x3,10x3,12x3,16x3" \
        --no-synthetic_require_solvable \
        --synthetic_min_states 5 \
        --synthetic_max_iters_search 5000 --synthetic_timeout_ms_search 2000 \
        --max_transitions_per_game 200000 \
        --n_updates 10000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 500 --ckpt_interval 5000 \
        --save_dir "$save_dir" \
        --seed $seed \
        > "$log" 2>&1
    echo "  [GPU $gpu] done $tag (exit=$?)"
}

# Iterate (depth, seed) pairs; run two at a time pinned to GPU 0 & 1.
PAIRS=()
for depth in 8 16 32 64; do
    for seed in 0 1 2; do
        PAIRS+=("$depth:$seed")
    done
done
echo "queue: ${#PAIRS[@]} runs"

i=0
while [ $i -lt ${#PAIRS[@]} ]; do
    p1=${PAIRS[$i]}; d1=${p1%%:*}; s1=${p1##*:}
    if [ $((i+1)) -lt ${#PAIRS[@]} ]; then
        p2=${PAIRS[$((i+1))]}; d2=${p2%%:*}; s2=${p2##*:}
        run_one 0 $d1 $s1 &
        pid0=$!
        run_one 1 $d2 $s2 &
        pid1=$!
        wait $pid0 $pid1
        i=$((i+2))
    else
        run_one 0 $d1 $s1
        i=$((i+1))
    fi
done
echo "ALL DONE"

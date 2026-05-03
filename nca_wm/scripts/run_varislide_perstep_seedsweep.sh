#!/bin/bash
# Per-step (un-shared) weights on multi-grid varislide, multi-seed.
# Directly tests "is the failure caused by weight-sharing?" — n_nca_repeats=1
# means each NCA step has its own conv/attn/out weights (~7x more params at
# n_nca_steps=16). Same recipe as the shared-weight depth sweep otherwise.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_canary
mkdir -p $LOGDIR

export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

run_one() {
    local gpu=$1; local seed=$2
    local tag="perstep_d16_s${seed}"
    local save_dir="$LOGDIR/varislide_$tag"
    local log="$LOGDIR/varislide_$tag.out"
    if [ -f "$save_dir/train_meta.json" ]; then
        echo "  skip $tag"
        return
    fi
    echo "  [GPU $gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY $REPO/nca_wm/train.py \
        --games varislide \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps 16 --n_nca_repeats 1 --n_slots 16 \
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

run_one 0 0 &
pid0=$!
run_one 1 1 &
pid1=$!
wait $pid0 $pid1
run_one 0 2
echo "ALL DONE"

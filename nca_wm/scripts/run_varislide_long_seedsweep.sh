#!/bin/bash
# Long-training (50k steps) seed sweep on multi-grid varislide at depth=16.
# Tests "did we just not train long enough?" against the depth sweep's flat
# negative result. 3 seeds, two in parallel pinned to GPU 0/1.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_canary
mkdir -p $LOGDIR
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

run_one() {
    local gpu=$1; local seed=$2
    local tag="long50k_d16_s${seed}"
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
        --n_hid 128 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels 64 --synthetic_multi_grid \
        --synthetic_grid_sizes "6x3,8x3,10x3,12x3,16x3" \
        --no-synthetic_require_solvable \
        --synthetic_min_states 5 \
        --synthetic_max_iters_search 5000 --synthetic_timeout_ms_search 2000 \
        --max_transitions_per_game 200000 \
        --n_updates 50000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 1000 --ckpt_interval 10000 \
        --save_dir "$save_dir" \
        --seed $seed \
        > "$log" 2>&1
    echo "  [GPU $gpu] done $tag (exit=$?)"
}

# 3 seeds, 2 in parallel, then 1 alone
run_one 0 0 &
pid0=$!
run_one 1 1 &
pid1=$!
wait $pid0 $pid1
run_one 0 2
echo "ALL DONE"

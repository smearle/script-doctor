#!/bin/bash
# Cross-check the multi-grid varislide failure on microban (sokoban-class with
# chain-push depth-bound dynamics). If multi-grid synth microban *also* fails,
# the failure generalizes to all depth-bound mechanics under varied-grid synth.
# If microban succeeds, something is specific to varislide (probably its
# trivially-simple rule structure).
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_canary
mkdir -p $LOGDIR

export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

run_one() {
    local gpu=$1; local seed=$2; local mg_flag=$3; local tag_suffix=$4
    local tag="microban_${tag_suffix}_s${seed}"
    local save_dir="$LOGDIR/$tag"
    local log="$LOGDIR/$tag.out"
    if [ -f "$save_dir/train_meta.json" ]; then
        echo "  skip $tag"
        return
    fi
    echo "  [GPU $gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY $REPO/nca_wm/train.py \
        --games Microban \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels 64 \
        $mg_flag \
        --synthetic_mode evolve \
        --synthetic_evolve_pop_size 24 \
        --synthetic_evolve_max_generations 30 \
        --synthetic_require_solvable \
        --synthetic_fallback_dynamics \
        --synthetic_min_states 5 \
        --synthetic_max_iters_search 1500 --synthetic_timeout_ms_search 400 \
        --max_transitions_per_game 200000 \
        --token_decoder_loss_weight 0.1 \
        --n_updates 10000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 500 --ckpt_interval 5000 \
        --save_dir "$save_dir" \
        --seed $seed \
        > "$log" 2>&1
    echo "  [GPU $gpu] done $tag (exit=$?)"
}

# Test 1: single-grid synth microban (control) — at one of microban's authored
#   sizes (12x12). Should work like prior reports show.
# Test 2: multi-grid synth microban (treatment) — synth at multiple sizes spanning
#   smaller-than-authored to authored-size. If this fails, the failure
#   generalizes beyond varislide.

# Single-grid control (1 seed for now)
run_one 0 0 "--synthetic_w 12 --synthetic_h 12" "single_w12" &
pid0=$!
# Multi-grid treatment (1 seed for now)
run_one 1 0 "--synthetic_multi_grid --synthetic_grid_sizes 8x8,10x10,12x12,14x14,16x16" "multigrid_8to16" &
pid1=$!
wait $pid0 $pid1
echo "ALL DONE"

#!/usr/bin/env bash
# Train Heroes_of_Sokoban models on evolved synthetic levels with three seed
# strategies — pure-evolve (no authored seeds), seed-from-L0, and seed-from-L07
# — at depths d=4 and d=8. Output dirs live under nca_wm/logs_heroes_synth/.
#
# Recipe matches the authored-data baselines (heroes_C_nopool_shared_d{4,8} and
# heroes_l07_C_d{4,8}): rule_attn, n_hid=256, conditional, input_skip, no pool,
# astar collection, balanced sampling, batch=16, lr=3e-4.
#
# Synth size matches the training/eval bucket: 13x13 (Heroes_of_Sokoban max
# authored extent). seed_from_authored will crop authored levels into 13x13
# with background-padded centering for smaller authored dims.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
GPU=${GPU:-1}
N_LEVELS=${N_LEVELS:-32}
N_UPDATES=${N_UPDATES:-20000}
EVOLVE_POP=${EVOLVE_POP:-16}
EVOLVE_GENS=${EVOLVE_GENS:-30}
SEARCH_TIMEOUT_MS=${SEARCH_TIMEOUT_MS:-30000}
N_SEARCH_STEPS=${N_SEARCH_STEPS:-50000}
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

LOG_BASE=$REPO/nca_wm/logs_heroes_synth
mkdir -p "$LOG_BASE"

train_one() {
    local tag=$1
    local depth=$2
    local extra_flags=$3
    local save_dir="$LOG_BASE/heroes_synth_${tag}_d${depth}"
    local out_log="$LOG_BASE/heroes_synth_${tag}_d${depth}.out"
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/eval_multigame.npz" ]; then
        echo "  skip $tag d=$depth: already complete"
        return
    fi
    echo "  [GPU $GPU] train $tag d=$depth → $save_dir"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Heroes_of_Sokoban \
        --architecture rule_attn --n_hid 256 \
        --n_nca_steps $depth --n_nca_repeats $depth \
        --conditional --input_skip \
        --no-axis_pool --no-axis_cummax --no-global_pool \
        --batch_size 16 --lr 3e-4 \
        --change_loss_weight 5.0 --win_loss_weight 1.0 \
        --balanced_sampling \
        --n_updates $N_UPDATES --log_interval 500 --ckpt_interval 10000 \
        --synthetic_levels $N_LEVELS \
        --synthetic_w 13 --synthetic_h 13 \
        --synthetic_mode evolve \
        --synthetic_evolve_pop_size $EVOLVE_POP \
        --synthetic_evolve_max_generations $EVOLVE_GENS \
        --no-synthetic_require_solvable \
        --synthetic_no_a_count_max 5 \
        --n_search_steps $N_SEARCH_STEPS --search_timeout_ms $SEARCH_TIMEOUT_MS \
        --search_algo astar \
        --seed 0 \
        --save_dir "$save_dir" \
        $extra_flags \
        > "$out_log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED $tag d=$depth (rc=$rc); see $out_log"
        tail -20 "$out_log"
    else
        echo "  done $tag d=$depth"
    fi
}

# Three seed strategies × two depths.
for depth in 4 8; do
    train_one pureEvolve $depth ""
    train_one seedL0 $depth "--synthetic_seed_from_authored --synthetic_seed_level_indices 0"
    train_one seedL07 $depth "--synthetic_seed_from_authored --synthetic_seed_level_indices 0,1,2,3,4,5,6,7"
done

echo "ALL DONE"

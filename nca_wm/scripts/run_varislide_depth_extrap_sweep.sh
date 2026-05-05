#!/usr/bin/env bash
# Varislide depth-extrapolation sweep.
#
# Question: does an NCA world model trained at *small* depth (and without
# pooling) extrapolate to longer slides when run for more steps at inference?
# Varislide's `again` rule slides the player one cell per engine iteration,
# so the slide distance of a single training transition can exceed any
# 3x3 local cone. With pooling OFF, the only way to predict a long slide
# is iterative application of the local rule.
#
# Train: pool OFF, shared weights (n_repeats == n_nca_steps), input_skip,
# mask_hidden, h=128, batch=16, 10k updates, 3 seeds. Train widths held
# small ({6x3, 8x3}) so the train-time max slide distance stays modest.
# Vary D_train ∈ {2, 4, 8}.
# Plus a pool-ON control at D_train=2 to show pooling shortcuts iteration.

set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_depth_extrap
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

COMMON=(
    --games varislide
    --conditional --architecture rule_attn
    --n_hid 128 --n_slots 16
    --mask_hidden --mask_padded_loss
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --synthetic_levels 64 --synthetic_multi_grid
    --synthetic_grid_sizes "6x3,8x3"
    --no-synthetic_require_solvable
    --synthetic_min_states 5
    --synthetic_max_iters_search 5000 --synthetic_timeout_ms_search 2000
    --max_transitions_per_game 200000
    --n_updates 10000 --patience 0 --min_delta 1e-6
    --batch_size 16 --lr 3e-4
    --log_interval 1000 --ckpt_interval 5000
)

run_one() {
    local gpu=$1; local tag=$2; shift 2
    local save_dir="$LOGDIR/$tag"
    local log="$LOGDIR/$tag.out"
    if [ -f "$save_dir/train_meta.json" ]; then
        echo "  skip $tag (already done)"
        return
    fi
    echo "  [GPU $gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        "${COMMON[@]}" "$@" \
        --save_dir "$save_dir" \
        > "$log" 2>&1
    local rc=$?
    if [ -f "$save_dir/params.pkl" ]; then
        echo "  [GPU $gpu] done $tag (train ok, eval rc=$rc)"
    else
        echo "  [GPU $gpu] FAILED $tag rc=$rc"
    fi
}

QUEUE=()
# Pool OFF (the main sweep). depth × seed.
for depth in 2 4 8; do
    for seed in 0 1 2; do
        QUEUE+=("nopool_d${depth}_s${seed}|$seed|$depth|$depth|--no-axis_pool --no-axis_cummax --no-global_pool --input_skip")
    done
done
# Pool ON control at the smallest depth (axis_cummax gives directional global info).
for seed in 0 1 2; do
    QUEUE+=("pool_d2_s${seed}|$seed|2|2|--axis_pool --axis_cummax --no-global_pool --input_skip")
done

run_from_spec() {
    local gpu=$1; local spec=$2
    IFS='|' read -r tag seed n_steps n_reps extra <<< "$spec"
    eval "run_one $gpu \"$tag\" --seed $seed --n_nca_steps $n_steps --n_nca_repeats $n_reps $extra"
}

echo "queue: ${#QUEUE[@]} runs"
i=0
while [ $i -lt ${#QUEUE[@]} ]; do
    s1="${QUEUE[$i]}"
    if [ $((i+1)) -lt ${#QUEUE[@]} ]; then
        s2="${QUEUE[$((i+1))]}"
        run_from_spec 0 "$s1" &
        pid0=$!
        run_from_spec 1 "$s2" &
        pid1=$!
        wait $pid0 $pid1
        i=$((i+2))
    else
        run_from_spec 0 "$s1"
        i=$((i+1))
    fi
done
echo "ALL DONE"

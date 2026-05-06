#!/usr/bin/env bash
# Varislide depth-extrapolation FOLLOW-UP: uniform-halt training.
#
# The fixed-D_train sweep showed the shared-weight body learns the
# per-iteration shift rule but is non-idempotent at the rest state — so
# D_eval > 2× D_train collapses. The fix per `_ponder_loss(halt_mode=uniform)`
# is to supervise *every* NCA step with the same target, which forces the
# body to be a fixed point at the rest state. With uniform halt and
# T_max = 32, the body should be valid for any D_eval ∈ [1, 32+] and the
# heatmap should light up the entire upper-right region instead of just
# the diagonal.
#
# Single bucket, 3 seeds. pool OFF + input_skip + mask_hidden, full
# multi-grid {6,8,10,12,16}x3, h=128, 10k updates, batch=16. T_max=32.

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
    
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --synthetic_levels 64 --synthetic_multi_grid
    --synthetic_grid_sizes "6x3,8x3,10x3,12x3,16x3"
    --no-synthetic_require_solvable
    --synthetic_min_states 5
    --synthetic_max_iters_search 5000 --synthetic_timeout_ms_search 2000
    --max_transitions_per_game 200000
    --n_updates 10000 --patience 0 --min_delta 1e-6
    --batch_size 16 --lr 3e-4
    --log_interval 1000 --ckpt_interval 5000
    --no-axis_pool --no-axis_cummax --no-global_pool --input_skip
    --n_nca_steps 32 --n_nca_repeats 32
    --adaptive_halt --halt_mode uniform --halt_kl_weight 0.0
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
for seed in 0 1 2; do
    QUEUE+=("nopool_uniform_T32_s${seed}|$seed")
done

run_from_spec() {
    local gpu=$1; local spec=$2
    IFS='|' read -r tag seed <<< "$spec"
    run_one "$gpu" "$tag" --seed "$seed"
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

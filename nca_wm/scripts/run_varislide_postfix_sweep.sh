#!/usr/bin/env bash
# Post-bitpack-fix (1fa557d) varislide ablation: depth, n_layers/n_repeats
# factor, and pooling on/off. All on multi-grid synth varislide
# (6x3,8x3,10x3,12x3,16x3) at h=128, 10k updates, batch=16, lr=3e-4.
# mask_hidden=True is the new default (per project_varislide_synth_transfer
# memory: required for translation-equivariance / authored-level transfer).
#
# Runs are dispatched two-at-a-time pinned to GPU 0 / GPU 1.
# Skips configs whose train_meta.json already exists.
#
# Buckets:
#   A: depth × seed sweep at full sharing (n_repeats = n_nca_steps)
#      depth ∈ {8, 16, 32}, seed ∈ {0, 1, 2}    → 9 runs
#   B: L × R factor at total=16, single seed
#      (L,R) ∈ {(1,16), (2,8), (4,4), (8,2), (16,1)} × seed ∈ {0, 1, 2} → 15 runs
#   C: pool OFF
#      depth ∈ {8, 16, 32} × seed ∈ {0, 1, 2}, --input_skip → 9 runs
#   C': pool OFF, no input_skip control
#      depth=16, seed=0 → 1 run

set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_canary
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# Common flags. Keep mask_hidden / mask_padded_loss explicit for clarity even
# though those are the post-1f59070 defaults.
COMMON=(
    --games varislide
    --conditional --architecture rule_attn
    --n_hid 128 --n_slots 16
    --mask_hidden --mask_padded_loss
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
)

run_one() {
    local gpu=$1; local tag=$2; shift 2
    local save_dir="$LOGDIR/varislide_$tag"
    local log="$LOGDIR/varislide_$tag.out"
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
    # Eval-time crash on OOD widths (L7=W19) is expected and non-fatal —
    # train_meta.json + params.pkl are saved before evaluate_multigame runs.
    if [ -f "$save_dir/params.pkl" ]; then
        echo "  [GPU $gpu] done $tag (train ok, eval rc=$rc)"
    else
        echo "  [GPU $gpu] FAILED $tag rc=$rc"
    fi
}

# Build queue. Each entry: "tag|seed|n_steps|n_repeats|extra_flags"
QUEUE=()

# A: depth × seed (fully-shared, n_repeats = n_steps)
for depth in 8 16 32; do
    for seed in 0 1 2; do
        QUEUE+=("postfixA_d${depth}_s${seed}|$seed|$depth|$depth|")
    done
done

# B: L × R factor at total=16
for spec in "1:16" "2:8" "4:4" "8:2" "16:1"; do
    L=${spec%:*}; R=${spec##*:}
    total=$((L*R))
    for seed in 0 1 2; do
        QUEUE+=("postfixB_L${L}R${R}_s${seed}|$seed|$total|$R|")
    done
done

# C: pool OFF + input_skip
for depth in 8 16 32; do
    for seed in 0 1 2; do
        QUEUE+=("postfixC_nopool_d${depth}_s${seed}|$seed|$depth|$depth|--no-axis_pool --no-axis_cummax --no-global_pool --input_skip")
    done
done

# C': pool OFF, no input_skip control
QUEUE+=("postfixCp_nopool_noskip_d16_s0|0|16|16|--no-axis_pool --no-axis_cummax --no-global_pool")

run_from_spec() {
    local gpu=$1; local spec=$2
    IFS='|' read -r tag seed n_steps n_reps extra <<< "$spec"
    # Use eval since extra may contain multiple flags
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

#!/usr/bin/env bash
# Nekopuzzle synth-level architecture sweep.
#
# Goal: with synthetically-generated training levels, find an architecture that
# generalizes perfectly to held-out (authored + held-out synth) levels of
# nekopuzzle. Sweep three axes:
#
#   - depth (n_nca_steps total)  ∈ {8, 16, 32}
#   - sharing                    ∈ {fully-shared (L=1, R=N), per-step (L=N, R=1)}
#   - pool                       ∈ {ON, OFF + input_skip}
#
# 3 × 2 × 2 = 12 configs. Recipe = h=256, batch=16, lr=3e-4, 15k updates,
# rule_attn, mask_hidden default, change_loss_weight=5.0, balanced_sampling.
# Synth: 64 levels at each of {5x5, 6x6, 7x7, 8x8} (multi-grid). Authored
# nekopuzzle is 8x7, so authored is held-out by aspect-ratio + layout.
#
# Two GPUs run in parallel; configs split alternately. Skip already-running.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_neko_arch
mkdir -p "$LOGDIR"

# 12 configs
CONFIGS=(
    # depth pool        sharing
    "8     pool        shared"
    "8     nopool      shared"
    "16    pool        shared"
    "16    nopool      shared"
    "32    pool        shared"
    "32    nopool      shared"
    "8     pool        perstep"
    "8     nopool      perstep"
    "16    pool        perstep"
    "16    nopool      perstep"
    "32    pool        perstep"
    "32    nopool      perstep"
)

run_one() {
    local gpu=$1
    local depth=$2
    local pool_kind=$3
    local share=$4

    if [ "$share" = "shared" ]; then
        local L=1
        local R=$depth
    else
        local L=$depth
        local R=1
    fi

    if [ "$pool_kind" = "pool" ]; then
        local pool_flags="--axis_pool --axis_cummax --global_pool"
        local skip_flags=""
    else
        local pool_flags="--no-axis_pool --no-axis_cummax --no-global_pool"
        local skip_flags="--input_skip"
    fi

    local tag="neko_d${depth}_${pool_kind}_${share}_s0"
    local save_dir="$LOGDIR/${tag}"
    local log="$LOGDIR/${tag}.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running pid=$pid"
            return
        fi
    fi
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/train_meta.json" ]; then
        echo "[$tag] SKIP — already complete"
        return
    fi

    echo "[$tag] gpu=$gpu starting (L=$L R=$R, $pool_kind, ${skip_flags:-no_skip})"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games nekopuzzle \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps "$depth" --n_nca_repeats "$R" \
        --n_slots 16 --n_app_slots 1 \
        $pool_flags $skip_flags \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels 64 --synthetic_multi_grid \
        --synthetic_grid_sizes "5x5,6x6,7x7,8x8" \
        --synthetic_min_states 5 \
        --synthetic_max_iters_search 5000 \
        --synthetic_timeout_ms_search 2000 \
        --max_transitions_per_game 100000 \
        --n_updates 15000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 500 --ckpt_interval 5000 \
        --save_dir "$save_dir" \
        --seed 0 \
        > "$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

# Schedule across two GPUs alternately.
GPU_A=0
GPU_B=1

# Run alternating: even index → GPU 0, odd → GPU 1, sequentially within each GPU.
PIDS=()

(
    for i in 0 2 4 6 8 10; do
        IFS=' ' read -ra parts <<< "${CONFIGS[$i]}"
        run_one "$GPU_A" "${parts[0]}" "${parts[1]}" "${parts[2]}"
    done
    echo "[gpu0] queue done"
) &
PIDS+=($!)

(
    for i in 1 3 5 7 9 11; do
        IFS=' ' read -ra parts <<< "${CONFIGS[$i]}"
        run_one "$GPU_B" "${parts[0]}" "${parts[1]}" "${parts[2]}"
    done
    echo "[gpu1] queue done"
) &
PIDS+=($!)

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo "[master] neko synth sweep done."

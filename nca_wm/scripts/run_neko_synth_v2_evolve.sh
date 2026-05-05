#!/usr/bin/env bash
# Followup to nekopuzzle synth-arch sweep: take the best architecture
# (d=16/perstep/pool=ON) and vary the synth-gen recipe to test the
# "synth distribution is the bottleneck" hypothesis.
#
# Configs:
#   evolve_n64  : evolve-mode synth, 64 total levels (16/size at multi-grid)
#   evolve_n256 : evolve-mode synth, 256 total levels (64/size at multi-grid)
#
# Both compared against the existing tile_pattern_empirical n=64 baseline
# (neko_d16_pool_perstep_s0 from the original sweep).

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_neko_arch
mkdir -p "$LOGDIR"

run_one() {
    local gpu=$1
    local mode=$2
    local n_levels=$3
    local tag=$4
    local save_dir="$LOGDIR/${tag}"
    local log="$LOGDIR/${tag}.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running pid=$pid"; return
        fi
    fi
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/train_meta.json" ]; then
        echo "[$tag] SKIP — already complete"; return
    fi

    local extra_evolve_flags=""
    if [ "$mode" = "evolve" ]; then
        extra_evolve_flags="--synthetic_evolve_pop_size 32 --synthetic_evolve_max_generations 100"
    fi

    echo "[$tag] gpu=$gpu starting (mode=$mode, n=$n_levels)"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games nekopuzzle \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps 16 --n_nca_repeats 1 \
        --n_slots 16 --n_app_slots 1 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --synthetic_levels "$n_levels" --synthetic_multi_grid \
        --synthetic_grid_sizes "5x5,6x6,7x7,8x8" \
        --synthetic_min_states 5 \
        --synthetic_max_iters_search 5000 \
        --synthetic_timeout_ms_search 2000 \
        --synthetic_mode "$mode" \
        $extra_evolve_flags \
        --max_transitions_per_game 100000 \
        --n_updates 15000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 500 --ckpt_interval 5000 \
        --save_dir "$save_dir" \
        --seed 0 \
        > "$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

# Two GPUs in parallel
(
    run_one 0 evolve 64 "neko_d16_pool_perstep_evolve_n64_s0"
    run_one 0 evolve 256 "neko_d16_pool_perstep_evolve_n256_s0"
    echo "[gpu0] queue done"
) &
PID_A=$!

(
    # GPU 1: tile_pattern_empirical with bumped n=256 (more variety, same mode)
    run_one 1 tile_pattern_empirical 256 "neko_d16_pool_perstep_tpe_n256_s0"
    echo "[gpu1] queue done"
) &
PID_B=$!

wait "$PID_A"
wait "$PID_B"
echo "[master] neko v2 followup done."

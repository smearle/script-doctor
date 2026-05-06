#!/usr/bin/env bash
# Synth-multi-grid pool / input_skip ablation on the NCA-shared recipe.
#
# Decomposes the NCA-shared advantage on the multi-grid synth setting
# (Table 2 in the paper) by dropping each pool feature and the
# input-skip individually, plus an "all pool features off" config.
#
# Compared against the existing baseline runs at
# nca_wm/logs_baselines/sokoban_basic_synth_nca_shared_s{0,1,2}, which
# encode the full recipe (--axis_pool --axis_cummax --global_pool
# --input_skip).
#
# 5 ablation configs × N_SEEDS seeds. Sequential on GPU 0.

set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_baselines
mkdir -p "$LOGDIR"

N_SEEDS=${N_SEEDS:-3}
N_UPDATES=${N_UPDATES:-10000}

GAMES=sokoban_basic
SYNTH=(--synthetic_levels 256 --synthetic_per_game_size --synthetic_multi_grid
       --synthetic_grid_sizes 5x5,6x6,7x7,8x8
       --synthetic_fallback_dynamics --synthetic_no_a_count_max 5
       --token_decoder_loss_weight 0.1 )

SHARED=(--n_hid 256 --n_slots 16 --d_slot 64
        --d_model 64 --n_enc_layers 2 --n_heads 4
        --change_loss_weight 5.0 --grad_clip 0.5
        --balanced_sampling --patience 200 --min_delta 1e-6
        --lr 3e-4 --lr_schedule cosine
        --architecture rule_attn --n_nca_steps 4 --n_nca_repeats 4)

run_one() {
    local tag=$1
    local seed=$2
    shift 2
    local pool_args=("$@")
    local save_dir="$LOGDIR/sokoban_basic_synth_${tag}_s${seed}"
    local log="$save_dir.out"

    if [ -f "$save_dir/curves_step${N_UPDATES}.npz" ]; then
        echo "[$tag s=$seed] SKIP — already done"
        return
    fi
    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid
        pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag s=$seed] SKIP — running (pid $pid)"
            return
        fi
    fi
    echo "[$tag s=$seed] starting in $save_dir"
    mkdir -p "$save_dir"
    echo $$ > "$save_dir/RUNNING.pid"
    "$PY" -m nca_wm.train \
        --games "$GAMES" \
        "${SHARED[@]}" \
        "${SYNTH[@]}" \
        "${pool_args[@]}" \
        --n_updates "$N_UPDATES" \
        --batch_size 32 --seed "$seed" \
        --log_interval 100 \
        --save_dir "$save_dir" \
        > "$log" 2>&1
    rm -f "$save_dir/RUNNING.pid"
}

for seed in $(seq 0 $((N_SEEDS - 1))); do
    # 1. No axis_pool (keeps cummax + global + input_skip)
    run_one "nca_shared_noap"  "$seed" \
        --no-axis_pool --axis_cummax --global_pool --input_skip

    # 2. No axis_cummax (keeps axis_pool + global + input_skip)
    run_one "nca_shared_noac"  "$seed" \
        --axis_pool --no-axis_cummax --global_pool --input_skip

    # 3. No global_pool (keeps axis + cummax + input_skip)
    run_one "nca_shared_nogp"  "$seed" \
        --axis_pool --axis_cummax --no-global_pool --input_skip

    # 4. No input_skip (keeps full pool stack)
    run_one "nca_shared_nois"  "$seed" \
        --axis_pool --axis_cummax --global_pool

    # 5. No pool at all (keeps input_skip)
    run_one "nca_shared_nopool" "$seed" \
        --no-axis_pool --no-axis_cummax --no-global_pool --input_skip
done
echo "[ablation] launched ${N_SEEDS} seeds × 5 configs = $((N_SEEDS * 5)) runs"

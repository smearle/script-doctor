#!/usr/bin/env bash
# Baselines-vs-NCA comparison for the NeurIPS paper.
#
# Compares the rule_attn NCA against three non-NCA architectures (CNN, U-Net,
# ViT) on two settings:
#
#   (A) single-game multi-grid synth on sokoban_basic — clean controlled
#       setting where the validated v3 synth recipe is well-understood.
#       Shows whether the NCA's iterated-local-update inductive bias matters
#       on a single dynamics ruleset.
#
#   (B) multi-game scaling_14 preset — 14 PuzzleScript games. Tests whether
#       the NCA still wins when the model has to share dynamics across rule
#       sets that all enter via the same slot-encoder conditioner.
#
# Each architecture runs at matched n_hid; we report parameter count alongside
# every metric so the comparison is honest. NCA runs in two modes:
#   - "nca_shared":  n_repeats=n_steps (paper default)  ~smallest param count
#   - "nca_perstep": n_repeats=1 (no sharing)           ~matches CNN params
#
# Set N_SEEDS to control replication; defaults to 3.
#
# Pass "single" or "multi" as $1 to run only that setting; default = "single"
# (faster, runs fully on one GPU). Multi-game runs take longer and should
# generally be launched separately on GPU 1.

set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_baselines
mkdir -p "$LOGDIR"

WHICH=${1:-single}
N_SEEDS=${N_SEEDS:-3}

# Shared training recipe — derived from ARCHITECTURE_REPORT default recipe.
SHARED=(
    --n_hid 256
    --n_slots 16 --d_slot 64
    --d_model 64 --n_enc_layers 2 --n_heads 4
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --patience 200 --min_delta 1e-6
    --lr 3e-4 --lr_schedule cosine
)

run_arch() {
    local tag=$1                                  # short name for path
    local seed=$2
    local games=$3                                # --games arg value
    local n_updates=$4
    local synth_args=$5                           # may be empty
    shift 5
    local arch_args=("$@")                        # remaining = arch-specific
    local save_dir="$LOGDIR/${games_tag}_${tag}_s${seed}"
    local log="$save_dir.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid
        pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag s=$seed] SKIP — already running (pid $pid)"
            return
        fi
    fi
    if [ -f "$save_dir/curves.npz" ] || [ -f "$save_dir/curves_step${n_updates}.npz" ]; then
        echo "[$tag s=$seed] SKIP — already done"
        return
    fi

    echo "[$tag s=$seed] starting in $save_dir (log: $log)"
    mkdir -p "$save_dir"
    echo $$ > "$save_dir/RUNNING.pid"
    # Sequential: block until the run finishes so only one job runs on the GPU
    # at a time. Wrap the whole launcher in `nohup ... &` if you want to
    # background the sweep itself.
    # shellcheck disable=SC2086
    "$PY" -m nca_wm.train \
        --games "$games" \
        "${SHARED[@]}" \
        ${synth_args} \
        --n_updates "$n_updates" \
        --batch_size 32 \
        --seed "$seed" \
        --save_dir "$save_dir" \
        --log_interval 100 \
        "${arch_args[@]}" \
        > "$log" 2>&1
    rm -f "$save_dir/RUNNING.pid"
}

# ---- (A) Single-game comparison ------------------------------------------
# Two modes selected by env var SINGLE_MODE:
#   "authored" (default): microban 10 authored levels. No synth code path.
#       Useful as a sanity check; small dataset → fast runs.
#   "synth":   sokoban_basic + multi-grid synthetic levels. The validated
#       v3 synth recipe; requires the rebuilt C++ binding (track_rules_fired
#       kwarg). The headline comparison: model has to learn rules, not memorize.
SINGLE_MODE=${SINGLE_MODE:-authored}
if [ "$WHICH" = "single" ] || [ "$WHICH" = "all" ]; then
    if [ "$SINGLE_MODE" = "synth" ]; then
        games_tag="sokoban_basic_synth"
        GAMES="sokoban_basic"
        N_UPDATES=${N_UPDATES_SINGLE:-10000}
        SYNTH="--synthetic_levels 256 --synthetic_per_game_size --synthetic_multi_grid \
            --synthetic_grid_sizes 5x5,6x6,7x7,8x8 \
            --synthetic_fallback_dynamics --synthetic_no_a_count_max 5 \
            --token_decoder_loss_weight 0.1 --mask_hidden"
    else
        games_tag="microban_authored"
        GAMES="microban"
        N_UPDATES=${N_UPDATES_SINGLE:-10000}
        SYNTH="--mask_hidden"
    fi

    for seed in $(seq 0 $((N_SEEDS - 1))); do
        # NCA — paper default (shared weights).
        run_arch "nca_shared" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture rule_attn --n_nca_steps 4 --n_nca_repeats 4 --input_skip \
            --axis_pool --axis_cummax --global_pool

        # NCA — per-step weights (no sharing) at matched depth = a stronger NCA.
        run_arch "nca_perstep" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture rule_attn --n_nca_steps 4 --n_nca_repeats 1 --input_skip \
            --axis_pool --axis_cummax --global_pool

        # CNN — 4 residual blocks, same pool feature stack. Matched-depth
        # to NCA n_steps=4; param-matched to NCA-perstep.
        run_arch "cnn_d4" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture cnn --baseline_n_blocks 4 \
            --axis_pool --axis_cummax --global_pool

        # U-Net — 2 levels of down/up sampling.
        run_arch "unet_l2" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture unet --baseline_n_levels 2

        # ViT — 4 transformer encoder layers (matched depth).
        run_arch "vit_l4" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture vit --baseline_n_layers 4
    done
    echo "[single] launched ${N_SEEDS} seeds × 5 architectures = $((N_SEEDS * 5)) runs"
fi

# ---- (B) Multi-game scaling_14 preset -------------------------------------
if [ "$WHICH" = "multi" ] || [ "$WHICH" = "all" ]; then
    games_tag="scaling_14"
    GAMES="scaling_14"
    N_UPDATES=${N_UPDATES_MULTI:-80000}
    SYNTH=""  # multi-game uses authored data only for now

    for seed in $(seq 0 $((N_SEEDS - 1))); do
        run_arch "nca_shared" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture rule_attn --n_nca_steps 4 --n_nca_repeats 4 --input_skip \
            --axis_pool --axis_cummax --global_pool --mask_hidden

        run_arch "nca_perstep" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture rule_attn --n_nca_steps 4 --n_nca_repeats 1 --input_skip \
            --axis_pool --axis_cummax --global_pool --mask_hidden

        run_arch "cnn_d4" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture cnn --baseline_n_blocks 4 \
            --axis_pool --axis_cummax --global_pool --mask_hidden

        run_arch "unet_l2" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture unet --baseline_n_levels 2 --mask_hidden

        run_arch "vit_l4" "$seed" "$GAMES" "$N_UPDATES" "$SYNTH" \
            --architecture vit --baseline_n_layers 4 --mask_hidden
    done
    echo "[multi] launched ${N_SEEDS} seeds × 5 architectures = $((N_SEEDS * 5)) runs"
fi

echo "Tail logs in $LOGDIR/*.out  |  per-run save dirs in $LOGDIR/<tag>_s<seed>/"

#!/usr/bin/env bash
# Multi-grid A/B: does padding mask matter when training samples DO vary in
# size? Same recipe as run_mask_ab.sh except --synthetic_multi_grid with four
# sizes. Each sample's padding pattern differs, so the masks are non-trivial.
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_mask_ab_mg
EVAL_SUBDIR=heldout_eval_dual_control
mkdir -p "$LOGDIR"

run_one() {
    local tag=$1
    shift
    local extra_args=("$@")
    local save_dir="$LOGDIR/mg_${tag}_seed0"
    local log="$LOGDIR/mg_${tag}_seed0.out"
    local train_done=0

    if "$PY" -c 'import json,sys; from pathlib import Path
d=Path(sys.argv[1]); meta=d/"train_meta.json"; cfg=d/"config.json"; ok=False
if meta.is_file() and cfg.is_file():
    m=json.load(open(meta)); ok=int(m.get("total_steps",0)) >= 20000
print(1 if ok else 0)' "$save_dir" | grep -q 1; then
        train_done=1
    fi

    if [ "$train_done" -eq 0 ]; then
        echo "[$tag] training -> $save_dir"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
            --games scaling_1 \
            --conditional --architecture rule_attn \
            --n_hid 256 --n_nca_steps 4 --n_slots 16 --n_app_slots 1 \
            --axis_pool --axis_cummax --global_pool \
            --change_loss_weight 5.0 --grad_clip 0.5 \
            --balanced_sampling \
            --synthetic_levels 256 \
            --synthetic_multi_grid --synthetic_grid_sizes "6x7,7x7,7x8,8x8" \
            --synthetic_seed 0 \
            --synthetic_mode tile_pattern_empirical \
            --synthetic_evolve_pop_size 64 \
            --synthetic_evolve_max_generations 200 \
            --synthetic_evolve_n_mutations_min 1 \
            --synthetic_evolve_n_mutations_max 3 \
            --synthetic_require_solvable \
            --synthetic_min_states 20 \
            --synthetic_max_attempts_per_level 1000 \
            --synthetic_max_iters_search 5000 \
            --synthetic_timeout_ms_search 2000 \
            --n_search_steps 100000 --search_timeout_ms 60000 \
            --max_transitions_per_game 200000 \
            --n_updates 20000 --patience 200 --min_delta 1e-8 \
            --batch_size 64 --lr 3e-4 \
            --log_interval 2000 --ckpt_interval 5000 \
            "${extra_args[@]}" \
            --save_dir "$save_dir" \
            --seed 0 \
            >"$log" 2>&1
        local train_exit=$?
        echo "[$tag] train exit=$train_exit"
        if [ "$train_exit" -ne 0 ]; then
            return $train_exit
        fi
    else
        echo "[$tag] training already complete -> $save_dir"
    fi

    if [ ! -f "$save_dir/$EVAL_SUBDIR/results.json" ]; then
        echo "[$tag] heldout eval"
        "$PY" "$REPO/nca_wm/heldout_eval.py" \
            --load "$save_dir" \
            --heldout_games sokoban_basic,Microban,Microban_I \
            --n_random_episodes 5 \
            --max_steps 30 \
            --include_train_sample 1 \
            --allow_training_games \
            --out_subdir "$EVAL_SUBDIR" \
            >>"$log" 2>&1
        echo "[$tag] eval exit=$?"
    else
        echo "[$tag] heldout eval already complete"
    fi
}

run_one nomask
run_one mask --mask_hidden --mask_padded_loss

echo "[master] done. Logs in $LOGDIR"

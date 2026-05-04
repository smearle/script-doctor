#!/usr/bin/env bash
# Test whether 8x8 synthetic sokoban failure is level/position memorization.
#
# Baseline symptom:
#   synth_sokoban_n256_w8h8_v7_seed0 trains well on 8x8 synthetic levels but
#   fails the authored 6x7 sokoban_basic control and is much worse on Microban
#   than the matched-size 7x7 synthetic run.
#
# Two direct probes:
#   1. More synthetic levels: if it is memorizing finite layouts, larger N
#      should recover authored/Microban transfer.
#   2. Less NCA depth: if the NCA stack is using depth/capacity to fit
#      grid-position artifacts, n_steps={1,2} should force a simpler/local rule.
#
# After every train run, heldout_eval scores Microban/Microban_I and includes
# the single training game as an authored sokoban_basic control.
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_sokoban_w8_mem_v3
EVAL_SUBDIR=heldout_eval_dual_control
MASK_OPTIONS=${MASK_OPTIONS:-"0 1"}
mkdir -p "$LOGDIR"

run_one() {
    local n_levels=$1
    local n_steps=$2
    local mask_pad=$3
    local suffix=""
    local mask_args=()
    if [ "$mask_pad" -eq 1 ]; then
        suffix="_maskpad"
        mask_args=(--mask_padded_loss)
    fi
    local save_dir="$LOGDIR/sokoban_w8_n${n_levels}_nca${n_steps}_seed0${suffix}"
    local log="$LOGDIR/sokoban_w8_n${n_levels}_nca${n_steps}_seed0${suffix}.out"
    local train_done=0

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid
        pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            local cmdline
            cmdline=$(ps -p "$pid" -o args= 2>/dev/null || echo "")
            if printf '%s' "$cmdline" | grep -F "$save_dir" >/dev/null 2>&1; then
                echo "[n=$n_levels depth=$n_steps mask=$mask_pad] SKIP - already running (pid $pid)"
                return
            fi
        fi
        echo "[n=$n_levels depth=$n_steps mask=$mask_pad] removing stale RUNNING.pid"
        rm -f "$save_dir/RUNNING.pid"
    fi
    if "$PY" -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); meta=d/"train_meta.json"; cfg=d/"config.json"; ok=False
if meta.is_file() and cfg.is_file():
    m=json.load(open(meta)); c=json.load(open(cfg))
    ok=int(m.get("total_steps",0)) >= int(sys.argv[2]) and "vocab_size" in c
print(1 if ok else 0)' "$save_dir" 20000 | grep -q 1; then
        train_done=1
    fi

    if [ "$train_done" -eq 1 ] && [ -f "$save_dir/$EVAL_SUBDIR/results.json" ]; then
        echo "[n=$n_levels depth=$n_steps mask=$mask_pad] SKIP - train+eval already complete"
        return
    fi

    if [ "$train_done" -eq 1 ]; then
        echo "[n=$n_levels depth=$n_steps mask=$mask_pad] training already complete -> $save_dir"
    else
        echo "[n=$n_levels depth=$n_steps mask=$mask_pad] training -> $save_dir"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
            --games scaling_1 \
            --conditional --architecture rule_attn \
            --n_hid 256 --n_nca_steps "$n_steps" --n_slots 16 --n_app_slots 1 \
            --axis_pool --axis_cummax --global_pool \
            --change_loss_weight 5.0 --grad_clip 0.5 \
            --balanced_sampling \
            --synthetic_levels "$n_levels" \
            --synthetic_w 8 --synthetic_h 8 \
            --synthetic_seed 0 \
            --synthetic_mode evolve \
            --synthetic_evolve_pop_size 32 \
            --synthetic_evolve_max_generations 200 \
            --synthetic_require_solvable \
            --synthetic_min_states 10 \
            --synthetic_max_iters_search 5000 \
            --synthetic_timeout_ms_search 2000 \
            --n_search_steps 100000 --search_timeout_ms 60000 \
            --max_transitions_per_game 200000 \
            --n_updates 20000 --patience 200 --min_delta 1e-8 \
            --batch_size 64 --lr 3e-4 \
            --log_interval 2000 --ckpt_interval 5000 \
            "${mask_args[@]}" \
            --save_dir "$save_dir" \
            --seed 0 \
            >"$log" 2>&1
        local train_exit=$?
        echo "[n=$n_levels depth=$n_steps mask=$mask_pad] train exit=$train_exit"
        if [ "$train_exit" -ne 0 ]; then
            return
        fi
    fi

    echo "[n=$n_levels depth=$n_steps mask=$mask_pad] heldout eval"
    "$PY" "$REPO/nca_wm/heldout_eval.py" \
        --load "$save_dir" \
        --heldout_games sokoban_basic,Microban,Microban_I \
        --n_random_episodes 5 \
        --max_steps 30 \
        --include_train_sample 1 \
        --allow_training_games \
        --out_subdir "$EVAL_SUBDIR" \
        >>"$log" 2>&1
    echo "[n=$n_levels depth=$n_steps mask=$mask_pad] eval exit=$?"
}

# Keep the first pass small enough to answer the question quickly.
# Add 1024 only after the lower-N/depth signal is clear.
for n_levels in 64 256 512; do
    for n_steps in 1 2 4; do
        for mask_pad in $MASK_OPTIONS; do
            run_one "$n_levels" "$n_steps" "$mask_pad"
        done
    done
done

"$PY" "$REPO/nca_wm/scripts/summarize_sokoban_w8_memorization_sweep.py" \
    --root "$LOGDIR" \
    --out "$LOGDIR/summary.md"

echo "[master] done. Summary: $LOGDIR/summary.md"

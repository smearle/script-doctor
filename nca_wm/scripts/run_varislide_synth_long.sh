#!/usr/bin/env bash
# Longer/bigger varislide synth run — does multi-size synth + sufficient
# compute let the model actually learn the slide rule (vs being under-trained
# at 10k/h=128 as in E15)?
set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch
mkdir -p "$LOGDIR"

SAVE_DIR="$LOGDIR/varislide_synth_pool_long_seed0"
LOG="$LOGDIR/varislide_synth_pool_long_seed0.out"
if [ -f "$SAVE_DIR/RUNNING.pid" ]; then
    pid=$(cat "$SAVE_DIR/RUNNING.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "skip"; exit 0; fi
fi

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games varislide \
    --conditional --architecture rule_attn \
    --n_hid 256 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
    --axis_pool --axis_cummax --global_pool \
    --change_loss_weight 5.0 --grad_clip 0.5 \
    --balanced_sampling \
    --synthetic_levels 64 --synthetic_multi_grid \
    --synthetic_grid_sizes "6x3,8x3,10x3,12x3,16x3" \
    --no-synthetic_require_solvable \
    --synthetic_min_states 5 \
    --n_search_steps 50000 --search_timeout_ms 60000 \
    --max_transitions_per_game 200000 \
    --n_updates 50000 --patience 0 --min_delta 1e-6 \
    --batch_size 16 --lr 3e-4 \
    --log_interval 500 --ckpt_interval 5000 \
    --save_dir "$SAVE_DIR" \
    --seed 0 \
    >"$LOG" 2>&1
echo "exit=$?"

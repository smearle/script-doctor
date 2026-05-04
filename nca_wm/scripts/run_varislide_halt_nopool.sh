#!/usr/bin/env bash
# Within-game adaptive-halt on varislide with all pool features OFF.
#
# Companion to run_varislide_halt.sh. The pool-on variant showed E[k]
# essentially constant (~4.75) across slide_distance ∈ {1,...,16} —
# pool features (axis_cummax in particular) let the model resolve
# "where's the wall?" in 1 NCA step, so depth wasn't binding and
# adaptive halt had no signal to learn per-instance variation.
#
# With --no-axis_pool --no-axis_cummax --no-global_pool, the NCA can
# only propagate one cell of context per step, so to predict where the
# player ends up it MUST iterate ≥ slide_distance times. Per F1 this
# regime caused identity-collapse on Collapse, but varislide is simpler
# (1 rule, no obstacles in the chain) — it should be trainable.
#
# Hypothesis: E[halt step] now tracks slide_distance.
set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch
mkdir -p "$LOGDIR"

SAVE_DIR="$LOGDIR/varislide_halt_nopool_seed0"
LOG="$LOGDIR/varislide_halt_nopool_seed0.out"

if [ -f "$SAVE_DIR/RUNNING.pid" ]; then
    pid=$(cat "$SAVE_DIR/RUNNING.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[varislide_nopool] SKIP — already running (pid $pid)"
        exit 0
    fi
fi

echo "[varislide_nopool] starting"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games varislide \
    --conditional --architecture rule_attn \
    --n_hid 128 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
    --adaptive_halt --halt_prior_p 0.2 --halt_kl_weight 0.01 \
    --no-axis_pool --no-axis_cummax --no-global_pool \
    --change_loss_weight 5.0 --grad_clip 0.5 \
    --balanced_sampling \
    --n_search_steps 20000 --search_timeout_ms 60000 \
    --max_transitions_per_game 100000 \
    --n_updates 8000 --patience 0 --min_delta 1e-6 \
    --batch_size 16 --lr 3e-4 \
    --log_interval 200 --ckpt_interval 1000 \
    --save_dir "$SAVE_DIR" \
    --seed 0 \
    >"$LOG" 2>&1
echo "[varislide_nopool] finished exit=$? -> $SAVE_DIR"

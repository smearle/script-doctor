#!/usr/bin/env bash
# Within-game adaptive-halt experiment on the custom `varislide` game.
#
# varislide is a single-rule slide game (player slides right via `again`
# until hitting a wall) with multiple levels placing the player at
# different distances from the wall (1, 2, 3, 4, 6, 8, 12, 16). Trained
# jointly on all levels so the same model has to handle the full
# distance gradient.
#
# Hypothesis: at eval, the halt distribution per (level, action) should
# have E[k] ≈ slide distance for "right" transitions and E[k] ≈ 1 for
# any other action (which is a no-op here).
#
# Single run — the within-game per-instance variation is the test, no
# need to sweep additional knobs.
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch
mkdir -p "$LOGDIR"

SAVE_DIR="$LOGDIR/varislide_halt_seed0"
LOG="$LOGDIR/varislide_halt_seed0.out"

if [ -f "$SAVE_DIR/RUNNING.pid" ]; then
    pid=$(cat "$SAVE_DIR/RUNNING.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[varislide] SKIP — already running (pid $pid)"
        exit 0
    fi
fi

echo "[varislide] starting"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games varislide \
    --conditional --architecture rule_attn \
    --n_hid 128 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
    --adaptive_halt --halt_prior_p 0.2 --halt_kl_weight 0.01 \
    --axis_pool --axis_cummax --global_pool \
    --change_loss_weight 5.0 --grad_clip 0.5 \
    --balanced_sampling \
    --n_search_steps 20000 --search_timeout_ms 60000 \
    --max_transitions_per_game 100000 \
    --n_updates 5000 --patience 0 --min_delta 1e-6 \
    --batch_size 16 --lr 3e-4 \
    --log_interval 200 --ckpt_interval 1000 \
    --save_dir "$SAVE_DIR" \
    --seed 0 \
    >"$LOG" 2>&1
echo "[varislide] finished exit=$? -> $SAVE_DIR"

#!/usr/bin/env bash
# Sweep n_nca_steps on nekopuzzle alone to test whether `...` rules
# (info-propagation across rows/columns) are the bottleneck.
# Sequential on GPU 1; balanced_sampling auto-disables (1 game).

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

GPU=1
STEPS=(8 16 32)

for s in "${STEPS[@]}"; do
    log="$LOGDIR/train_neko_steps${s}_gpu${GPU}.log"
    echo "[GPU $GPU] starting neko n_nca_steps=$s -> $log"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games scaling_1_neko \
        --conditional \
        --n_hid 128 \
        --n_nca_steps "$s" \
        --n_updates 200000 \
        --wandb \
        --sweep_name nca_steps_neko \
        >"$log" 2>&1
    echo "[GPU $GPU] finished neko n_nca_steps=$s (exit=$?)"
done
echo "All n_nca_steps configs finished."

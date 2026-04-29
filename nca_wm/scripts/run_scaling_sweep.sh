#!/usr/bin/env bash
# Launches n_games_scaling sweep on GPU 1 only (GPU 0 in use by labmate).
# Configs run sequentially. Uses balanced_sampling to fix per-game imbalance.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

GPU=1
CONFIGS=(scaling_1 scaling_2 scaling_4 scaling_6 small)

for games in "${CONFIGS[@]}"; do
    log="$LOGDIR/train_${games}_bal_gpu${GPU}.log"
    echo "[GPU $GPU] starting $games (balanced) -> $log"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$REPO/nca_wm/train.py" \
        --games "$games" \
        --conditional \
        --balanced_sampling \
        --n_hid 128 \
        --n_updates 200000 \
        --wandb \
        --sweep_name n_games_scaling \
        >"$log" 2>&1
    echo "[GPU $GPU] finished $games (exit=$?)"
done
echo "All scaling configs finished."

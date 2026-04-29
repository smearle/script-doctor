#!/usr/bin/env bash
# Retry clearing training with --max_transitions_per_game 200000 after
# nirvana finishes on GPU 1. The previous clearing run OOM'd assembling
# a 3.8M-transition merged dataset; capping per-game cuts that to 200K.

set -u
REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
GPU=1

# Wait for nirvana to finish
while pgrep -f 'nca_wm/train.py --games global_nirvana' >/dev/null; do
    sleep 60
done
echo "[retry-clearing] nirvana finished; starting clearing with transition cap"

# Clean the half-made clearing dir from the OOM run (it only has config.json,
# no checkpoint; safe to wipe for a clean retry).
save_dir="$REPO/nca_wm/logs/multi_global_clearing_cond_bal_ap_ac_gp_level-None_nca-4_hid-128_lr-0.001_pat-80_s-0"
rm -rf "$save_dir"

log="$LOGDIR/followup_global_clearing_CAPPED_gpu1.log"
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games global_clearing \
    --conditional \
    --balanced_sampling \
    --axis_pool --axis_cummax --global_pool \
    --grad_clip 1.0 \
    --n_hid 128 \
    --n_nca_steps 4 \
    --max_transitions_per_game 200000 \
    --n_updates 80000 \
    --patience 80 --min_delta 1e-6 \
    --wandb \
    --sweep_name "arch_followup_with_gradclip" \
    >"$log" 2>&1
echo "[retry-clearing] clearing finished, exit=$?"

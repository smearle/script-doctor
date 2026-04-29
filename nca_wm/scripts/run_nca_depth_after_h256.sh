#!/usr/bin/env bash
# Queue scaling_large with deeper NCA (n_nca_steps=8) on GPU 1 after the
# current FiLM h256 run exits eval. Direct ablation against the just-
# completed n_nca_steps=4 run (which finished at change_err=19% with worst
# games kettle/Travelling/Modality at 47-81% — all cascading-dynamics games
# the 4-step NCA likely can't simulate).
#
# Hypothesis: games with multi-stage rule applications per PS tick need
# n_nca_steps >= the longest rule chain. 8 should cover most gallery games.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs

# Wait for GPU 1 h256 FiLM to finish (eval phase)
while pgrep -f 'nca_wm/train\.py.*--games scaling_large.*--n_hid 256.*--sweep_name post_encfix' >/dev/null; do
    sleep 60
done
echo "[depth] GPU 1 freed; starting scaling_large @ n_nca_steps=8"

log="$LOGDIR/ncadepth_scaling_large_h256_nca8_gpu1.log"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games scaling_large \
    --conditional \
    --balanced_sampling \
    --axis_pool --axis_cummax --global_pool \
    --n_hid 256 \
    --n_nca_steps 8 \
    --grad_clip 1.0 \
    --max_transitions_per_game 200000 \
    --change_loss_weight 5.0 \
    --n_updates 120000 \
    --patience 120 --min_delta 1e-6 \
    --wandb \
    --sweep_name "nca_depth" \
    >"$log" 2>&1
echo "[depth] exit=$?"

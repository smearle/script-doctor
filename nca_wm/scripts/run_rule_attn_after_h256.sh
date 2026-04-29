#!/usr/bin/env bash
# Queue scaling_large @ rule_attn on GPU 1 after the current h256 FiLM run
# finishes its eval phase. This is the vector-latent vs rule-latent A/B for
# the paper comparison. Uses identical hyperparameters (clw=5, encfix) so
# the only varying axis is the conditioning architecture.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs

# Wait for the scaling_large FiLM run on GPU 1 to exit
while pgrep -f 'nca_wm/train\.py.*--games scaling_large.*--n_hid 256.*--sweep_name post_encfix' >/dev/null; do
    sleep 60
done
echo "[rule_attn] GPU 1 FiLM scaling_large@h256 finished; starting rule_attn"

log="$LOGDIR/rule_attn_scaling_large_h256_gpu1.log"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games scaling_large \
    --conditional \
    --balanced_sampling \
    --axis_pool --axis_cummax --global_pool \
    --architecture rule_attn \
    --n_slots 16 --d_slot 64 \
    --n_hid 256 \
    --n_nca_steps 4 \
    --grad_clip 1.0 \
    --max_transitions_per_game 200000 \
    --change_loss_weight 5.0 \
    --n_updates 120000 \
    --patience 120 --min_delta 1e-6 \
    --wandb \
    --sweep_name "rule_attn_vs_film" \
    >"$log" 2>&1
echo "[rule_attn] exit=$?"

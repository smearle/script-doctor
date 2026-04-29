#!/usr/bin/env bash
# After wide_film eval finishes, run rule_attn @ K=32 (double the slots)
# on GPU 1. Tests whether the rule-attention win comes from "more
# conditioning capacity" (would scale with K) or "structured conditioning
# at any K" (would saturate quickly). With K=16 we got 9.4%; if K=32 gets
# substantially lower, slot count is the lever.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs

while pgrep -f 'nca_wm/train\.py.*--sweep_name wide_film' >/dev/null; do
    sleep 60
done
echo "[k32] GPU 1 freed; starting rule_attn @ K=32"

log="$LOGDIR/rule_attn_k32_scaling_large_h256_gpu1.log"
save_dir="$REPO/nca_wm/logs/multi_scaling_large_cond_bal_ap_ac_gp_clw5_arch-rule_attn_K32_level-None_nca-4_hid-256_lr-0.001_pat-120_s-0"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games scaling_large \
    --conditional \
    --balanced_sampling \
    --axis_pool --axis_cummax --global_pool \
    --architecture rule_attn \
    --n_slots 32 --d_slot 64 \
    --n_hid 256 \
    --n_nca_steps 4 \
    --grad_clip 1.0 \
    --max_transitions_per_game 200000 \
    --change_loss_weight 5.0 \
    --n_updates 120000 \
    --patience 120 --min_delta 1e-6 \
    --save_dir "$save_dir" \
    --wandb \
    --sweep_name "rule_attn_K_scaling" \
    >"$log" 2>&1
echo "[k32] exit=$?"

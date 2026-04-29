#!/usr/bin/env bash
# Queue scaling_large with WIDER FiLM encoder on GPU 1 after current h256
# FiLM run exits eval. Direct ablation: is the 64-d z bottleneck limiting
# multi-game fit? 1-vector conditioning might be too narrow for 19
# distinct games' rule sets.
#
# Delta vs. canonical: d_z 64 -> 256, d_model 64 -> 128, n_enc_layers 2 -> 4.
# Keeps everything else the same (encfix, clw=5, h=256, n_nca_steps=4).

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs

while pgrep -f 'nca_wm/train\.py.*--games scaling_large.*--n_hid 256.*--sweep_name post_encfix' >/dev/null; do
    sleep 60
done
echo "[wide_film] GPU 1 freed; starting wider-FiLM run"

log="$LOGDIR/wide_film_scaling_large_h256_gpu1.log"
save_dir="$REPO/nca_wm/logs/multi_scaling_large_cond_bal_ap_ac_gp_clw5_widefilm_level-None_nca-4_hid-256_dz-256_dmodel-128_encL-4_lr-0.001_pat-120_s-0"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
    --games scaling_large \
    --conditional \
    --balanced_sampling \
    --axis_pool --axis_cummax --global_pool \
    --n_hid 256 \
    --n_nca_steps 4 \
    --d_z 256 --d_model 128 --n_enc_layers 4 --n_heads 4 \
    --grad_clip 1.0 \
    --max_transitions_per_game 200000 \
    --change_loss_weight 5.0 \
    --n_updates 120000 \
    --patience 120 --min_delta 1e-6 \
    --save_dir "$save_dir" \
    --wandb \
    --sweep_name "wide_film" \
    >"$log" 2>&1
echo "[wide_film] exit=$?"

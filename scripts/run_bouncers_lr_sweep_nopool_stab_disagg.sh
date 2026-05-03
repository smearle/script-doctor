#!/usr/bin/env bash
# Disaggregate the LN vs input_skip components of the "stab patch" on the
# two regimes that broke without pooling:
#   1. (L=1, R=16) — max-sharing, regressed to 3.59% bfs (vs 0.98% at L=1,R=8)
#   2. (L=16, R=1) — per-step deep; the architecture report's canonical case
#      where the bundled patch is supposed to help most.
#
# Each config runs four cells:
#   bare        (already covered by the un-stab nopool sweep — skipped here)
#   LN-only     (--use_layernorm)
#   skip-only   (--input_skip)
#   bundled     (already covered by the bundled stab sweep — skipped here)
#
# Reuses Bouncers transitions cache. Runs ~12-15 min/config.
set -e
cd /home/jupyter-smearle/script-doctor
GAME=Bouncers
GPU=${GPU:-1}

run() {
    local L=$1 R=$2 component=$3 ln_flag=$4 skip_flag=$5
    local NS=$((L * R))
    local tag="${GAME}_nopool_${component}_L${L}_R${R}"
    local logdir="nca_wm/logs/single_${tag}"
    local logfile="/tmp/${tag}.log"
    if [[ -f "$logdir/eval_multigame.npz" ]]; then
        echo "[skip] $tag — already done"
        return
    fi
    echo "=== $tag : n_steps=$NS, n_repeats=$R, n_layers=$L (NO POOL + ${component}) ==="
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python3 -u nca_wm/train.py \
        --games $GAME \
        --architecture rule_attn \
        --conditional \
        --no-axis_pool \
        --no-axis_cummax \
        --no-global_pool \
        $ln_flag $skip_flag \
        --n_nca_steps $NS \
        --n_nca_repeats $R \
        --n_hid 256 \
        --n_slots 16 \
        --batch_size 16 \
        --n_updates 15000 \
        --lr 3e-4 \
        --grad_clip 0.5 \
        --change_loss_weight 5.0 \
        --patience 2000 \
        --search_timeout_ms 60000 \
        --log_interval 250 \
        --ckpt_interval 2500 \
        --save_dir "$logdir" \
        > "$logfile" 2>&1
    echo "  done — $logfile"
}

# (L=1, R=16): worst max-sharing degradation
run 1 16 lnonly  "--use_layernorm" ""
run 1 16 skiponly "" "--input_skip"
# (L=16, R=1): canonical per-step deep
run 16 1 lnonly  "--use_layernorm" ""
run 16 1 skiponly "" "--input_skip"

echo "All Bouncers nopool LN/skip-disaggregation configs done."

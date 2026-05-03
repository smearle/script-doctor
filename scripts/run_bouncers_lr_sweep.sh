#!/usr/bin/env bash
# Sweep n_layers × n_repeats factoring of n_steps on Bouncers (single game,
# `again`-heavy gameplay). Each config has the same total NCA iteration count
# but varies how iterations are split between distinct layers (n_layers,
# per-step weights) and shared repeats (n_repeats, weight-shared `again`-style
# loop). Total iters held at 8 to mirror the v3 recipe (n_nca_steps=8).
#
# Single-game training, batch=16, lr=3e-4, n_updates=15000 — same envelope as
# the architecture-report Collapse sweep so numbers are comparable.
set -e
cd /home/jupyter-smearle/script-doctor
GAME=Bouncers
TOTAL=${TOTAL:-8}
GPU=${GPU:-1}

run() {
    local L=$1 R=$2
    local NS=$((L * R))
    local tag="${GAME}_L${L}_R${R}"
    local logdir="nca_wm/logs/single_${tag}"
    local logfile="/tmp/${tag}.log"
    if [[ -f "$logdir/eval_multigame.npz" || -f "$logdir/curves_step15000.npz" ]]; then
        echo "[skip] $tag — already done ($logdir)"
        return
    fi
    echo "=== $tag : n_steps=$NS, n_repeats=$R, n_layers=$L ==="
    # `--games $GAME` (plural) forces the multi-game code path which supports
    # rule_attn + n_nca_repeats. The single-game `--game` path hardcodes
    # NCAWorldModel (unconditional) and would silently ignore the L×R knob.
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python3 -u nca_wm/train.py \
        --games $GAME \
        --architecture rule_attn \
        --conditional \
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

# Sweep at total=$TOTAL iterations. Order: per-step first (warms data cache),
# then progressively more sharing.
run 8 1   # per-step n=8 (current default)
run 4 2   # 4-layer block × 2 repeats
run 2 4   # 2-layer block × 4 repeats (engine-faithful)
run 1 8   # one shared layer × 8 repeats (max sharing)

# Two more at total=4 as shallow controls
run 4 1   # per-step n=4
run 1 4   # shared n=4

# Deep: crank n_repeats and inner-block depth past total=8 to find the
# saturation point on this game. Per-step weights blow up param count fast,
# so deep configs are concentrated on the max-shared (L=1) and engine-
# faithful (L=2) curves.
run 2 8    # total=16, engine-faithful
run 1 16   # total=16, max-shared
run 4 4    # total=16, balanced
run 2 16   # total=32, engine-faithful
run 1 32   # total=32, max-shared
run 1 64   # total=64, max-shared deep

echo "All Bouncers (L,R) configs done."

#!/usr/bin/env bash
# Multi-game adaptive-halt sweep: paper-figure substrate.
#
# Three games chosen for an increasing-complexity gradient:
#   sokoban_basic  — 1 rule, axis-aligned chain push, no `again`
#   sokoban_match3 — 2 rules, push + late-match-3 cascade
#   Atlas_Shrank   — ~30 rules, gravity (4× `again`), crate pickup,
#                    shadow-door logic
#
# Hypothesis (the plot we want to produce): the learned halt
# distribution should shift to deeper steps as game complexity
# increases, with E[halt step] (the headline number) tracking the
# difficulty.
#
# Same shared body across games (n_steps=8, n_repeats=8, h=128) so
# the only thing differing across panels is the game spec and the
# halt distribution the model converges to.
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch
mkdir -p "$LOGDIR"

run_one() {
    local game=$1
    local tag=$2
    local save_dir="$LOGDIR/multigame_halt_${tag}_seed0"
    local log="$LOGDIR/multigame_halt_${tag}_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting halt on $game"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games "$game" --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps 8 --n_nca_repeats 8 --n_slots 16 \
        --adaptive_halt --halt_prior_p 0.2 --halt_kl_weight 0.01 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 100000 --search_timeout_ms 60000 \
        --max_transitions_per_game 100000 \
        --n_updates 5000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 200 --ckpt_interval 1000 \
        --save_dir "$save_dir" \
        --seed 0 \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

run_one sokoban_basic   "sokoban_basic"
run_one sokoban_match3  "sokoban_match3"
run_one Atlas_Shrank    "atlas_shrank"

echo "[master] multi-game halt sweep done."

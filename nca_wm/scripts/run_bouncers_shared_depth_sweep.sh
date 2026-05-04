#!/usr/bin/env bash
# Bouncers single-game shared-weights depth sweep — the real
# depth-helps-looping test.
#
# Why Bouncers and not Microban / Collapse:
#   - Bouncers' core dynamics are a ball-trajectory simulation driven by
#     `again`: the ball moves one cell per `again` iteration, bouncing
#     off bouncers and changing direction. To predict the next state, the
#     model must simulate the trajectory cell by cell.
#   - Non-axis-aligned: the ball's trajectory turns at bouncers, so
#     pool/cummax features can't express "where does the ball end up"
#     as a one-step lookup.
#   - Sequential-state-dependent: each `again` iteration sees a state
#     modified by the prior one (the ball's position).
#   - Long: a ball can traverse most of the level in one tick.
#
# This is the Q1 test in ARCHITECTURE_REPORT (does NCA depth help on
# games with genuine looping dynamics). If shared n=2 saturates here,
# pool is somehow still hacking. If best-loss / rollout monotonically
# improves with depth, the hypothesis is supported.
#
# Bouncers level 0 is 12x10 with multiple object channels — slightly
# larger than Collapse-L0 but manageable. If BFS is too expensive at
# the default budget, increase --search_timeout_ms.
set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_bouncers_arch
mkdir -p "$LOGDIR"

run_one() {
    local n_steps=$1
    local tag=$2
    local save_dir="$LOGDIR/bouncers_L0_n${n_steps}_shared_seed0"
    local log="$LOGDIR/bouncers_L0_n${n_steps}_shared_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting Bouncers L0 n=$n_steps shared"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Bouncers --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps "$n_steps" --n_nca_repeats "$n_steps" \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 200000 --search_timeout_ms 120000 \
        --max_transitions_per_game 100000 \
        --n_updates 15000 --patience 200 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 200 --ckpt_interval 1000 \
        --save_dir "$save_dir" \
        --seed 0 \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

run_one 2  "shared_n2"
run_one 4  "shared_n4"
run_one 8  "shared_n8"
run_one 16 "shared_n16"

echo "[master] bouncers shared-weights sweep done."

#!/usr/bin/env bash
# Argmax-ST halt mode sweep — does selecting one step per example reduce
# the shortcut pressure that all our prior loss aggregations had?
#
# Two runs:
#   - Collapse-L0 (matches the existing ponder/uniform/none Collapse comparison)
#   - varislide pool-off (the test bed where halt collapsed under all v1/v2)
#
# Same architecture (n_steps=8, n_repeats=8, h=128) and recipe as the
# uniform_halt_collapse sweep, just with --halt_mode argmax_st.
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_halt_arch
mkdir -p "$LOGDIR"

run_collapse() {
    local save_dir="$LOGDIR/halt_collapse_argmax_st_seed0"
    local log="$LOGDIR/halt_collapse_argmax_st_seed0.out"
    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[collapse] SKIP — already running (pid $pid)"; return
        fi
    fi
    echo "[collapse] starting argmax_st"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Collapse --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps 8 --n_nca_repeats 8 --n_slots 16 \
        --adaptive_halt --halt_prior_p 0.2 --halt_kl_weight 0.01 --halt_mode argmax_st \
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
    echo "[collapse] finished exit=$? -> $save_dir"
}

run_varislide_nopool() {
    local save_dir="$LOGDIR/varislide_halt_nopool_argmax_st_seed0"
    local log="$LOGDIR/varislide_halt_nopool_argmax_st_seed0.out"
    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[varislide] SKIP — already running (pid $pid)"; return
        fi
    fi
    echo "[varislide] starting argmax_st (no-pool)"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games varislide \
        --conditional --architecture rule_attn \
        --n_hid 128 --n_nca_steps 16 --n_nca_repeats 16 --n_slots 16 \
        --adaptive_halt --halt_prior_p 0.2 --halt_kl_weight 0.01 --halt_mode argmax_st \
        --no-axis_pool --no-axis_cummax --no-global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 20000 --search_timeout_ms 60000 \
        --max_transitions_per_game 100000 \
        --n_updates 8000 --patience 0 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 200 --ckpt_interval 1000 \
        --save_dir "$save_dir" \
        --seed 0 \
        >"$log" 2>&1
    echo "[varislide] finished exit=$? -> $save_dir"
}

run_collapse
run_varislide_nopool

echo "[master] argmax_st sweep done."

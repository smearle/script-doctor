#!/usr/bin/env bash
# Microban (single-level) shared-weights depth sweep.
#
# Microban is the simplest classic Sokoban in the gallery (1 rule:
# `[> Player | Crate] -> [> Player | > Crate]`), and chain-pushing of
# multiple crates is exactly the kind of looping in-tick dynamics that
# motivated the depth-helps-looping question — the engine fires that
# rule iteratively per tick, once per crate in the push chain.
#
# Pairs with the Collapse shared-weights sweep (run_collapse_shared_weights_sweep.sh)
# to test whether the shared-weights inductive bias generalizes off
# Collapse, and whether depth matters more on a chain-pushing game.
# Same n_steps ladder so the table lines up.
set -u

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_microban_arch
mkdir -p "$LOGDIR"

run_one() {
    local n_steps=$1
    local extra=$2
    local tag=$3
    local stab=$4   # tag suffix only
    local save_dir="$LOGDIR/microban_L0_n${n_steps}_shared${stab:+_$stab}_seed0"
    local log="$LOGDIR/microban_L0_n${n_steps}_shared${stab:+_$stab}_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting n=$n_steps shared $stab"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Microban --level 0 \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps "$n_steps" --n_nca_repeats "$n_steps" \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 100000 --search_timeout_ms 60000 \
        --max_transitions_per_game 100000 \
        --n_updates 15000 --patience 200 --min_delta 1e-6 \
        --batch_size 16 --lr 3e-4 \
        --log_interval 200 --ckpt_interval 1000 \
        --save_dir "$save_dir" \
        --seed 0 \
        $extra \
        >"$log" 2>&1
    echo "[$tag] finished exit=$? -> $save_dir"
}

# Mirror the Collapse shared ladder.
run_one 2  ""                              "shared_n2"   ""
run_one 4  ""                              "shared_n4"   ""
run_one 8  ""                              "shared_n8"   ""
run_one 16 ""                              "shared_n16"  ""

echo "[master] microban shared-weights sweep done."

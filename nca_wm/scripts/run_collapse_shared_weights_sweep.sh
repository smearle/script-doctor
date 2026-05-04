#!/usr/bin/env bash
# Collapse shared-weight depth ablation.
#
# Companion to run_collapse_depth_sweep.sh. Same Collapse-L0 setup; the only
# axis varied here is `--shared_weights`. Question this sweep is built to
# answer (per ARCHITECTURE_REPORT, "Open architectural question: depth"):
#
#   Is per-step weight independence in rule_attn doing real work, or is the
#   PuzzleScript-natural inductive bias (one rule set, applied repeatedly
#   until the state stops changing) the right architecture?
#
# Reads side-by-side with the existing un-shared sweep:
#   - Equal n_steps, equal pool stack, equal stab patch on the n=16 row.
#   - Param count drops ~n_steps×; if rollout error matches or beats the
#     un-shared body, the per-step weights were redundant capacity that
#     mostly enabled overfitting (consistent with the n=8/n=16 stock
#     finding that train loss is uncorrelated with rollout error).
#
# Single-GPU serial sweep (small game, cached dataset reused across runs).
set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_collapse_arch
mkdir -p "$LOGDIR"

run_one() {
    local n_steps=$1
    local stab=$2     # "" or "stab"
    local extra=$3    # extra flags string
    local tag=$4
    local save_dir="$LOGDIR/collapse_L0_n${n_steps}_shared${stab:+_$stab}_seed0"
    local log="$LOGDIR/collapse_L0_n${n_steps}_shared${stab:+_$stab}_seed0.out"

    if [ -f "$save_dir/RUNNING.pid" ]; then
        local pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "[$tag] SKIP — already running (pid $pid)"
            return
        fi
    fi

    echo "[$tag] starting SHARED n_nca_steps=$n_steps $stab"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games Collapse --level 0 \
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

# Mirror the un-shared depth ladder so the two tables line up.
run_one 2  ""    ""                              "shared_n2"
run_one 4  ""    ""                              "shared_n4"
run_one 8  ""    ""                              "shared_n8"
run_one 16 ""    ""                              "shared_n16"

# Stab patch on the deepest depth — note that with shared weights this is a
# very different regime: pre-norm LN now folds into the same shared LN
# applied at every step (already shared in the un-shared model too), and
# input_skip just feeds the original observation back in at each iteration.
# Worth running because the per-step LN/skip combination interacts with the
# residual recurrence; a single shared body iterating 16 times is closer to
# a shallow RNN than to a deep transformer stack.
run_one 16 "stab" "--use_layernorm --input_skip"  "shared_n16_stab"

echo "[master] shared-weights sweep done."

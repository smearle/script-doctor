#!/usr/bin/env bash
# Post-encoder-fix sweep. The inverted-mask bug in GameSpecEncoder was
# producing near-identical z's for every game (cos-sim 0.77-0.99 across
# different games). With the fix, cos-sim drops to 0.3-0.8 between distinct
# games and 1.0 for duplicates. This run tests whether the fix alone (or
# combined with weighted loss) is enough to escape the 19-game collapse
# that we previously blamed purely on capacity.
#
# GPU 0: scaling_14 @ h=256 + change_loss_weight=5 (A/B vs preEncFix run)
# GPU 1: scaling_large @ h=256 + change_loss_weight=5 (the big test —
#         if this escapes collapse at h=256, it's strong evidence the
#         encoder bug was the dominant issue)

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

run_one() {
    local gpu=$1 games=$2 hid=$3 n_updates=$4 tag=$5
    local log="$LOGDIR/post_encfix_${games}_h${hid}_gpu${gpu}.log"
    echo "[$tag] starting $games @ hid=$hid on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games "$games" \
        --conditional \
        --balanced_sampling \
        --axis_pool --axis_cummax --global_pool \
        --n_hid "$hid" \
        --n_nca_steps 4 \
        --grad_clip 1.0 \
        --max_transitions_per_game 200000 \
        --change_loss_weight 5.0 \
        --n_updates "$n_updates" \
        --patience 120 --min_delta 1e-6 \
        --wandb \
        --sweep_name "post_encfix" \
        >"$log" 2>&1 &
    local pid=$!
    echo "[$tag] pid=$pid -> $log"
    wait "$pid"
    echo "[$tag] finished $games@h$hid exit=$?"
}

run_one 0 scaling_14    256 100000 gpu0 &
gpu0=$!
sleep 30  # stagger
run_one 1 scaling_large 256 120000 gpu1 &
gpu1=$!

wait $gpu0
wait $gpu1
echo "[master] post-encfix runs finished."

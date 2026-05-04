#!/usr/bin/env bash
# Extended held-out eval covering the persistently-hard gallery games that
# are NOT in scaling_14, on the 3 scaling_14 transfer checkpoints.
# Hard list: Take_Heart_Lass, notsnake, Lightdown, It_Dies_In_The_Light,
# the_art_of_cloning. (All in scaling_gallery_v1+, not in scaling_14.)
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs

HELDOUT="Take_Heart_Lass,notsnake,Lightdown,It_Dies_In_The_Light,the_art_of_cloning"

run_eval() {
    local gpu=$1; local run_dir=$2
    local out_subdir="heldout_hard_v1"
    local log="$LOGDIR/$(basename $run_dir).heldout_hard.log"
    if [ -f "$run_dir/$out_subdir/results.json" ]; then
        echo "  skip $(basename $run_dir): already done"; return
    fi
    echo "  [GPU $gpu] start $(basename $run_dir)"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/heldout_eval.py" \
        --load "$run_dir" \
        --heldout_games "$HELDOUT" \
        --n_random_episodes 5 \
        --max_steps 30 \
        --include_train_sample 1 \
        --out_subdir "$out_subdir" \
        > "$log" 2>&1
    echo "  [GPU $gpu] done $(basename $run_dir) (exit=$?)"
}

run_eval 0 "$LOGDIR/multi_scaling_14_v3recipe" &
pid0=$!
run_eval 1 "$LOGDIR/multi_scaling_14_mask_v1" &
pid1=$!
wait $pid0 $pid1

run_eval 0 "$LOGDIR/multi_scaling_14_mask_v2_perstep"
echo "ALL DONE"

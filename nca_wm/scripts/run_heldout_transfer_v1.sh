#!/usr/bin/env bash
# Run heldout_eval on the best multi-game checkpoints we have, post the
# 2026-05-04 _build_model fix (mask_hidden / input_skip / use_layernorm /
# adaptive_halt now plumbed). This is the first valid transfer measurement.
#
# Targets:
#   - multi_scaling_14_v3recipe         (14g, batch=32, n_nca=8, 150k)
#   - multi_scaling_gallery_v2_long200k (59g, batch=64, n_nca=4, 200k)
#   - multi_scaling_gallery_v2_nca8     (59g, batch=32, n_nca=8, 80k)
#
# Held-out games (default): blank, sumo, the_undertaking, wrappingrecipe,
# rigidfail1, constellationz — these are in scaling_large but NOT scaling_14.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs

run_eval() {
    local gpu=$1; local run_dir=$2
    local out_subdir="heldout_transfer_v1"
    if [ ! -d "$run_dir" ]; then
        echo "  skip $(basename $run_dir): no run dir"; return
    fi
    if [ -f "$run_dir/$out_subdir/results.json" ]; then
        echo "  skip $(basename $run_dir): $out_subdir already exists"; return
    fi
    local log="$LOGDIR/$(basename $run_dir).heldout_v1.log"
    echo "  [GPU $gpu] start $(basename $run_dir)"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/heldout_eval.py" \
        --load "$run_dir" \
        --n_random_episodes 5 \
        --max_steps 30 \
        --include_train_sample 1 \
        --allow_training_games \
        --out_subdir "$out_subdir" \
        > "$log" 2>&1
    echo "  [GPU $gpu] done $(basename $run_dir) (exit=$?)"
}

run_eval 0 "$LOGDIR/multi_scaling_14_v3recipe" &
pid0=$!
run_eval 1 "$LOGDIR/multi_scaling_gallery_v2_long200k" &
pid1=$!
wait $pid0 $pid1

run_eval 0 "$LOGDIR/multi_scaling_gallery_v2_nca8"
echo "ALL DONE"

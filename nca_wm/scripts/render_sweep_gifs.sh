#!/usr/bin/env bash
# Render comparison GIFs for every completed run of a given sweep.
# Reads sweep_name from arg, finds matching run dirs, and re-invokes
# nca_wm/train.py with --render_only --render_gif on each.
#
# Usage:
#   ./render_sweep_gifs.sh n_games_scaling [GPU=1]
# Each render is sequential (avoids JAX preallocate contention with
# concurrent training on the same GPU).

set -u

SWEEP_NAME="${1:?usage: $0 SWEEP_NAME [GPU]}"
GPU="${2:-1}"
REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3

cd "$REPO"
LOG=$REPO/nca_wm/sweep_launch_logs/render_${SWEEP_NAME}_gpu${GPU}.log
mkdir -p "$(dirname "$LOG")"

echo "Rendering GIFs for sweep '$SWEEP_NAME' on GPU $GPU"
echo "Log: $LOG"
: > "$LOG"

for cfg in "$REPO"/nca_wm/logs/*/config.json; do
    sweep=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('sweep_name',''))" "$cfg")
    if [ "$sweep" != "$SWEEP_NAME" ]; then continue; fi

    rundir="$(dirname "$cfg")"
    # Reconstruct minimum CLI to load the run; --load + --render_only do the heavy lifting
    games=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['games'] or '')" "$cfg")
    cond=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('conditional', False))" "$cfg")
    n_hid=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('n_hid', 128))" "$cfg")
    bal=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('balanced_sampling', False))" "$cfg")

    cond_arg=""
    [ "$cond" = "True" ] && cond_arg="--conditional"
    bal_arg=""
    [ "$bal" = "True" ] && bal_arg="--balanced_sampling"

    echo "[render] $rundir (games=$games, conditional=$cond, n_hid=$n_hid)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$REPO/nca_wm/train.py" \
        --games "$games" \
        $cond_arg $bal_arg \
        --n_hid "$n_hid" \
        --render_only --render_gif \
        --load "$rundir" \
        --save_dir "$rundir" \
        --max_episode_steps 30 \
        >>"$LOG" 2>&1
    echo "[render] done $rundir (exit=$?)" | tee -a "$LOG"
done
echo "All renders finished."

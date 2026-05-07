#!/usr/bin/env bash
# Run heldout_eval.py (AR rollout + identity baseline) for a trained
# checkpoint against the games in data/heldout_v4_n30.json.
#
# Usage:
#   nca_wm/scripts/run_heldout_rollout_eval.sh <run_save_dir>
# Example:
#   CUDA_VISIBLE_DEVICES=0 nca_wm/scripts/run_heldout_rollout_eval.sh \
#       nca_wm/logs/multi_scaling_gallery_v3
set -euo pipefail

RUN_DIR=${1:?usage: $0 <run_save_dir>}
HELDOUT_FILE=${2:-data/heldout_v4_n30.json}

cd "$(dirname "$0")/../.."

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR does not exist"
    exit 1
fi
if [ ! -f "$HELDOUT_FILE" ]; then
    echo "ERROR: $HELDOUT_FILE does not exist"
    exit 1
fi

# Extract heldout names from JSON.
# Use ';' as separator: heldout_eval.py deliberately doesn't split on ','
# because some PuzzleScript filenames contain commas (commit d5ddd82).
NAMES=$(.venv/bin/python3 -c "
import json
d = json.load(open('$HELDOUT_FILE'))
print(';'.join(g['name'] for g in d['heldout']))
")
N=$(echo "$NAMES" | tr ';' '\n' | wc -l)
LOG=/tmp/heldout_rollout_$(basename "$RUN_DIR").log
echo "Running heldout AR rollout eval on $N games -> log $LOG"

.venv/bin/python3 -m nca_wm.heldout_eval \
    --load "$RUN_DIR" \
    --heldout_games "$NAMES" \
    --n_random_episodes 3 \
    --max_steps 30 \
    --include_train_sample 0 \
    --max_levels_per_game 3 \
    --out_subdir "heldout_v4_n30" \
    --seed 0 \
    >> "$LOG" 2>&1

echo "Done. Results: $RUN_DIR/heldout_v4_n30/"

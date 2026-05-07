#!/usr/bin/env bash
# Re-run heldout_eval.py on all matched-recipe checkpoints to refresh
# nca_wm/logs/<run>/heldout_v4_n30/results.json with the new BFS/A*
# (AR) regimes added in heldout_eval. Search trajectories are cached
# at nca_wm/data_cache/heldout_search/ via precache_heldout_search.py,
# so this pass is just model rollouts.
#
# Designed to run on CPU (no GPU contention with training queues).
#
# Usage:
#   nohup nca_wm/scripts/refresh_heldout_with_bfs_astar.sh \
#     > /tmp/refresh_heldout.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

HELDOUT_FILE="data/heldout_v4_n30.json"

# Grab the 30 heldout names as a single ';'-separated string. Some
# filenames contain commas, hence ';' rather than ',' as separator.
names=$(.venv/bin/python3 -c "
import json
hd = json.load(open('$HELDOUT_FILE'))
print(';'.join(h['name'] for h in hd['heldout']))
")

# Every checkpoint we want refreshed. Add new ones (s1, s2, ...) here as
# their training finishes. We rerun even if heldout_v4_n30/results.json
# exists, because the existing file is the pre-bfs/astar version.
RUNS=(
    nca_wm/logs/multi_scaling_14_cond_match_s0
    nca_wm/logs/multi_scaling_14_uncond_match_s0
    nca_wm/logs/multi_scaling_gallery_v2_cond_match_s0
    nca_wm/logs/multi_scaling_gallery_v2_uncond_match_s0
    nca_wm/logs/multi_scaling_gallery_v4_cond_match_s0
)

for run in "${RUNS[@]}"; do
    echo
    echo "=== [$(date '+%F %T')] $run ==="
    if [ ! -f "$run/eval_multigame.npz" ]; then
        echo "[skip] $run: training not finished (no eval_multigame.npz)"
        continue
    fi
    # Check whether the heldout already includes bfs (the new column)
    if [ -f "$run/heldout_v4_n30/results.json" ]; then
        if .venv/bin/python3 -c "
import json,sys
r = json.load(open('$run/heldout_v4_n30/results.json'))
ho = r.get('heldout', {})
if not ho:
    sys.exit(1)
g = next(iter(ho.values()))
lvl = next(iter(g.values()))
sys.exit(0 if 'bfs' in lvl else 1)
" 2>/dev/null; then
            echo "[skip] $run: heldout already has bfs/astar"
            continue
        fi
    fi

    JAX_PLATFORM_NAME=cpu .venv/bin/python3 -m nca_wm.heldout_eval \
        --load "$run" \
        --heldout_games "$names" \
        --out_subdir heldout_v4_n30 \
        --max_levels_per_game 2 \
        --n_random_episodes 5 \
        --max_steps 30 \
        --include_train_sample 0 2>&1 | tail -120
done

echo
echo "=== refresh_heldout done at $(date '+%F %T') ==="

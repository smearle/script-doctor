#!/usr/bin/env bash
# Post-training scaling-curve eval for a multi-game checkpoint:
# 1. Latent overlay (geometric: 1-NN cos dist)
# 2. Heldout AR rollout per-game (predictive: vs identity baseline)
# 3. Aggregate latent overlays across all known runs into scaling-curve plot
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nca_wm/scripts/postrun_scaling_eval.sh <run_dir>
set -euo pipefail

RUN_DIR=${1:?usage: $0 <run_save_dir>}
HELDOUT_FILE=${2:-data/heldout_v4_n30.json}

cd "$(dirname "$0")/../.."

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR does not exist"
    exit 1
fi

RUN_NAME=$(basename "$RUN_DIR")

echo "=== [1/3] Latent overlay (heldout_v4_n30, reduce=flat) ==="
.venv/bin/python3 -m nca_wm.latent_overlay_heldout \
    --load "$RUN_DIR" \
    --heldout_file "$HELDOUT_FILE" \
    --reduce flat 2>&1 | grep -v "Conjoined\|already in use" | tail -8

echo
echo "=== [2/3] Heldout AR rollout (per-game subprocess) ==="
.venv/bin/python3 -m nca_wm.scripts.heldout_rollout_per_game \
    --load "$RUN_DIR" \
    --heldout_file "$HELDOUT_FILE" \
    --max_levels_per_game 2 \
    --n_random_episodes 2 \
    --max_steps 20 \
    --skip_done 2>&1 | tail -40

echo
echo "=== [3/3] Aggregate scaling curve ==="
RUNS=(
    nca_wm/logs/multi_scaling_14_v3recipe
    nca_wm/logs/multi_scaling_14_mask_v1
    nca_wm/logs/multi_scaling_14_mask_v2_perstep
    nca_wm/logs/multi_scaling_gallery_v2_nca8
    nca_wm/logs/multi_scaling_gallery_v3_combined
)
for r in nca_wm/logs/multi_scaling_gallery_v3 nca_wm/logs/multi_scaling_gallery_v4; do
    if [ -d "$r" ] && [ -f "$r/interp/heldout_overlay_flat.json" ]; then
        RUNS+=("$r")
    fi
done
.venv/bin/python3 -m nca_wm.scripts.aggregate_heldout_overlay \
    --runs "${RUNS[@]}" \
    --reduce flat 2>&1 | tail -20

echo
echo "Done. Results in $RUN_DIR/{interp,heldout_v4_n30_per_game}/."

#!/usr/bin/env bash
# Submit the history x rule-cond 2x2 across seeds on torch, one job per
# (KIND, HIST, SEED). 4 arms x |SEEDS| jobs. Run from /scratch/se2161/script-doctor.
#
# Prereq: the matching merged dataset caches must already be present under
# rollout_data/_merged/ (rsync'd from the local box) so jobs reuse them instead
# of re-collecting. KIND=uncond reuses one cache (h0+h4); KIND=cond another.
#
# Usage:
#   SEEDS="1 2 3" KINDS="cond uncond" nca_wm/scripts/torch_history_2x2_submit.sh
#   SEEDS="1 2 3" KINDS=uncond nca_wm/scripts/torch_history_2x2_submit.sh   # uncond only (cache ready first)
set -uo pipefail
cd "$(dirname "$0")/../.."

SEEDS=${SEEDS:-"1 2 3"}
KINDS=${KINDS:-"cond uncond"}
HISTS=${HISTS:-"0 4"}

for seed in $SEEDS; do
  for kind in $KINDS; do
    for hist in $HISTS; do
      jn="h2x2_${kind}_h${hist}_s${seed}"
      sbatch --job-name="$jn" \
        --export=ALL,KIND=$kind,HIST=$hist,SEED=$seed \
        nca_wm/scripts/torch_history_2x2.sbatch
    done
  done
done
echo "submitted; squeue:"; squeue -u "$USER" -o "%.18i %.28j %.2t %.10M %R" 2>/dev/null | head -40

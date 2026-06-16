#!/usr/bin/env bash
# Submit n_per_rule seed cells to SLURM on torch, one job per (KIND, N, SEED),
# with the seed-0-matching per-n budget. Run from /scratch/se2161/script-doctor.
#
# Usage:
#   SEED=1 NS="1 2 3 5 20 100 400" STAGES=both nca_wm/scripts/torch_n_per_rule_submit.sh
#   SEED=2 NS="10 50 200"          STAGES=both nca_wm/scripts/torch_n_per_rule_submit.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

SEED=${SEED:-1}
NS=${NS:-"1 2 3 5 10 20 50 100 200 400"}
STAGES=${STAGES:-both}   # cond | uncond | both
declare -A BUDGETS=( [1]=30000 [2]=30000 [3]=30000 [5]=30000 \
    [10]=150000 [20]=150000 [50]=150000 [100]=150000 [200]=150000 [400]=200000 )

submit() {
    local kind=$1 nhid=$2 n=$3
    local jn="npr_${n}_${kind}_s${SEED}"
    sbatch --job-name="$jn" \
        --export=ALL,KIND=$kind,NHID=$nhid,N=$n,SEED=$SEED,NUP=${BUDGETS[$n]} \
        nca_wm/scripts/torch_n_per_rule_seed.sbatch
}

for n in $NS; do
    [ "$STAGES" = "cond" ]   || [ "$STAGES" = "both" ] && submit cond   256 "$n"
    [ "$STAGES" = "uncond" ] || [ "$STAGES" = "both" ] && submit uncond 288 "$n"
done
echo "submitted; squeue:"; squeue -u "$USER" -o "%.18i %.28j %.2t %.10M %R" 2>/dev/null | head -40

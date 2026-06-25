#!/bin/bash
# One Autumn world-model recurrent training run as a SLURM job on torch.
# Training is ENGINE-FREE (loads {GAME}_train_seq.npz, trains a torch model), so MARA
# is NOT needed on torch -- only the npz data (rsynced from the local box) + the code.
#
# Usage (submit one job per (GAME, POOL, NHID, SEED) cell so they parallelize):
#   sbatch --export=ALL,GAME=nim,POOL=meanmax,NHID=96,SEED=0,UPDATES=8000 \
#     --job-name=aut_nim_meanmax_h96_s0 nca_wm/scripts/torch_autumn_train.sbatch
# (torch_autumn_submit.sh wraps the grid.)
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=nca_wm/logs/autumn_slurm/%x.slurm.out
set -u

cd /scratch/se2161/script-doctor || exit 1
mkdir -p nca_wm/logs/autumn_slurm
source .venv/bin/activate 2>/dev/null || true

GAME=${GAME:?set GAME}
POOL=${POOL:-meanmax}
NHID=${NHID:-96}
SEED=${SEED:-0}
UPDATES=${UPDATES:-8000}
SCHED=${SCHED:-0.0}
TAG=${TAG:-${POOL}_h${NHID}_s${SEED}}

DATA=nca_wm/autumn/data/${GAME}_train_seq.npz
SAVE=nca_wm/autumn/runs/${GAME}_sweep_${TAG}
if [ ! -f "$DATA" ]; then echo "MISSING DATA $DATA"; exit 2; fi

echo "[autumn-train] game=$GAME pool=$POOL nhid=$NHID seed=$SEED updates=$UPDATES sched=$SCHED -> $SAVE"
python -m nca_wm.autumn.train_recurrent \
  --data "$DATA" --save_dir "$SAVE" \
  --updates "$UPDATES" --pool "$POOL" --n_hid "$NHID" --sched_samp "$SCHED" --seed "$SEED"
echo "[autumn-train] done $GAME/$TAG"

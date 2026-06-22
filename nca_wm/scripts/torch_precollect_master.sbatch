#!/bin/bash
# SLURM array: pre-collect BFS transitions for the deduped master corpus.
# The A*/BFS state-space search is CPU-only, but torch gives our project NO
# cpu-only path (cpu_* partitions reject our QoS; gpu partitions reject jobs
# without a GPU). So, like the lab's other torch jobs, each task holds one GPU
# (idle) and uses that GPU's fair 16-core share — 128 cores / 8 GPUs per node.
# Each array task processes one shard (candidate index % NUM_SHARDS == task id),
# writes to the shared rollout_data/, and skips already-cached games — so
# preemption just resumes.
#
# Pilot (one stratified shard of 64 ~= 340 games):
#   NUM_SHARDS=64 sbatch --array=0 nca_wm/scripts/torch_precollect_master.sbatch
# Full fan-out (mind the gpu48 per-user GPU cap = concurrent shards):
#   NUM_SHARDS=128 sbatch --array=0-127%24 nca_wm/scripts/torch_precollect_master.sbatch
#
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# 16 workers can each transiently materialize a ~500k-transition array on games
# that hit max_iters before the write-time cap, so peak RAM spikes; 110G fits
# both the a100 (128G/GPU-share) and h200 (257G/GPU-share) granularity.
#SBATCH --mem=110000
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=nca_wm/logs/precollect_master/%x_%A_%a.out
set -u
cd /scratch/se2161/script-doctor
mkdir -p nca_wm/logs/precollect_master

export PATH=/scratch/se2161/.nvm/versions/node/v24.13.0/bin:$PATH
export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=1

NUM_SHARDS=${NUM_SHARDS:-128}
ALGO=${ALGO:-bfs}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
MASTER=${MASTER:-/scratch/se2161/puzzlescript-gists}

echo "host=$(hostname) shard=${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} algo=${ALGO} workers=${WORKERS}"
srun .venv/bin/python3 -m nca_wm.scripts.precollect_master \
    --master-dir "$MASTER" \
    --search_algo "$ALGO" \
    --workers "$WORKERS" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "$NUM_SHARDS"

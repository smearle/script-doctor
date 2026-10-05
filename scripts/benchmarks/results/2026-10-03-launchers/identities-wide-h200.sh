#!/bin/bash
#SBATCH --job-name=puzzlejax-identities
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --array=0-3%2
#SBATCH --output=/scratch/se2161/puzzlejax-perf-20261002/round3/identities-%A_%a.log
set -euo pipefail
cd /scratch/se2161/puzzlejax-perf-20261002/round3
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
games=(Slidings Slidings Take_Heart_Lass Take_Heart_Lass)
levels=(0 1 0 1)
index=${SLURM_ARRAY_TASK_ID}
python -u -m scripts.benchmarks.benchmark_movement_identities --games "${games[$index]}" --levels "${levels[$index]}" --batches 4096 --seeds 42 1042 --trials 9 --output "identities-wide-${index}-h200.json"

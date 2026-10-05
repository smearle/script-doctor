#!/bin/bash
#SBATCH --job-name=puzzlejax-updates-large
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=large-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python -u -m scripts.benchmarks.benchmark_movement_updates \
  --candidates production --games blocks Slidings nekopuzzle \
  --batches 16384 --steps 100 --seeds 42 1042 --trials 7 \
  --output large-batch-h200.json

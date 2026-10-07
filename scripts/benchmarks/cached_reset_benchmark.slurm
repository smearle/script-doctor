#!/bin/bash
#SBATCH --job-name=puzzlejax-cached-reset
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --array=0-3%4
#SBATCH --output=cached-reset-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/puzzlejax-mpl
export PYTHONDONTWRITEBYTECODE=1
mkdir -p data/game_trees
cases=(sokoban_basic "atlas shrank" Beam_Islands Caramelban)
game="${cases[$SLURM_ARRAY_TASK_ID]}"
batch="${CACHE_BENCHMARK_BATCH:-256}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -u -m scripts.benchmarks.benchmark_cached_reset --games "$game" --batch "$batch" \
  --steps 64 --trials 7 --safe-api --output "results/cached-reset-h200/$game-b$batch.json"

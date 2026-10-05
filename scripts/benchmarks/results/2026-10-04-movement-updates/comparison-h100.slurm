#!/bin/bash
#SBATCH --job-name=puzzlejax-updates-h100
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=comparison-h100-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -u -m scripts.benchmarks.benchmark_movement_updates \
  --candidates production --games sokoban_basic blocks Zen_Puzzle_Garden Slidings nekopuzzle \
  --batches 4096 --steps 100 --seeds 42 1042 --trials 7 \
  --output comparison-h100.json

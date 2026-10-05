#!/bin/bash
# Submit from the repository root with an H200 account and partition.
#SBATCH --job-name=puzzlejax-updates-wide
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --array=0-11%3
#SBATCH --output=updates-wide-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
games=(sokoban_basic blocks Zen_Puzzle_Garden notsnake Slidings limerick kettle Take_Heart_Lass 'atlas shrank' nekopuzzle sokoban_match3 Travelling_salesman)
index="${SLURM_ARRAY_TASK_ID:?This script requires a job array}"
output_dir="${MOVEMENT_UPDATES_OUTPUT_DIR:-updates-wide-h200}"
python -u -m scripts.benchmarks.benchmark_movement_updates \
    --candidates production --games "${games[$index]}" \
    --batches 1 256 4096 --steps 100 --seeds 42 1042 --trials 7 \
    --output "$output_dir/game-$index.json"

#!/bin/bash
#SBATCH --job-name=puzzlejax-updates
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --array=0-2%2
#SBATCH --output=/scratch/se2161/puzzlejax-perf-20261002/round4/updates-%A_%a.log
set -euo pipefail
cd /scratch/se2161/puzzlejax-perf-20261002/round4
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
games=(sokoban_basic blocks Zen_Puzzle_Garden)
python -u -m scripts.benchmarks.benchmark_movement_updates --candidates masked batched --games "${games[$SLURM_ARRAY_TASK_ID]}" --batches 256 4096 --seeds 42 1042 --trials 9 --output "updates-${SLURM_ARRAY_TASK_ID}-h200.json"

#!/bin/bash
#SBATCH --job-name=puzzlejax-wide
#SBATCH --account=torch_pr_84_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --array=0-8%2
#SBATCH --output=/scratch/se2161/puzzlejax-perf-20261002/round3/wide-%A_%a.log
set -euo pipefail
cd /scratch/se2161/puzzlejax-perf-20261002/round3
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
GAMES=(notsnake Slidings limerick kettle Take_Heart_Lass 'atlas shrank' nekopuzzle sokoban_match3 Travelling_salesman)
python -u -m scripts.benchmarks.benchmark_movement_refinements --experiment combined --candidates all --games "${GAMES[$SLURM_ARRAY_TASK_ID]}" --batches 256 4096 --steps 100 --trials 7 --output "wide-${SLURM_ARRAY_TASK_ID}-h200.json"

#!/bin/bash
#SBATCH --job-name=puzzlejax-identities
#SBATCH --account=torch_pr_84_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/se2161/puzzlejax-perf-20261002/round3/identities-%j.log
set -euo pipefail
cd /scratch/se2161/puzzlejax-perf-20261002/round3
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
python -u -m scripts.benchmarks.benchmark_movement_identities --games sokoban_basic blocks Zen_Puzzle_Garden --batches 256 4096 --seeds 42 1042 --trials 9 --output identities-pilot-h200.json

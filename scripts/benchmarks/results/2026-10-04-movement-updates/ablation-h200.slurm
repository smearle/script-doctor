#!/bin/bash
#SBATCH --job-name=puzzlejax-loop-ablation
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:07:00
#SBATCH --output=/scratch/se2161/puzzlejax-perf-20261002/round4/ablation-%j.log
set -euo pipefail
cd /scratch/se2161/puzzlejax-perf-20261002/round4
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
python -u -m scripts.benchmarks.benchmark_movement_updates --candidates batched_select --batches 4096 --seeds 42 1042 --trials 9 --output ablation-h200.json

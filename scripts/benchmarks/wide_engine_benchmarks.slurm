#!/bin/bash
# Submit from the repository root with an account authorized for H200 use.
#SBATCH --job-name=puzzlejax-wide
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --array=0-8%2
#SBATCH --output=wide-engine-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
games=(notsnake Slidings limerick kettle Take_Heart_Lass 'atlas shrank' nekopuzzle sokoban_match3 Travelling_salesman)
index="${SLURM_ARRAY_TASK_ID:?This script requires a job array}"
output_dir="${WIDE_BENCHMARK_OUTPUT_DIR:-wide-engine-h200}"
python -u -m scripts.benchmarks.benchmark_movement_refinements \
    --experiment combined --candidates all --games "${games[$index]}" \
    --batches 256 4096 --steps 100 --trials 7 --output "$output_dir/wide-$index.json"

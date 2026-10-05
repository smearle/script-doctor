#!/bin/bash
# Submit from the repository root using an account with H200 access.
# sbatch --account=torch_pr_84_general scripts/benchmarks/movement_refinements.slurm
#SBATCH --job-name=puzzlejax-refinements
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-6%2
#SBATCH --output=movement-refinements-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
output_dir="${MOVEMENT_BENCHMARK_OUTPUT_DIR:-movement-refinements-h200}"
games=(sokoban_basic blocks Zen_Puzzle_Garden 'atlas shrank')
index="${SLURM_ARRAY_TASK_ID:?This script requires a job array}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if ((index < 3)); then
    python -u -m scripts.benchmarks.benchmark_movement_refinements \
        --experiment combined --candidates cleanup_barriers all \
        --games "${games[$index]}" --batches 256 1024 4096 16384 \
        --steps 100 --trials 15 --output "$output_dir/combined-$index.json"
else
    python -u -m scripts.benchmarks.benchmark_chunked_rollouts \
        --games "${games[$((index - 3))]}" --batch 16384 --chunks 4096 \
        --steps 100 --trials 9 --output "$output_dir/chunks-$((index - 3)).json"
fi

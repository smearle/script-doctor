#!/bin/bash
# Submit from the repository root with your cluster account, for example:
# sbatch --account=torch_pr_84_general scripts/benchmarks/paper_throughput.slurm
#SBATCH --job-name=puzzlejax-paper
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-7%4
#SBATCH --output=paper-throughput-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -u -m scripts.benchmarks.benchmark_paper_throughput \
    --game-index "${SLURM_ARRAY_TASK_ID:?This script requires a job array}" \
    --output-dir "${PAPER_BENCHMARK_OUTPUT_DIR:-paper-throughput-h200}" "$@"

#!/bin/bash
# Eight paper games plus eight additional games; up to four GPUs concurrently.
# sbatch --account=YOUR_ACCOUNT --partition=h200_tandon scripts/benchmarks/adaptive_throughput.slurm
#SBATCH --job-name=puzzlejax-adaptive
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%4
#SBATCH --output=adaptive-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -u -m scripts.benchmarks.benchmark_paper_throughput \
    --game-index "${SLURM_ARRAY_TASK_ID:?}" --adaptive --resume \
    --output-dir "${PAPER_BENCHMARK_OUTPUT_DIR:-adaptive-throughput-h200}" "$@"

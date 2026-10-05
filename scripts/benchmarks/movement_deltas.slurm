#!/bin/bash
# Override account/partition/GPU at submission for the available device.
#SBATCH --job-name=puzzlejax-deltas
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=deltas-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u -m scripts.benchmarks.benchmark_movement_deltas \
    --batches 1 4096 16384 --trials 9 --output movement-deltas.json "$@"

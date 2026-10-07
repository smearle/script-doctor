#!/bin/bash
#SBATCH --job-name=puzzlejax-cpp-validation
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cpp-validation-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/puzzlejax-mpl PYTHONDONTWRITEBYTECODE=1
mkdir -p results
python -u -m scripts.benchmarks.validate_complex_cpp --library cpp-validation-library.so \
  --output results/cpp-original-js-validation.json

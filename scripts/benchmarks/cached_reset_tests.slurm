#!/bin/bash
#SBATCH --job-name=puzzlejax-reset-tests
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=reset-tests-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/puzzlejax-mpl PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/test-support:$PWD"
mkdir -p data/game_trees results
python -m pytest -q tests/test_cached_reset.py tests/test_env_switch.py tests/test_validation_wide_objects.py > results/cached-reset-tests.log 2>&1

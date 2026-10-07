#!/bin/bash
#SBATCH --job-name=puzzlejax-reachable-rules
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=reachable-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/puzzlejax-mpl
export PYTHONDONTWRITEBYTECODE=1
mkdir -p data/game_trees
python -u -m scripts.benchmarks.benchmark_reachable_rules --game Crate_Assembler \
  --compiled-json compiled-crate.json --output results/reachable-crate.json

#!/bin/bash
#SBATCH --job-name=puzzlejax-complex-resume
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=resume-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/puzzlejax-mpl
export PYTHONDONTWRITEBYTECODE=1
cases=(Beam_Islands IceCrates)
game="${cases[$SLURM_ARRAY_TASK_ID]}"
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "results/validation/$game.json"
python -u -m scripts.benchmarks.benchmark_paper_throughput --game "$game" \
  --batches 16 256 1024 4096 --adaptive --max-batch 1048576 --resume \
  --output-dir results/complex-throughput-h200

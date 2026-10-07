#!/bin/bash
# Submit from the frozen benchmark root with the appropriate site account.
#SBATCH --job-name=puzzlejax-complex
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%4
#SBATCH --output=complex-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/puzzlejax-mpl
export PYTHONDONTWRITEBYTECODE=1
mkdir -p data/game_trees
game=$(python -c 'from scripts.benchmarks.complex_games import COMPLEX_GAMES; import sys; print(COMPLEX_GAMES[int(sys.argv[1])])' "${SLURM_ARRAY_TASK_ID}")
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -u -m scripts.benchmarks.validate_complex_game --game "$game" --output "results/validation/$game.json"
python -u -m scripts.benchmarks.benchmark_paper_throughput --game "$game" \
  --batches 16 256 1024 4096 --adaptive --max-batch 262144 --resume \
  --output-dir results/complex-throughput-h200 "$@"

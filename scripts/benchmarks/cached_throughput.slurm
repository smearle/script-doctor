#!/bin/bash
#SBATCH --job-name=puzzlejax-prepared-curves
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-10%6
#SBATCH --output=prepared-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/puzzlejax-mpl
export PYTHONDONTWRITEBYTECODE=1
mkdir -p data/game_trees
# Every sampled game with run_rules_on_level_start; the same fixed level and
# paper protocol are used for the ordinary and prepared-parameter curves.
cases=("atlas shrank" Magnet_Jack Beam_Islands Caramelban IceCrates \
       Heroes_of_Sokoban_III__The_Bard_and_The_Druid ParaLands SwapBot \
       Symbolism Transition Indigestion)
game="${cases[$SLURM_ARRAY_TASK_ID]}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -u -m scripts.benchmarks.validate_complex_game --game "$game" --stop-on-win --prepare-params \
  --output "results/validation/$game.json"
python -u -m scripts.benchmarks.benchmark_paper_throughput --game "$game" --prepare-params \
  --batches 16 256 1024 4096 --adaptive --max-batch 262144 --resume \
  --output-dir results/prepared-throughput-h200

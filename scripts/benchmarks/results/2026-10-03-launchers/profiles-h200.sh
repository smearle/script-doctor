#!/bin/bash
#SBATCH --job-name=puzzlejax-profile
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/se2161/puzzlejax-perf-20261002/round3/profile-%j.log
set -euo pipefail
cd /scratch/se2161/puzzlejax-perf-20261002/round3
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
profile_dir=$(mktemp -d /tmp/puzzlejax-final-profile-${SLURM_JOB_ID}.XXXXXX)
python -u -m scripts.benchmarks.benchmark_rollout_scaling --games sokoban_basic blocks Zen_Puzzle_Garden --batches 4096 --steps 100 --trials 9 --profile-dir "$profile_dir" --output final-profile-timings-h200.json
mapfile -d '' traces < <(find "$profile_dir" -name perfetto_trace.json.gz -print0)
python -m scripts.benchmarks.summarize_gpu_trace "${traces[@]}" --output final-profiles-h200.json
(cd "$profile_dir" && find . -type f -exec sha256sum {} +) > final-profiles-h200.sha256
tar czf final-profiles-h200.tar.gz -C "$profile_dir" .
rm -r -- "$profile_dir"

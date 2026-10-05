#!/bin/bash
#SBATCH --job-name=puzzlejax-updates-profile
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=profile-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source .venv/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4
trace_dir=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/puzzlejax-movement-${SLURM_JOB_ID}-XXXXXX")
python -u -m scripts.benchmarks.benchmark_rollout_scaling \
    --games sokoban_basic blocks Zen_Puzzle_Garden --batches 4096 --trials 9 \
    --profile-dir "$trace_dir" --output profile-timings.json
python - "$trace_dir" <<'PY'
import hashlib, json, pathlib, sys
from scripts.benchmarks.summarize_gpu_trace import summarize
root=pathlib.Path(sys.argv[1])
traces=[summarize(p) for p in sorted(root.rglob('perfetto_trace.json.gz'))]
pathlib.Path('profile-summary.json').write_text(json.dumps({'traces':traces},indent=2)+'\n')
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
pathlib.Path('profile-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
PY
tar czf profiles.tar.gz -C "$trace_dir" .
sha256sum profiles.tar.gz > profiles.tar.gz.sha256

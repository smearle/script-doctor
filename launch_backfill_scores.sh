#!/bin/bash
# Launch backfill_initial_scores.py on SLURM via submitit.
#
# Usage:
#   ./launch_backfill_scores.sh                          # js backend, pedro dataset
#   ./launch_backfill_scores.sh backend=cpp              # cpp backend
#   ./launch_backfill_scores.sh backend=cpp dataset=increpare
#   ./launch_backfill_scores.sh n_games_per_job=10       # fewer, larger jobs
#   ./launch_backfill_scores.sh overwrite=True            # recompute existing scores

set -euo pipefail
cd "$(dirname "$0")"

export SLURM_ACCOUNT="${SLURM_ACCOUNT:-torch_pr_84_tandon_advanced}"

.venv/bin/python3 backfill_initial_scores.py slurm=True "$@"

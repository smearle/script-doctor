#!/bin/bash
#SBATCH --job-name=indist
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000
#SBATCH --time=03:00:00
#SBATCH --output=nca_wm/logs/in_dist_next/%x.slurm.out
set -u
cd /scratch/se2161/script-doctor
export PATH=/scratch/se2161/.nvm/versions/node/v24.13.0/bin:$PATH
export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "host=$(hostname) node=$(command -v node) job=$JOB_NAME game=$JOB_GAME"
.venv/bin/python3 -c "import jax; print(\"jax devices:\", jax.devices())"
PYTHONUNBUFFERED=1 .venv/bin/python3 nca_wm/train.py \
  --conditional --architecture rule_attn \
  --n_hid 256 --n_slots 16 --n_app_slots 1 \
  --change_loss_weight 5.0 --grad_clip 0.5 --balanced_sampling \
  --n_search_steps 200000 --search_timeout_ms 120000 \
  --batch_size 16 --lr 3e-4 \
  --val_frac 0.1 --val_eval_interval 500 \
  --patience 0 --min_delta 1e-8 --log_interval 500 --ckpt_interval 5000 \
  --input_skip --seed 0 \
  $JOB_EXTRA \
  --games $JOB_GAME \
  --save_dir nca_wm/logs/in_dist_next/$JOB_NAME

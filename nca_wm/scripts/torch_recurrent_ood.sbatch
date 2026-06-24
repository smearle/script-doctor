#!/bin/bash
# Recurrent ("memory") NCA world model — OOD held-out-games eval — on torch.
# Long-context truncated BPTT (large K, small BPTT_WINDOW) needs the full L=K+1
# forward in memory, so request the high-VRAM partitions (H100/H200, 80GB+).
# Reuses the n_per_rule-800 uncond merged cache the history-2x2 primer built
# (same collect params -> same _dataset_cache_key), so no re-collection.
#
# Usage (via a submit loop):
#   sbatch --export=ALL,K=16,BPTT_WINDOW=4,BATCH=8,SEED=0 \
#     --job-name=ood_k16w4_s0 nca_wm/scripts/torch_recurrent_ood.sbatch
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=nca_wm/logs/ood_recurrent_slurm/%x.slurm.out
set -u
cd /scratch/se2161/script-doctor
mkdir -p nca_wm/logs/ood_recurrent_slurm

export PATH=/scratch/se2161/.nvm/versions/node/v24.13.0/bin:$PATH
export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1

: "${K:?set K}"; : "${BPTT_WINDOW:?set BPTT_WINDOW}"; : "${BATCH:?set BATCH}"
: "${SEED:?set SEED}"
SAVE="nca_wm/logs/ood_recurrent/k${K}w${BPTT_WINDOW}_b${BATCH}_s${SEED}"
if [ -f "$SAVE/config.json" ] && grep -q '"best_val_change_err"' "$SAVE/config.json" 2>/dev/null; then
  echo "[skip] $SAVE complete"; exit 0; fi
echo "host=$(hostname) node=$(command -v node) K=$K window=$BPTT_WINDOW batch=$BATCH seed=$SEED"
.venv/bin/python3 -c "import jax; print('jax devices:', jax.devices())"

.venv/bin/python3 -u -m nca_wm.train_recurrent \
  --n_games 800 --heldout_game_frac 0.1 \
  --k "$K" --bptt_window "$BPTT_WINDOW" --batch_size "$BATCH" \
  --n_hid 288 --n_steps 8 --n_updates 50000 \
  --lr 3e-4 --grad_clip 0.5 --eval_interval 2500 \
  --val_frac 0.10 --max_transitions_per_game 20000 --max_grid_dim 30 \
  --max_eval_rows 4096 --seed "$SEED" --save_dir "$SAVE"

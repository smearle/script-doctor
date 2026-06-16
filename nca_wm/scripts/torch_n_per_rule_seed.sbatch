#!/bin/bash
# One n_per_rule dedup_pool training cell as a SLURM job on torch (H200).
# Mirrors run_n_per_rule_seed.sh / run_n_per_rule_dp_sweep.sh exactly so
# torch-trained seeds are comparable to the local box's. Submit one job per
# (KIND, N, SEED) so cells parallelize across the cluster.
#
# Usage:
#   for n in 1 2 3 5 20 100 400; do
#     sbatch --export=ALL,KIND=cond,NHID=256,N=$n,SEED=1,NUP=$BUDGET \
#       --job-name=npr_${n}_cond_s1 nca_wm/scripts/torch_n_per_rule_seed.sbatch
#   done
# (a submit helper, torch_n_per_rule_submit.sh, wraps the budget map.)
#SBATCH --account=torch_pr_84_tandon_advanced
# GPU-agnostic: each tandon partition has its own per-user GPU QOS cap, so
# listing all three lets SLURM place each job wherever a slot frees first and
# multiplies the effective concurrent-GPU allowance. Any of A100/H100/H200 has
# ample memory for these jobs (n_hid<=288, batch 32).
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000
#SBATCH --time=12:00:00
#SBATCH --output=nca_wm/logs/n_per_rule_slurm/%x.slurm.out
set -u
cd /scratch/se2161/script-doctor
mkdir -p nca_wm/logs/n_per_rule_slurm

export PATH=/scratch/se2161/.nvm/versions/node/v24.13.0/bin:$PATH
export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1

: "${KIND:?set KIND=cond|uncond}"; : "${N:?set N}"; : "${SEED:?set SEED}"
: "${NHID:?set NHID}"; : "${NUP:?set NUP}"; VAL_FRAC=${VAL_FRAC:-0.10}

if [ "$KIND" = "cond" ]; then
  COND_FLAG="--conditional"
  SAVE="nca_wm/logs/n_per_rule_${N}_cond_dp_val${VAL_FRAC}_s${SEED}"
  EXTRA=( --n_slots 16 --d_slot 64 --n_app_slots 1 --n_enc_layers 2 --n_heads 4
          --d_model 64 --d_z 64 --encode_sprites --token_decoder_loss_weight 1.0
          --decoder_d_model 128 --decoder_n_layers 4 --decoder_n_heads 4 )
else
  COND_FLAG="--no-conditional"
  SAVE="nca_wm/logs/n_per_rule_${N}_uncond_h288_dp_val${VAL_FRAC}_s${SEED}"
  EXTRA=( --n_app_slots 1 )
fi

if [ -f "$SAVE/eval_multigame.npz" ]; then echo "[skip] $SAVE complete"; exit 0; fi
echo "host=$(hostname) node=$(command -v node) cell=$KIND n=$N seed=$SEED nup=$NUP"
.venv/bin/python3 -c "import jax; print('jax devices:', jax.devices())"

.venv/bin/python3 -u -m nca_wm.train \
  --n_per_rule_games "$N" --n_per_rule_universe dedup_pool --n_per_rule_max_area 30 \
  $COND_FLAG --architecture rule_attn --n_hid "$NHID" --n_nca_steps 8 \
  --batch_size 32 --n_updates "$NUP" \
  --lr 3e-4 --lr_schedule cosine --lr_min 1e-7 --grad_clip 0.5 \
  --change_loss_weight 5.0 --balanced_sampling --max_transitions_per_game 200000 \
  --search_timeout_ms 60000 --n_search_steps 100000 \
  --input_skip --axis_pool --axis_cummax --global_pool \
  --patience 0 --ckpt_interval 5000 --log_interval 250 \
  --val_frac "$VAL_FRAC" --seed "$SEED" --save_dir "$SAVE" "${EXTRA[@]}"

#!/bin/bash
# One cell of the history x rule-cond 2x2 as a SLURM job on torch.
# 4 arms = KIND {cond,uncond} x HIST {0,4}; sweep SEED for error bars.
#
# This is an EXACT replica of the local first-read driver (run_2x2_final.sh):
# same flags -> same _dataset_cache_key, so the merged dataset caches rsync'd
# from the local box (uncond: 8f6dc413, cond: <built when rulecond finishes>)
# are reused verbatim and NO collection happens on torch. History is not part
# of the cache key, so one uncond cache serves h0+h4 and one cond cache serves
# h0+h4.
#
# Usage (via torch_history_2x2_submit.sh, which wraps the matrix):
#   sbatch --export=ALL,KIND=cond,HIST=4,SEED=1 \
#     --job-name=h2x2_cond_h4_s1 nca_wm/scripts/torch_history_2x2.sbatch
#SBATCH --account=torch_pr_84_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=nca_wm/logs/history_2x2_slurm/%x.slurm.out
set -u
cd /scratch/se2161/script-doctor
mkdir -p nca_wm/logs/history_2x2_slurm

export PATH=/scratch/se2161/.nvm/versions/node/v24.13.0/bin:$PATH
export JAX_COMPILATION_CACHE_DIR=$PWD/.jax_compile_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1

: "${KIND:?set KIND=cond|uncond}"; : "${HIST:?set HIST=0|4}"; : "${SEED:?set SEED}"
VAL_FRAC=${VAL_FRAC:-0.10}

if [ "$KIND" = "cond" ]; then
  COND_FLAG="--conditional"
  EXTRA=( --architecture rule_attn --n_hid 256 --n_slots 16 --d_slot 64 --n_app_slots 1
          --n_enc_layers 2 --n_heads 4 --d_model 64 --d_z 64 --encode_sprites
          --token_decoder_loss_weight 1.0 --decoder_d_model 128 --decoder_n_layers 4
          --decoder_n_heads 4 )
else
  COND_FLAG="--no-conditional"
  EXTRA=( --n_hid 288 )
fi
SAVE="nca_wm/logs/history_2x2/${KIND}_h${HIST}_s${SEED}"

if [ -f "$SAVE/eval_multigame.npz" ]; then echo "[skip] $SAVE complete"; exit 0; fi
echo "host=$(hostname) node=$(command -v node) cell=$KIND hist=$HIST seed=$SEED"
.venv/bin/python3 -c "import jax; print('jax devices:', jax.devices())"

.venv/bin/python3 -u -m nca_wm.train \
  --n_per_rule_games 800 --n_per_rule_universe dedup_pool --n_per_rule_max_area 30 \
  --max_transitions_per_game 20000 --n_nca_steps 8 --batch_size 32 --n_updates 50000 \
  --lr 3e-4 --lr_schedule cosine --lr_min 1e-7 --grad_clip 0.5 \
  --input_skip --axis_pool --axis_cummax --global_pool \
  --val_frac "$VAL_FRAC" --val_eval_interval 10000 --per_game_eval_interval 1000000 \
  --ancestor_closed_subsample --search_timeout_ms 60000 \
  --patience 0 --ckpt_interval 2000 --log_interval 250 \
  $COND_FLAG --history "$HIST" --seed "$SEED" --save_dir "$SAVE" "${EXTRA[@]}"

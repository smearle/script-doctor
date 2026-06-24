#!/bin/bash
# Controlled test of the "multiple resets" hypothesis: does a restart action
# help a recurrent agent on a dead-end-prone single level (sokoban_basic
# level=0), where plain recurrent PPO collapses?
# Runs a matched baseline (no restart) and a restart-action agent.
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}
export PYTHONUNBUFFERED=1
PY=.venv/bin/python3
GAME=sokoban_basic
STEPS=${STEPS:-3000000}

for ra in false true; do
  echo "=================================================================="
  echo "=== sokoban_basic level=0  restart_action=$ra ==="
  echo "=================================================================="
  $PY -m puzzlejax.train_jax_rnn game=$GAME level=0 n_envs=256 num_steps=64 \
      hidden_dims=[128,128] total_timesteps=$STEPS render_freq=0 ckpt_freq=100000000000 \
      ENT_COEF=0.02 restart_action=$ra wandb_mode=disabled overwrite=true seed=0 \
      2>&1 | tr '\r' '\n' | grep -iE "step=|ret=|win|error|Traceback|logged at" | tail -15
done
echo "=== DONE restart experiment ==="

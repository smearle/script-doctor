#!/bin/bash
# Train recurrent (GRU) PPO agents on a variety of RL-winnable human-authored
# PuzzleScript games, then render greedy GIFs of the best agent per game.
# Runs sequentially on one GPU. Usage: bash scripts/training/run_rnn_showcase.sh
set -u
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}
PY=.venv/bin/python3

# Travelling_salesman already completed in an earlier run; GIFs in the showcase dir.
GAMES=(${GAMES_OVERRIDE:-"sokodig" "kettle" "Multi-word_Dictionary_Game" "sokoban_match3" "Slidings"})
STEPS=${STEPS:-1500000}
NENVS=${NENVS:-256}

SHOWCASE_DIR=rl_logs_jax/_recurrent_showcase
mkdir -p "$SHOWCASE_DIR"

for game in "${GAMES[@]}"; do
  echo "=================================================================="
  echo "=== TRAIN recurrent agent: $game ==="
  echo "=================================================================="
  # Train render-free (in-graph GIF rendering makes the RNN graph compile
  # very slowly); GIFs are produced separately by the enjoy step below.
  $PY -m puzzlejax.train_jax_rnn game="$game" level=-1 n_envs=$NENVS num_steps=64 \
      hidden_dims=[128,128] total_timesteps=$STEPS render_freq=0 \
      ckpt_freq=500000 ENT_COEF=0.02 wandb_mode=disabled overwrite=true seed=0 \
      2>&1 | tr '\r' '\n' | grep -iE "step=|win|error|Traceback" | tail -20

  echo "=== ENJOY (render best agent): $game ==="
  $PY -m puzzlejax.enjoy_rnn game="$game" level=-1 n_envs=$NENVS \
      hidden_dims=[128,128] n_render_eps=6 wandb_mode=disabled 2>&1 | tail -20

  # Copy any winning greedy gif into the showcase dir
  enjoy_dir="rl_logs_jax/$game/level--1/n-envs-${NENVS}_rnn-128-128_seed-0_ep-len-200/enjoy"
  cp "$enjoy_dir/${game}"*WIN*.gif "$SHOWCASE_DIR/" 2>/dev/null && echo "  -> copied WIN gif(s) to $SHOWCASE_DIR"
done

echo "=== Showcase gifs ==="
ls -la "$SHOWCASE_DIR"

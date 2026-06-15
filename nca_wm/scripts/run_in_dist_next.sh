#!/usr/bin/env bash
# Next in-distribution single-game set (2026-06-15): climb the mechanic ladder
# past the games already modeled perfectly (sokoban_basic, nekopuzzle,
# Travelling_salesman). Each game is a clean single-game perfect-modeling target:
#
#   Microban           data-scale  : canonical 1-rule sokoban, ~150 small authored
#                                     levels. Validates the water-fill per-game
#                                     budget on a large authored level bank.
#   Heroes_of_Sokoban  local-rules : many distinct conditional rules / unit types,
#                                     no `again`. Old single-level grid: 0% L0 but
#                                     26% held-out levels; all-levels should close it.
#   Bouncers           iterative   : `again`-driven continuous motion. Stresses the
#                                     multi-step until-convergence axis (the known
#                                     iteration-extrapolation weak spot).
#
# Canonical in-dist recipe (matches run_ln_ablation.sh COMMON): rule_attn h=256,
# n_slots=16, n_app_slots=1, clw=5.0, grad_clip=0.5, balanced_sampling, batch=16,
# lr=3e-4 cosine, val_frac=0.1, 30k steps, patience 0, LN OFF, input_skip ON,
# depth 8 (n_nca_steps==n_nca_repeats==8, max-shared). Depth-8 is the shipped
# default; if Bouncers shows iteration-limited rollout residual, re-run at depth 16.
#
# Usage (one game per free GPU/box):
#   GPU=1 CONFIGS="Microban Bouncers" bash nca_wm/scripts/run_in_dist_next.sh   # local
#   GPU=0 CONFIGS="Heroes_of_Sokoban" bash nca_wm/scripts/run_in_dist_next.sh   # ssh 210
set -u
REPO=/home/jupyter-smearle/script-doctor
cd "$REPO"
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs/in_dist_next
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache
GPU="${GPU:-1}"
MEM="${MEM:-0.45}"
N_UPDATES="${N_UPDATES:-30000}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON=(
    --conditional --architecture rule_attn
    --n_hid 256 --n_slots 16 --n_app_slots 1
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --n_search_steps 200000 --search_timeout_ms 120000
    --batch_size 16 --lr 3e-4
    --val_frac 0.1 --val_eval_interval 500
    --n_updates "$N_UPDATES" --patience 0 --min_delta 1e-8
    --log_interval 500 --ckpt_interval 5000
    --n_nca_steps 8 --n_nca_repeats 8 --input_skip
    --max_transitions_per_game 300000
    --seed 0
)

CONFIGS="${CONFIGS:-Microban Heroes_of_Sokoban Bouncers}"

for game in $CONFIGS; do
    save_dir="$LOGDIR/$game"
    log="$LOGDIR/$game.out"
    if [ -f "$save_dir/train_meta.json" ]; then
        echo "[GPU $GPU] skip $game (already done)"; continue
    fi
    echo "[GPU $GPU] === $(date '+%H:%M:%S') start $game ==="
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        "${COMMON[@]}" --games "$game" \
        --save_dir "$save_dir" > "$log" 2>&1
    echo "[GPU $GPU] === $(date '+%H:%M:%S') done  $game (rc=$?) ==="
done
echo "[GPU $GPU] ALL DONE"

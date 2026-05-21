#!/usr/bin/env bash
# use_layernorm ablation (in-distribution single-game paradigm).
#
# Question: --use_layernorm defaults OFF. The only prior evidence (ARCHITECTURE
# F4 / SCALING_RESULTS 2026-05-03, Bouncers+Collapse, multi-game) said LN is "at
# best neutral, at worst harmful" — but that was BEFORE input_skip became the
# default and BEFORE the in-distribution single-game pivot. No in-dist single-game
# LN data exists. This sweep regenerates it under the current canonical recipe.
#
# `--use_layernorm` adds one shared pre-norm LayerNorm on h at the start of every
# NCA step (the attn/slot/win LayerNorms are always present regardless).
#
# Grid (all: rule_attn h=256, n_slots=16, n_app_slots=1, clw=5.0, grad_clip=0.5,
#       balanced_sampling, batch=16, lr=3e-4, val_frac=0.1, 30k steps, patience 0):
#
#   A. DEFAULT regime (pool ON + input_skip ON) — does enabling LN cost us on the
#      games/recipe we ship? Depth 8 (current default) and depth 32 (max-shared,
#      where the "stabilize deep unrolls" claim should bite). Games: TSM, THL.
#   B. STRESS regime (TSM, depth 32, pool OFF) with skip on/off — the historical
#      regime where bare training diverged and LN gave PARTIAL recovery. Does LN
#      still add value on top of (or instead of) input_skip now?
#
# Usage (split across two sequential queues, both on the free GPU):
#   GPU=1 MEM=0.45 CONFIGS="tsm_d8_lnoff tsm_d8_lnon tsm_d32_lnoff tsm_d32_lnon \
#       tsm_pooloff_skipoff_d32_lnoff tsm_pooloff_skipoff_d32_lnon \
#       tsm_pooloff_skipon_d32_lnoff tsm_pooloff_skipon_d32_lnon" \
#       bash nca_wm/scripts/run_ln_ablation.sh
#   GPU=1 MEM=0.45 CONFIGS="thl_d8_lnoff thl_d8_lnon thl_d32_lnoff thl_d32_lnon" \
#       bash nca_wm/scripts/run_ln_ablation.sh
set -u
REPO=/home/jupyter-smearle/script-doctor
cd "$REPO"
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs/ln_ablation
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
    --seed 0
)

# Per-config flag strings: game, depth (n_steps==n_repeats → max-shared), pool,
# skip, LN, and per-game transition cap.
declare -A EXTRA
# --- A. default regime (pool ON, skip ON) ---
EXTRA[tsm_d8_lnoff]="--games Travelling_salesman --max_transitions_per_game 100000 --n_nca_steps 8  --n_nca_repeats 8  --input_skip"
EXTRA[tsm_d8_lnon]="--games Travelling_salesman --max_transitions_per_game 100000 --n_nca_steps 8  --n_nca_repeats 8  --input_skip --use_layernorm"
EXTRA[tsm_d32_lnoff]="--games Travelling_salesman --max_transitions_per_game 100000 --n_nca_steps 32 --n_nca_repeats 32 --input_skip"
EXTRA[tsm_d32_lnon]="--games Travelling_salesman --max_transitions_per_game 100000 --n_nca_steps 32 --n_nca_repeats 32 --input_skip --use_layernorm"
EXTRA[thl_d8_lnoff]="--games Take_Heart_Lass --max_transitions_per_game 300000 --n_nca_steps 8  --n_nca_repeats 8  --input_skip"
EXTRA[thl_d8_lnon]="--games Take_Heart_Lass --max_transitions_per_game 300000 --n_nca_steps 8  --n_nca_repeats 8  --input_skip --use_layernorm"
EXTRA[thl_d32_lnoff]="--games Take_Heart_Lass --max_transitions_per_game 300000 --n_nca_steps 32 --n_nca_repeats 32 --input_skip"
EXTRA[thl_d32_lnon]="--games Take_Heart_Lass --max_transitions_per_game 300000 --n_nca_steps 32 --n_nca_repeats 32 --input_skip --use_layernorm"
# --- B. stress regime (TSM, pool OFF, depth 32) ---
EXTRA[tsm_pooloff_skipon_d32_lnoff]="--games Travelling_salesman --max_transitions_per_game 100000 --n_nca_steps 32 --n_nca_repeats 32 --no-axis_pool --no-axis_cummax --no-global_pool --input_skip"
EXTRA[tsm_pooloff_skipon_d32_lnon]="--games Travelling_salesman --max_transitions_per_game 100000 --n_nca_steps 32 --n_nca_repeats 32 --no-axis_pool --no-axis_cummax --no-global_pool --input_skip --use_layernorm"
EXTRA[tsm_pooloff_skipoff_d32_lnoff]="--games Travelling_salesman --max_transitions_per_game 100000 --n_nca_steps 32 --n_nca_repeats 32 --no-axis_pool --no-axis_cummax --no-global_pool"
EXTRA[tsm_pooloff_skipoff_d32_lnon]="--games Travelling_salesman --max_transitions_per_game 100000 --n_nca_steps 32 --n_nca_repeats 32 --no-axis_pool --no-axis_cummax --no-global_pool --use_layernorm"

CONFIGS="${CONFIGS:-${!EXTRA[@]}}"

for cfg in $CONFIGS; do
    if [ -z "${EXTRA[$cfg]+x}" ]; then
        echo "[GPU $GPU] UNKNOWN config '$cfg' — skipping"; continue
    fi
    save_dir="$LOGDIR/$cfg"
    log="$LOGDIR/$cfg.out"
    if [ -f "$save_dir/train_meta.json" ]; then
        echo "[GPU $GPU] skip $cfg (already done)"; continue
    fi
    echo "[GPU $GPU] === $(date '+%H:%M:%S') start $cfg : ${EXTRA[$cfg]} ==="
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        "${COMMON[@]}" ${EXTRA[$cfg]} \
        --save_dir "$save_dir" > "$log" 2>&1
    echo "[GPU $GPU] === $(date '+%H:%M:%S') done  $cfg (rc=$?) ==="
done
echo "[GPU $GPU] ALL DONE"

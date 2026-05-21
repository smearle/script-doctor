#!/usr/bin/env bash
# LR-schedule ablation: is the default cosine LR schedule necessary?
#
# Motivation: the default recipe uses --lr_schedule cosine, which ties LR decay
# (down to --lr_min=1e-7) to a fixed --n_updates budget. That makes resuming /
# extending a run unnatural: a finished run sits at LR~0, and extending it
# restarts cosine over the *remaining* steps, jumping LR back to full. A constant
# LR resumes as a no-op continuation. The code comment claims cosine is needed
# "for sharp-minimum cases where the model walks out of optima without decay."
# This ablation tests whether constant LR reaches and HOLDS the same val
# change_err as cosine on the in-distribution single-game recipe.
#
# Per game x {cosine, constant}, all else identical to the in-dist single-game
# recipe (== run_tsm_pool_diag COMMON, pool ON + input_skip). val_frac 0.1 held
# out for measurement only (patience 0 = no early stop). Travelling_salesman is
# the sharp-minimum / pooling stress case where cosine's justification bites
# hardest; sokoban_basic + nekopuzzle are easy controls.
#
# Usage:
#   GPU=0 GAMES="Travelling_salesman sokoban_basic nekopuzzle" \
#     bash nca_wm/scripts/run_lr_schedule_ablation.sh
set -u
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"   # cache paths are cwd-relative (rollout_data/ at repo root)
PY="$REPO/.venv/bin/python3"
LOGDIR="$REPO/nca_wm/logs/lr_sched_ablation"
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR="$REPO/.jax_compile_cache"
GPU="${GPU:-0}"
N_UPDATES="${N_UPDATES:-30000}"
GAMES="${GAMES:-Travelling_salesman sokoban_basic nekopuzzle}"
# Run conditions sequentially per game by default; set MAXJOBS>1 to overlap.
MAXJOBS="${MAXJOBS:-2}"

COMMON=(
    --conditional --architecture rule_attn
    --n_hid 256 --n_slots 16 --n_app_slots 1
    --n_nca_steps 8 --n_nca_repeats 8
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling --input_skip
    --n_search_steps 200000 --search_timeout_ms 120000
    --max_transitions_per_game 100000
    --batch_size 16 --lr 3e-4
    --val_frac 0.1 --val_eval_interval 500
    --n_updates "$N_UPDATES" --patience 0 --min_delta 1e-8
    --log_interval 500 --ckpt_interval 5000
    --seed 0
)

declare -A EXTRA
EXTRA[cosine]=""                          # default schedule
EXTRA[constant]="--lr_schedule constant"

launch() {
    local game="$1" cond="$2"
    local save_dir="$LOGDIR/${game}__${cond}"
    local log="$LOGDIR/${game}__${cond}.out"
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/train_meta.json" ]; then
        echo "[GPU $GPU] skip ${game}/${cond} (already done)"
        return
    fi
    echo "[GPU $GPU] === $(date '+%H:%M:%S') start ${game}/${cond} : ${EXTRA[$cond]:-<cosine default>} ==="
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        "${COMMON[@]}" --games "$game" ${EXTRA[$cond]} \
        --save_dir "$save_dir" > "$log" 2>&1
}

for game in $GAMES; do
    for cond in cosine constant; do
        launch "$game" "$cond" &
        # Throttle to MAXJOBS concurrent runs on the single GPU.
        while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
    done
done
wait
echo "[GPU $GPU] === all done $(date '+%H:%M:%S') ==="

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
# Cap per-process GPU memory so MAXJOBS runs fit on one device (JAX otherwise
# preallocates ~75-90%, and two unbounded runs already saturate a 24GB card).
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
GPU="${GPU:-0}"
N_UPDATES="${N_UPDATES:-30000}"
GAMES="${GAMES:-Travelling_salesman sokoban_basic nekopuzzle}"
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

# Condition-major with a hard barrier between waves. The cosine wave builds each
# game's per-level A*/dataset cache (distinct games => distinct files => safe to
# overlap MAXJOBS at a time). The barrier guarantees every cache is fully written
# before any constant run starts, so the constant wave only ever READS caches.
# Running both conditions of the SAME game concurrently corrupts the shared
# per-game cache (concurrent writers -> truncated npz -> EOFError), which is what
# this ordering prevents.
for cond in cosine constant; do
    for game in $GAMES; do
        launch "$game" "$cond" &
        while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n; done
    done
    wait   # barrier: finish all of this condition before starting the next
done
echo "[GPU $GPU] === all done $(date '+%H:%M:%S') ==="

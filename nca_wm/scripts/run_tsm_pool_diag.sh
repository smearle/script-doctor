#!/usr/bin/env bash
# TSM global-pooling diagnostic.
#
# Question: Travelling_salesman has only LOCAL rules (2-3 cell patterns, no
# `...` / no `[X][Y]` global rules), yet the per_game_arch grid showed global
# pooling breaks it (45%/27% trained-level rollout) while pool-OFF fits it (0%).
# Is that pool-ON failure a real architectural ceiling, or just under-training?
#
# 2x2 over (pooling x input_skip), all-levels single-game training, 10% of
# transitions held out for validation (NOT used for early stopping — patience 0
# disables stopping).
#
# Usage (run two instances, one per GPU, in parallel):
#   GPU=0 CONFIGS="pool_on_skip_on pool_off_skip_off"  bash nca_wm/scripts/run_tsm_pool_diag.sh
#   GPU=1 CONFIGS="pool_off_skip_on pool_on_skip_off"  bash nca_wm/scripts/run_tsm_pool_diag.sh
set -u
REPO=/home/jupyter-smearle/script-doctor
cd "$REPO"   # cache paths are cwd-relative (rollout_data/ at repo root)
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs/tsm_pool_diag
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache
GPU="${GPU:-1}"
N_UPDATES="${N_UPDATES:-30000}"
CONFIGS="${CONFIGS:-pool_off_skip_on pool_on_skip_on pool_on_skip_off pool_off_skip_off}"

COMMON=(
    --games Travelling_salesman
    --conditional --architecture rule_attn
    --n_hid 256 --n_slots 16 --n_app_slots 1
    --n_nca_steps 8 --n_nca_repeats 8
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --n_search_steps 200000 --search_timeout_ms 120000
    --max_transitions_per_game 100000
    --batch_size 16 --lr 3e-4
    --val_frac 0.1 --val_eval_interval 500
    --n_updates "$N_UPDATES" --patience 0 --min_delta 1e-8
    --log_interval 500 --ckpt_interval 5000
    --seed 0
)

# config -> extra flags (pooling x skip). Pool flags default ON.
declare -A EXTRA
EXTRA[pool_off_skip_on]="--no-axis_pool --no-axis_cummax --no-global_pool --input_skip"
EXTRA[pool_on_skip_on]="--input_skip"
EXTRA[pool_on_skip_off]=""
EXTRA[pool_off_skip_off]="--no-axis_pool --no-axis_cummax --no-global_pool"

for cfg in $CONFIGS; do
    save_dir="$LOGDIR/$cfg"
    log="$LOGDIR/$cfg.out"
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/train_meta.json" ]; then
        echo "[GPU $GPU] skip $cfg (already done)"
        continue
    fi
    echo "[GPU $GPU] === $(date '+%H:%M:%S') start $cfg : ${EXTRA[$cfg]:-<pool ON, no skip>} ==="
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        "${COMMON[@]}" ${EXTRA[$cfg]} \
        --save_dir "$save_dir" > "$log" 2>&1
    echo "[GPU $GPU] === $(date '+%H:%M:%S') done  $cfg (rc=$?) ==="
done
echo "[GPU $GPU] ALL DONE"

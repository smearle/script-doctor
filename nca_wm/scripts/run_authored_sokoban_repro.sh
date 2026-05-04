#!/usr/bin/env bash
# Reproduce the human-authored sokoban -> Microban generalization anchor.
set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_authored_sokoban_repro
SAVE_DIR=$LOGDIR/scaling1_authored_seed0_current
LOG=$LOGDIR/scaling1_authored_seed0_current.out
EVAL_SUBDIR=heldout_eval_dual_control

mkdir -p "$LOGDIR"

train_done=0
if "$PY" -c 'import json,sys; from pathlib import Path
d=Path(sys.argv[1]); meta=d/"train_meta.json"; cfg=d/"config.json"; ok=False
if meta.is_file() and cfg.is_file():
    m=json.load(open(meta)); c=json.load(open(cfg))
    ok=int(m.get("total_steps",0)) >= 20000 and c.get("games") == "scaling_1"
print(1 if ok else 0)' "$SAVE_DIR" | grep -q 1; then
    train_done=1
fi

if [ "$train_done" -eq 0 ]; then
    echo "[authored-repro] training -> $SAVE_DIR"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games scaling_1 \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps 4 --n_slots 16 --n_app_slots 1 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --n_search_steps 100000 --search_timeout_ms 60000 \
        --max_transitions_per_game 200000 \
        --n_updates 20000 --patience 200 --min_delta 1e-8 \
        --batch_size 64 --lr 3e-4 \
        --log_interval 2000 --ckpt_interval 5000 \
        --save_dir "$SAVE_DIR" \
        --seed 0 \
        >"$LOG" 2>&1
    train_exit=$?
    echo "[authored-repro] train exit=$train_exit"
    if [ "$train_exit" -ne 0 ]; then
        exit "$train_exit"
    fi
else
    echo "[authored-repro] training already complete -> $SAVE_DIR"
fi

if [ ! -f "$SAVE_DIR/$EVAL_SUBDIR/results.json" ]; then
    echo "[authored-repro] heldout eval"
    "$PY" "$REPO/nca_wm/heldout_eval.py" \
        --load "$SAVE_DIR" \
        --heldout_games sokoban_basic,Microban,Microban_I \
        --n_random_episodes 5 \
        --max_steps 30 \
        --include_train_sample 1 \
        --allow_training_games \
        --out_subdir "$EVAL_SUBDIR" \
        >>"$LOG" 2>&1
    echo "[authored-repro] eval exit=$?"
else
    echo "[authored-repro] heldout eval already complete"
fi

echo "[authored-repro] done: $SAVE_DIR/$EVAL_SUBDIR/results.json"

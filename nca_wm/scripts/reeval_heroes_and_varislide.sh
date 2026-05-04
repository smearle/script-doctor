#!/usr/bin/env bash
# Re-run auto-eval on all heroes and varislide_postfix checkpoints with the
# 2026-05-04 top-left-slicing fix applied. The original eval_multigame.npz
# files were generated before the fix and used centered slicing — which
# inflates cell-error for any level whose actual dims are smaller than the
# bucket dims (the model puts predictions at the top-left corner, but eval
# extracted from the center → comparing real obs against padded zeros).
#
# Saves new files as eval_multigame_tlfix.npz alongside the old ones, leaving
# the buggy npz in place for diff-ability.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

run_eval() {
    local gpu=$1
    local save_dir=$2
    local out_npz="$save_dir/eval_multigame_tlfix.npz"
    if [ -f "$out_npz" ]; then
        echo "  skip $(basename $save_dir): tlfix already exists"
        return
    fi
    echo "  [GPU $gpu] re-eval $(basename $save_dir)"
    # Use --render_only to load checkpoint and run eval without re-training.
    # Patch eval_multigame.npz path by symlinking the existing path elsewhere
    # then renaming after.
    local backup="$save_dir/eval_multigame_buggy.npz"
    [ -f "$save_dir/eval_multigame.npz" ] && [ ! -f "$backup" ] && \
        mv "$save_dir/eval_multigame.npz" "$backup"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --load "$save_dir" \
        --render_only \
        --no-render_gif \
        > "$save_dir/reeval.log" 2>&1
    if [ -f "$save_dir/eval_multigame.npz" ]; then
        mv "$save_dir/eval_multigame.npz" "$out_npz"
        echo "  [GPU $gpu] done $(basename $save_dir)"
    else
        echo "  [GPU $gpu] FAILED $(basename $save_dir)"
    fi
}

# Build queue
QUEUE=()
for d in $REPO/nca_wm/logs_heroes/heroes_*/ \
         $REPO/nca_wm/logs_canary/varislide_postfix[ABC]*/; do
    if [ -d "$d" ] && [ -f "$d/train_meta.json" ]; then
        QUEUE+=("$d")
    fi
done
echo "queue: ${#QUEUE[@]} dirs"

i=0
while [ $i -lt ${#QUEUE[@]} ]; do
    d1="${QUEUE[$i]}"
    if [ $((i+1)) -lt ${#QUEUE[@]} ]; then
        d2="${QUEUE[$((i+1))]}"
        run_eval 0 "$d1" &
        pid0=$!
        run_eval 1 "$d2" &
        pid1=$!
        wait $pid0 $pid1
        i=$((i+2))
    else
        run_eval 0 "$d1"
        i=$((i+1))
    fi
done
echo "ALL DONE"

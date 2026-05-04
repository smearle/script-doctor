#!/usr/bin/env bash
# Run after the post-bitpack-fix varislide sweep finishes. Generates the
# summary CSV/MD and the paper-style figures, then prints what to update.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
OUT=$REPO/nca_wm/figures/varislide_postfix
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="" "$PY" nca_wm/scripts/summarize_varislide_postfix_sweep.py \
    --out_csv "$OUT/summary.csv" --out_md "$OUT/summary.md"

CUDA_VISIBLE_DEVICES="" "$PY" nca_wm/scripts/plot_varislide_postfix_sweep.py \
    --csv "$OUT/summary.csv" --outdir "$OUT"

echo
echo "=== summary ==="
cat "$OUT/summary.md"
echo
echo "=== artifacts ==="
ls -la "$OUT/"

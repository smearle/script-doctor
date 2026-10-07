#!/usr/bin/env bash
# End-to-end smoke test beside the real queue: a tiny model trains, resumes from last.pt
# with a longer horizon, then sample_eval.py runs every arm at small counts.
set -euo pipefail
W=/workspace/psgen
S=$W/smoke
export PATH=/opt/node/bin:$PATH OMP_NUM_THREADS=8
cd "$W/code"
rm -rf "$S"
tiny=(--data "$W/data" --out "$S/run" --n-layer 2 --n-head 2 --d-model 128 --no-compile
      --batch-tokens 32768 --warmup 10 --evals-per-epoch 60 --device cpu)
python train_lm.py "${tiny[@]}" --epochs 0.03
python train_lm.py "${tiny[@]}" --epochs 0.06  # resumes from last.pt
python sample_eval.py --data "$W/data" --run "$S/run" --out "$S/eval" --engine-dir "$W/engine" \
  --n-uncond 8 --n-uncond-cold 4 --n-level-prompts 3 --level-samples 2 --level-max-new 256 \
  --batch 8 --workers 4 --max-human 8 --device cpu
echo SMOKE_OK

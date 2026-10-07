#!/usr/bin/env bash
# Stage 2 on the pod: canonical corpus -> training -> select -> eval -> canonical readouts,
# plus the re-check of stage 1's eval arms with the fixed ps_check.js.
#   bash run_canon.sh DATASET_REVISION
# Finished stages are skipped (marker files) and training resumes from last.pt, so a
# re-run continues where a crash or stop left off. Any failure writes $C/FAILED.
set -uo pipefail
W=/workspace/psgen
C=$W/canon
REV="$1"
cd "$W/code-canon" || exit 1
mkdir -p "$C"
exec >>"$C/run.log" 2>&1
STAGE=start
fail() {
  printf '{"stage": "%s", "time": %s, "rc": %s}\n' "$STAGE" "$(date +%s)" "$1" > "$C/FAILED"
  rm -f /root/PSGEN_RUNNING
  exit 1
}
stage() {
  STAGE=$1
  echo "===== $1 $(date -u +%FT%TZ)"
  printf '{"stage": "%s", "time": %s}\n' "$1" "$(date +%s)" > "$C/stage.json"
}
rm -f "$C/FAILED"
touch /root/PSGEN_RUNNING  # holds the idle watchdog off while the pipeline runs

stage setup
if [ ! -x /opt/node/bin/node ]; then  # container disk: a pod stop erases it, so reinstall
  mkdir -p /opt/node
  curl -fsSL https://nodejs.org/dist/v20.19.4/node-v20.19.4-linux-x64.tar.xz |
    tar --no-same-owner --strip-components=1 -xJ -C /opt/node || fail 1
fi
export PATH="/opt/node/bin:$PATH"
export PYTHONUNBUFFERED=1  # run.log is the controller's liveness signal
python -c "import tokenizers, huggingface_hub" 2>/dev/null || pip install -q tokenizers huggingface_hub || fail 1

stage prep
if [ ! -e "$C/data/prep_report.json" ]; then
  python prepare_canonical.py --out "$C/data" --revision "$REV" --engine-dir "$W/engine" --workers 24 || fail $?
fi

stage recheck-stage1
for e in eval eval-m85-e8-d01; do
  if [ ! -e "$C/recheck/$e/eval_report.json" ]; then
    python recheck_eval.py --eval "$W/$e" --out "$C/recheck/$e" --engine-dir "$W/engine" --workers 24 || fail $?
  fi
done

train() {  # NAME, then train_lm.py arguments
  local name=$1
  shift
  stage "train-$name"
  [ -e "$C/runs/$name/DONE" ] && return 0
  python train_lm.py --data "$C/data" --out "$C/runs/$name" "$@" || fail $?
  touch "$C/runs/$name/DONE"
}
# Two cosine horizons for the stage-1 recipe: 8 epochs, and the epoch count that gives
# stage 1's 2,808 optimizer steps on this (smaller) corpus (skipped below 10 epochs).
EMATCH=$(python - "$C/data" <<'PY'
import random, sys
from pathlib import Path
from train_lm import load_split, make_batches
_, _, lens = load_split(Path(sys.argv[1]), "train", 8192)
e = 2808 / len(make_batches(lens, 8192, 131072, random.Random(0)))
print(f"{e:.2f}" if e > 10 else "skip")
PY
) || fail $?
echo "epochs matching 2808 steps: $EMATCH"
M30="--n-layer 8 --n-head 8 --d-model 512 --dropout 0.1 --lr 1e-3"
train m30-e8-d01 $M30 --epochs 8
[ "$EMATCH" = skip ] || train m30-steps2808-d01 $M30 --epochs "$EMATCH"

stage select
BEST=$(python - "$C" <<'PY'
import json, pathlib, sys
c = pathlib.Path(sys.argv[1])
val = {p.parent.name: json.loads(p.read_text())["best_val"] for p in (c / "runs").glob("*/status.json")}
best = min(val, key=val.get)
(c / "selection.json").write_text(json.dumps({"best_val_loss": val, "selected": best}, indent=2))
print(best)
PY
) || fail $?

stage eval
if [ ! -e "$C/eval/eval_report.json" ]; then
  python sample_eval.py --data "$C/data" --run "$C/runs/$BEST" --out "$C/eval" \
    --engine-dir "$W/engine" --workers 24 || fail $?
fi

stage canonical-eval
if [ ! -e "$C/canonical_eval_stage2.json" ]; then
  python canonical_eval.py --eval "$C/eval" --data "$C/data" --canonical "$C/data" \
    --engine-dir "$W/engine" --out "$C/canonical_eval_stage2.json" || fail $?
fi
if [ ! -e "$C/canonical_eval_stage1.json" ]; then
  python canonical_eval.py --eval "$C/recheck/eval" --data "$W/data" --canonical "$C/data" \
    --raw-train-docs "$W/data/train_docs.jsonl" --engine-dir "$W/engine" \
    --out "$C/canonical_eval_stage1.json" || fail $?
fi

stage gallery
if [ ! -e "$C/eval/gallery/gallery_human_test.png" ]; then
  python gallery.py --eval "$C/eval" --data "$C/data" --engine-dir "$W/engine" --out "$C/eval/gallery" || fail $?
fi

stage done
touch "$C/ALL_DONE"
rm -f /root/PSGEN_RUNNING

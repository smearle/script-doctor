#!/usr/bin/env bash
# Stage 3 on the pod: level-first corpus -> unit tests -> training -> mechanics for held-out
# levels (level_eval.py).
#   bash run_level_first.sh DATASET_REVISION
# Finished stages are skipped (marker files) and training resumes from last.pt, so a
# re-run continues where a crash or stop left off. Any failure writes $L/FAILED.
set -uo pipefail
W=/workspace/psgen
L=$W/lf
REV="$1"
cd "$W/code-lf" || exit 1
mkdir -p "$L"
exec >>"$L/run.log" 2>&1
STAGE=start
fail() {
  printf '{"stage": "%s", "time": %s, "rc": %s}\n' "$STAGE" "$(date +%s)" "$1" > "$L/FAILED"
  rm -f /root/PSGEN_RUNNING
  exit 1
}
stage() {
  STAGE=$1
  echo "===== $1 $(date -u +%FT%TZ)"
  printf '{"stage": "%s", "time": %s}\n' "$1" "$(date +%s)" > "$L/stage.json"
}
rm -f "$L/FAILED"
touch /root/PSGEN_RUNNING  # holds the idle watchdog off while the pipeline runs

stage setup
if [ ! -x /opt/node/bin/node ]; then  # container disk: a pod stop erases it, so reinstall
  mkdir -p /opt/node
  curl -fsSL https://nodejs.org/dist/v20.19.4/node-v20.19.4-linux-x64.tar.xz |
    tar --no-same-owner --strip-components=1 -xJ -C /opt/node || fail 1
fi
export PATH="/opt/node/bin:$PATH"
export PYTHONUNBUFFERED=1  # run.log is the controller's liveness signal
python -c "import tokenizers, huggingface_hub, PIL" 2>/dev/null || pip install -q tokenizers huggingface_hub pillow || fail 1

stage prep
if [ ! -e "$L/data/prep_report.json" ]; then
  # --stall-s 600: a slow game is not a hung one (stage 2 lost one game to the 120 s default)
  python prepare_level_first.py --out "$L/data" --revision "$REV" --engine-dir "$W/engine" --workers 24 \
    --stall-s 600 || fail $?
fi

stage unit-tests
if [ ! -e "$L/unit_tests.log" ]; then
  python test_level_first.py --engine-dir "$W/engine" --extract "$L/data/extract.jsonl" --per-feature 8 \
    --workers 24 > "$L/unit_tests.tmp" 2>&1 || { cat "$L/unit_tests.tmp"; fail 1; }
  mv "$L/unit_tests.tmp" "$L/unit_tests.log"
fi

train() {  # NAME, then train_lm.py arguments
  local name=$1
  shift
  stage "train-$name"
  [ -e "$L/runs/$name/DONE" ] && return 0
  python train_lm.py --data "$L/data" --out "$L/runs/$name" "$@" || fail $?
  touch "$L/runs/$name/DONE"
}
# Stage 2's selected recipe at two horizons: stage 2's 2,808 optimizer steps, and twice that
# (this corpus is larger). Epochs are the cosine horizon, so convert steps to epochs.
epochs_for() {
  python - "$L/data" "$1" <<'PY'
import random, sys
from pathlib import Path
from train_lm import load_split, make_batches
_, _, lens = load_split(Path(sys.argv[1]), "train", 8192)
print(f"{int(sys.argv[2]) / len(make_batches(lens, 8192, 131072, random.Random(0))):.3f}")
PY
}
E1=$(epochs_for 2808) || fail $?
E2=$(epochs_for 5616) || fail $?
echo "epochs for 2808 / 5616 steps: $E1 / $E2"
M30="--n-layer 8 --n-head 8 --d-model 512 --dropout 0.1 --lr 1e-3"
train m30-s2808-d01 $M30 --epochs "$E1"
train m30-s5616-d01 $M30 --epochs "$E2"

stage select
BEST=$(python - "$L" <<'PY'
import json, pathlib, sys
c = pathlib.Path(sys.argv[1])
val = {p.parent.name: json.loads(p.read_text())["best_val"] for p in (c / "runs").glob("*/status.json")}
best = min(val, key=val.get)
(c / "selection.json").write_text(json.dumps({"best_val_loss": val, "selected": best}, indent=2))
print(best)
PY
) || fail $?

stage level-eval
if [ ! -e "$L/eval/level_eval_report.json" ]; then
  python level_eval.py --data "$L/data" --run "$L/runs/$BEST" --out "$L/eval" --engine-dir "$W/engine" \
    --workers 24 || fail $?
fi

stage done
touch "$L/ALL_DONE"
rm -f /root/PSGEN_RUNNING

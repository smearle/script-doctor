#!/usr/bin/env bash
# PuzzleScript generator pipeline on the pod: setup -> data -> training queue -> select -> eval.
#   bash run_pod.sh DATASET_REVISION
# Finished stages are skipped (marker files) and training resumes from last.pt, so a
# re-run continues where a crash or stop left off. Any failure writes $W/FAILED.
set -uo pipefail
W=/workspace/psgen
REV="$1"
cd "$W/code" || exit 1
exec >>"$W/run.log" 2>&1
STAGE=start
fail() {
  printf '{"stage": "%s", "time": %s, "rc": %s}\n' "$STAGE" "$(date +%s)" "$1" > "$W/FAILED"
  rm -f /root/PSGEN_RUNNING
  exit 1
}
stage() {
  STAGE=$1
  echo "===== $1 $(date -u +%FT%TZ)"
  printf '{"stage": "%s", "time": %s}\n' "$1" "$(date +%s)" > "$W/stage.json"
}
rm -f "$W/FAILED"
touch /root/PSGEN_RUNNING  # holds the idle watchdog off while the pipeline runs

stage setup
if [ ! -x /opt/node/bin/node ]; then  # container disk: a pod stop erases it, so reinstall
  mkdir -p /opt/node
  curl -fsSL https://nodejs.org/dist/v20.19.4/node-v20.19.4-linux-x64.tar.xz |
    tar --no-same-owner --strip-components=1 -xJ -C /opt/node || fail 1
fi
export PATH="/opt/node/bin:$PATH"
python -c "import tokenizers, huggingface_hub" 2>/dev/null || pip install -q tokenizers huggingface_hub || fail 1

stage prep
if [ ! -e "$W/data/prep_report.json" ]; then
  python prepare_data.py --out "$W/data" --revision "$REV" --engine-dir "$W/engine" --workers 24 --pretok lines || fail $?
fi

train() {  # NAME, then train_lm.py arguments
  local name=$1
  shift
  stage "train-$name"
  [ -e "$W/runs/$name/DONE" ] && return 0
  python train_lm.py --data "$W/data" --out "$W/runs/$name" "$@" || fail $?
  touch "$W/runs/$name/DONE"
}
train m85-e8-d01 --n-layer 12 --n-head 12 --d-model 768 --dropout 0.1 --epochs 8 --lr 6e-4
train m85-e4-d0 --n-layer 12 --n-head 12 --d-model 768 --dropout 0.0 --epochs 4 --lr 6e-4
train m30-e8-d01 --n-layer 8 --n-head 8 --d-model 512 --dropout 0.1 --epochs 8 --lr 1e-3

stage select
BEST=$(python - "$W" <<'PY'
import json, pathlib, sys
w = pathlib.Path(sys.argv[1])
val = {p.parent.name: json.loads(p.read_text())["best_val"] for p in (w / "runs").glob("*/status.json")}
best = min(val, key=val.get)
(w / "selection.json").write_text(json.dumps({"best_val_loss": val, "selected": best}, indent=2))
print(best)
PY
) || fail $?

stage eval
if [ ! -e "$W/eval/eval_report.json" ]; then
  python sample_eval.py --data "$W/data" --run "$W/runs/$BEST" --out "$W/eval" \
    --engine-dir "$W/engine" --workers 24 || fail $?
fi

stage done
touch "$W/ALL_DONE"
rm -f /root/PSGEN_RUNNING

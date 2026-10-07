#!/usr/bin/env bash
# Stage 3b on the pod: masked rules-only sampling (constrained.py) on stage 3's 24 rich candidate
# levels, against stage 3's unmasked diag-rules samples (control on file, not rerun).
#   bash run_masked_ab.sh
# Inputs under /workspace/psgen/ab: data/ (stage 3 corpus files), runs/m30-s2808-d01/ (selected model),
# diag-rules-samples.jsonl (control); /workspace/psgen/engine (script-doctor 315e01fc + PuzzleScript dfdeabcd).
# Steps: smoke (2 levels, 4 samples), checker validation on every val and test document, then the arms
# names and full. Finished steps are skipped (marker files); any failure writes $A/FAILED.
set -uo pipefail
W=/workspace/psgen
A=$W/ab
RUN=$A/runs/m30-s2808-d01
cd "$W/code-ab" || exit 1
mkdir -p "$A"
exec >>"$A/run.log" 2>&1
STAGE=start
fail() {
  printf '{"stage": "%s", "time": %s, "rc": %s}\n' "$STAGE" "$(date +%s)" "$1" > "$A/FAILED"
  rm -f /root/PSGEN_RUNNING
  exit 1
}
stage() {
  STAGE=$1
  echo "===== $1 $(date -u +%FT%TZ)"
  printf '{"stage": "%s", "time": %s}\n' "$1" "$(date +%s)" > "$A/stage.json"
}
rm -f "$A/FAILED"
touch /root/PSGEN_RUNNING  # holds the idle watchdog off while the pipeline runs

stage setup
sha256sum -c --quiet ../ab_code_manifest.sha256 || fail 1
if [ ! -x /opt/node/bin/node ]; then  # container disk: a pod stop erases it, so reinstall
  mkdir -p /opt/node
  curl -fsSL https://nodejs.org/dist/v20.19.4/node-v20.19.4-linux-x64.tar.xz |
    tar --no-same-owner --strip-components=1 -xJ -C /opt/node || fail 1
fi
export PATH="/opt/node/bin:$PATH"
export PYTHONUNBUFFERED=1  # run.log is the controller's liveness signal
python -c "import tokenizers, PIL" 2>/dev/null || pip install -q tokenizers pillow || fail 1

arm() {  # NAME, then level_eval.py options
  local name=$1
  shift
  stage "$name"
  [ -e "$A/$name/level_eval_report.json" ] && return 0
  python level_eval.py --data "$A/data" --run "$RUN" --out "$A/$name" --engine-dir "$W/engine" --workers 24 \
    --prompt-until RULES "$@" || fail $?
}

arm smoke --constrain full --n-levels 2 --samples 4 --temps 0.8 --max-new 2048

# CPU only: runs beside the sampling arms. Exit status 1 means some engine-accepted text was refused;
# the report lists every case for review.
if [ ! -e "$A/checker_validation.json" ]; then
  python test_constrained.py --data "$A/data" --samples "$A/diag-rules-samples.jsonl" --docs 1000000 \
    --out "$A/checker_validation.tmp" > "$A/checker_validation.log" 2>&1 &
  CHECKER=$!
fi

arm diag-names --constrain names --temps 0.6 0.8
arm diag-full --constrain full --temps 0.6 0.8

stage checker-validation
if [ -n "${CHECKER:-}" ]; then
  wait "$CHECKER"
  rc=$?
  [ "$rc" -le 1 ] && [ -s "$A/checker_validation.tmp" ] || fail $rc
  mv "$A/checker_validation.tmp" "$A/checker_validation.json"
fi

stage done
touch "$A/ALL_DONE"
rm -f /root/PSGEN_RUNNING

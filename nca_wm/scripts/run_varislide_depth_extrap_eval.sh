#!/usr/bin/env bash
# Eval + plot for the varislide depth-extrapolation sweep.
# Run after run_varislide_depth_extrap_sweep.sh has finished training.

set -e
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_depth_extrap

# Eval each run with overridden inference depths.
RUNS=()
for d in $(ls -d "$LOGDIR"/*/ 2>/dev/null); do
    if [ -f "$d/params.pkl" ]; then
        RUNS+=("${d%/}")
    fi
done

if [ ${#RUNS[@]} -eq 0 ]; then
    echo "no runs in $LOGDIR — train first"
    exit 1
fi

echo "evaluating ${#RUNS[@]} runs"
CUDA_VISIBLE_DEVICES=0 "$PY" "$REPO/nca_wm/scripts/eval_varislide_depth_extrap.py" \
    --runs "${RUNS[@]}" \
    --depths 1,2,4,8,16,32,64 \
    --widths 6,8,10,12,16

echo
echo "plotting"
"$PY" "$REPO/nca_wm/scripts/plot_varislide_depth_extrap.py" \
    --runs "${RUNS[@]}"

echo
echo "ALL DONE"

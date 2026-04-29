#!/usr/bin/env bash
# Overnight plan: incrementally train world models on growing game sets to
# move toward the "model all gallery games" milestone. All no-sprite for now
# (sprite decoder verified separately; trivial to add back).
#
# Config choices baked in based on the day's experiments:
#   --axis_pool --axis_cummax --global_pool : full global-context stack —
#       cheap (no new params), strict superset of any one alone.
#   --balanced_sampling    : fixes per-game dataset imbalance
#   --grad_clip 1.0        : cheap stability insurance (standard BPTT trick)
#   --n_hid 128            : sweet-spot for single-game; re-evaluate if insufficient
#   --conditional          : game-conditional FiLM
#   --patience 80 --min_delta 1e-6 : less aggressive than initial pass. The
#       arch sweep at 30/1e-5 stopped constellationz variants at <15K steps,
#       which looked confounded; loosening gives the model more room.
#
# Each run skips if already >= n_updates completed.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

GPU=${GPU:-0}

run_one() {
    local games=$1 hid=$2 n_updates=$3 tag=$4
    local log="$LOGDIR/growth_${games}_h${hid}_${tag}_gpu${GPU}.log"

    # Skip-check: expected save_dir pattern (tags match train.py's logic:
    # _bal for balanced_sampling, _ap/_ac/_gp for pool flags, applied
    # alphabetically in that order). Note: directory includes _pat-80 tag
    # since patience > 0.
    local pool_tag="_ap_ac_gp"
    local save_dir="$REPO/nca_wm/logs/multi_${games}_cond_bal${pool_tag}_level-None_nca-4_hid-${hid}_lr-0.001_pat-80_s-0"
    if [ -f "$save_dir/train_meta.json" ]; then
        # "Done" = reached full n_updates OR early-stopped at the same/larger
        # requested budget. Avoid rerunning a converged run just because its
        # step count is < n_updates.
        local done_status=$("$PY" -c "
import json, sys
m = json.load(open(sys.argv[1]))
steps = m.get('total_steps', 0)
early = m.get('early_stopped', False)
req = m.get('n_updates_requested', 0)
target = int(sys.argv[2])
if steps >= target: print('done')
elif early and req >= target: print('done')
else: print('not_done')
" "$save_dir/train_meta.json" "$n_updates" 2>/dev/null || echo "not_done")
        if [ "$done_status" = "done" ]; then
            echo "[$tag] SKIP $games (hid=$hid) — already finished (early_stopped or >= $n_updates steps)"
            return
        fi
    fi

    # Parallel-launcher lock: if another process is currently training this
    # same save_dir (e.g. GPU 1 companion script), skip to avoid clobber.
    # Stale locks (dead pid) are ignored.
    if [ -f "$save_dir/RUNNING.pid" ]; then
        local lock_pid=$(cat "$save_dir/RUNNING.pid" 2>/dev/null || echo "")
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            echo "[$tag] SKIP $games (hid=$hid) — already being trained by pid $lock_pid (parallel launcher)"
            return
        fi
    fi

    echo "[$tag] starting $games (hid=$hid, n_updates=$n_updates) -> $log"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games "$games" \
        --conditional \
        --balanced_sampling \
        --axis_pool --axis_cummax --global_pool \
        --n_hid "$hid" \
        --n_nca_steps 4 \
        --grad_clip 1.0 \
        --max_transitions_per_game 200000 \
        --n_updates "$n_updates" \
        --patience 80 --min_delta 1e-6 \
        --wandb \
        --sweep_name "overnight_growth" \
        >"$log" 2>&1
    echo "[$tag] finished $games (hid=$hid), exit=$?"
}

# Phase 1: confirm we can nail scaling_6 and small with the "good" config.
run_one scaling_6     128  80000   phase1
run_one small         128  80000   phase1

# Phase 1b: verify a single "hard" multi-bracket game (constellationz)
# given patience=80 and full pool stack — isolates whether the stuck
# arch-sweep result (~5% change_err) was early-stopping or capacity.
run_one global_constellationz 128  80000  phase1b

# Phase 1c: singleton runs of the scaling_6 bottleneck games (kettle, Zen
# Puzzle Garden, Travelling_salesman) to see if they fit individually at
# hid=128 — if yes, the multi-game stall is a capacity-sharing issue.
run_one global_kettle             128  40000  phase1c
run_one global_zen                128  40000  phase1c
run_one global_travelling_salesman 128 40000  phase1c

# Phase 2: hid=256 for both scaling_6 and small. scaling_6@hid=128 plateaus
# around 5% cerr (observed in flight), so bigger may unlock sub-1%.
run_one scaling_6     256  80000   phase2
run_one small         256  80000   phase2

# Phase 3: scale game count up. scaling_large extends "small" with more
# gallery games.
run_one scaling_large 128  100000  phase3
run_one scaling_large 256  100000  phase3

echo "All growth phases finished."

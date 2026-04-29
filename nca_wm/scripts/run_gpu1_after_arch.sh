#!/usr/bin/env bash
# After the current arch sweep (GPU 1) finishes, kick off experiments on
# games that the arch sweep left unfinished due to premature early-stopping:
#   - constellationz alone with full pool + grad_clip + patience=80
#   - clearing alone same
#   - nirvana alone same
# These are the cleanest reruns to see whether grad_clip + loose stopping
# actually lets these games fit.
#
# Auto-waits for the current arch sweep launcher to exit.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

echo "[gpu1-followup] starting followup runs (clearing + nirvana) on GPU 1"

GPU=1
# constellationz excluded: overnight_growth phase 1b is already training it on GPU 0.
# Same save_dir → would clobber.
GAMES=(global_clearing global_nirvana)

for game in "${GAMES[@]}"; do
    save_dir="$REPO/nca_wm/logs/multi_${game}_cond_bal_ap_ac_gp_level-None_nca-4_hid-128_lr-0.001_pat-80_s-0"
    if [ -f "$save_dir/train_meta.json" ]; then
        steps=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('total_steps',0))" "$save_dir/train_meta.json" 2>/dev/null || echo 0)
        if [ "$steps" -ge 80000 ] 2>/dev/null; then
            echo "[gpu1-followup] SKIP $game (already at >=80K steps)"
            continue
        fi
    fi
    log="$LOGDIR/followup_${game}_gpu1.log"
    echo "[gpu1-followup] starting $game -> $log"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games "$game" \
        --conditional \
        --balanced_sampling \
        --axis_pool --axis_cummax --global_pool \
        --grad_clip 1.0 \
        --n_hid 128 \
        --n_nca_steps 4 \
        --max_transitions_per_game 200000 \
        --n_updates 80000 \
        --patience 80 --min_delta 1e-6 \
        --wandb \
        --sweep_name "arch_followup_with_gradclip" \
        >"$log" 2>&1
    echo "[gpu1-followup] finished $game (exit=$?)"
done

echo "[gpu1-followup] All followup runs finished."

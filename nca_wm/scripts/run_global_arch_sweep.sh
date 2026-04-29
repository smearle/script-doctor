#!/usr/bin/env bash
# Sweep architectural variants {baseline, axis_pool, global_pool, cummax+global}
# on each of 4 single-game presets (neko, constellationz, clearing, nirvana).
# Sequential on GPU 1.
#
# Auto-waits for the in-flight nca_steps sweep on neko to finish so it
# doesn't compete for GPU 1.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/sweep_launch_logs
mkdir -p "$LOGDIR"

GPU=1
GAMES=(neko constellationz clearing nirvana)
VARIANTS=(baseline axis_pool global_pool cummax_global)

echo "[arch sweep] starting architecture sweep"

# Resolve variant tag → CLI flags
flags_for_variant() {
    case "$1" in
        baseline)       echo "" ;;
        axis_pool)      echo "--axis_pool" ;;
        axis_cummax)    echo "--axis_cummax" ;;
        global_pool)    echo "--global_pool" ;;
        axis_global)    echo "--axis_pool --global_pool" ;;
        cummax_global)  echo "--axis_cummax --global_pool" ;;
        all)            echo "--axis_pool --axis_cummax --global_pool" ;;
        *) echo "UNKNOWN_VARIANT_$1" ;;
    esac
}

# Compute pool tag for save_dir (matches train script's logic)
pool_tag_for_variant() {
    case "$1" in
        baseline)       echo "" ;;
        axis_pool)      echo "_ap" ;;
        axis_cummax)    echo "_ac" ;;
        global_pool)    echo "_gp" ;;
        axis_global)    echo "_ap_gp" ;;
        cummax_global)  echo "_ac_gp" ;;
        all)            echo "_ap_ac_gp" ;;
    esac
}

N_UPDATES=50000

# Consider a config done if it reached N_UPDATES OR if early-stopping kicked in
# (train_meta.json records total_steps; anything >= N_UPDATES/2 is plausibly
# done — be strict and require at least N_UPDATES).
already_done() {
    local game=$1 variant=$2
    local pt=$(pool_tag_for_variant "$variant")
    local d="$REPO/nca_wm/logs/multi_global_${game}_cond${pt}_level-None_nca-4_hid-128_lr-0.001_s-0"
    [ -f "$d/train_meta.json" ] || return 1
    local steps=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('total_steps',0))" "$d/train_meta.json" 2>/dev/null)
    [ "$steps" -ge "$N_UPDATES" ] 2>/dev/null
}

for game in "${GAMES[@]}"; do
    for v in "${VARIANTS[@]}"; do
        if already_done "$game" "$v"; then
            echo "[GPU $GPU] SKIP global_${game} / variant=${v} (already at >=${N_UPDATES} steps)"
            continue
        fi
        log="$LOGDIR/train_global_${game}_${v}_gpu${GPU}.log"
        flags=$(flags_for_variant "$v")
        echo "[GPU $GPU] starting global_${game} / variant=${v} -> $log"
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
            --games "global_${game}" \
            --conditional \
            --n_hid 128 \
            --n_nca_steps 4 \
            --n_updates "$N_UPDATES" \
            --patience 30 \
            --min_delta 1e-5 \
            --wandb \
            --sweep_name "global_arch_${game}" \
            $flags \
            >"$log" 2>&1
        echo "[GPU $GPU] finished global_${game} / variant=${v} (exit=$?)"
    done
done
echo "All global-arch configs finished."

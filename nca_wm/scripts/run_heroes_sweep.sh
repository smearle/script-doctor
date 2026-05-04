#!/usr/bin/env bash
# Heroes_of_Sokoban L0 depth × sharing × pool sweep.
# Heroes has genuinely non-local rules — `[> Wizard] -> [Wizard > Temp]` then
# `[> Temp | no Moveable no Static] -> [ | > Temp]` (chain projectile travel),
# `[Action Fighter] [SThief] -> ...` (multi-bracket character swap), and
# `late [Weighing YellowSwitch] [YellowDoor] -> ...` (multi-bracket door state).
# Pool features can't substitute for actual iteration here, so this is a
# clean depth-stress test for picking a scaling recipe.
#
# Recipe matches run_bouncers_shared_depth_sweep.sh so results compare directly
# to the prior Bouncers L×R findings.
#
# Buckets:
#   A: pool ON, fully-shared (n_repeats=depth)        × depth ∈ {4, 8, 16, 32} (4 runs)
#   B: pool ON, per-step      (n_repeats=1)           × depth ∈ {4, 8, 16, 32} (4 runs)
#   C: pool OFF + input_skip, fully-shared            × depth ∈ {4, 8, 16, 32} (4 runs)
#   D: pool OFF + input_skip, per-step                × depth ∈ {4, 8, 16, 32} (4 runs)
# Total 16 runs. Pairs of 2 on GPU 0 / GPU 1.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_heroes
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

COMMON=(
    --games Heroes_of_Sokoban --level 0
    --conditional --architecture rule_attn
    --n_hid 256 --n_slots 16 --n_app_slots 1
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --n_search_steps 200000 --search_timeout_ms 120000
    --max_transitions_per_game 100000
    --n_updates 15000 --patience 0 --min_delta 1e-6
    --batch_size 16 --lr 3e-4
    --log_interval 500 --ckpt_interval 5000
)

run_one() {
    local gpu=$1; local tag=$2; shift 2
    local save_dir="$LOGDIR/heroes_$tag"
    local log="$LOGDIR/heroes_$tag.out"
    if [ -f "$save_dir/train_meta.json" ]; then
        echo "  skip $tag"; return
    fi
    echo "  [GPU $gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        "${COMMON[@]}" "$@" \
        --save_dir "$save_dir" \
        > "$log" 2>&1
    if [ -f "$save_dir/params.pkl" ]; then
        echo "  [GPU $gpu] done $tag (train ok)"
    else
        echo "  [GPU $gpu] FAILED $tag"
    fi
}

# Build queue: tag|depth|n_repeats|extra_flags
QUEUE=()
for depth in 4 8 16 32; do
    QUEUE+=("A_pool_shared_d${depth}|$depth|$depth|")
    QUEUE+=("B_pool_perstep_d${depth}|$depth|1|")
    QUEUE+=("C_nopool_shared_d${depth}|$depth|$depth|--no-axis_pool --no-axis_cummax --no-global_pool --input_skip")
    QUEUE+=("D_nopool_perstep_d${depth}|$depth|1|--no-axis_pool --no-axis_cummax --no-global_pool --input_skip")
done

run_from_spec() {
    local gpu=$1; local spec=$2
    IFS='|' read -r tag depth n_reps extra <<< "$spec"
    eval "run_one $gpu \"$tag\" --seed 0 --n_nca_steps $depth --n_nca_repeats $n_reps $extra"
}

echo "queue: ${#QUEUE[@]} runs"
i=0
while [ $i -lt ${#QUEUE[@]} ]; do
    s1="${QUEUE[$i]}"
    if [ $((i+1)) -lt ${#QUEUE[@]} ]; then
        s2="${QUEUE[$((i+1))]}"
        run_from_spec 0 "$s1" &
        pid0=$!
        run_from_spec 1 "$s2" &
        pid1=$!
        wait $pid0 $pid1
        i=$((i+2))
    else
        run_from_spec 0 "$s1"
        i=$((i+1))
    fi
done
echo "ALL DONE"

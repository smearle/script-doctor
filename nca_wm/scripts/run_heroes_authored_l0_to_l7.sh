#!/usr/bin/env bash
# Heroes_of_Sokoban authored L0–L7 transfer experiment.
#
# Motivation: the L0-only sweep (run_heroes_sweep.sh) hit 0% on L0 with the
# robust C recipe (pool OFF + input_skip + shared) at every depth, but
# generalized to held-out authored levels (L1–L21) at only ~21% cell-error
# even at the best depth (d=4). Heroes has rule firings that L0 alone barely
# exposes (Wizard projectile chains in long corridors, Fighter/SThief swap,
# late-rule Weighing/Door state). Here we train on L0–L7 (more rule
# coverage at realistic distributions) and eval on L8–L21 to measure
# whether seeing more authored levels closes the transfer gap.
#
# Recipe (matches sweep bucket C, the winner): pool OFF + input_skip +
# shared body. Sweep depth ∈ {4, 8, 16}; d=32 is unlikely to help per the
# original sweep (d=4 was best on heldout for every variant).
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_heroes_authored
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

COMMON=(
    --games Heroes_of_Sokoban --train_levels 0,1,2,3,4,5,6,7
    --conditional --architecture rule_attn
    --n_hid 256 --n_slots 16 --n_app_slots 1
    --no-axis_pool --no-axis_cummax --no-global_pool --input_skip
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --n_search_steps 200000 --search_timeout_ms 120000
    --max_transitions_per_game 100000
    --n_updates 20000 --patience 0 --min_delta 1e-6
    --batch_size 16 --lr 3e-4
    --log_interval 500 --ckpt_interval 5000
)

run_one() {
    local gpu=$1; local tag=$2; shift 2
    local save_dir="$LOGDIR/heroes_l07_$tag"
    local log="$LOGDIR/heroes_l07_$tag.out"
    if [ -f "$save_dir/train_meta.json" ]; then
        echo "  skip $tag"; return
    fi
    echo "  [GPU $gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        "${COMMON[@]}" "$@" \
        --save_dir "$save_dir" \
        > "$log" 2>&1
    if [ -f "$save_dir/params.pkl" ]; then
        echo "  [GPU $gpu] done $tag"
    else
        echo "  [GPU $gpu] FAILED $tag"
    fi
}

QUEUE=(
    "C_d4|4|4"
    "C_d8|8|8"
)

run_from_spec() {
    local gpu=$1; local spec=$2
    IFS='|' read -r tag depth n_reps <<< "$spec"
    run_one $gpu "$tag" --seed 0 --n_nca_steps $depth --n_nca_repeats $n_reps
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

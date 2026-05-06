#!/usr/bin/env bash
# Synth-only variant of run_per_game_arch_grid.sh.
#
# Same per-game × per-architecture grid (5 mechanic-classes × 4 buckets at
# n_nca_steps=8), but every cell is trained on synthetic levels generated at
# the game's authored max-dim — *no authored levels enter the training set*.
# The headline question is: does the rule-conditioned NCA generalize from
# pure-synth supervision to the full set of authored levels for each game?
# Because no authored level is L0, every authored level here counts as
# held-out for evaluation purposes (use summarize_per_game_arch_grid.py
# --all-authored-as-heldout).
#
# Recipe matches project_synth_recipe.md (validated post-1fa557d), plus
# rule-coverage-driven evolution (newly wired through train.py CLI):
#   --synthetic_levels 64
#   --synthetic_per_game_size
#   --synthetic_fallback_dynamics
#   --synthetic_no_a_count_max 5
#   --synthetic_mode evolve --synthetic_evolve_pop_size 24
#   --synthetic_evolve_max_generations 60
#   --synthetic_require_solvable --synthetic_min_states 5
#   --synthetic_max_iters_search 1500 --synthetic_timeout_ms_search 400
#   --token_decoder_loss_weight 0.1
#   --synthetic_track_rules_fired                                  # log + enable below
#   --synthetic_rule_coverage_weight 100                           # GA fitness += 100 * |new rules fired|
#   --synthetic_coverage_select_topk                               # greedy union pick at end
#
# The coverage knobs matter most when K (--synthetic_levels) is small
# relative to the rule-set size — the GA's iterations-only fitness can
# otherwise pick K near-duplicates of the easiest mechanic. 100 is on
# the high side; lower it (e.g. 25–50) if you observe the GA losing
# solvability rate. Cache key encodes the weight, so existing iterations-
# only caches are not clobbered.
#
# Buckets at fixed n_nca_steps=8 (rule_attn defaults: h=256, K=16, batch=16):
#   A: pool ON,            shared (n_repeats=8)
#   B: pool ON,            per-step (n_repeats=1)
#   C: pool OFF + skip,    shared
#   D: pool OFF + skip,    per-step
#
# Usage:
#   bash nca_wm/scripts/run_per_game_arch_synth_grid.sh
#   GAMES="sokoban_basic Microban" BUCKETS="A D" bash ...
#   N_UPDATES=30000 bash ...                        # tighter / wider budget
set -u
cd /home/jupyter-earle/script-doctor

REPO=/home/jupyter-earle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_per_game_arch_synth
mkdir -p "$LOGDIR"
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

GAMES="${GAMES:-Microban Heroes_of_Sokoban Bouncers nekopuzzle Travelling_salesman}"
BUCKETS="${BUCKETS:-A B C D}"
DEPTH="${DEPTH:-8}"
N_UPDATES="${N_UPDATES:-30000}"
SEED="${SEED:-0}"
N_SYNTH="${N_SYNTH:-64}"

COMMON=(
    --conditional --architecture rule_attn
    --n_hid 256 --n_slots 16 --n_app_slots 1
    --change_loss_weight 5.0 --grad_clip 0.5
    --balanced_sampling
    --n_search_steps 200000 --search_timeout_ms 120000
    --max_transitions_per_game 100000
    --n_updates "$N_UPDATES" --patience 0 --min_delta 1e-6
    --batch_size 16 --lr 3e-4
    --log_interval 500 --ckpt_interval 5000
    --seed "$SEED"
    # Synth-only training set (no authored level is fed to the model).
    --synthetic_levels "$N_SYNTH"
    --synthetic_per_game_size
    --synthetic_fallback_dynamics
    --synthetic_no_a_count_max 5
    --synthetic_mode evolve
    --synthetic_evolve_pop_size 24
    --synthetic_evolve_max_generations 60
    --synthetic_require_solvable
    --synthetic_min_states 5
    --synthetic_max_iters_search 1500
    --synthetic_timeout_ms_search 400
    # Rule-coverage-driven evolution: tracks rule firings, biases GA fitness
    # toward coverage, and greedy-union-picks the final K. Override at launch
    # with COVERAGE_WEIGHT=0 to disable.
    --synthetic_track_rules_fired
    --synthetic_rule_coverage_weight "${COVERAGE_WEIGHT:-100}"
    --synthetic_coverage_select_topk
    --token_decoder_loss_weight 0.1
)

flags_for_bucket() {
    case "$1" in
        A) echo "--n_nca_steps $DEPTH --n_nca_repeats $DEPTH" ;;
        B) echo "--n_nca_steps $DEPTH --n_nca_repeats 1" ;;
        C) echo "--n_nca_steps $DEPTH --n_nca_repeats $DEPTH --no-axis_pool --no-axis_cummax --no-global_pool --input_skip" ;;
        D) echo "--n_nca_steps $DEPTH --n_nca_repeats 1       --no-axis_pool --no-axis_cummax --no-global_pool --input_skip" ;;
        *) echo "UNKNOWN_$1" ;;
    esac
}

run_one() {
    local gpu=$1; local game=$2; local bucket=$3
    local tag="${game}__${bucket}_d${DEPTH}"
    local save_dir="$LOGDIR/$tag"
    local log="$LOGDIR/$tag.out"
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/train_meta.json" ]; then
        echo "  [GPU $gpu] skip $tag (already done)"
        return
    fi
    local extra
    extra=$(flags_for_bucket "$bucket")
    echo "  [GPU $gpu] start $tag"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games "$game" \
        "${COMMON[@]}" $extra \
        --save_dir "$save_dir" \
        > "$log" 2>&1
    if [ -f "$save_dir/params.pkl" ]; then
        echo "  [GPU $gpu] done $tag"
    else
        echo "  [GPU $gpu] FAILED $tag (see $log)"
    fi
}

QUEUE=()
for g in $GAMES; do
    for b in $BUCKETS; do
        QUEUE+=("${g}|${b}")
    done
done

GPUS_STR="${GPUS:-0}"
read -r -a GPUS <<< "$GPUS_STR"
N_GPUS=${#GPUS[@]}

echo "queue: ${#QUEUE[@]} cells (depth=$DEPTH, n_updates=$N_UPDATES, n_synth=$N_SYNTH, seed=$SEED, gpus=${GPUS_STR})"

i=0
while [ $i -lt ${#QUEUE[@]} ]; do
    pids=()
    for ((g_idx=0; g_idx<N_GPUS && i<${#QUEUE[@]}; g_idx++, i++)); do
        spec="${QUEUE[$i]}"
        IFS='|' read -r game bucket <<< "$spec"
        gpu="${GPUS[$g_idx]}"
        run_one "$gpu" "$game" "$bucket" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
done
echo "ALL DONE"

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
# Recipe matches project_synth_recipe.md (validated post-1fa557d), plus:
# (a) Per-game *smallest* authored size — picked from the union of authored
#     observation_shapes — instead of either the bounding-box max
#     (--synthetic_per_game_size, which picks a size few authored levels
#     actually have) or multi_grid (which dilutes per-size data density).
#     Validated 2026-05-05: Microban w=7 K=128 30k → 0.60% BFS / 0.02%
#     1-step on all 10 authored levels (1-step beats the authored-only
#     baseline of 0.11%). Size-up generalization carries the smaller
#     synth grid to bigger authored levels per project_synth_recipe.md.
#     See feedback_per_game_size_pitfall.md.
# (b) Rule-coverage-driven evolution (newly wired through train.py CLI):
#   --synthetic_track_rules_fired                                  # log + enable below
#   --synthetic_rule_coverage_weight 100                           # GA fitness += 100 * |new rules fired|
#   --synthetic_coverage_select_topk                               # greedy union pick at end
#
# Full recipe used:
#   --synthetic_levels 128
#   --synthetic_w <min_W>  --synthetic_h <min_H>     # per-game lookup at launch
#   --synthetic_fallback_dynamics
#   --synthetic_no_a_count_max 5
#   --synthetic_mode evolve --synthetic_evolve_pop_size 24
#   --synthetic_evolve_max_generations 60
#   --synthetic_require_solvable --synthetic_min_states 5
#   --synthetic_max_iters_search 1500 --synthetic_timeout_ms_search 400
#   --token_decoder_loss_weight 0.1
#
# The coverage knobs matter most when K is small relative to the rule-set
# size — iterations-only fitness can otherwise pick K near-duplicates of
# the easiest mechanic. Cache key encodes the weight, so the baseline-
# recipe (rc_weight=0) caches written by prewarm_synth_caches.py are kept
# side-by-side for the appendix comparison.
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
# K=128 gives ≥10 levels per unique authored size on the worst game (Heroes,
# 12 unique sizes); single-size sokoban_basic-style work used K=64 just fine.
N_SYNTH="${N_SYNTH:-128}"

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
    # synthetic_w/h are filled in per-game by min_size_for() inside run_one
    --synthetic_fallback_dynamics
    --synthetic_no_a_count_max 5
    --synthetic_mode evolve
    --synthetic_evolve_pop_size 24
    --synthetic_evolve_max_generations 60
    --synthetic_require_solvable
    --synthetic_min_states 5
    --synthetic_max_iters_search 1500
    --synthetic_timeout_ms_search 400
    --token_decoder_loss_weight 0.1
)

# Rule-coverage-driven evolution. ON by default. Set COVERAGE=0 (or
# RECIPE=baseline) to disable all three coverage knobs in unison —
# this hits the appendix's iterations-only baseline cache. The cache
# key encodes the coverage state, so coverage and baseline runs do not
# share caches.
COVERAGE="${COVERAGE:-1}"
if [ "${RECIPE:-}" = "baseline" ]; then
    COVERAGE=0
fi
if [ "$COVERAGE" = "1" ]; then
    COMMON+=(
        --synthetic_track_rules_fired
        --synthetic_rule_coverage_weight "${COVERAGE_WEIGHT:-100}"
        --synthetic_coverage_select_topk
    )
    RECIPE_TAG="rc100_cstop"
else
    RECIPE_TAG="baseline"
fi

flags_for_bucket() {
    case "$1" in
        A) echo "--n_nca_steps $DEPTH --n_nca_repeats $DEPTH" ;;
        B) echo "--n_nca_steps $DEPTH --n_nca_repeats 1" ;;
        C) echo "--n_nca_steps $DEPTH --n_nca_repeats $DEPTH --no-axis_pool --no-axis_cummax --no-global_pool --input_skip" ;;
        D) echo "--n_nca_steps $DEPTH --n_nca_repeats 1       --no-axis_pool --no-axis_cummax --no-global_pool --input_skip" ;;
        *) echo "UNKNOWN_$1" ;;
    esac
}

# Pick the smallest-by-area authored size for this game (echoes "<W> <H>").
# Using an actual authored size (not a per-axis composite min) avoids picking
# a (min_W, min_H) that no real level has — for Microban that would be 6x6
# while no authored level is 6x6. Size-up generalization is the validated
# direction (project_synth_recipe.md); size-down is not.
min_size_for() {
    local game=$1
    "$PY" - <<PYEOF
from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.utils import init_ps_lark_parser
parser = init_ps_lark_parser()
backend = CppPuzzleScriptBackend()
js = backend.compile_and_serialize(parser, "$game")
env0 = CppPuzzleScriptEnv(js, level_i=0, max_episode_steps=10)
sizes = []
for li in range(int(env0.num_levels)):
    e = CppPuzzleScriptEnv(js, level_i=li, max_episode_steps=10)
    _, h, w = e.observation_shape
    sizes.append((int(w), int(h)))
# Smallest-by-area authored size; tiebreak by sum, then by (w, h)
mw, mh = min(set(sizes), key=lambda wh: (wh[0]*wh[1], wh[0]+wh[1], wh))
print(f"{mw} {mh}")
PYEOF
}

run_one() {
    local gpu=$1; local game=$2; local bucket=$3
    local tag="${game}__${bucket}_d${DEPTH}"
    # Baseline runs land in a parallel set of save dirs so they don't
    # overwrite the coverage runs from the canonical sweep.
    if [ "$RECIPE_TAG" != "rc100_cstop" ]; then
        tag="${tag}__${RECIPE_TAG}"
    fi
    local save_dir="$LOGDIR/$tag"
    local log="$LOGDIR/$tag.out"
    if [ -f "$save_dir/params.pkl" ] && [ -f "$save_dir/train_meta.json" ]; then
        echo "  [GPU $gpu] skip $tag (already done)"
        return
    fi
    local extra wh
    extra=$(flags_for_bucket "$bucket")
    wh=$(min_size_for "$game" 2>>"$LOGDIR/min_size.err" | tail -1)
    if [ -z "$wh" ]; then
        echo "  [GPU $gpu] FAILED $tag — could not resolve min synth size for $game"
        return
    fi
    local mw mh
    mw=${wh%% *}; mh=${wh##* }
    echo "  [GPU $gpu] start $tag  (synth ${mw}x${mh})"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games "$game" \
        --synthetic_w "$mw" --synthetic_h "$mh" \
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

#!/usr/bin/env bash
# Re-run train.py --render_only on a list of run dirs, reconstructing the
# original training CLI from config.json so the model is built with the
# correct flags. Saves the corrected eval as eval_multigame_tlfix.npz and
# preserves the buggy original as eval_multigame_buggy.npz.
set -u
cd /home/jupyter-smearle/script-doctor

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
export JAX_COMPILATION_CACHE_DIR=$REPO/.jax_compile_cache

reeval_one() {
    local gpu=$1
    local run_dir=$2
    local out_npz="$run_dir/eval_multigame_tlfix.npz"
    if [ -f "$out_npz" ]; then
        echo "  skip $(basename $run_dir): tlfix exists"; return
    fi
    if [ ! -f "$run_dir/config.json" ] || [ ! -f "$run_dir/params.pkl" ]; then
        echo "  skip $(basename $run_dir): missing config or params"; return
    fi

    # Build CLI from config.json
    local args
    args=$("$PY" - <<'PY' "$run_dir"
import sys, json, shlex
cfg = json.load(open(sys.argv[1] + "/config.json"))
out = []
# BooleanOptionalAction flags accept --no-X. store_true flags only accept --X.
BOA = {"conditional","vq_codebook","balanced_sampling",
       "axis_pool","axis_cummax","global_pool",
       "synthetic_require_solvable"}
ST = {"encode_sprites","use_layernorm","input_skip","adaptive_halt",
      "synthetic_per_game_size","synthetic_multi_grid",
      "synthetic_fallback_dynamics","render_gif","render_only","play","wandb"}
def add(k, v):
    if v is None or v == "" or (isinstance(v, list) and not v):
        return
    flag = "--" + k
    if isinstance(v, bool):
        if k in BOA:
            out.append(flag if v else "--no-" + k)
        elif k in ST:
            if v:
                out.append(flag)
        else:
            # Unknown bool — try BOA-style; if it fails the caller fixes it.
            out.append(flag if v else "--no-" + k)
    elif isinstance(v, list):
        out.extend([flag, ",".join(str(x) for x in v)])
    else:
        out.extend([flag, str(v)])

# All meaningful flags. argparse names follow cfg keys.
keys = [
    "games","game","level","train_levels","conditional","architecture",
    "n_hid","n_nca_steps","n_nca_repeats","n_slots","n_app_slots","d_slot",
    "d_model","n_enc_layers","n_heads","d_z",
    "axis_pool","axis_cummax","global_pool",
    "use_layernorm","input_skip",
    "adaptive_halt","halt_prior_p","halt_kl_weight","halt_mode",
    "vq_codebook","vq_codebook_size","vq_commitment_weight","vq_loss_weight",
    "token_decoder_loss_weight","decoder_d_model","decoder_n_layers","decoder_n_heads",
    "lr","lr_schedule","lr_min","grad_clip","change_loss_weight",
    "win_loss_weight","win_pos_weight","encode_sprites","sprite_loss_weight",
    "balanced_sampling",
    "n_search_steps","search_timeout_ms","max_episode_steps","search_algo",
    "max_transitions_per_game",
    "synthetic_levels","synthetic_w","synthetic_h","synthetic_seed",
    "synthetic_min_states","synthetic_mode","synthetic_per_game_size",
    "synthetic_multi_grid","synthetic_grid_sizes","synthetic_fallback_dynamics",
    "synthetic_no_a_count_max","synthetic_evolve_pop_size",
    "synthetic_evolve_max_generations","synthetic_evolve_n_mutations_min",
    "synthetic_evolve_n_mutations_max","synthetic_require_solvable",
    "synthetic_max_attempts_per_level","synthetic_max_iters_search",
    "synthetic_timeout_ms_search",
    "n_updates","batch_size","seed","log_interval","patience","min_delta",
    "ckpt_interval",
]
for k in keys:
    if k in cfg:
        add(k, cfg[k])
print(" ".join(shlex.quote(s) for s in out))
PY
)

    # Backup buggy eval if not already
    [ -f "$run_dir/eval_multigame.npz" ] && [ ! -f "$run_dir/eval_multigame_buggy.npz" ] && \
        mv "$run_dir/eval_multigame.npz" "$run_dir/eval_multigame_buggy.npz"

    echo "  [GPU $gpu] re-eval $(basename $run_dir)"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
        eval "\"$PY\" \"$REPO/nca_wm/train.py\" \
            $args \
            --save_dir \"$run_dir\" --load \"$run_dir\" --render_only" \
        > "$run_dir/reeval.log" 2>&1
    rc=$?
    if [ -f "$run_dir/eval_multigame.npz" ]; then
        mv "$run_dir/eval_multigame.npz" "$out_npz"
        echo "  [GPU $gpu] done $(basename $run_dir)"
    else
        echo "  [GPU $gpu] FAILED $(basename $run_dir) rc=$rc"
    fi
}

QUEUE=()
for d in $REPO/nca_wm/logs_heroes/heroes_*/ \
         $REPO/nca_wm/logs_canary/varislide_postfix[ABC]*/; do
    [ -d "$d" ] && [ -f "$d/train_meta.json" ] && QUEUE+=("$d")
done
echo "queue: ${#QUEUE[@]} dirs"

i=0
while [ $i -lt ${#QUEUE[@]} ]; do
    d1="${QUEUE[$i]}"
    if [ $((i+1)) -lt ${#QUEUE[@]} ]; then
        d2="${QUEUE[$((i+1))]}"
        reeval_one 0 "$d1" &
        pid0=$!
        reeval_one 1 "$d2" &
        pid1=$!
        wait $pid0 $pid1
        i=$((i+2))
    else
        reeval_one 0 "$d1"
        i=$((i+1))
    fi
done
echo "ALL DONE"

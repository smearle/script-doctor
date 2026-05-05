#!/usr/bin/env bash
# Render comparison GIFs for the two best nekopuzzle synth-trained NCA WMs.
#   v4: tpe_n256_5sizes  — best BFS authored (0.31%)
#   v5: tpe_n512_5sizes  — best random-AR authored (1.00%)
# Per memory feedback_rule_attn_load_flags.md: pass every rule_attn boolean
# flag explicitly, since omitting one silently changes the forward pass.

set -u

REPO=/home/jupyter-smearle/script-doctor
PY=$REPO/.venv/bin/python3
LOGDIR=$REPO/nca_wm/logs_neko_arch

render_one() {
    local gpu=$1
    local tag=$2
    local rundir="$LOGDIR/${tag}"
    local log="$LOGDIR/${tag}.render.out"

    if [ ! -f "$rundir/params_best.pkl" ]; then
        echo "[$tag] SKIP — no params_best.pkl"; return
    fi
    echo "[$tag] gpu=$gpu rendering -> $rundir"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 "$PY" "$REPO/nca_wm/train.py" \
        --games nekopuzzle \
        --conditional --architecture rule_attn \
        --n_hid 256 --n_nca_steps 16 --n_nca_repeats 1 \
        --n_slots 16 --n_app_slots 1 \
        --axis_pool --axis_cummax --global_pool \
        --change_loss_weight 5.0 --grad_clip 0.5 \
        --balanced_sampling \
        --max_transitions_per_game 200000 \
        --batch_size 16 --lr 3e-4 \
        --n_updates 15000 --patience 0 --min_delta 1e-6 \
        --log_interval 500 --ckpt_interval 5000 \
        --max_episode_steps 30 \
        --render_only --render_gif \
        --load "$rundir" \
        --save_dir "$rundir" \
        --seed 0 \
        > "$log" 2>&1
    echo "[$tag] finished exit=$? -> $rundir (log: $log)"
}

(
    render_one 0 "neko_d16_pool_perstep_tpe_n256_5sizes_s0"
    echo "[gpu0] done"
) &
PA=$!

(
    render_one 1 "neko_d16_pool_perstep_tpe_n512_5sizes_s0"
    echo "[gpu1] done"
) &
PB=$!

wait "$PA"; wait "$PB"
echo "[master] renders done."

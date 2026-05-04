"""Curriculum loop: alternate training with population-based level mutation,
where fitness = world-model prediction error on each level's transitions.

Pipeline per generation:
  1. Score the current level pool by ``change_err`` (1 − change_acc) on
     each level's transitions, using the current model params.
  2. Mutate top-error parents (rank-weighted) to produce M children via
     ``LevelMutator`` (swap/place/remove/move + player-count-respecting).
  3. Filter children for validity (BFS-based, same checks as
     ``synthetic_levels.search_validate``).
  4. Score children by ``change_err``; combine with parents and keep the
     top ``pop_size`` by error.
  5. Train for ``steps_per_generation`` updates on the refreshed pool.

Output is heldout_eval-compatible (writes ``config.json``, ``params.pkl``,
``game_infos.pkl``), so you can re-use ``heldout_eval.py`` to test
generalization.

Usage:
    python -m nca_wm.curriculum --game sokoban_basic --pop_size 16 \\
        --warmup_steps 2000 --n_generations 5 --steps_per_generation 2000 \\
        --n_children_per_gen 16 --save_dir nca_wm/logs/curr_sokoban_demo
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp._puzzlescript_cpp import Engine
from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.utils import init_ps_lark_parser
from evolve_level_cpp import LevelMutator

from nca_wm.synthetic_levels import (
    LevelGenerator,
    search_validate,
    _bitpacked_to_multihot,
    _evolve_levels,
)
from nca_wm.tokenize_game import (
    tokenize_game,
    get_game_tree_from_js,
    VOCAB_SIZE_BASE,
    VOCAB_SIZE_EXT,
    VOCAB_SIZE_EXT_V2,
)
from nca_wm.train import (
    N_ACTIONS,
    make_train_step,
    make_eval_forward,
    _build_sprite_tensor,
    _wm_p,
    ConditionalNCAWorldModel,
)
from nca_wm.rule_attn_model import RuleAttnNCAWorldModel


# ---------------------------------------------------------------------------
# Payload <-> training tensors
# ---------------------------------------------------------------------------

def _materialize_multihot(payload: dict, n_objs: int, stride_obj: int,
                          width: int, height: int) -> dict:
    """Cache (states, next_states) as (N, C, H, W) uint8 inside the payload."""
    if "mh_states" in payload:
        return payload
    mh_s = np.stack([
        _bitpacked_to_multihot(s, width, height, n_objs, stride_obj)
        for s in payload["states"]
    ]).astype(np.uint8) if payload["states"] else np.zeros(
        (0, n_objs, height, width), dtype=np.uint8
    )
    mh_n = np.stack([
        _bitpacked_to_multihot(s, width, height, n_objs, stride_obj)
        for s in payload["next_states"]
    ]).astype(np.uint8) if payload["next_states"] else np.zeros(
        (0, n_objs, height, width), dtype=np.uint8
    )
    payload["mh_states"] = mh_s
    payload["mh_next_states"] = mh_n
    payload["np_actions"] = np.array(payload["actions"], dtype=np.int32)
    payload["np_wons"] = np.array(payload["wons"], dtype=np.uint8)
    return payload


def _build_dataset_arrays(payloads: list[dict]) -> dict:
    """Concat (states, actions, next_states, wons) across the level pool."""
    if not payloads:
        return None
    states = np.concatenate([p["mh_states"] for p in payloads])
    next_states = np.concatenate([p["mh_next_states"] for p in payloads])
    actions = np.concatenate([p["np_actions"] for p in payloads])
    wons = np.concatenate([p["np_wons"] for p in payloads])
    return {
        "states": states, "next_states": next_states,
        "actions": actions, "wons": wons,
    }


# ---------------------------------------------------------------------------
# Per-level scoring
# ---------------------------------------------------------------------------

def _make_score_forward(model):
    """JIT'd per-level scoring with model AND identity-baseline per-cell errors.

    Returns (change_err, total_err, identity_total_err, regret) where:
      - change_err = fraction of changed cells the model gets wrong
      - total_err  = fraction of all cells the model gets wrong
      - identity_total_err = fraction of all cells where state != next_state
        (i.e., the per-cell error of a "predict no change" baseline)
      - regret = total_err − identity_total_err (positive = model is worse
        than the trivial copy-forward baseline → strong "needs more data" signal)
    """
    @jax.jit
    def _fwd(wm_params, states, action_onehots, next_states, game_tokens, game_masks):
        logits, _, _ = model.apply(
            wm_params, states, action_onehots, game_tokens, game_masks,
        )
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        wrong = (preds != next_states).astype(jnp.float32)
        total_err = wrong.mean()
        changed = (states != next_states).astype(jnp.float32)
        n_changed = changed.sum()
        change_wrong = (wrong * changed).sum()
        change_err = jnp.where(n_changed > 0, change_wrong / n_changed, 0.0)
        identity_err = changed.mean()
        regret = total_err - identity_err
        return change_err, total_err, identity_err, regret

    def score(params, states, action_onehots, next_states, game_tokens, game_masks):
        return _fwd(_wm_p(params), states, action_onehots,
                     next_states, game_tokens, game_masks)
    return score


def _score_level(score_fn, params, payload: dict,
                  tokens: np.ndarray, mask: np.ndarray) -> dict:
    """Returns dict {change_err, total_err, identity_err, regret}."""
    states = jnp.asarray(payload["mh_states"], dtype=jnp.float32)
    next_states = jnp.asarray(payload["mh_next_states"], dtype=jnp.float32)
    actions_oh = jnp.asarray(
        np.eye(N_ACTIONS, dtype=np.float32)[payload["np_actions"]],
    )
    n = states.shape[0]
    tokens_b = jnp.broadcast_to(tokens[None], (n, tokens.shape[0]))
    mask_b = jnp.broadcast_to(mask[None], (n, mask.shape[0]))
    change_err, total_err, identity_err, regret = score_fn(
        params, states, actions_oh, next_states, tokens_b, mask_b,
    )
    return {
        "change_err": float(change_err),
        "total_err": float(total_err),
        "identity_err": float(identity_err),
        "regret": float(regret),
    }


# ---------------------------------------------------------------------------
# Initial pool generation (rejection-sampling)
# ---------------------------------------------------------------------------

def _generate_initial_pool(
    gen: LevelGenerator,
    engine: Engine,
    width: int,
    height: int,
    n_target: int,
    rng: np.random.Generator,
    *,
    require_solvable: bool,
    min_states: int,
    max_iters_search: int,
    timeout_ms_search: int,
    max_attempts: int,
) -> tuple[list[list[int]], list[dict]]:
    accepted_dats: list[list[int]] = []
    accepted_payloads: list[dict] = []
    seen: set[tuple] = set()
    n_attempts = 0
    while len(accepted_dats) < n_target and n_attempts < max_attempts:
        n_attempts += 1
        d = gen.random_dat(rng, width, height)
        if d is None or not gen.structurally_valid(d, width, height):
            continue
        key = tuple(d)
        if key in seen:
            continue
        result = search_validate(
            engine, d, width, height,
            max_iters=max_iters_search,
            timeout_ms=timeout_ms_search,
            min_states=min_states,
            require_solvable=require_solvable,
        )
        if result is None:
            continue
        seen.add(key)
        accepted_dats.append(d)
        accepted_payloads.append(result)
    return accepted_dats, accepted_payloads


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default="sokoban_basic")
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--height", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)

    # Pool
    ap.add_argument("--init_pop_size", type=int, default=16,
                    help="Number of levels in the initial rejection-sampled pool.")
    ap.add_argument("--pop_size", type=int, default=16,
                    help="Population size kept across generations (top-K by model error).")
    ap.add_argument("--n_generations", type=int, default=5)
    ap.add_argument("--n_children_per_gen", type=int, default=16)
    ap.add_argument("--n_mutations_min", type=int, default=1)
    ap.add_argument("--n_mutations_max", type=int, default=3)
    ap.add_argument("--selection_metric", default="change_err",
                    choices=["change_err", "total_err", "regret"],
                    help="Per-level fitness used for parent-selection and pool-keep. "
                         "'change_err' = wrong-changed-cell fraction (default, used so far). "
                         "'total_err' = wrong-cell fraction over all cells. "
                         "'regret' = total_err - identity_err (mines levels where the model "
                         "loses to the copy-forward baseline, which filters out hard-because-"
                         "underexposed levels and focuses on hard-because-rule-rare levels).")
    ap.add_argument("--replay_buffer", action="store_true",
                    help="Replay-buffer mode: keep every accepted level forever instead of "
                         "pool-replacement. Combined with --replay_sample_weight, lets hard "
                         "levels get more gradient without sacrificing i.i.d. coverage.")
    ap.add_argument("--replay_sample_weight", default="uniform",
                    choices=["uniform", "softmax", "linear"],
                    help="When --replay_buffer is on: how to weight per-level sampling. "
                         "'uniform' = i.i.d. across the whole buffer (just adds data). "
                         "'softmax' = exp(error / temp) across levels. "
                         "'linear' = error + epsilon, normalized across levels.")
    ap.add_argument("--replay_softmax_temp", type=float, default=0.05,
                    help="Temperature for softmax weighting (smaller = more concentrated on hard).")
    ap.add_argument("--replay_buffer_cap", type=int, default=4096,
                    help="Hard cap on replay buffer size (drops oldest if exceeded).")
    ap.add_argument("--max_attempts_init", type=int, default=10000,
                    help="Per-pool attempt cap. With require_solvable=True at 7x7 "
                         "sokoban, acceptance is ~0.3%%, so plan ~600 attempts/level.")
    ap.add_argument("--max_attempts_per_gen_factor", type=float, default=10.0,
                    help="Per-gen child-validity attempts cap = factor * n_children_per_gen.")
    ap.add_argument("--parent_rank_temperature", type=float, default=0.1,
                    help="Lower = more concentrated on the highest-error parent (~exp(-rank*T)).")

    # Training
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--warmup_steps", type=int, default=2000)
    ap.add_argument("--steps_per_generation", type=int, default=2000)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--change_loss_weight", type=float, default=5.0)
    ap.add_argument("--win_loss_weight", type=float, default=1.0)
    ap.add_argument("--win_pos_weight", type=float, default=1.0)
    ap.add_argument("--log_interval", type=int, default=500)

    # Model
    ap.add_argument("--architecture", default="rule_attn",
                    choices=["rule_attn", "film"])
    ap.add_argument("--n_hid", type=int, default=128)
    ap.add_argument("--n_nca_steps", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_enc_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_slots", type=int, default=16)
    ap.add_argument("--d_slot", type=int, default=64)
    ap.add_argument("--n_app_slots", type=int, default=1)
    ap.add_argument("--d_z", type=int, default=64,
                    help="Latent dim for FiLM encoder (only used with --architecture film)")
    ap.add_argument("--axis_pool", action="store_true", default=True)
    ap.add_argument("--no-axis_pool", dest="axis_pool", action="store_false")
    ap.add_argument("--axis_cummax", action="store_true", default=True)
    ap.add_argument("--no-axis_cummax", dest="axis_cummax", action="store_false")
    ap.add_argument("--global_pool", action="store_true", default=True)
    ap.add_argument("--no-global_pool", dest="global_pool", action="store_false")

    # Synth/validity
    ap.add_argument("--require_solvable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--search_max_iters", type=int, default=5000)
    ap.add_argument("--search_timeout_ms", type=int, default=2000)
    ap.add_argument("--min_states", type=int, default=20)
    ap.add_argument("--synthetic_mode", default="tile_pattern_empirical",
                    choices=["tile_pattern_empirical", "tile_pattern_uniform"])
    ap.add_argument("--init_mode", default="rejection",
                    choices=["rejection", "evolve"],
                    help="How to seed the initial pool. 'rejection' samples and "
                         "filters; 'evolve' uses a GA with BFS-iterations fitness "
                         "(use this when rejection-sampling acceptance is near-zero).")
    ap.add_argument("--init_evolve_pop_size", type=int, default=64)
    ap.add_argument("--init_evolve_max_generations", type=int, default=200)
    ap.add_argument("--init_evolve_n_mutations_min", type=int, default=1)
    ap.add_argument("--init_evolve_n_mutations_max", type=int, default=3)

    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()

    if args.save_dir is None:
        args.save_dir = (
            f"nca_wm/logs/curr_{args.game}_w{args.width}h{args.height}_"
            f"pop{args.pop_size}_gens{args.n_generations}_"
            f"warm{args.warmup_steps}_step{args.steps_per_generation}_"
            f"hid{args.n_hid}_seed{args.seed}"
        )
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[curriculum] save_dir = {args.save_dir}")

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Compile game + setup level-gen primitives
    # ------------------------------------------------------------------
    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(parser, args.game)
    json_state = json.loads(json_str)
    gen = LevelGenerator(json_state, mode=args.synthetic_mode)
    mutator = LevelMutator(
        json_state, width=args.width, height=args.height,
        allowed_tile_patterns=gen.patterns,
    )
    n_objs = gen.n_objs
    stride_obj = gen.stride

    engine = Engine()
    engine.load_from_json(json_str)
    engine.load_level(0)

    # ------------------------------------------------------------------
    # Tokenize game spec for the conditional model
    # ------------------------------------------------------------------
    tree, canonical_ids = get_game_tree_from_js(parser, args.game)
    token_ids = tokenize_game(
        tree, canonical_ids, encode_sprites=False,
    )
    max_tok_len = max(len(token_ids), 1)
    tokens_arr = np.zeros(max_tok_len, dtype=np.int32)
    tokens_arr[:len(token_ids)] = token_ids
    mask_arr = np.zeros(max_tok_len, dtype=np.bool_)
    mask_arr[:len(token_ids)] = True

    # ------------------------------------------------------------------
    # Initial pool + materialize multihot tensors
    # ------------------------------------------------------------------
    print(f"[curriculum] Generating initial pool of {args.init_pop_size} levels "
          f"(mode={args.init_mode})...")
    t0 = time.time()
    if args.init_mode == "evolve":
        accepted_dats, accepted_payloads = _evolve_levels(
            engine, json_state, gen, args.width, args.height,
            args.init_pop_size,
            rng=rng,
            require_solvable=args.require_solvable,
            min_states=args.min_states,
            max_iters_search=args.search_max_iters,
            timeout_ms_search=args.search_timeout_ms,
            pop_size=args.init_evolve_pop_size,
            n_mutations_min=args.init_evolve_n_mutations_min,
            n_mutations_max=args.init_evolve_n_mutations_max,
            max_generations=args.init_evolve_max_generations,
            verbose=True,
        )
    else:
        accepted_dats, accepted_payloads = _generate_initial_pool(
            gen, engine, args.width, args.height, args.init_pop_size, rng,
            require_solvable=args.require_solvable,
            min_states=args.min_states,
            max_iters_search=args.search_max_iters,
            timeout_ms_search=args.search_timeout_ms,
            max_attempts=args.max_attempts_init,
        )
    for p in accepted_payloads:
        _materialize_multihot(p, n_objs, stride_obj, args.width, args.height)
    print(f"[curriculum] Pool: {len(accepted_dats)}/{args.init_pop_size} levels "
          f"in {time.time()-t0:.1f}s")
    if not accepted_payloads:
        raise RuntimeError("No valid levels generated; cannot proceed.")

    # ------------------------------------------------------------------
    # Build model + optimizer
    # ------------------------------------------------------------------
    vocab_size = VOCAB_SIZE_EXT_V2  # kernel_sep=True
    if args.architecture == "rule_attn":
        model = RuleAttnNCAWorldModel(
            n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=n_objs,
            vocab_size=vocab_size + 1,
            enc_d_model=args.d_model, enc_n_self_layers=args.n_enc_layers,
            n_slots=args.n_slots, n_app_slots=args.n_app_slots,
            d_slot=args.d_slot,
            n_attn_heads=args.n_heads,
            max_seq_len=max_tok_len + 1,
            axis_pool=args.axis_pool,
            axis_cummax=args.axis_cummax,
            global_pool=args.global_pool,
            n_repeats=getattr(args, "n_nca_repeats", 1),
        )
    else:
        model = ConditionalNCAWorldModel(
            n_hid=args.n_hid, n_steps=args.n_nca_steps, n_out=n_objs,
            vocab_size=vocab_size + 1,
            d_model=args.d_model, n_heads=args.n_heads,
            n_enc_layers=args.n_enc_layers, d_z=args.d_z,
            max_seq_len=max_tok_len + 1,
            sprite_decoder=False,
            axis_pool=args.axis_pool,
            axis_cummax=args.axis_cummax,
            global_pool=args.global_pool,
        )

    rng_jax = jax.random.PRNGKey(args.seed)
    dummy_state = jnp.zeros((1, n_objs, args.height, args.width), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, N_ACTIONS), dtype=jnp.float32)
    dummy_tokens = jnp.array(tokens_arr[None])
    dummy_mask = jnp.array(mask_arr[None])
    init_params = model.init(
        rng_jax, dummy_state, dummy_action, dummy_tokens, dummy_mask,
    )
    n_model_params = sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(init_params))
    print(f"[curriculum] Model: {type(model).__name__}, params={n_model_params:,}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(args.lr),
    )
    opt_state = optimizer.init(init_params)

    train_step = make_train_step(
        model, optimizer, conditional=True,
        win_loss_weight=args.win_loss_weight,
        win_pos_weight=args.win_pos_weight,
        change_loss_weight=args.change_loss_weight,
    )
    eval_forward = make_eval_forward(model, conditional=True)
    score_fn = _make_score_forward(model)
    params = init_params

    # ------------------------------------------------------------------
    # Build initial dataset + sampler
    # ------------------------------------------------------------------
    dataset = _build_dataset_arrays(accepted_payloads)
    # Per-transition level-id (which payload it came from) for replay sampling.
    transition_level_ids = np.concatenate([
        np.full(len(p["np_actions"]), i, dtype=np.int32)
        for i, p in enumerate(accepted_payloads)
    ]) if accepted_payloads else np.empty((0,), dtype=np.int32)
    # Per-level sampling weight (uniform initially; updated each gen).
    level_sample_probs = np.full(len(accepted_payloads),
                                  1.0 / max(len(accepted_payloads), 1),
                                  dtype=np.float64)

    def _build_per_transition_probs():
        """Per-transition prob = level_prob / n_transitions_in_level."""
        if not accepted_payloads:
            return np.array([], dtype=np.float64)
        per_level_n = np.array([len(p["np_actions"]) for p in accepted_payloads],
                                dtype=np.float64)
        per_level_n[per_level_n == 0] = 1.0
        per_tx_level_prob = level_sample_probs / per_level_n
        return per_tx_level_prob[transition_level_ids]

    per_tx_probs = _build_per_transition_probs()

    def sample_batch(ds: dict):
        n = len(ds["actions"])
        if args.replay_buffer and args.replay_sample_weight != "uniform" and per_tx_probs.size == n:
            idx = rng.choice(n, size=args.batch_size, p=per_tx_probs, replace=True)
        else:
            idx = rng.integers(0, n, size=args.batch_size)
        states = jnp.asarray(ds["states"][idx], dtype=jnp.float32)
        actions_oh = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[ds["actions"][idx]])
        next_states = jnp.asarray(ds["next_states"][idx], dtype=jnp.float32)
        wons = jnp.asarray(ds["wons"][idx], dtype=jnp.float32)
        tokens_b = jnp.broadcast_to(tokens_arr[None], (args.batch_size, max_tok_len))
        mask_b = jnp.broadcast_to(mask_arr[None], (args.batch_size, max_tok_len))
        return states, actions_oh, next_states, wons, tokens_b, mask_b

    print(f"[curriculum] Initial dataset: {len(dataset['actions']):,} transitions, "
          f"{int(dataset['wons'].sum()):,} winning")

    # ------------------------------------------------------------------
    # Helper: train for K steps, returning final loss
    # ------------------------------------------------------------------
    def train_n(params, opt_state, ds, n_steps: int, label: str):
        t = time.time()
        last_loss = float("nan")
        for step in range(n_steps):
            batch = sample_batch(ds)
            params, opt_state, loss, *_aux = train_step(params, opt_state, *batch)
            if (step + 1) % args.log_interval == 0:
                last_loss = float(loss)
                print(
                    f"  [{label}] step {step+1}/{n_steps} "
                    f"loss={last_loss:.4e} ({time.time()-t:.1f}s)"
                )
        last_loss = float(loss)
        return params, opt_state, last_loss

    # ------------------------------------------------------------------
    # Persistent run-config so heldout_eval can rebuild the model
    # ------------------------------------------------------------------
    cfg_for_eval = {
        # used by heldout_eval._build_model + _load_run + analysis
        "architecture": args.architecture,
        "n_hid": args.n_hid,
        "n_nca_steps": args.n_nca_steps,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_enc_layers": args.n_enc_layers,
        "d_z": args.d_z,
        "n_slots": args.n_slots,
        "n_app_slots": args.n_app_slots,
        "d_slot": args.d_slot,
        "axis_pool": args.axis_pool,
        "axis_cummax": args.axis_cummax,
        "global_pool": args.global_pool,
        "conditional": True,
        "encode_sprites": False,
        "kernel_sep": True,
        "sprite_loss_weight": 0.0,
        # curriculum-specific bookkeeping
        "curriculum": True,
        "synthetic_levels": args.pop_size,
        "synthetic_w": args.width,
        "synthetic_h": args.height,
        "synthetic_seed": args.seed,
        "synthetic_mode": args.synthetic_mode,
        "synthetic_require_solvable": args.require_solvable,
        "synthetic_min_states": args.min_states,
        "n_generations": args.n_generations,
        "warmup_steps": args.warmup_steps,
        "steps_per_generation": args.steps_per_generation,
        "n_children_per_gen": args.n_children_per_gen,
        "selection_metric": args.selection_metric,
        "replay_buffer": args.replay_buffer,
        "replay_sample_weight": args.replay_sample_weight,
        "replay_softmax_temp": args.replay_softmax_temp,
        "replay_buffer_cap": args.replay_buffer_cap,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "change_loss_weight": args.change_loss_weight,
        "win_loss_weight": args.win_loss_weight,
        "win_pos_weight": args.win_pos_weight,
        "seed": args.seed,
        "game": args.game,
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(cfg_for_eval, f, indent=2)

    # game_infos.pkl for heldout_eval (single-game, native shape)
    sprite_tensor = None
    try:
        sprite_tensor = _build_sprite_tensor(tree, canonical_ids)
    except Exception:
        pass
    try:
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
        n_authored_levels = int(env0.num_levels)
    except Exception:
        n_authored_levels = 1
    game_info = {
        "name": args.game,
        "json_str": json_str,
        "n_objs": n_objs,
        "H": args.height,
        "W": args.width,
        "n_levels": n_authored_levels,
        "token_ids": list(token_ids),
        "sprite_tensor": sprite_tensor,
    }
    with open(os.path.join(args.save_dir, "game_infos.pkl"), "wb") as f:
        pickle.dump([game_info], f)

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------
    print(f"\n[curriculum] Warmup: {args.warmup_steps} training steps on "
          f"{len(accepted_payloads)} initial levels...")
    params, opt_state, warmup_loss = train_n(
        params, opt_state, dataset, args.warmup_steps, "warmup",
    )
    # Save params after warmup so they're recoverable
    with open(os.path.join(args.save_dir, "params.pkl"), "wb") as f:
        pickle.dump(jax.device_get(params), f)

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    history: list[dict] = []
    history.append({
        "phase": "warmup",
        "n_steps": args.warmup_steps,
        "loss": warmup_loss,
        "n_levels": len(accepted_payloads),
        "n_transitions": int(len(dataset["actions"])),
    })

    for gen_idx in range(args.n_generations):
        t_gen = time.time()

        # 1. Score current population
        pop_metrics = [
            _score_level(score_fn, params, p, tokens_arr, mask_arr)
            for p in accepted_payloads
        ]
        pop_change_errs = np.array([m["change_err"] for m in pop_metrics])
        pop_total_errs = np.array([m["total_err"] for m in pop_metrics])
        pop_regrets = np.array([m["regret"] for m in pop_metrics])
        # Selection metric for ranking (higher = harder = mine)
        if args.selection_metric == "change_err":
            pop_errors = pop_change_errs
        elif args.selection_metric == "total_err":
            pop_errors = pop_total_errs
        elif args.selection_metric == "regret":
            pop_errors = pop_regrets
        else:
            pop_errors = pop_change_errs

        # 2. Mutate to produce children, biased toward high-error parents
        ranks = np.argsort(np.argsort(-pop_errors))  # rank 0 = highest error
        weights = np.exp(-ranks * args.parent_rank_temperature)
        weights /= weights.sum()

        children_dats: list[list[int]] = []
        children_payloads: list[dict] = []
        seen_local = set(tuple(d) for d in accepted_dats)
        max_attempts_gen = int(args.n_children_per_gen * args.max_attempts_per_gen_factor)
        n_attempts_gen = 0
        n_struct_rej = 0
        n_search_rej = 0
        while (len(children_dats) < args.n_children_per_gen
               and n_attempts_gen < max_attempts_gen):
            n_attempts_gen += 1
            parent_idx = int(rng.choice(len(accepted_dats), p=weights))
            parent = accepted_dats[parent_idx]
            n_muts = int(rng.integers(args.n_mutations_min, args.n_mutations_max + 1))
            child = mutator.mutate(
                list(parent), rng, n_mutations=n_muts,
                required_player_count=1,
            )
            key = tuple(child)
            if key in seen_local:
                continue
            # LevelMutator already enforces player count; we still want the
            # full structural+search check before accepting.
            if not gen.structurally_valid(child, args.width, args.height):
                n_struct_rej += 1
                continue
            result = search_validate(
                engine, child, args.width, args.height,
                max_iters=args.search_max_iters,
                timeout_ms=args.search_timeout_ms,
                min_states=args.min_states,
                require_solvable=args.require_solvable,
            )
            if result is None:
                n_search_rej += 1
                continue
            seen_local.add(key)
            _materialize_multihot(result, n_objs, stride_obj, args.width, args.height)
            children_dats.append(child)
            children_payloads.append(result)

        # 3. Score children
        children_metrics = [
            _score_level(score_fn, params, p, tokens_arr, mask_arr)
            for p in children_payloads
        ]
        if args.selection_metric == "change_err":
            children_errors = [m["change_err"] for m in children_metrics]
        elif args.selection_metric == "total_err":
            children_errors = [m["total_err"] for m in children_metrics]
        elif args.selection_metric == "regret":
            children_errors = [m["regret"] for m in children_metrics]
        else:
            children_errors = [m["change_err"] for m in children_metrics]
        children_errors_arr = np.array(children_errors) if children_errors else np.array([])

        # 4. Combine + select.
        all_dats = accepted_dats + children_dats
        all_payloads = accepted_payloads + children_payloads
        all_errors = np.concatenate([pop_errors, children_errors_arr]) if children_errors_arr.size \
                     else pop_errors

        if args.replay_buffer:
            # Keep everything (subject to buffer cap, drop-oldest-first).
            if len(all_dats) > args.replay_buffer_cap:
                drop_n = len(all_dats) - args.replay_buffer_cap
                all_dats = all_dats[drop_n:]
                all_payloads = all_payloads[drop_n:]
                all_errors = all_errors[drop_n:]
            accepted_dats = all_dats
            accepted_payloads = all_payloads
            kept_errors = all_errors
        else:
            order = np.argsort(-all_errors)[:args.pop_size]
            kept_errors = all_errors[order]
            accepted_dats = [all_dats[i] for i in order]
            accepted_payloads = [all_payloads[i] for i in order]

        # Update sampling weights (used only in replay mode w/ non-uniform weight).
        if args.replay_buffer and args.replay_sample_weight != "uniform" and len(kept_errors) > 0:
            errs = np.asarray(kept_errors, dtype=np.float64)
            if args.replay_sample_weight == "softmax":
                w = np.exp(errs / max(args.replay_softmax_temp, 1e-8))
            else:  # linear
                w = errs - errs.min() + 1e-3
            w = np.clip(w, 1e-12, None)
            level_sample_probs = w / w.sum()
        else:
            level_sample_probs = np.full(len(accepted_payloads),
                                          1.0 / max(len(accepted_payloads), 1),
                                          dtype=np.float64)

        # Rebuild flat dataset + per-transition probs over the (possibly grown) buffer.
        dataset = _build_dataset_arrays(accepted_payloads)
        transition_level_ids = np.concatenate([
            np.full(len(p["np_actions"]), i, dtype=np.int32)
            for i, p in enumerate(accepted_payloads)
        ]) if accepted_payloads else np.empty((0,), dtype=np.int32)
        per_tx_probs = _build_per_transition_probs()

        # 5. Train on refreshed pool
        params, opt_state, loss_after = train_n(
            params, opt_state, dataset, args.steps_per_generation,
            f"gen{gen_idx}",
        )

        gen_time = time.time() - t_gen
        gen_stats = {
            "phase": "generation",
            "gen": gen_idx,
            "selection_metric": args.selection_metric,
            "n_attempts": n_attempts_gen,
            "n_struct_rej": n_struct_rej,
            "n_search_rej": n_search_rej,
            "n_children_accepted": len(children_dats),
            "pop_err_mean": float(pop_errors.mean()),
            "pop_err_max": float(pop_errors.max()),
            "pop_err_min": float(pop_errors.min()),
            "pop_change_err_mean": float(pop_change_errs.mean()),
            "pop_total_err_mean": float(pop_total_errs.mean()),
            "pop_regret_mean": float(pop_regrets.mean()),
            "pop_regret_max": float(pop_regrets.max()),
            "children_err_mean": (float(children_errors_arr.mean())
                                   if children_errors_arr.size else None),
            "children_err_max": (float(children_errors_arr.max())
                                  if children_errors_arr.size else None),
            "kept_err_mean": float(kept_errors.mean()),
            "kept_err_max": float(kept_errors.max()),
            "n_transitions": int(len(dataset["actions"])),
            "loss_after": loss_after,
            "time_s": gen_time,
        }
        history.append(gen_stats)
        ce_max_str = (f"{gen_stats['children_err_max']:.4f}"
                      if gen_stats["children_err_max"] is not None else "n/a")
        print(
            f"\n[gen {gen_idx}/{args.n_generations}] "
            f"children={len(children_dats)}/{args.n_children_per_gen} "
            f"(struct-rej={n_struct_rej}, search-rej={n_search_rej})  "
            f"pop_err mean={gen_stats['pop_err_mean']:.4f} "
            f"max={gen_stats['pop_err_max']:.4f}  "
            f"children_err max={ce_max_str}  "
            f"kept_err mean={gen_stats['kept_err_mean']:.4f} "
            f"max={gen_stats['kept_err_max']:.4f}  "
            f"loss_after={loss_after:.4e}  "
            f"trans={gen_stats['n_transitions']}  "
            f"time={gen_time:.1f}s"
        )

        # Save checkpoint after every generation
        with open(os.path.join(args.save_dir, "params.pkl"), "wb") as f:
            pickle.dump(jax.device_get(params), f)
        with open(os.path.join(args.save_dir, "curriculum_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # Save level pool dats as a final artifact
    pool_path = os.path.join(args.save_dir, "final_pool.npz")
    np.savez_compressed(
        pool_path,
        level_dats=np.array(accepted_dats, dtype=np.int64),
        kept_errors=np.array(kept_errors, dtype=np.float32),
    )
    print(f"\n[curriculum] DONE. params + history + pool saved to {args.save_dir}")


if __name__ == "__main__":
    main()

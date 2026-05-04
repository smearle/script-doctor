"""Score one or more trained checkpoints on a fresh held-out synthetic level pool.

Avoids the size-mismatch issue when trained-on synthetic_w/h is smaller than the
game's authored levels. Generates a held-out pool with a separate seed using
the same generator as ``nca_wm.synthetic_levels`` and reports per-checkpoint
mean change-error.

Usage:
    python -m nca_wm.scripts.eval_on_synth_holdout \\
        --game Travelling_salesman --width 8 --height 8 \\
        --n_levels 32 --holdout_seed 7 --mode evolve --min_states 5 \\
        --checkpoints curr=nca_wm/logs/curr_TSP_w8h8_pop16_gens5_hid128_seed0 \\
                       static=nca_wm/logs/synth_static_TSP_n16_hid128_seed0
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp._puzzlescript_cpp import Engine
from puzzlescript_cpp import CppPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser

from nca_wm.synthetic_levels import (
    LevelGenerator as _LG,
    _evolve_levels as _EV,
    search_validate as _SV,
    _bitpacked_to_multihot as _MH,
)
from nca_wm.tokenize_game import (
    tokenize_game, get_game_tree_from_js,
    VOCAB_SIZE_EXT_V2,
)
from nca_wm.train import N_ACTIONS, make_eval_forward
from nca_wm.heldout_eval import _build_model


def _generate_holdout_pool(
    json_state, gen, engine, width, height, n_target, rng, *,
    mode, require_solvable, min_states, max_iters_search, timeout_ms_search,
    evolve_pop_size, evolve_max_generations, max_attempts,
):
    if mode == "evolve":
        return _EV(
            engine, json_state, gen, width, height, n_target,
            rng=rng,
            require_solvable=require_solvable,
            min_states=min_states,
            max_iters_search=max_iters_search,
            timeout_ms_search=timeout_ms_search,
            pop_size=evolve_pop_size,
            n_mutations_min=1, n_mutations_max=3,
            max_generations=evolve_max_generations,
            verbose=False,
        )
    # rejection
    accepted_dats, accepted_payloads = [], []
    seen = set()
    n_attempts = 0
    while len(accepted_dats) < n_target and n_attempts < max_attempts:
        n_attempts += 1
        d = gen.random_dat(rng, width, height)
        if d is None or not gen.structurally_valid(d, width, height):
            continue
        key = tuple(d)
        if key in seen:
            continue
        result = _SV(
            engine, d, width, height,
            max_iters=max_iters_search, timeout_ms=timeout_ms_search,
            min_states=min_states, require_solvable=require_solvable,
        )
        if result is None:
            continue
        seen.add(key)
        accepted_dats.append(d)
        accepted_payloads.append(result)
    return accepted_dats, accepted_payloads


def _materialize(payload, n_objs, stride_obj, w, h):
    if "mh_states" in payload:
        return payload
    payload["mh_states"] = (
        np.stack([_MH(s, w, h, n_objs, stride_obj) for s in payload["states"]]).astype(np.uint8)
        if payload["states"]
        else np.zeros((0, n_objs, h, w), dtype=np.uint8)
    )
    payload["mh_next_states"] = (
        np.stack([_MH(s, w, h, n_objs, stride_obj) for s in payload["next_states"]]).astype(np.uint8)
        if payload["next_states"]
        else np.zeros((0, n_objs, h, w), dtype=np.uint8)
    )
    payload["np_actions"] = np.array(payload["actions"], dtype=np.int32)
    return payload


def _load_run_for_game(save_dir: str):
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(save_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    pp = os.path.join(save_dir, "params_best.pkl")
    if not os.path.isfile(pp):
        pp = os.path.join(save_dir, "params.pkl")
    with open(pp, "rb") as f:
        params = pickle.load(f)
    # Override cfg["vocab_size"] from the actual checkpoint params so older
    # runs (where vocab_size wasn't recorded but the embedding was sized to
    # max-used-token-id) load with the right embedding shape. The minus-1 is
    # because we pass vocab_size+1 (for the CLS token) when building.
    try:
        leaves = jax.tree_util.tree_leaves_with_path(params)
        for k, v in leaves:
            ks = jax.tree_util.keystr(k)
            if "tok_embed" in ks and "embedding" in ks:
                cfg["vocab_size"] = int(v.shape[0]) - 1
                break
    except Exception:
        pass
    return cfg, params, game_infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--n_levels", type=int, default=32)
    ap.add_argument("--holdout_seed", type=int, default=42)
    ap.add_argument("--mode", default="evolve",
                    choices=["evolve", "tile_pattern_empirical", "tile_pattern_uniform"])
    ap.add_argument("--require_solvable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min_states", type=int, default=5)
    ap.add_argument("--max_iters_search", type=int, default=5000)
    ap.add_argument("--timeout_ms_search", type=int, default=2000)
    ap.add_argument("--evolve_pop_size", type=int, default=64)
    ap.add_argument("--evolve_max_generations", type=int, default=200)
    ap.add_argument("--max_attempts", type=int, default=5000)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="List of label=path entries.")
    args = ap.parse_args()

    # ----- compile + level-gen primitives -----
    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(parser, args.game)
    json_state = json.loads(json_str)
    construction_mode = ("tile_pattern_empirical" if args.mode == "evolve"
                         else args.mode)
    gen = _LG(json_state, mode=construction_mode)
    n_objs = gen.n_objs
    stride_obj = gen.stride
    engine = Engine()
    engine.load_from_json(json_str)
    engine.load_level(0)

    # ----- holdout pool -----
    rng = np.random.default_rng(args.holdout_seed)
    print(f"Generating holdout pool: {args.n_levels} levels, mode={args.mode}, "
          f"seed={args.holdout_seed}")
    t0 = time.time()
    dats, payloads = _generate_holdout_pool(
        json_state, gen, engine, args.width, args.height, args.n_levels, rng,
        mode=args.mode, require_solvable=args.require_solvable,
        min_states=args.min_states,
        max_iters_search=args.max_iters_search,
        timeout_ms_search=args.timeout_ms_search,
        evolve_pop_size=args.evolve_pop_size,
        evolve_max_generations=args.evolve_max_generations,
        max_attempts=args.max_attempts,
    )
    for p in payloads:
        _materialize(p, n_objs, stride_obj, args.width, args.height)
    print(f"  holdout: {len(payloads)} levels, "
          f"{sum(len(p['actions']) for p in payloads):,} transitions, "
          f"{time.time()-t0:.1f}s")
    if not payloads:
        print("No holdout levels generated; aborting.")
        sys.exit(1)

    # ----- per-checkpoint scoring -----
    # Tokenize game spec once (assume all checkpoints used the same vocab/kernel_sep)
    tree, canonical_ids = get_game_tree_from_js(parser, args.game)

    results = {}
    for spec in args.checkpoints:
        if "=" not in spec:
            label, path = spec, spec
        else:
            label, path = spec.split("=", 1)
        print(f"\n=== checkpoint: {label}  ({path}) ===")
        cfg, params, game_infos = _load_run_for_game(path)

        # Tokenize using the checkpoint's encode_sprites; kernel_sep is implicit/always-on now.
        encode_sprites = cfg.get("encode_sprites", False)
        token_ids = tokenize_game(tree, canonical_ids,
                                   encode_sprites=encode_sprites)
        # Pad tokens to whatever max_seq_len the checkpoint expected.
        train_max_seq_len = max(len(g.get("token_ids", [])) for g in game_infos)
        train_max_seq_len = max(train_max_seq_len, 1)
        max_tok_len = train_max_seq_len  # heldout_eval uses train_max_seq_len, +1 for CLS in pos-embed
        if len(token_ids) > max_tok_len:
            token_ids = token_ids[:max_tok_len]
        tokens = np.zeros(max_tok_len, dtype=np.int32)
        tokens[:len(token_ids)] = token_ids
        mask = np.zeros(max_tok_len, dtype=np.bool_)
        mask[:len(token_ids)] = True

        # Build model from cfg via heldout_eval._build_model (handles film vs rule_attn)
        model = _build_model(cfg, game_infos)
        eval_forward = make_eval_forward(model, conditional=True)

        # Score each holdout level. Batch transitions to bound peak GPU memory:
        # a 16K-transition level in one shot blows past 20 GB for the rule_attn
        # model. Use a fixed CHUNK so jit only compiles once per checkpoint.
        CHUNK = 256
        per_level = []
        for li, p in enumerate(payloads):
            states_full = np.asarray(p["mh_states"], dtype=np.float32)
            nexts_full = np.asarray(p["mh_next_states"], dtype=np.float32)
            actions_full = np.eye(N_ACTIONS, dtype=np.float32)[p["np_actions"]]
            n = states_full.shape[0]
            if n == 0:
                per_level.append({
                    "level": li, "n_transitions": 0,
                    "change_err": 0.0, "bce": 0.0, "acc": 0.0,
                })
                continue
            n_pad = (CHUNK - n % CHUNK) % CHUNK
            if n_pad:
                states_full = np.concatenate([states_full,
                    np.zeros((n_pad, *states_full.shape[1:]), states_full.dtype)])
                nexts_full = np.concatenate([nexts_full,
                    np.zeros((n_pad, *nexts_full.shape[1:]), nexts_full.dtype)])
                actions_full = np.concatenate([actions_full,
                    np.zeros((n_pad, N_ACTIONS), actions_full.dtype)])
            tokens_b = jnp.broadcast_to(jnp.asarray(tokens)[None], (CHUNK, len(tokens)))
            mask_b = jnp.broadcast_to(jnp.asarray(mask)[None], (CHUNK, len(mask)))
            bce_sum = 0.0
            acc_w = 0.0
            cacc_w = 0.0
            real = 0
            for s0 in range(0, states_full.shape[0], CHUNK):
                sl = slice(s0, s0 + CHUNK)
                s = jnp.asarray(states_full[sl])
                nx = jnp.asarray(nexts_full[sl])
                a = jnp.asarray(actions_full[sl])
                bce, acc, change_acc = eval_forward(
                    params, s, a, nx, tokens_b, mask_b,
                )
                # weight each chunk by # of real transitions in it
                lo, hi = s0, min(s0 + CHUNK, n)
                w = max(0, hi - lo)
                if w > 0:
                    bce_sum += float(bce) * w
                    acc_w += float(acc) * w
                    cacc_w += float(change_acc) * w
                    real += w
            ce = float(1.0 - cacc_w / max(real, 1))
            per_level.append({
                "level": li,
                "n_transitions": int(n),
                "change_err": ce,
                "bce": bce_sum / max(real, 1),
                "acc": acc_w / max(real, 1),
            })
        change_errs = np.array([r["change_err"] for r in per_level])
        print(f"  mean change_err: {change_errs.mean():.4f}")
        print(f"   max change_err: {change_errs.max():.4f}")
        print(f"   med change_err: {np.median(change_errs):.4f}")
        print(f"  n levels at 0%:  {int((change_errs < 1e-6).sum())}/{len(change_errs)}")
        results[label] = {
            "mean_change_err": float(change_errs.mean()),
            "max_change_err": float(change_errs.max()),
            "median_change_err": float(np.median(change_errs)),
            "n_perfect": int((change_errs < 1e-6).sum()),
            "per_level": per_level,
        }

    # ----- print summary -----
    print(f"\n=== Summary on holdout ({len(payloads)} levels, mode={args.mode}, "
          f"seed={args.holdout_seed}) ===")
    print(f"{'label':12s}  {'mean':>9s}  {'max':>9s}  {'median':>9s}  {'n_perfect':>10s}")
    for label, r in results.items():
        print(f"{label:12s}  "
              f"{100*r['mean_change_err']:>8.3f}%  "
              f"{100*r['max_change_err']:>8.3f}%  "
              f"{100*r['median_change_err']:>8.3f}%  "
              f"{r['n_perfect']:>5d}/{len(payloads):>4d}")


if __name__ == "__main__":
    main()

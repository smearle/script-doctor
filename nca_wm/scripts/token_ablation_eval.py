"""Token-ablation eval: does the encoder actually steer the NCA?

For each run dir, reload the trained model and run the same per-game eval
as `evaluate_multigame` but with the game tokens replaced by an ablated
version (zero / random / shuffle). Compare against the baseline (real
tokens).

If a model performs identically with ablated tokens, the encoder is being
ignored and the NCA is doing pure spatial-feature memorization.

Usage:
    python -m nca_wm.scripts.token_ablation_eval \\
        --runs nca_wm/logs/<dir1> nca_wm/logs/<dir2> ... \\
        --modes none zero shuffle \\
        --n_episodes 3 --max_steps 50 \\
        --out_json nca_wm/refine-logs/token_ablation_results.json
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nca_wm.train import (
    _run_eval_rollout, _pad_offsets, _pad_state_for_model,
    VOCAB_SIZE_BASE, VOCAB_SIZE_EXT, N_ACTIONS,
)
from nca_wm.tokenize_game import VOCAB_SIZE_BASE as _VBASE


def build_model(cfg, max_C, max_tok_len, vocab_size_override=None, max_seq_override=None):
    """Reconstruct the model class used by this run, matching train.py."""
    arch = cfg.get("architecture", "rule_attn")
    if vocab_size_override is not None:
        vocab_size = vocab_size_override - 1   # model takes vocab_size+1
    else:
        vocab_size = VOCAB_SIZE_EXT if cfg.get("encode_sprites", False) else VOCAB_SIZE_BASE
    if max_seq_override is not None:
        max_tok_len = max_seq_override - 1     # model takes max_tok_len+1
    pool_kwargs = {
        "axis_pool": cfg.get("axis_pool", False),
        "axis_cummax": cfg.get("axis_cummax", False),
        "global_pool": cfg.get("global_pool", False),
    }
    if arch == "rule_attn":
        from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
        return RuleAttnNCAWorldModel(
            n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
            vocab_size=vocab_size + 1,
            enc_d_model=cfg.get("d_model", 64),
            enc_n_self_layers=cfg.get("n_enc_layers", 2),
            n_slots=cfg.get("n_slots", 16),
            n_app_slots=cfg.get("n_app_slots", 1),
            d_slot=cfg.get("d_slot", 64),
            n_attn_heads=cfg.get("n_heads", 4),
            max_seq_len=max_tok_len + 1,
            use_vq=cfg.get("vq_codebook", False),
            vq_codebook_size=cfg.get("vq_codebook_size", 512),
            **pool_kwargs,
        )
    else:
        from nca_wm.train import ConditionalNCAWorldModel
        return ConditionalNCAWorldModel(
            n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=max_C,
            vocab_size=vocab_size + 1,
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            n_enc_layers=cfg.get("n_enc_layers", 2),
            d_z=cfg.get("d_z", 64),
            max_seq_len=max_tok_len + 1,
            sprite_decoder=cfg.get("sprite_loss_weight", 0.0) > 0,
            **pool_kwargs,
        )


def ablate_tokens(tokens: np.ndarray, mask: np.ndarray, mode: str,
                  rng: np.random.RandomState, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply ablation to a (S,) token sequence + (S,) mask.

    Modes:
      none     — pass through unchanged.
      zero     — set all tokens to 0 (id of the pad/null token).
      random   — replace masked (real) positions with uniform random ids.
      shuffle  — random permutation of masked positions in-place.
    """
    if mode == "none":
        return tokens, mask
    t = tokens.copy()
    if mode == "zero":
        t[:] = 0
        # keep mask so the encoder still attends over the same length
        return t, mask
    if mode == "random":
        valid = mask.astype(bool)
        t[valid] = rng.randint(1, vocab_size, size=int(valid.sum()), dtype=tokens.dtype)
        return t, mask
    if mode == "shuffle":
        valid = mask.astype(bool)
        idx = np.where(valid)[0]
        perm = rng.permutation(idx)
        t[idx] = tokens[perm]
        return t, mask
    raise ValueError(f"unknown ablation mode: {mode}")


def eval_one_run(run_dir: str, modes: list[str], n_episodes: int,
                 max_steps: int, n_levels_max: int, seed: int,
                 game_infos_from: str | None = None,
                 filter_games: list[str] | None = None) -> dict:
    """Reload a run and eval under each ablation mode."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    games_csv = cfg["games"] or cfg.get("game", "")
    games = [g.strip() for g in games_csv.split(",") if g.strip()]

    # Always load the run's own game_infos for model-dim sizing.
    with open(os.path.join(run_dir, "game_infos.pkl"), "rb") as f:
        train_game_infos = pickle.load(f)
    if game_infos_from:
        with open(os.path.join(game_infos_from, "game_infos.pkl"), "rb") as f:
            game_infos = pickle.load(f)
    else:
        game_infos = train_game_infos
    if filter_games:
        keep = set(filter_games)
        game_infos = [g for g in game_infos if g["name"] in keep]
        print(f"  filtered to {len(game_infos)} games: {[g['name'] for g in game_infos]}", flush=True)
    # Skip eval games whose n_objs exceeds the trained model's capacity.
    train_max_C = int(max(g["n_objs"] for g in train_game_infos))
    skipped = [g["name"] for g in game_infos if g["n_objs"] > train_max_C]
    if skipped:
        print(f"  skipping {len(skipped)} games exceeding trained max_C={train_max_C}: {skipped}", flush=True)
        game_infos = [g for g in game_infos if g["n_objs"] <= train_max_C]
    with open(os.path.join(run_dir, "params.pkl"), "rb") as f:
        loaded = pickle.load(f)
    # Two on-disk shapes:
    #   joint decoder: {"wm": {"params": {...}}, "dec": {"params": {...}}}
    #   non-joint:     {"params": {...}}
    # `model.apply` expects the FULL variables dict (with the outer "params" key),
    # so for joint we pull out loaded["wm"]; otherwise loaded already is what we want.
    if isinstance(loaded, dict) and "wm" in loaded and "dec" in loaded:
        params = loaded["wm"]
    else:
        params = loaded

    max_C = train_max_C
    max_tok_len = int(max(len(g.get("token_ids", [])) for g in train_game_infos))
    max_tok_len = max(max_tok_len, 1)

    # Read vocab + seq dims from the checkpoint to handle codebase constant
    # drift (training-time VOCAB_SIZE_BASE may differ from current).
    inner = params["params"] if "params" in params else params
    enc = inner.get("game_encoder", {})
    tok_emb_shape = enc.get("tok_embed", {}).get("embedding", None)
    pos_emb_shape = enc.get("pos_embed", {}).get("embedding", None)
    vocab_override = int(tok_emb_shape.shape[0]) if tok_emb_shape is not None else None
    seq_override = int(pos_emb_shape.shape[0]) if pos_emb_shape is not None else None
    print(f"  ckpt vocab={vocab_override}  max_seq={seq_override}", flush=True)

    model = build_model(cfg, max_C=max_C, max_tok_len=max_tok_len,
                         vocab_size_override=vocab_override, max_seq_override=seq_override)
    apply_fn = jax.jit(model.apply)

    vocab_size = VOCAB_SIZE_EXT if cfg.get("encode_sprites", False) else VOCAB_SIZE_BASE
    rng = np.random.RandomState(seed)

    results: dict[str, dict] = {}
    for mode in modes:
        per_game = {}
        t0 = time.time()
        for info in game_infos:
            name = info["name"]
            json_str = info["json_str"]
            n_objs = info["n_objs"]
            n_levels = min(int(info["n_levels"]), n_levels_max)
            tids = np.asarray(info.get("token_ids", []), dtype=np.int32)
            # Pad tokens / mask to the model's max_seq_len (read from ckpt)
            S = seq_override if seq_override is not None else (max_tok_len + 1)
            padded = np.zeros(S, dtype=np.int32)
            mask = np.zeros(S, dtype=np.bool_)
            n = min(len(tids), S)
            padded[:n] = tids[:n]
            mask[:n] = True
            vs = vocab_override if vocab_override is not None else vocab_size
            ab_tokens, ab_mask = ablate_tokens(padded, mask, mode, rng, vs)

            game_results = {}
            for level_i in range(n_levels):
                level_results = {}
                for tf_label, tf in [("random", False), ("random_tf", True)]:
                    bits, cells, first_divs = [], [], []
                    per_step_curves = []
                    for ep in range(n_episodes):
                        np.random.seed(seed + 1000 * level_i + ep)
                        try:
                            r = _run_eval_rollout(
                                apply_fn, params, json_str, level_i, n_objs,
                                max_C=max_C,
                                max_H=int(info["H"]), max_W=int(info["W"]),
                                max_steps=max_steps,
                                game_tokens=ab_tokens, game_mask=ab_mask,
                                teacher_forced=tf,
                            )
                            bits.append(int(np.sum(r["wrong_tiles"])))
                            cells.append(int(np.sum(r["wrong_cells"])))
                            first_divs.append(int(r["first_div_step"]))
                            per_step_curves.append(r["tile_error_rate"].tolist())
                        except Exception as e:
                            print(f"    [{name} L{level_i} {tf_label} ep{ep}] failed: {e}", flush=True)
                            bits.append(np.nan); cells.append(np.nan); first_divs.append(-1)
                    denom = n_episodes * n_objs * info["H"] * info["W"] * max_steps
                    err = float(np.nansum(bits)) / max(1, denom)
                    level_results[tf_label] = err
                    level_results[tf_label + "_first_div"] = float(np.mean([d for d in first_divs if d >= 0]) if any(d >= 0 for d in first_divs) else -1)
                    level_results[tf_label + "_curve"] = per_step_curves
                game_results[level_i] = level_results
            # Aggregate across levels
            agg = {}
            for tf_label in ("random", "random_tf"):
                vals = [game_results[l][tf_label] for l in game_results]
                agg[tf_label] = float(np.mean(vals))
                fd_vals = [game_results[l].get(tf_label + "_first_div", -1) for l in game_results]
                fd_vals = [v for v in fd_vals if v >= 0]
                agg[tf_label + "_first_div"] = float(np.mean(fd_vals)) if fd_vals else -1
            agg["per_level"] = game_results
            per_game[name] = agg
            print(f"  [{mode}] {name:<28} random_tf={agg['random_tf']:.4%}  random={agg['random']:.4%}", flush=True)
        elapsed = time.time() - t0
        macro_tf = float(np.mean([v["random_tf"] for v in per_game.values()]))
        macro_ar = float(np.mean([v["random"] for v in per_game.values()]))
        results[mode] = {
            "per_game": per_game,
            "macro_random_tf": macro_tf,
            "macro_random": macro_ar,
            "elapsed_s": elapsed,
        }
        print(f"  [{mode}] MACRO  random_tf={macro_tf:.4%}  random={macro_ar:.4%}  ({elapsed:.0f}s)", flush=True)
    return {"config": cfg, "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run dirs (must contain params.pkl, game_infos.pkl, config.json).")
    p.add_argument("--modes", nargs="+", default=["none", "zero", "shuffle"],
                   choices=["none", "zero", "random", "shuffle"])
    p.add_argument("--n_episodes", type=int, default=3,
                   help="Eval episodes per (game, level, mode, ablation).")
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--n_levels_max", type=int, default=2,
                   help="Eval up to this many levels per game (faster).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default="nca_wm/refine-logs/token_ablation_results.json")
    p.add_argument("--game_infos_from", default=None,
                   help="Borrow game_infos.pkl from this run dir instead of the eval target. "
                        "Use to eval held-out games with their info present in another run.")
    p.add_argument("--filter_games", nargs="*", default=None,
                   help="Only eval these game names (subset of game_infos).")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    all_results = {}
    for run in args.runs:
        label = os.path.basename(run.rstrip("/"))
        print(f"\n=== {label} ===", flush=True)
        try:
            all_results[label] = eval_one_run(
                run, args.modes, args.n_episodes, args.max_steps,
                args.n_levels_max, args.seed,
                game_infos_from=args.game_infos_from,
                filter_games=args.filter_games,
            )
        except Exception as e:
            print(f"FAILED: {label}: {e}\n{traceback.format_exc()}", flush=True)
            all_results[label] = {"error": str(e), "trace": traceback.format_exc()[-2000:]}
        # Persist incrementally so a crash mid-run doesn't lose progress.
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote {args.out_json}", flush=True)

    # Print compact summary table
    print("\n\n========== ABLATION SUMMARY ==========")
    modes = args.modes
    print(f"{'run':<55}  " + "  ".join(f"{m:>10}" for m in modes))
    for label, r in all_results.items():
        if "error" in r:
            print(f"{label:<55}  ERROR")
            continue
        cells_tf = [f"{r['results'][m]['macro_random_tf']*100:>9.3f}%" for m in modes]
        cells_ar = [f"{r['results'][m]['macro_random']*100:>9.3f}%" for m in modes]
        print(f"{label:<55} TF " + "  ".join(cells_tf))
        print(f"{'':<55} AR " + "  ".join(cells_ar))


if __name__ == "__main__":
    main()

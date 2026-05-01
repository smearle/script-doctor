"""Random-walk-init eval: does the model generalize to unseen start states?

Training data = A* solver trajectories (biased toward states near the goal).
This script takes each game, plays K random actions to drift the state into
a region the solver never explored, then rolls out the model and the engine
in parallel from that state and compares.

A model that learned game rules should still predict transitions correctly
from these unseen states. A memorizing model should fail catastrophically.

Usage:
    python -m nca_wm.scripts.random_init_eval \\
        --runs nca_wm/logs/<dir1> nca_wm/logs/<dir2> ... \\
        --burnin_steps 100 --eval_steps 50 --n_episodes 5 \\
        --out_json nca_wm/refine-logs/random_init_results.json
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

from puzzlescript_cpp import CppPuzzleScriptEnv

from nca_wm.train import (
    _pad_state_for_model, _pad_offsets,
    VOCAB_SIZE_BASE, VOCAB_SIZE_EXT, N_ACTIONS,
)
# Reuse model-rebuild logic from the token-ablation script.
from nca_wm.scripts.token_ablation_eval import build_model


def _load_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    with open(os.path.join(run_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    with open(os.path.join(run_dir, "params.pkl"), "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, dict) and "wm" in loaded and "dec" in loaded:
        params = loaded["wm"]
    else:
        params = loaded
    inner = params["params"] if "params" in params else params
    enc = inner.get("game_encoder", {})
    tok_emb = enc.get("tok_embed", {}).get("embedding", None)
    pos_emb = enc.get("pos_embed", {}).get("embedding", None)
    vocab_override = int(tok_emb.shape[0]) if tok_emb is not None else None
    seq_override = int(pos_emb.shape[0]) if pos_emb is not None else None
    max_C = int(max(g["n_objs"] for g in game_infos))
    max_tok_len = int(max(len(g.get("token_ids", [])) for g in game_infos))
    max_tok_len = max(max_tok_len, 1)
    model = build_model(cfg, max_C=max_C, max_tok_len=max_tok_len,
                         vocab_size_override=vocab_override,
                         max_seq_override=seq_override)
    apply_fn = jax.jit(model.apply)
    return cfg, game_infos, params, apply_fn, max_C, seq_override


def _pad_tokens(tids, S):
    padded = np.zeros(S, dtype=np.int32)
    mask = np.zeros(S, dtype=np.bool_)
    n = min(len(tids), S)
    padded[:n] = tids[:n]
    mask[:n] = True
    return padded, mask


def random_init_episode(apply_fn, params, json_str, level_i, n_objs,
                         max_C, max_H, max_W, game_tokens, game_mask,
                         burnin_steps: int, eval_steps: int,
                         seed: int) -> dict:
    """Reset env, do `burnin_steps` random actions to drift to an unseen
    state, then roll out engine + model in parallel for `eval_steps`."""
    rng = np.random.RandomState(seed)
    env = CppPuzzleScriptEnv(
        json_str, level_i=level_i,
        max_episode_steps=burnin_steps + eval_steps + 8,
    )
    real_obs, _ = env.reset()

    # Burn-in: random actions until either we hit the limit or env terminates.
    burnin_done = False
    for t in range(burnin_steps):
        action = int(rng.randint(N_ACTIONS))
        real_obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            burnin_done = True
            break
    # Note: if env terminated mid-burnin we still use real_obs as the init.
    # That's fine — it's still an unseen state from the model's perspective
    # (terminal states are rarely reached by solver-only training).

    # state_0 (model side) and state_0 (env side) both = real_obs after burnin.
    pred_state = _pad_state_for_model(real_obs, max_C, max_H, max_W)
    _, H, W = real_obs.shape
    if game_tokens is not None:
        gt = jnp.array(game_tokens[None]); gm = jnp.array(game_mask[None])

    wrong_tiles = []; wrong_cells = []
    first_div = -1
    actual_steps = 0
    for t in range(eval_steps):
        action = int(rng.randint(N_ACTIONS))
        a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[action][None])

        if game_tokens is not None:
            logits, *_ = apply_fn(params, pred_state, a_oh, gt, gm)
        else:
            logits, *_ = apply_fn(params, pred_state, a_oh)
        pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)

        try:
            real_obs, _, done, truncated, _ = env.step(action)
        except Exception:
            break

        _, pad_H, pad_W = pred_next.shape[1:]
        oy_c, _ = _pad_offsets(H, int(pad_H))
        ox_c, _ = _pad_offsets(W, int(pad_W))
        pred_binary = np.array(
            pred_next[0, :n_objs, oy_c:oy_c+H, ox_c:ox_c+W] > 0.5, dtype=np.uint8,
        )
        mismatch = (pred_binary != real_obs)
        n_wrong_bits = int(mismatch.sum())
        n_wrong_cells = int(mismatch.any(axis=0).sum())
        wrong_tiles.append(n_wrong_bits)
        wrong_cells.append(n_wrong_cells)
        if first_div == -1 and n_wrong_cells > 0:
            first_div = t
        actual_steps += 1
        pred_state = pred_next  # autoregressive
        if done or truncated:
            break

    total_tiles = n_objs * H * W
    return {
        "wrong_tiles": np.array(wrong_tiles),
        "wrong_cells": np.array(wrong_cells),
        "first_div_step": first_div,
        "total_tiles_per_step": total_tiles,
        "actual_steps": actual_steps,
        "burnin_terminated_early": burnin_done,
    }


def eval_one_run(run_dir, burnin_steps, eval_steps, n_episodes, n_levels_max, seed):
    cfg, game_infos, params, apply_fn, max_C, seq_override = _load_run(run_dir)
    print(f"  ckpt: vocab={seq_override and 'ok'}  max_C={max_C}", flush=True)

    per_game = {}
    t0 = time.time()
    for info in game_infos:
        name = info["name"]
        json_str = info["json_str"]
        n_objs = info["n_objs"]
        n_levels = min(int(info["n_levels"]), n_levels_max)
        S = seq_override if seq_override is not None else 1
        tids = np.asarray(info.get("token_ids", []), dtype=np.int32)
        gt, gm = _pad_tokens(tids, S)

        per_level = {}
        for level_i in range(n_levels):
            ep_results = []
            for ep in range(n_episodes):
                try:
                    r = random_init_episode(
                        apply_fn, params, json_str, level_i, n_objs,
                        max_C=max(max_C, n_objs), max_H=int(info["H"]), max_W=int(info["W"]),
                        game_tokens=gt, game_mask=gm,
                        burnin_steps=burnin_steps, eval_steps=eval_steps,
                        seed=seed + 10000 * level_i + 100 * ep,
                    )
                    ep_results.append(r)
                except Exception as e:
                    print(f"    [{name} L{level_i} ep{ep}] failed: {e}", flush=True)
            if not ep_results:
                continue
            # aggregate over episodes
            total_wrong = sum(int(np.sum(r["wrong_tiles"])) for r in ep_results)
            total_steps_seen = sum(r["actual_steps"] for r in ep_results)
            total_denom = sum(r["actual_steps"] * r["total_tiles_per_step"]
                              for r in ep_results)
            err = total_wrong / max(1, total_denom)
            fdvals = [r["first_div_step"] for r in ep_results if r["first_div_step"] >= 0]
            fdmean = float(np.mean(fdvals)) if fdvals else -1
            per_level[level_i] = {
                "err": err,
                "first_div": fdmean,
                "n_eps": len(ep_results),
                "mean_actual_steps": float(np.mean([r["actual_steps"] for r in ep_results])),
            }
        if not per_level:
            continue
        # aggregate over levels (unweighted mean)
        agg = {
            "err": float(np.mean([v["err"] for v in per_level.values()])),
            "first_div": float(np.mean([v["first_div"] for v in per_level.values()
                                          if v["first_div"] >= 0]) or -1),
            "per_level": per_level,
        }
        per_game[name] = agg
        print(f"  {name:<28} err={agg['err']:.4%}  first_div={agg['first_div']:.1f}", flush=True)

    macro_err = float(np.mean([v["err"] for v in per_game.values()]))
    fds = [v["first_div"] for v in per_game.values() if v["first_div"] >= 0]
    macro_fd = float(np.mean(fds)) if fds else -1
    elapsed = time.time() - t0
    print(f"  MACRO  err={macro_err:.4%}  first_div={macro_fd:.1f}  ({elapsed:.0f}s)", flush=True)
    return {
        "config": cfg,
        "macro_err": macro_err,
        "macro_first_div": macro_fd,
        "per_game": per_game,
        "elapsed_s": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--burnin_steps", type=int, default=100,
                   help="Random actions before starting model rollout. Larger "
                        "= further from training distribution.")
    p.add_argument("--eval_steps", type=int, default=50,
                   help="Autoregressive rollout length after burn-in.")
    p.add_argument("--n_episodes", type=int, default=5,
                   help="Eval episodes per (game, level). Each picks a "
                        "different random seed for both burn-in and eval "
                        "actions.")
    p.add_argument("--n_levels_max", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default="nca_wm/refine-logs/random_init_results.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    all_results = {}
    for run in args.runs:
        label = os.path.basename(run.rstrip("/"))
        print(f"\n=== {label} ===", flush=True)
        try:
            all_results[label] = eval_one_run(
                run, args.burnin_steps, args.eval_steps,
                args.n_episodes, args.n_levels_max, args.seed,
            )
        except Exception as e:
            print(f"FAILED: {label}: {e}\n{traceback.format_exc()}", flush=True)
            all_results[label] = {"error": str(e), "trace": traceback.format_exc()[-2000:]}
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Wrote {args.out_json}", flush=True)

    # Compact summary
    print("\n\n========== RANDOM-INIT SUMMARY ==========")
    print(f"  burnin={args.burnin_steps} eval={args.eval_steps} eps={args.n_episodes}")
    print(f"\n{'run':<60}  {'err':>8}  {'first_div':>10}")
    for label, r in all_results.items():
        if "error" in r:
            print(f"{label:<60}  ERROR")
            continue
        print(f"{label:<60}  {r['macro_err']*100:>7.3f}%  {r['macro_first_div']:>10.1f}")


if __name__ == "__main__":
    main()

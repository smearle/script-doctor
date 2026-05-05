"""Rule inference from transitions: likelihood-ratio test over candidate games.

Sanity check for "can the rule-conditioned WM identify a game's dynamics from
its transitions alone?" Loads a trained checkpoint, samples random
(s, a, s') transitions from each target game, then scores them under every
candidate game's rule conditioning. Argmin loss = predicted game.

This is the no-SGD baseline: we use each candidate game's *known* token
sequence rather than optimizing a free slot tensor. If the model's rule
conditioning isn't discriminative enough to win this test, gradient-based
inversion over the slot space won't help.

Usage:
    python nca_wm/rule_inference.py --load nca_wm/logs/scaling_14_joint_v1
    python nca_wm/rule_inference.py --load nca_wm/logs/scaling_2_joint_v1 \\
        --n_transitions 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.heldout_eval import _build_model, _load_run
from nca_wm.train import N_ACTIONS, _pad_state_for_model, _wm_p
from puzzlescript_cpp import CppPuzzleScriptEnv


def _next_pow2(x: int, min_val: int = 8) -> int:
    v = max(min_val, int(x))
    p = 1
    while p < v:
        p <<= 1
    return p


def _pad_tokens(token_ids, max_seq_len):
    tids = list(token_ids)[:max_seq_len]
    tok = np.zeros(max_seq_len, dtype=np.int32)
    msk = np.zeros(max_seq_len, dtype=np.bool_)
    tok[: len(tids)] = tids
    msk[: len(tids)] = True
    return tok, msk


def collect_transitions(info, n_transitions, max_C, seed,
                        require_change: bool = True,
                        max_attempts_factor: int = 50):
    """Sample (s, a, s') tuples by random rollouts in `info`'s game.

    With require_change=True, skips no-op transitions — most discriminative
    signal lives in cells that actually flipped between s and s'. Caps the
    total step budget at n_transitions * max_attempts_factor to avoid
    spinning on games where the random policy rarely changes state.
    """
    rng = np.random.RandomState(seed)
    H_eval = _next_pow2(info["H"])
    W_eval = _next_pow2(info["W"])
    states, actions, next_states = [], [], []
    attempts = 0
    budget = n_transitions * max_attempts_factor
    while len(states) < n_transitions and attempts < budget:
        li = int(rng.randint(info["n_levels"]))
        try:
            env = CppPuzzleScriptEnv(info["json_str"], level_i=li, max_episode_steps=10_000)
        except Exception as e:
            print(f"    env init failed (L{li}): {e}")
            continue
        obs, _ = env.reset()
        ep_len = int(rng.randint(2, 20))
        for _ in range(ep_len):
            if len(states) >= n_transitions or attempts >= budget:
                break
            attempts += 1
            a = int(rng.randint(N_ACTIONS))
            s_padded = _pad_state_for_model(obs, max_C, H_eval, W_eval)[0]
            nobs, _, done, trunc, _ = env.step(a)
            ns_padded = _pad_state_for_model(nobs, max_C, H_eval, W_eval)[0]
            s_arr = np.asarray(s_padded)
            ns_arr = np.asarray(ns_padded)
            if not require_change or not np.array_equal(s_arr, ns_arr):
                states.append(s_arr)
                actions.append(a)
                next_states.append(ns_arr)
            obs = nobs
            if done or trunc:
                break
    if len(states) < n_transitions:
        print(f"    WARN: only collected {len(states)}/{n_transitions} "
              f"changing transitions in {attempts} attempts")
    return (
        np.stack(states).astype(np.float32),
        np.asarray(actions, dtype=np.int32),
        np.stack(next_states).astype(np.float32),
        H_eval,
        W_eval,
    )


def score_candidate(apply_fn, params, states, actions, next_states, tokens, mask,
                    changed_mask=None):
    """BCE for target transitions under one candidate's rule tokens.

    Returns (full_bce, changed_only_bce). full_bce averages over all cells;
    changed_only_bce averages only over cells that actually flipped between
    s and s' (the discriminative subset).
    """
    B = states.shape[0]
    a_oh = np.eye(N_ACTIONS, dtype=np.float32)[actions]
    gt = np.broadcast_to(tokens[None], (B,) + tokens.shape).copy()
    gm = np.broadcast_to(mask[None], (B,) + mask.shape).copy()
    logits, _wl, _sl = apply_fn(
        _wm_p(params),
        jnp.asarray(states),
        jnp.asarray(a_oh),
        jnp.asarray(gt),
        jnp.asarray(gm),
    )
    ns = jnp.asarray(next_states)
    bce_per = optax.sigmoid_binary_cross_entropy(logits, ns)
    full_bce = float(bce_per.mean())
    if changed_mask is None:
        return full_bce, full_bce
    cm = jnp.asarray(changed_mask)
    if cm.sum() == 0:
        return full_bce, float("nan")
    changed_bce = float((bce_per * cm).sum() / cm.sum())
    return full_bce, changed_bce


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--load", required=True, help="Path to a trained save_dir.")
    ap.add_argument("--target_games", default=None,
                    help="Comma-separated; default = all training games.")
    ap.add_argument("--candidate_games", default=None,
                    help="Comma-separated; default = same as target_games.")
    ap.add_argument("--n_transitions", type=int, default=32,
                    help="(s, a, s') transitions sampled per target game.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="JSON output path. Default: <load>/rule_inference/ratio.json")
    args = ap.parse_args()

    print(f"\n=== Loading checkpoint from {args.load} ===")
    cfg, params, train_game_infos = _load_run(args.load)
    if not cfg.get("conditional", False):
        raise SystemExit("This script needs a conditional (rule-conditioned) model.")
    if "vocab_size" not in cfg:
        try:
            saved_vocab = params["wm"]["params"]["game_encoder"]["tok_embed"]["embedding"].shape[0]
            cfg["vocab_size"] = saved_vocab - 1
            print(f"  recovered vocab_size={cfg['vocab_size']} from saved tok_embed")
        except (KeyError, AttributeError):
            pass
    model = _build_model(cfg, train_game_infos)
    max_C = max(g["n_objs"] for g in train_game_infos)
    train_max_seq_len = max(len(g.get("token_ids", [])) for g in train_game_infos)
    model_max_seq_len = train_max_seq_len + 1
    print(f"  arch={cfg.get('architecture', 'film')}  n_hid={cfg['n_hid']}  "
          f"n_out={max_C}  max_seq_len={model_max_seq_len}  "
          f"games={[g['name'] for g in train_game_infos]}")

    by_name = {g["name"]: g for g in train_game_infos}

    target_names = (
        [n.strip() for n in args.target_games.split(",") if n.strip()]
        if args.target_games else [g["name"] for g in train_game_infos]
    )
    cand_names = (
        [n.strip() for n in args.candidate_games.split(",") if n.strip()]
        if args.candidate_games else target_names
    )
    target_names = [n for n in target_names if n in by_name]
    cand_names = [n for n in cand_names if n in by_name]
    print(f"  targets   ({len(target_names)}): {target_names}")
    print(f"  candidates({len(cand_names)}): {cand_names}")

    cand_tok = {n: _pad_tokens(by_name[n]["token_ids"], model_max_seq_len)
                for n in cand_names}

    apply_fn = jax.jit(model.apply)

    # loss_mat[i, j] = mean BCE for target_i's transitions conditioned on cand_j.
    loss_mat = np.full((len(target_names), len(cand_names)), np.nan)
    loss_mat_changed = np.full((len(target_names), len(cand_names)), np.nan)
    correct = 0
    correct_changed = 0
    total = 0
    for ti, tn in enumerate(target_names):
        info = by_name[tn]
        t0 = time.time()
        states, actions, next_states, H_eval, W_eval = collect_transitions(
            info, args.n_transitions, max_C, seed=args.seed + ti,
            require_change=True,
        )
        if len(states) == 0:
            print(f"\n  target={tn}  SKIP (no changing transitions found)")
            continue
        # Per-cell mask of cells that flipped between s and s' (any channel
        # changed). Shape (B, C, H, W) — broadcast-equal to logits.
        changed = (states != next_states).astype(np.float32)
        change_frac = float(changed.any(axis=1).mean())
        print(f"\n  target={tn}  shape=({H_eval},{W_eval})  "
              f"n_transitions={len(states)}  changed_any_cell_frac={change_frac:.3f}")
        for ci, cn in enumerate(cand_names):
            tok, msk = cand_tok[cn]
            full_bce, ch_bce = score_candidate(
                apply_fn, params, states, actions, next_states, tok, msk,
                changed_mask=changed,
            )
            loss_mat[ti, ci] = full_bce
            loss_mat_changed[ti, ci] = ch_bce

        order_full = np.argsort(loss_mat[ti])
        rank_full = [cand_names[j] for j in order_full]
        pred_full = rank_full[0]
        order_ch = np.argsort(loss_mat_changed[ti])
        rank_ch = [cand_names[j] for j in order_ch]
        pred_ch = rank_ch[0]
        ok_full = (pred_full == tn)
        ok_ch = (pred_ch == tn)
        correct += int(ok_full)
        correct_changed += int(ok_ch)
        total += 1
        true_full = loss_mat[ti, cand_names.index(tn)] if tn in cand_names else float("nan")
        true_ch = loss_mat_changed[ti, cand_names.index(tn)] if tn in cand_names else float("nan")
        print(f"    [full ]  pred={pred_full}  {'OK' if ok_full else 'WRONG'}  "
              f"true={true_full:.6f}  min={loss_mat[ti, order_full[0]]:.6f}  "
              f"top3={[(rank_full[i], f'{loss_mat[ti, order_full[i]]:.5f}') for i in range(min(3, len(cand_names)))]}")
        print(f"    [chg  ]  pred={pred_ch }  {'OK' if ok_ch   else 'WRONG'}  "
              f"true={true_ch:.4f}  min={loss_mat_changed[ti, order_ch[0]]:.4f}  "
              f"top3={[(rank_ch[i], f'{loss_mat_changed[ti, order_ch[i]]:.4f}') for i in range(min(3, len(cand_names)))]}  "
              f"({time.time()-t0:.1f}s)")

    acc_full = correct / max(total, 1)
    acc_changed = correct_changed / max(total, 1)
    print(f"\n=== Accuracy (full BCE):    {correct}/{total} = {acc_full:.3f} ===")
    print(f"=== Accuracy (changed BCE): {correct_changed}/{total} = {acc_changed:.3f} ===")

    out_path = args.out
    if out_path is None:
        out_dir = os.path.join(args.load, "rule_inference")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "ratio.json")
    with open(out_path, "w") as f:
        json.dump({
            "load": args.load,
            "n_transitions": args.n_transitions,
            "target_names": target_names,
            "cand_names": cand_names,
            "loss_mat_full": loss_mat.tolist(),
            "loss_mat_changed": loss_mat_changed.tolist(),
            "accuracy_full": acc_full,
            "accuracy_changed": acc_changed,
            "correct_full": correct,
            "correct_changed": correct_changed,
            "total": total,
        }, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""Evaluate whether varislide NCA steps match engine intermediate states.

This complements final-state distance accuracy: a model can predict the final
slide destination while ignoring the iterative "move one cell per rule pass"
semantics. This script measures per-NCA-step predictions against hand-simulated
varislide engine iterations.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
import jax.numpy as jnp

from nca_wm.scripts.inspect_varislide_distance import _build_model_from_cfg
from nca_wm.scripts.train_varislide_perstep import (
    N_ACTIONS,
    load_dataset,
    precompute_intermediate_states,
)


def _load_run(run_dir: str):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    # train_varislide_perstep always constructs RuleAttnNCAWorldModel with
    # adaptive_halt=True to expose per-step logits, but early configs did not
    # persist that flag.
    cfg.setdefault("adaptive_halt", True)
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p_path = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p_path):
        p_path = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p_path, "rb"))
    g = gi[0]
    embed_in = params["params"]["embed"]["kernel"].shape[0]
    c_pad = embed_in - N_ACTIONS
    toks_np = np.asarray(g["token_ids"])
    eff_len = max(toks_np.shape[0], 1)
    model = _build_model_from_cfg(cfg, toks_np.shape[0], c_pad)
    return cfg, params, model, toks_np, eff_len, c_pad


def _pad_channels(x: np.ndarray, c_pad: int) -> np.ndarray:
    if x.shape[1] == c_pad:
        return x
    if x.shape[1] > c_pad:
        raise ValueError(f"state has {x.shape[1]} channels but model expects {c_pad}")
    return np.pad(x, [(0, 0), (0, c_pad - x.shape[1]), (0, 0), (0, 0)])


def evaluate(run_dir: str, train_seed: int, widths: set[int] | None,
             batch_size: int):
    cfg, params, model, toks_np, eff_len, c_pad = _load_run(run_dir)
    n_steps = int(cfg.get("n_nca_steps", cfg.get("n_steps")))
    states, actions, next_states, native_h, native_w, _, _ = load_dataset(
        "varislide", train_seed=train_seed, widths=widths)
    states = _pad_channels(states.astype(np.float32), c_pad)
    next_states = _pad_channels(next_states.astype(np.float32), c_pad)
    inter = precompute_intermediate_states(
        states, actions, native_h, native_w, n_steps)

    # Evaluate all transitions, but report right-action separately because it
    # is the only action that exercises the slide rule.
    sum_wrong = np.zeros(n_steps, dtype=np.float64)
    sum_total = np.zeros(n_steps, dtype=np.float64)
    sum_wrong_chg = np.zeros(n_steps, dtype=np.float64)
    sum_total_chg = np.zeros(n_steps, dtype=np.float64)
    sum_delta = np.zeros(n_steps - 1, dtype=np.float64)
    sum_delta_total = np.zeros(n_steps - 1, dtype=np.float64)

    right = actions == 3
    idx_all = np.where(right)[0]
    for start in range(0, len(idx_all), batch_size):
        idx = idx_all[start:start + batch_size]
        s_b = jnp.asarray(states[idx])
        a_b = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[actions[idx]])
        toks = np.zeros((len(idx), eff_len), dtype=np.int32)
        if toks_np.shape[0] > 0:
            toks[:, :toks_np.shape[0]] = toks_np
        mask = np.zeros((len(idx), eff_len), dtype=bool)
        if toks_np.shape[0] > 0:
            mask[:, :toks_np.shape[0]] = True
        out = model.apply(params, s_b, a_b, jnp.asarray(toks), jnp.asarray(mask))
        if not isinstance(out[-1], tuple):
            raise ValueError(f"{run_dir} does not expose per-step logits")
        logits_per_step = out[-1][0]
        preds = np.asarray((jax.nn.sigmoid(logits_per_step) > 0.5), dtype=np.float32)
        targets = inter[idx, 1:n_steps + 1].transpose(1, 0, 2, 3, 4)
        prev = inter[idx, 0:1].transpose(1, 0, 2, 3, 4)
        prev = np.broadcast_to(prev, targets.shape)
        chg = targets != prev
        wrong = preds != targets
        sum_wrong += wrong.sum(axis=(1, 2, 3, 4))
        sum_total += np.prod(wrong.shape[1:])
        sum_wrong_chg += (wrong & chg).sum(axis=(1, 2, 3, 4))
        sum_total_chg += chg.sum(axis=(1, 2, 3, 4))
        if n_steps > 1:
            delta = preds[1:] != preds[:-1]
            sum_delta += delta.sum(axis=(1, 2, 3, 4))
            sum_delta_total += np.prod(delta.shape[1:])

    err = sum_wrong / np.maximum(sum_total, 1)
    chg_err = sum_wrong_chg / np.maximum(sum_total_chg, 1)
    delta_frac = sum_delta / np.maximum(sum_delta_total, 1)

    print(f"\n=== {run_dir} ===")
    print(f"  n_steps={n_steps} vq={cfg.get('vq_codebook', False)} "
          f"right_transitions={len(idx_all)}")
    print("  step_err:   " + " ".join(f"{x:.4f}" for x in err))
    print("  chg_err:    " + " ".join(f"{x:.4f}" for x in chg_err))
    if len(delta_frac):
        print("  pred_delta: " + " ".join(f"{x:.4f}" for x in delta_frac))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--train_seed", type=int, default=43)
    p.add_argument("--widths", default="6,8,10,12,16")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()
    widths = ({int(x) for x in args.widths.split(",") if x.strip()}
              if args.widths else None)
    for run_dir in args.runs:
        evaluate(run_dir, args.train_seed, widths, args.batch_size)


if __name__ == "__main__":
    main()

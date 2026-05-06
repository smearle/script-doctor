"""Evaluate a uniform-mode trained model with convergence-based halting.

Given a checkpoint trained with `--adaptive_halt --halt_mode uniform
--halt_kl_weight 0`, runs the model on cached transitions and reports:

  - Per-step prediction error / change-error trajectory (does the body
    actually converge, or does it drift?)
  - Effective halt step under "stop when consecutive predictions agree
    on ≥ (1-ε) fraction of cells" for several ε thresholds
  - Prediction quality at the convergence-halt step vs at fixed n_steps

Usage:
    python -m nca_wm.scripts.eval_convergence_halt \
        --run nca_wm/logs_halt_arch/<uniform_run>_seed0 \
        --eps_list "0.001,0.005,0.01,0.05" \
        --action 3   # optional filter
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nca_wm.rule_attn_model import RuleAttnNCAWorldModel  # noqa: E402
from nca_wm.train import _unpack_states  # noqa: E402

ROLLOUT_CACHE_DIR = os.path.join(_REPO, "rollout_data")
N_ACTIONS = 5


def _build_model(cfg, gtoks_len, n_objs):
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg["d_model"], enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg["d_slot"], n_attn_heads=cfg["n_heads"],
        max_seq_len=gtoks_len + 1,
        axis_pool=cfg["axis_pool"], axis_cummax=cfg["axis_cummax"],
        global_pool=cfg["global_pool"],
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=cfg["n_nca_repeats"],
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def convergence_halt_steps(preds: np.ndarray, eps: float) -> np.ndarray:
    """Per-batch-element halt step under "fraction of cells changing < eps".

    preds: (T, B, C, H, W) discrete predictions.
    Returns: (B,) int array, the smallest k≥2 where < eps fraction of cells
             changed between step k-1 and step k. If never converges, returns T.
    """
    T = preds.shape[0]
    B = preds.shape[1]
    cells_per_b = preds.shape[2] * preds.shape[3] * preds.shape[4]
    halt = np.full(B, T, dtype=np.int32)
    halted = np.zeros(B, dtype=bool)
    for k in range(1, T):
        diff = (preds[k] != preds[k - 1]).reshape(B, -1).sum(axis=1) / cells_per_b
        new_halt = (~halted) & (diff < eps)
        halt = np.where(new_halt, k + 1, halt)  # +1: 1-indexed step k+1
        halted = halted | new_halt
        if halted.all(): break
    return halt


def run_eval(run_dir: str, eps_list: list[float], action_filter: int | None,
             batch_size: int) -> None:
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p_path = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p_path):
        p_path = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p_path, "rb"))

    g = gi[0]
    embed_in = params["params"]["embed"]["kernel"].shape[0]
    C_pad = embed_in - N_ACTIONS
    maxH, maxW = g["H"], g["W"]
    toks_np = np.asarray(g["token_ids"])
    cache = sorted(glob.glob(os.path.join(
        ROLLOUT_CACHE_DIR, g["name"], "level_0", "*transitions*.npz")),
        key=lambda p: -os.path.getsize(p))[0]
    z = np.load(cache)
    lW = int(z["W"])
    # Cache stores states bit-packed along W; unpack before padding.
    s_np = _unpack_states(z["states"], lW).astype(np.float32)
    n_np = _unpack_states(z["next_states"], lW).astype(np.float32)
    a_np = z["actions"]
    if action_filter is not None:
        mask = a_np == action_filter
        s_np, a_np, n_np = s_np[mask], a_np[mask], n_np[mask]
    B = min(batch_size, s_np.shape[0])
    pad = lambda arr: np.pad(
        arr[:B],
        [(0, 0), (0, C_pad - arr.shape[1]),
         (0, maxH - arr.shape[2]), (0, maxW - arr.shape[3])],
    )
    sb = jnp.asarray(pad(s_np))
    nb = jnp.asarray(pad(n_np))
    ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[a_np[:B]])
    toks = jnp.asarray(np.tile(toks_np[None], (B, 1)))
    gm = jnp.ones_like(toks, dtype=bool)

    model = _build_model(cfg, toks_np.shape[0], C_pad)
    out = model.apply(params, sb, ab, toks, gm)
    if not cfg.get("adaptive_halt", False):
        # Final-step-only output. Just report fixed-T performance.
        logits = out[0]                                         # (B, C, H, W)
        preds_T = np.asarray((jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32))
        nb_np = np.asarray(nb)
        sb_np = np.asarray(sb)
        changed = (sb_np != nb_np).astype(np.float32)
        n_changed = changed.sum()
        err_T = (preds_T != nb_np).mean()
        chg_err_T = ((preds_T != nb_np).astype(np.float32) * changed
                      ).sum() / max(n_changed, 1.0)
        print(f"\n=== {run_dir} (no adaptive_halt — fixed-T only) ===")
        print(f"  fixed_T err = {err_T:.4f}, change_err = {chg_err_T:.4f}")
        return
    halt_aux = out[-1]
    ps_logits = halt_aux[0]                                    # (T, B, C, H, W)
    T = ps_logits.shape[0]
    preds = np.asarray((jax.nn.sigmoid(ps_logits) > 0.5).astype(jnp.float32))
    nb_np = np.asarray(nb)
    sb_np = np.asarray(sb)

    # Per-step error trajectory.
    err_per_step = (preds != nb_np[None]).mean(axis=(1, 2, 3, 4))
    # Per-step change-error
    changed = (sb_np != nb_np).astype(np.float32)
    n_changed = changed.sum()
    changed_full = np.broadcast_to(changed[None], preds.shape)
    nb_full = np.broadcast_to(nb_np[None], preds.shape)
    wrong_changed_per_step = ((preds != nb_full).astype(np.float32) * changed_full
                              ).sum(axis=(1, 2, 3, 4))
    chg_err_per_step = wrong_changed_per_step / max(n_changed, 1.0)

    print(f"\n=== {run_dir} ({T} steps, {B} examples"
          f"{f', action={action_filter}' if action_filter is not None else ''}) ===")
    print(f"  Per-step error trajectory:")
    print(f"    step:        {' '.join(f'{i+1:>5d}' for i in range(T))}")
    print(f"    err:         {' '.join(f'{x:>5.3f}' for x in err_per_step)}")
    print(f"    change_err:  {' '.join(f'{x:>5.3f}' for x in chg_err_per_step)}")

    # Convergence-halt analysis at multiple eps thresholds.
    print(f"\n  Convergence halt (stop when <eps cells change between steps):")
    print(f"    {'eps':<10} {'mean halt':<11} {'median':<8} "
          f"{'pct converged':<14} {'err@halt':<10} {'chg_err@halt':<14}")
    for eps in eps_list:
        halt_k = convergence_halt_steps(preds, eps)              # (B,)
        # Predictions at the halt step (per-batch index into T).
        pred_at_halt = preds[halt_k - 1, np.arange(B)]            # (B, C, H, W)
        err_at_halt = (pred_at_halt != nb_np).mean()
        chg_err_at_halt = (((pred_at_halt != nb_np).astype(np.float32) * changed
                             ).sum() / max(n_changed, 1.0))
        pct_converged = (halt_k < T).mean() * 100
        print(f"    {eps:<10.4f} {halt_k.mean():<11.2f} {np.median(halt_k):<8.0f} "
              f"{pct_converged:<14.1f} {err_at_halt:<10.4f} {chg_err_at_halt:<14.4f}")
    # Baseline: pred at fixed step T.
    err_T = (preds[-1] != nb_np).mean()
    chg_err_T = (((preds[-1] != nb_np).astype(np.float32) * changed).sum() / max(n_changed, 1.0))
    print(f"    {'fixed_T':<10} {T:<11.2f} {T:<8} -              "
          f"{err_T:<10.4f} {chg_err_T:<14.4f}")
    # Baseline: best per-step picked oracle-style.
    best_k = int(err_per_step.argmin())
    print(f"    {'oracle_min':<10} {best_k+1:<11} {best_k+1:<8} -              "
          f"{err_per_step[best_k]:<10.4f} {chg_err_per_step[best_k]:<14.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--eps_list", default="0.001,0.005,0.01,0.05")
    p.add_argument("--action", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    eps_list = [float(x) for x in args.eps_list.split(",")]
    run_eval(args.run, eps_list, args.action, args.batch_size)


if __name__ == "__main__":
    main()

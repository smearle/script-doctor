"""Per-slide-distance accuracy inspector for varislide multi-grid checkpoints.

Reports for each slide distance d:
  - n: # of right-action transitions where player slides exactly d cells
  - sig@gt: mean sigmoid at the GT player-arrival cell
  - max_sig: mean of (max sigmoid in the player's row) — model's confidence
  - argmax_correct: % of cases where argmax player position = GT position
  - mean_pred_d: mean predicted slide distance (negative = undershoot)

This is the diagnostic for "did the model learn the again-iteration?" — if
slide_d=1 is fine but slide_d≥2 is at chance, the model is firing the rule
once but not iterating.

Usage:
    .venv/bin/python3 nca_wm/scripts/inspect_varislide_distance.py \\
        --runs nca_wm/logs_canary/varislide_reproduce_h128 \\
                nca_wm/logs_canary/varislide_input_skip_h128 \\
                nca_wm/logs_canary/varislide_clw50_h128
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
RIGHT_ACTION = 3
PLAYER = 2


def _build_model_from_cfg(cfg, gtoks_len, n_objs):
    eff_seq_len = max(gtoks_len, 1) + 1
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg["d_model"], enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg["d_slot"], n_attn_heads=cfg["n_heads"],
        max_seq_len=eff_seq_len,
        axis_pool=cfg["axis_pool"], axis_cummax=cfg["axis_cummax"],
        global_pool=cfg["global_pool"],
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=cfg["n_nca_repeats"],
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def inspect(run_dir: str):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p_path = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p_path):
        p_path = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p_path, "rb"))
    g = gi[0]
    embed_in = params["params"]["embed"]["kernel"].shape[0]
    C_pad = embed_in - N_ACTIONS
    toks_np = np.asarray(g["token_ids"])
    eff_len = max(toks_np.shape[0], 1)
    train_H, train_W = g["H"], g["W"]

    model = _build_model_from_cfg(cfg, toks_np.shape[0], C_pad)

    dist_to_acc = {}
    for cache in sorted(glob.glob(os.path.join(ROLLOUT_CACHE_DIR, "varislide",
                                                 "synthetic_*x3",
                                                 "seed0_n12_v9*solv0*.npz"))):
        z = np.load(cache)
        s_raw, a, n_raw = z["states"], z["actions"], z["next_states"]
        if "W" in z.files:
            lW = int(z["W"])
            s = _unpack_states(s_raw, lW).astype(np.float32)
            n = _unpack_states(n_raw, lW).astype(np.float32)
        else:
            s = s_raw.astype(np.float32)
            n = n_raw.astype(np.float32)
        mask = (a == RIGHT_ACTION)
        s, a, n = s[mask], a[mask], n[mask]
        if len(s) == 0:
            continue
        delta = (s != n).any(axis=(1, 2, 3))
        s, a, n = s[delta], a[delta], n[delta]
        if len(s) == 0:
            continue
        lH, lW = s.shape[2], s.shape[3]
        eval_H, eval_W = max(train_H, lH), max(train_W, lW)
        pad = lambda arr: np.pad(arr, [(0, 0), (0, C_pad-arr.shape[1]),
                                       (0, eval_H-arr.shape[2]),
                                       (0, eval_W-arr.shape[3])])
        sb = jnp.asarray(pad(s))
        nb = jnp.asarray(pad(n))
        ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[a])
        toks_padded = np.zeros((len(s), eff_len), dtype=np.int32)
        if toks_np.shape[0] > 0:
            toks_padded[:, :toks_np.shape[0]] = toks_np
        toks = jnp.asarray(toks_padded)
        gmask = np.zeros((len(s), eff_len), dtype=bool)
        if toks_np.shape[0] > 0:
            gmask[:, :toks_np.shape[0]] = True
        gm = jnp.asarray(gmask)
        out = model.apply(params, sb, ab, toks, gm)
        sig = np.asarray(jax.nn.sigmoid(out[0]))
        sb_np = np.asarray(sb)[:, :, :lH, :lW]
        nb_np = np.asarray(nb)[:, :, :lH, :lW]
        sig_np = sig[:, :, :lH, :lW]

        for i in range(len(s)):
            in_p = np.argwhere(sb_np[i, PLAYER] > 0.5)
            out_p = np.argwhere(nb_np[i, PLAYER] > 0.5)
            if len(in_p) == 0 or len(out_p) == 0:
                continue
            in_row, in_col = int(in_p[0, 0]), int(in_p[0, 1])
            out_row, out_col = int(out_p[0, 0]), int(out_p[0, 1])
            if in_row != out_row:
                continue
            d = out_col - in_col
            if d <= 0:
                continue
            sig_at_arrival = float(sig_np[i, PLAYER, out_row, out_col])
            row_sig = sig_np[i, PLAYER, out_row, :]
            argmax = int(np.argmax(row_sig))
            max_sig = float(row_sig.max())
            dist_to_acc.setdefault(d, []).append(
                (sig_at_arrival, argmax == out_col, max_sig, argmax - in_col))

    print(f"\n=== {run_dir} ===")
    print(f"  n_nca_steps={cfg['n_nca_steps']} n_hid={cfg['n_hid']} "
          f"input_skip={cfg.get('input_skip', False)} "
          f"clw={cfg.get('change_loss_weight', 5.0)}")
    print(f"  {'slide_d':>8}  {'n':>5}  {'sig@gt':>8}  {'max_sig':>8}  "
          f"{'argmax_correct':>15}  {'mean_pred_d':>12}")
    for d in sorted(dist_to_acc.keys()):
        arr = dist_to_acc[d]
        sigs = [x[0] for x in arr]
        correct = [x[1] for x in arr]
        max_sigs = [x[2] for x in arr]
        pred_ds = [x[3] for x in arr]
        print(f"  {d:>8d}  {len(arr):>5d}  {np.mean(sigs):>8.3f}  "
              f"{np.mean(max_sigs):>8.3f}  {np.mean(correct):>14.1%}  "
              f"{np.mean(pred_ds):>11.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args()
    for r in args.runs:
        inspect(r)


if __name__ == "__main__":
    main()

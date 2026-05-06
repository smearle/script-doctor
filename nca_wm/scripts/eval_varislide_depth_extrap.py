"""Depth-extrapolation evaluation for varislide NCA WM checkpoints.

Loads a checkpoint trained at D_train shared NCA steps, then re-runs the
same model at *D_eval* steps for D_eval ∈ a configurable list. Because
the shared-weight body has only `conv_0`, `cell_slot_xattn_0`, ... etc
(n_layers = 1), the same params apply at any larger n_steps.

For each (run, D_eval, eval-width), we evaluate every right-action
transition with a non-trivial player movement and report:
  - argmax_correct[d]: % of slide-distance-d transitions where the model's
    argmax in the player's row lands at the GT arrival cell.
  - mean_pred_d[d]: mean predicted slide distance.
  - n[d]: count of transitions per distance bucket.

Output: a JSON file per run-dir at <run>/depth_extrap_eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

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


def _build_model(cfg, gtoks_len, n_objs, *, n_steps_override: int):
    eff_seq_len = max(gtoks_len, 1) + 1
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"],
        n_steps=n_steps_override,
        n_repeats=n_steps_override,  # keep n_layers=1 (one shared body)
        n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg.get("d_model", 64),
        enc_n_self_layers=cfg.get("n_enc_layers", 2),
        n_slots=cfg["n_slots"],
        n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg.get("d_slot", 64),
        n_attn_heads=cfg.get("n_heads", 4),
        max_seq_len=eff_seq_len,
        axis_pool=cfg.get("axis_pool", True),
        axis_cummax=cfg.get("axis_cummax", True),
        global_pool=cfg.get("global_pool", True),
        use_vq=cfg.get("vq_codebook", False),
        vq_codebook_size=cfg.get("vq_codebook_size", 512),
        vq_commitment_weight=cfg.get("vq_commitment_weight", 0.25),
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def _load_transitions(train_seed: int, widths):
    """Load right-action delta transitions for every requested width."""
    out = {}  # w -> (s, a, n) arrays
    for w in widths:
        wh = f"synthetic_{w}x3"
        pat = os.path.join(ROLLOUT_CACHE_DIR, "varislide", wh,
                           f"seed{train_seed}_n*_v*_mode-tile_pattern_empirical_solv0*.npz")
        files = sorted(glob.glob(pat))
        if not files:
            print(f"  [warn] no cache files for width {w} at {pat}")
            continue
        s_chunks, a_chunks, n_chunks = [], [], []
        for cache in files:
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
            s_chunks.append(s)
            a_chunks.append(a)
            n_chunks.append(n)
        if not s_chunks:
            continue
        out[w] = (np.concatenate(s_chunks, axis=0),
                  np.concatenate(a_chunks, axis=0),
                  np.concatenate(n_chunks, axis=0))
        print(f"  width {w}: {len(out[w][0])} right-action delta transitions")
    return out


def _pad_to(arr, c_pad, h, w):
    return np.pad(arr,
                  [(0, 0), (0, c_pad - arr.shape[1]),
                   (0, h - arr.shape[2]), (0, w - arr.shape[3])])


def _evaluate_run(run_dir: str, depths: list[int], train_seed: int,
                  eval_widths: list[int], batch_size: int):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
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
    train_H, train_W = g["H"], g["W"]
    print(f"\n=== {run_dir} ===")
    print(f"  cfg: n_nca_steps={cfg.get('n_nca_steps')} pool=({cfg.get('axis_pool')}|"
          f"{cfg.get('axis_cummax')}|{cfg.get('global_pool')}) input_skip={cfg.get('input_skip')}")

    # Materialize datasets once.
    width_to_data = _load_transitions(train_seed, eval_widths)
    if not width_to_data:
        return {}

    out = {
        "run_dir": run_dir,
        "config": {k: cfg.get(k) for k in [
            "n_nca_steps", "n_nca_repeats", "n_hid", "n_slots", "n_app_slots",
            "axis_pool", "axis_cummax", "global_pool", "input_skip",
            "synthetic_grid_sizes",
        ]},
        "train_seed": train_seed,
        "eval_widths": eval_widths,
        "depths": [],
    }

    for d_eval in depths:
        print(f"\n  -- D_eval = {d_eval} --")
        model = _build_model(cfg, toks_np.shape[0], c_pad, n_steps_override=d_eval)
        per_width = []
        for w in eval_widths:
            if w not in width_to_data:
                continue
            s, a, n = width_to_data[w]
            lH, lW = s.shape[2], s.shape[3]
            eval_H = max(train_H, lH)
            eval_W = max(train_W, lW)
            dist_buckets = {}  # d -> [n_total, n_correct, sum_pred_d]

            t0 = time.time()
            for start in range(0, len(s), batch_size):
                end = min(start + batch_size, len(s))
                sb = jnp.asarray(_pad_to(s[start:end], c_pad, eval_H, eval_W))
                nb = _pad_to(n[start:end], c_pad, eval_H, eval_W)
                ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[a[start:end]])
                toks = np.zeros((end - start, eff_len), dtype=np.int32)
                if toks_np.shape[0] > 0:
                    toks[:, :toks_np.shape[0]] = toks_np
                gmask = np.zeros((end - start, eff_len), dtype=bool)
                if toks_np.shape[0] > 0:
                    gmask[:, :toks_np.shape[0]] = True
                out_logits = model.apply(
                    params, sb, ab, jnp.asarray(toks), jnp.asarray(gmask))
                sig = np.asarray(jax.nn.sigmoid(out_logits[0]))
                sb_np = np.asarray(sb)[:, :, :lH, :lW]
                nb_np = nb[:, :, :lH, :lW]
                sig_np = sig[:, :, :lH, :lW]

                for i in range(end - start):
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
                    row_sig = sig_np[i, PLAYER, out_row, :]
                    argmax = int(np.argmax(row_sig))
                    pred_d = argmax - in_col
                    bucket = dist_buckets.setdefault(d, [0, 0, 0])
                    bucket[0] += 1
                    bucket[1] += int(argmax == out_col)
                    bucket[2] += pred_d
            elapsed = time.time() - t0
            per_d = [
                {"d": d, "n": v[0], "argmax_correct": v[1] / max(v[0], 1),
                 "mean_pred_d": v[2] / max(v[0], 1)}
                for d, v in sorted(dist_buckets.items())
            ]
            print(f"    width {w} (eff {eval_H}x{eval_W}): "
                  + " ".join(f"d{x['d']}={x['argmax_correct']:.1%}(n={x['n']})"
                             for x in per_d)
                  + f"  [{elapsed:.1f}s]")
            per_width.append({"width": w, "per_d": per_d})
        out["depths"].append({"D_eval": d_eval, "per_width": per_width})

    out_path = os.path.join(run_dir, "depth_extrap_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> wrote {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--depths", default="1,2,4,8,16,32,64",
                   help="Comma-separated list of D_eval values.")
    p.add_argument("--widths", default="6,8,10,12,16",
                   help="Comma-separated synthetic widths to evaluate on.")
    p.add_argument("--train_seed", type=int, default=0,
                   help="Synthetic-data seed used during training (for cache lookup).")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    for r in args.runs:
        _evaluate_run(r, depths, args.train_seed, widths, args.batch_size)


if __name__ == "__main__":
    main()

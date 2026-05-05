"""2D scatter of game latents from a rule_attn checkpoint.

The rule_attn encoder produces (K, d_slot) per game (K rule slots, d_slot
each). For a single-vector-per-game scatter we have several reduction
options:

- `flat`: reshape to (K * d_slot,). Highest dimensional, preserves slot
  identity. Default.
- `mean`: mean-pool over the K slots → (d_slot,). Lower-dim but loses
  slot-by-slot structure.
- `dyn_mean`: mean over only the dynamic slots (K - n_app_slots), exclude
  appearance slots. Closer to "rules only."

Usage:
    .venv/bin/python3 -m nca_wm.latent_scatter_rule_attn \
        --load nca_wm/logs/multi_scaling_gallery_v3_combined \
        --reduce flat
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def pca_2d(X: np.ndarray):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    var = (S ** 2) / max(len(X) - 1, 1)
    total = var.sum()
    frac = var[:2] / total if total > 0 else np.zeros(2)
    return proj, frac


def load_run(save_dir: str):
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(save_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(save_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    return cfg, params, game_infos


def build_encoder(cfg, max_tok_len):
    """Build a fresh RuleSlotEncoder matching the saved checkpoint cfg."""
    from nca_wm.rule_attn_model import RuleSlotEncoder
    enc = RuleSlotEncoder(
        vocab_size=cfg["vocab_size"] + 1,  # +1 for CLS, matches train.py
        max_seq_len=max_tok_len + 1,
        d_model=cfg["d_model"],
        n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"],
        d_slot=cfg["d_slot"],
        n_heads=cfg["n_heads"],
    )
    return enc


def pad_tokens(token_ids, max_seq_len):
    pad = np.zeros(max_seq_len, dtype=np.int32)
    mask = np.zeros(max_seq_len, dtype=np.bool_)
    L = min(len(token_ids), max_seq_len)
    pad[:L] = token_ids[:L]
    mask[:L] = True
    return pad, mask


def encode_all(cfg, params, game_infos, reduce: str):
    """Returns (z, names) where z is (N, d_reduced)."""
    max_tok_len = max(max((len(g.get("token_ids", [])) for g in game_infos),
                          default=1), 1)
    enc = build_encoder(cfg, max_tok_len)
    # Handle joint-decoder layout {"wm": ..., "dec": ...} vs flat layout.
    if isinstance(params, dict) and "wm" in params:
        enc_params = {"params": params["wm"]["params"]["game_encoder"]}
    else:
        enc_params = {"params": params["params"]["game_encoder"]}

    @jax.jit
    def _enc_one(tids, mask):
        return enc.apply(enc_params, tids[None], mask[None])  # (1, K, d)

    out = []
    names = []
    K = cfg["n_slots"]
    n_app = cfg.get("n_app_slots", 0)
    for info in game_infos:
        pad, mask = pad_tokens(info.get("token_ids", []), max_tok_len + 1)
        slots = np.array(_enc_one(jnp.array(pad), jnp.array(mask))[0])  # (K, d)
        if reduce == "flat":
            v = slots.reshape(-1)
        elif reduce == "mean":
            v = slots.mean(axis=0)
        elif reduce == "dyn_mean":
            n_dyn = K - n_app
            v = slots[:n_dyn].mean(axis=0)
        else:
            raise ValueError(reduce)
        out.append(v)
        names.append(info["name"])
    return np.stack(out), names


def per_game_pairwise(z: np.ndarray) -> np.ndarray:
    """Cosine distance pairwise."""
    norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    z_n = z / norms
    sim = z_n @ z_n.T
    return 1.0 - sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True,
                    help="Trained rule_attn run dir.")
    ap.add_argument("--reduce", default="flat",
                    choices=["flat", "mean", "dyn_mean"])
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--label_top_n", type=int, default=20,
                    help="Annotate the N most-extreme games on the scatter.")
    args = ap.parse_args()

    cfg, params, game_infos = load_run(args.load)
    if cfg.get("architecture") != "rule_attn":
        print(f"WARN: cfg.architecture={cfg.get('architecture')!r} (expected "
              f"rule_attn). Run anyway.", file=sys.stderr)

    save_dir = args.save_dir or os.path.join(args.load, "interp")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Encoding {len(game_infos)} games via rule_attn encoder "
          f"(reduce={args.reduce})...", file=sys.stderr)
    z, names = encode_all(cfg, params, game_infos, args.reduce)
    print(f"  z.shape = {z.shape}", file=sys.stderr)

    proj, var_frac = pca_2d(z)
    print(f"  PC1+PC2 explain {sum(var_frac)*100:.1f}% of variance",
          file=sys.stderr)

    # Pairwise distance summary
    D = per_game_pairwise(z)
    np.fill_diagonal(D, np.nan)
    pair_min = np.nanmin(D)
    pair_mean = np.nanmean(D)
    pair_med = np.nanmedian(D)
    # Find tightest pair
    iu = np.triu_indices_from(D, k=1)
    flat = D[iu]
    order = np.argsort(flat)
    closest_pairs = [
        (names[iu[0][k]], names[iu[1][k]], flat[k])
        for k in order[:8]
    ]
    most_distinct = [
        (names[iu[0][k]], names[iu[1][k]], flat[k])
        for k in order[-8:][::-1]
    ]
    print(f"\nCosine-distance summary (over {len(names)} games, "
          f"{len(flat)} pairs):", file=sys.stderr)
    print(f"  min={pair_min:.4f}  median={pair_med:.4f}  mean={pair_mean:.4f}",
          file=sys.stderr)
    print(f"\n  Tightest pairs (most similar latents):", file=sys.stderr)
    for a, b, d in closest_pairs:
        print(f"    {d:.4f}  {a}  ↔  {b}", file=sys.stderr)
    print(f"\n  Most-distinct pairs:", file=sys.stderr)
    for a, b, d in most_distinct:
        print(f"    {d:.4f}  {a}  ↔  {b}", file=sys.stderr)

    # Plot scatter
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(proj[:, 0], proj[:, 1], c="#1f77b4", s=40,
               edgecolors="black", linewidths=0.5, zorder=3)
    # Annotate the N games furthest from the centroid (most "interesting")
    centroid = proj.mean(axis=0)
    dists = np.linalg.norm(proj - centroid, axis=1)
    annot_idx = set(np.argsort(-dists)[:args.label_top_n].tolist())
    for k, name in enumerate(names):
        if k in annot_idx:
            ax.annotate(name, (proj[k, 0], proj[k, 1]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var_frac[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_frac[1]*100:.1f}%)")
    title = (f"rule_attn z — reduce={args.reduce} | "
             f"d_z={z.shape[1]} | N={len(names)} games | "
             f"min cos dist = {pair_min:.3f}")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(save_dir, f"latent_scatter_{args.reduce}.png")
    out_pdf = os.path.join(save_dir, f"latent_scatter_{args.reduce}.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\nSaved {out_png}\nSaved {out_pdf}", file=sys.stderr)

    # Persist the data
    out_npz = os.path.join(save_dir, f"latent_scatter_{args.reduce}.npz")
    np.savez(out_npz, z=z, names=np.array(names, dtype=object),
             proj=proj, var_frac=var_frac, D=D)
    print(f"Saved {out_npz}", file=sys.stderr)

    # Summary JSON for downstream tooling
    summary = {
        "n_games": len(names),
        "d_z": int(z.shape[1]),
        "reduce": args.reduce,
        "var_frac_pc1": float(var_frac[0]),
        "var_frac_pc2": float(var_frac[1]),
        "cos_dist_min": float(pair_min),
        "cos_dist_median": float(pair_med),
        "cos_dist_mean": float(pair_mean),
        "tightest_pairs": [
            {"a": a, "b": b, "cos_dist": float(d)}
            for a, b, d in closest_pairs
        ],
        "most_distinct_pairs": [
            {"a": a, "b": b, "cos_dist": float(d)}
            for a, b, d in most_distinct
        ],
    }
    with open(os.path.join(save_dir, f"latent_scatter_{args.reduce}.json"),
              "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

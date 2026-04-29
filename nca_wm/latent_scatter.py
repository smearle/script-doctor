"""2D scatter of game latents (NCA encoder, optionally vs. token-AE baseline).

Encodes every training game in a ConditionalNCAWorldModel run to z via the
trained `GameSpecEncoder`, projects to 2D via PCA, and plots a labeled
scatter. With `--ae_path token_ae.pkl`, plots the token-AE's z's side-by-side
on the same axes (still PCA'd in their own space) so one can compare the
geometry of behavior-grounded vs reconstruction-grounded latents.

A specific `--game_a`/`--game_b` pair (with `--n_interp`) draws the linear
interpolation path projected into the same 2D plane — sanity-check for the
rollout-interpolation strip.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.interpolate_rollout import (
    build_model, encode_z, find_game, load_run,
)


def pca_2d(X: np.ndarray):
    """Center X (N, D) and project to top-2 PCs.

    Returns (proj (N, 2), explained_var_fraction (2,)).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S Vt; PCs are rows of Vt.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    var = (S ** 2) / max(len(X) - 1, 1)
    total_var = var.sum()
    frac = var[:2] / total_var if total_var > 0 else np.zeros(2)
    return proj, frac


def encode_all_nca(model, params, game_infos):
    """Run the trained encoder on every game's tokens. Returns (N, d_z)."""
    zs = []
    max_seq_len = model.max_seq_len - 1
    for info in game_infos:
        z, _, _ = encode_z(model, params, info, max_seq_len)
        zs.append(z)
    return np.stack(zs, axis=0)


def _scatter(ax, proj_2d, labels, title, var_frac, highlight_pair=None,
             interp_proj=None):
    """Render one PCA scatter into ax.

    highlight_pair: (i, j) — color these two points distinctly.
    interp_proj: optional (n_interp, 2) projected interpolation path.
    """
    n = len(labels)
    base_colors = ["#888888"] * n
    if highlight_pair is not None:
        i, j = highlight_pair
        base_colors[i] = "#1f77b4"
        base_colors[j] = "#d62728"
    ax.scatter(proj_2d[:, 0], proj_2d[:, 1], c=base_colors, s=60,
               edgecolors="black", linewidths=0.6, zorder=3)
    for k, name in enumerate(labels):
        ax.annotate(name, (proj_2d[k, 0], proj_2d[k, 1]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    if interp_proj is not None and len(interp_proj) >= 2:
        ax.plot(interp_proj[:, 0], interp_proj[:, 1],
                color="#2ca02c", linewidth=1.4, alpha=0.8,
                marker="o", markersize=4, zorder=2,
                label=f"interp (n={len(interp_proj)})")
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(f"PC1 ({var_frac[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_frac[1]*100:.1f}%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True,
                   help="Trained ConditionalNCAWorldModel run dir.")
    p.add_argument("--ae_path", default=None,
                   help="Optional token-AE checkpoint (token_ae.pkl) for "
                        "side-by-side comparison.")
    p.add_argument("--game_a", default=None,
                   help="If set with --game_b, draw projected interpolation path.")
    p.add_argument("--game_b", default=None)
    p.add_argument("--n_interp", type=int, default=11,
                   help="Number of points along A→B in latent space to project.")
    p.add_argument("--save_dir", default=None,
                   help="Output dir (default: <load>/interp).")
    args = p.parse_args()

    cfg, params, game_infos = load_run(args.load)
    model = build_model(cfg, game_infos)
    save_dir = args.save_dir or os.path.join(args.load, "interp")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Encoding {len(game_infos)} games via NCA encoder...")
    z_nca = encode_all_nca(model, params, game_infos)
    names = [g["name"] for g in game_infos]

    proj_nca, var_nca = pca_2d(z_nca)
    print(f"NCA latent: d_z={z_nca.shape[1]}, "
          f"PC1+PC2 explain {sum(var_nca)*100:.1f}% of variance")

    # Interpolation path projection (NCA latent)
    interp_proj_nca = None
    pair_idx = None
    if args.game_a and args.game_b:
        i, _ = find_game(game_infos, args.game_a)
        j, _ = find_game(game_infos, args.game_b)
        pair_idx = (i, j)
        ts = np.linspace(0.0, 1.0, args.n_interp)
        z_path = (1 - ts[:, None]) * z_nca[i:i+1] + ts[:, None] * z_nca[j:j+1]
        # Project using the same centering + Vt as the scatter (re-derive)
        Xc = z_nca - z_nca.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        path_c = z_path - z_nca.mean(axis=0, keepdims=True)
        interp_proj_nca = path_c @ Vt[:2].T

    # Optional AE side
    ae_panel = args.ae_path is not None
    z_ae, names_ae, proj_ae, var_ae, interp_proj_ae, pair_idx_ae = (
        None, None, None, None, None, None
    )
    if ae_panel:
        with open(args.ae_path, "rb") as f:
            ae = pickle.load(f)
        z_ae = np.array(ae["z_all"])
        names_ae = list(ae["game_names"])
        print(f"AE latent: d_z={z_ae.shape[1]}, N={z_ae.shape[0]} games")
        proj_ae, var_ae = pca_2d(z_ae)
        print(f"AE PC1+PC2 explain {sum(var_ae)*100:.1f}% of variance")
        if args.game_a and args.game_b:
            try:
                ia = names_ae.index(args.game_a)
                ib = names_ae.index(args.game_b)
                pair_idx_ae = (ia, ib)
                ts = np.linspace(0.0, 1.0, args.n_interp)
                z_path = (1 - ts[:, None]) * z_ae[ia:ia+1] + ts[:, None] * z_ae[ib:ib+1]
                Xc = z_ae - z_ae.mean(axis=0, keepdims=True)
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                path_c = z_path - z_ae.mean(axis=0, keepdims=True)
                interp_proj_ae = path_c @ Vt[:2].T
            except ValueError:
                print(f"  game_a/game_b not in AE training set; skipping AE path")

    if ae_panel:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        _scatter(axes[0], proj_nca, names, "NCA encoder z (behavior-grounded)",
                 var_nca, highlight_pair=pair_idx, interp_proj=interp_proj_nca)
        _scatter(axes[1], proj_ae, names_ae, "Token-AE z (reconstruction-grounded)",
                 var_ae, highlight_pair=pair_idx_ae, interp_proj=interp_proj_ae)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        _scatter(ax, proj_nca, names, "NCA encoder z",
                 var_nca, highlight_pair=pair_idx, interp_proj=interp_proj_nca)

    fig.tight_layout()
    out_path = os.path.join(save_dir, "latent_scatter.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # Also dump the raw arrays for downstream analysis.
    np.savez(
        os.path.join(save_dir, "latent_scatter.npz"),
        z_nca=z_nca,
        names=np.array(names, dtype=object),
        var_frac_nca=var_nca,
        proj_nca=proj_nca,
        **({"z_ae": z_ae, "names_ae": np.array(names_ae, dtype=object),
            "var_frac_ae": var_ae, "proj_ae": proj_ae} if ae_panel else {}),
    )


if __name__ == "__main__":
    main()

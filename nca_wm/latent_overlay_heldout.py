"""Overlay held-out games on a trained rule_attn run's latent scatter.

Encodes a list of held-out games (from `data/heldout_v*_n*.json`) through
the same `RuleSlotEncoder` used to encode the training games. PCA is fit on
the *training* latents so held-out points are projected into the training
space (not refit per heldout). For each held-out game we report:

- 1-NN training game by cosine distance
- 3-NN training games and their distances
- The held-out's PC1/PC2 coordinates

Plot: training games as gray dots, held-out games as colored stars labeled.

Usage:
    .venv/bin/python3 -m nca_wm.latent_overlay_heldout \
        --load nca_wm/logs/multi_scaling_gallery_v3 \
        --heldout_file data/heldout_v4_n30.json \
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


def load_run(save_dir: str):
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(save_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(save_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    return cfg, params, game_infos


def build_encoder(cfg, max_tok_len):
    from nca_wm.rule_attn_model import RuleSlotEncoder
    return RuleSlotEncoder(
        vocab_size=cfg["vocab_size"] + 1,
        max_seq_len=max_tok_len + 1,
        d_model=cfg["d_model"],
        n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"],
        d_slot=cfg["d_slot"],
        n_heads=cfg["n_heads"],
    )


def pad_tokens(token_ids, max_seq_len):
    pad = np.zeros(max_seq_len, dtype=np.int32)
    mask = np.zeros(max_seq_len, dtype=np.bool_)
    L = min(len(token_ids), max_seq_len)
    pad[:L] = token_ids[:L]
    mask[:L] = True
    return pad, mask


def reduce_slots(slots, reduce, n_slots, n_app_slots):
    """slots: (K, d_slot)."""
    if reduce == "flat":
        return slots.reshape(-1)
    if reduce == "mean":
        return slots.mean(axis=0)
    if reduce == "dyn_mean":
        n_dyn = n_slots - n_app_slots
        return slots[:n_dyn].mean(axis=0)
    raise ValueError(reduce)


def tokenize_game_for_encoder(name: str, encode_sprites: bool, vocab_size: int):
    """Run the standard tokenizer on a held-out game's cached lark tree.

    Returns (token_ids, ok). Raises if not findable.
    """
    from puzzlescript_jax.gen_tree import GenPSTree
    from nca_wm.tokenize_game import tokenize_game

    fp = os.path.join(REPO, "data", "game_trees", name + ".pkl")
    if not os.path.exists(fp):
        return None
    with open(fp, "rb") as f:
        lt = pickle.load(f)
    ps = GenPSTree().transform(lt)
    objs = ps.objects
    cids = (list(objs.keys()) if isinstance(objs, dict)
            else [o.name for o in objs])
    toks = tokenize_game(ps, cids, encode_sprites=encode_sprites)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--heldout_file", required=True)
    ap.add_argument("--reduce", default="flat",
                    choices=["flat", "mean", "dyn_mean"])
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--k_nn", type=int, default=3)
    args = ap.parse_args()

    cfg, params, game_infos = load_run(args.load)
    if cfg.get("architecture") != "rule_attn":
        print(f"WARN: cfg.architecture={cfg.get('architecture')!r}; expected "
              f"rule_attn", file=sys.stderr)

    save_dir = args.save_dir or os.path.join(args.load, "interp")
    os.makedirs(save_dir, exist_ok=True)

    # max_tok_len matches the training-time encoder dims
    max_tok_len = max(max((len(g.get("token_ids", [])) for g in game_infos),
                          default=1), 1)
    enc = build_encoder(cfg, max_tok_len)
    enc_params = {"params": params["params"]["game_encoder"]}

    @jax.jit
    def _enc(tids, mask):
        return enc.apply(enc_params, tids[None], mask[None])  # (1, K, d)

    print(f"Encoding {len(game_infos)} training games + held-outs...",
          file=sys.stderr)

    # Encode training games (must match training tokenization order/length)
    z_train = []
    train_names = []
    for info in game_infos:
        pad, mask = pad_tokens(info.get("token_ids", []), max_tok_len + 1)
        slots = np.array(_enc(jnp.array(pad), jnp.array(mask))[0])
        v = reduce_slots(slots, args.reduce,
                         cfg["n_slots"], cfg.get("n_app_slots", 0))
        z_train.append(v)
        train_names.append(info["name"])
    z_train = np.stack(z_train)

    # Encode held-out games — re-tokenize from cached trees
    with open(args.heldout_file) as f:
        heldout_spec = json.load(f)
    heldout_names_raw = [h["name"] for h in heldout_spec["heldout"]]
    z_heldout = []
    heldout_names = []
    skipped = []
    encode_sprites = cfg.get("encode_sprites", False)
    for name in heldout_names_raw:
        toks = tokenize_game_for_encoder(name, encode_sprites,
                                          cfg["vocab_size"])
        if toks is None:
            skipped.append((name, "no cached tree"))
            continue
        if len(toks) > max_tok_len:
            skipped.append((name, f"too long: {len(toks)} > {max_tok_len}"))
            continue
        pad, mask = pad_tokens(toks, max_tok_len + 1)
        slots = np.array(_enc(jnp.array(pad), jnp.array(mask))[0])
        v = reduce_slots(slots, args.reduce,
                         cfg["n_slots"], cfg.get("n_app_slots", 0))
        z_heldout.append(v)
        heldout_names.append(name)
    if skipped:
        print(f"  skipped {len(skipped)} held-outs:", file=sys.stderr)
        for n, why in skipped[:10]:
            print(f"    {n}: {why}", file=sys.stderr)
    if not z_heldout:
        print("ERROR: no held-out games encoded", file=sys.stderr)
        sys.exit(1)
    z_heldout = np.stack(z_heldout)

    # PCA fit on training latents
    mu = z_train.mean(axis=0, keepdims=True)
    Xc = z_train - mu
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj_train = Xc @ Vt[:2].T
    proj_heldout = (z_heldout - mu) @ Vt[:2].T
    var = (S ** 2) / max(len(z_train) - 1, 1)
    var_frac = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)

    # k-NN: for each held-out, nearest training games by cosine distance
    n_train = len(train_names)
    norms_t = np.linalg.norm(z_train, axis=1, keepdims=True) + 1e-9
    z_train_n = z_train / norms_t
    norms_h = np.linalg.norm(z_heldout, axis=1, keepdims=True) + 1e-9
    z_heldout_n = z_heldout / norms_h
    sim = z_heldout_n @ z_train_n.T  # (n_h, n_train)
    cos_dist = 1.0 - sim
    knn_summary = []
    print(f"\nk-NN summary (k={args.k_nn}):", file=sys.stderr)
    for i, hn in enumerate(heldout_names):
        order = np.argsort(cos_dist[i])
        top = [(train_names[j], float(cos_dist[i, j]))
               for j in order[:args.k_nn]]
        knn_summary.append({"heldout": hn,
                            "knn": [{"train": t, "cos_dist": d}
                                    for t, d in top]})
        nn1, nn1d = top[0]
        print(f"  {hn:55s} -> 1-NN: {nn1} ({nn1d:.3f})",
              file=sys.stderr)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.scatter(proj_train[:, 0], proj_train[:, 1], c="#888888", s=35,
               edgecolors="black", linewidths=0.4, zorder=2,
               label=f"train ({len(train_names)})")
    ax.scatter(proj_heldout[:, 0], proj_heldout[:, 1], c="#d62728",
               marker="*", s=180, edgecolors="black", linewidths=0.7,
               zorder=4, label=f"heldout ({len(heldout_names)})")
    # Annotate held-outs (compact labels)
    for i, name in enumerate(heldout_names):
        short = name if len(name) <= 30 else name[:27] + "..."
        ax.annotate(short, (proj_heldout[i, 0], proj_heldout[i, 1]),
                    fontsize=7, xytext=(5, 5), textcoords="offset points",
                    color="#d62728")
    # Annotate the training games at the boundary (most extreme PCs)
    centroid = proj_train.mean(axis=0)
    dists = np.linalg.norm(proj_train - centroid, axis=1)
    extreme = np.argsort(-dists)[:10]
    for k in extreme:
        ax.annotate(train_names[k], (proj_train[k, 0], proj_train[k, 1]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points",
                    color="#444444")

    ax.set_xlabel(f"PC1 ({var_frac[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_frac[1]*100:.1f}%)")
    ax.set_title(f"rule_attn z — train + heldout overlay "
                 f"(reduce={args.reduce}, d_z={z_train.shape[1]}, "
                 f"N_train={len(train_names)}, N_held={len(heldout_names)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    base = os.path.join(save_dir, f"heldout_overlay_{args.reduce}")
    fig.savefig(base + ".png", dpi=150)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"\nSaved {base}.png\nSaved {base}.pdf", file=sys.stderr)

    np.savez(base + ".npz",
             z_train=z_train, z_heldout=z_heldout,
             train_names=np.array(train_names, dtype=object),
             heldout_names=np.array(heldout_names, dtype=object),
             proj_train=proj_train, proj_heldout=proj_heldout,
             var_frac=var_frac, cos_dist=cos_dist)

    # k-NN summary stats
    nn1_dists = np.array([s["knn"][0]["cos_dist"] for s in knn_summary])
    summary = {
        "n_train": len(train_names),
        "n_heldout": len(heldout_names),
        "skipped": skipped,
        "reduce": args.reduce,
        "var_frac_pc1": float(var_frac[0]),
        "var_frac_pc2": float(var_frac[1]),
        "nn1_cos_dist_mean": float(nn1_dists.mean()),
        "nn1_cos_dist_median": float(np.median(nn1_dists)),
        "nn1_cos_dist_min": float(nn1_dists.min()),
        "nn1_cos_dist_max": float(nn1_dists.max()),
        "knn": knn_summary,
    }
    with open(base + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {base}.json", file=sys.stderr)
    print(f"\n1-NN cos dist: min={nn1_dists.min():.3f} median="
          f"{np.median(nn1_dists):.3f} mean={nn1_dists.mean():.3f} "
          f"max={nn1_dists.max():.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()

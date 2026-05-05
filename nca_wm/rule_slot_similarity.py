"""Pairwise similarity of encoded rule slots, per training game.

Encodes each game's tokens via the rule_attn encoder, dumps the resulting
(K, d_slot) slot tensor per game, computes pairwise similarity, and
optionally overlays the prediction-confusion matrix from rule_inference.py.

Diagnoses where prediction confusion comes from:
  - high slot-sim + high pred-confusion = encoder collapsed similar rules
  - low  slot-sim + high pred-confusion = downstream NCA threw away distinction

Usage:
    python nca_wm/rule_slot_similarity.py --load nca_wm/logs/scaling_14_joint_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.heldout_eval import _build_model, _load_run
from nca_wm.train import N_ACTIONS, _wm_p
from nca_wm.rule_inference import _pad_tokens


def _flat_cosine(A, B):
    a, b = A.flatten(), B.flatten()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float("nan") if (na == 0 or nb == 0) else float(a @ b / (na * nb))


def _greedy_per_slot_cosine(A, B):
    """Per-slot best cosine in B for each slot in A; return mean (and the
    matrix-max-row mean of the reverse direction, averaged for symmetry)."""
    A_n = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-12)
    B_n = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-12)
    sim = A_n @ B_n.T  # (K, K)
    return 0.5 * (float(sim.max(axis=-1).mean()) + float(sim.max(axis=0).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--load", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ratio_json", default=None,
                    help="Path to ratio.json from rule_inference.py to overlay. "
                    "Defaults to <load>/rule_inference/ratio.json if present.")
    args = ap.parse_args()

    print(f"\n=== Loading checkpoint from {args.load} ===")
    cfg, params, gi = _load_run(args.load)
    if not cfg.get("conditional", False):
        raise SystemExit("Need a conditional model.")
    if cfg.get("architecture", "film") != "rule_attn":
        raise SystemExit("This script targets rule_attn (FiLM has no slots).")
    if "vocab_size" not in cfg:
        try:
            v = params["wm"]["params"]["game_encoder"]["tok_embed"]["embedding"].shape[0]
            cfg["vocab_size"] = v - 1
            print(f"  recovered vocab_size={cfg['vocab_size']}")
        except (KeyError, AttributeError):
            pass

    model = _build_model(cfg, gi)
    max_C = max(g["n_objs"] for g in gi)
    train_max_seq_len = max(len(g.get("token_ids", [])) for g in gi)
    model_max_seq_len = train_max_seq_len + 1
    n_slots = cfg["n_slots"]
    n_app_slots = cfg.get("n_app_slots", 0)
    n_dyn = n_slots - n_app_slots
    print(f"  n_slots={n_slots}  n_app_slots={n_app_slots}  n_dyn={n_dyn}  "
          f"d_slot={cfg['d_slot']}  vq={cfg.get('vq_codebook', False)}")

    dummy_state = jnp.zeros((1, max_C, 8, 8), jnp.float32)
    dummy_action = jnp.zeros((1, N_ACTIONS), jnp.float32)

    names = [g["name"] for g in gi]
    G = len(names)
    slots_per_game = []
    print(f"\nEncoding {G} games...")
    for g in gi:
        tok, msk = _pad_tokens(g["token_ids"], model_max_seq_len)
        out = model.apply(
            _wm_p(params), dummy_state, dummy_action,
            jnp.asarray(tok[None]), jnp.asarray(msk[None]),
            return_slots=True,
        )
        all_slots = np.asarray(out[3][0])  # (n_slots, d_slot)
        slots_per_game.append(all_slots)
    slots_per_game = np.stack(slots_per_game)  # (G, n_slots, d_slot)
    dyn_slots = slots_per_game[:, :n_dyn, :]

    # Sanity: slot norms (collapsed slots would be near-zero or near-uniform).
    norms = np.linalg.norm(dyn_slots, axis=-1)  # (G, n_dyn)
    print(f"  per-game dyn-slot norm: mean={norms.mean(1).mean():.3f} "
          f"std-across-games={norms.mean(1).std():.4f} "
          f"per-slot-std-mean-across-games={norms.std(0).mean():.4f}")

    flat_sim = np.eye(G)
    greedy_sim = np.eye(G)
    for i in range(G):
        for j in range(i + 1, G):
            f = _flat_cosine(dyn_slots[i], dyn_slots[j])
            g_ = _greedy_per_slot_cosine(dyn_slots[i], dyn_slots[j])
            flat_sim[i, j] = flat_sim[j, i] = f
            greedy_sim[i, j] = greedy_sim[j, i] = g_

    pairs = [(i, j, flat_sim[i, j], greedy_sim[i, j])
             for i in range(G) for j in range(i + 1, G)]
    print("\nTop-10 pairs by flat cosine of dyn slots:")
    for i, j, fs, gs in sorted(pairs, key=lambda x: -x[2])[:10]:
        print(f"  {names[i]:32s} <-> {names[j]:32s}  flat={fs:+.3f}  greedy={gs:+.3f}")
    print("\nTop-10 pairs by greedy per-slot cosine:")
    for i, j, fs, gs in sorted(pairs, key=lambda x: -x[3])[:10]:
        print(f"  {names[i]:32s} <-> {names[j]:32s}  greedy={gs:+.3f}  flat={fs:+.3f}")

    # Overlay confusion margins from rule_inference.
    if args.ratio_json is None:
        guess = os.path.join(args.load, "rule_inference", "ratio.json")
        if os.path.exists(guess):
            args.ratio_json = guess
    overlay = None
    if args.ratio_json and os.path.exists(args.ratio_json):
        rj = json.load(open(args.ratio_json))
        loss_mat = np.array(rj["loss_mat_changed"])
        rj_targets = rj["target_names"]
        rj_cands = rj["cand_names"]
        rows = []
        for i, j, fs, gs in pairs:
            if names[i] not in rj_targets or names[j] not in rj_targets: continue
            if names[i] not in rj_cands or names[j] not in rj_cands: continue
            ti, tj = rj_targets.index(names[i]), rj_targets.index(names[j])
            ci, cj = rj_cands.index(names[i]), rj_cands.index(names[j])
            self_i, other_i = loss_mat[ti, ci], loss_mat[ti, cj]
            self_j, other_j = loss_mat[tj, cj], loss_mat[tj, ci]
            margin = (other_i - self_i) + (other_j - self_j)
            rows.append((i, j, fs, gs, margin))
        rows.sort(key=lambda x: x[4])  # smallest margin = most confused
        overlay = rows
        print(f"\n=== Slot-sim vs prediction-confusion ({args.ratio_json}) ===")
        print("Most prediction-confused pairs (smallest margin first):")
        print(f"  {'pair':<70s}  {'margin':>8s}  {'slot_flat':>9s}  {'slot_greedy':>11s}")
        for i, j, fs, gs, m in rows[:10]:
            pair = f"{names[i]} <-> {names[j]}"
            print(f"  {pair:<70s}  {m:+8.4f}  {fs:+9.3f}  {gs:+11.3f}")
        # Quick correlation: do slot-sim and confusion track?
        flat_arr = np.array([r[2] for r in rows])
        greedy_arr = np.array([r[3] for r in rows])
        margin_arr = np.array([r[4] for r in rows])
        # high slot-sim should correlate with low margin → negative correlation.
        if len(rows) > 2:
            r_flat = np.corrcoef(flat_arr, margin_arr)[0, 1]
            r_greedy = np.corrcoef(greedy_arr, margin_arr)[0, 1]
            print(f"\n  corr(slot_flat,   confusion_margin) = {r_flat:+.3f} "
                  f"  (negative = encoder explains confusion)")
            print(f"  corr(slot_greedy, confusion_margin) = {r_greedy:+.3f}")

    out_path = args.out
    if out_path is None:
        out_dir = os.path.join(args.load, "rule_inference")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "slot_sim.json")
    with open(out_path, "w") as f:
        json.dump({
            "names": names,
            "n_dyn": int(n_dyn),
            "d_slot": int(cfg["d_slot"]),
            "flat_sim": flat_sim.tolist(),
            "greedy_sim": greedy_sim.tolist(),
            "overlay": [
                {"i": int(i), "j": int(j),
                 "flat": float(fs), "greedy": float(gs), "margin": float(m)}
                for i, j, fs, gs, m in (overlay or [])
            ],
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, mat, title in zip(axes, [flat_sim, greedy_sim],
                              ["flat cosine (flatten K×d)", "greedy per-slot cosine"]):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(G)); ax.set_xticklabels(names, rotation=80, fontsize=8)
        ax.set_yticks(range(G)); ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"slot similarity: {title}")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    png = os.path.splitext(out_path)[0] + ".png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    print(f"Saved {png}")


if __name__ == "__main__":
    main()

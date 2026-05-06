"""Aggregate varislide depth sweep results and produce paper-ready figure.

For each (depth, seed) checkpoint in nca_wm/logs_canary/varislide_depth{D}_s{S}/:
  - Compute right-action change_err on the multi-grid synth dataset
  - Compute per-slide-distance argmax accuracy (the iteration metric)

Then plot:
  - Per-distance argmax accuracy vs depth, error bars over seeds
  - Right-action change_err vs depth, error bars over seeds

Saves both figures (PDF + PNG) and a summary CSV.

Usage:
    .venv/bin/python3 nca_wm/scripts/analyze_varislide_depth_sweep.py \\
        --runs_glob 'nca_wm/logs_canary/varislide_depth*_s*' \\
        --out_dir nca_wm/figures/varislide_depth_sweep
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import re
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
RIGHT = 3
PLAYER = 2

_TAG_RE = re.compile(r"varislide_depth(\d+)_s(\d+)")


def _build(cfg, gtoks_len, n_objs):
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg["d_model"], enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg["d_slot"], n_attn_heads=cfg["n_heads"],
        max_seq_len=max(gtoks_len, 1) + 1,
        axis_pool=cfg["axis_pool"], axis_cummax=cfg["axis_cummax"],
        global_pool=cfg["global_pool"],
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=cfg["n_nca_repeats"],
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def evaluate_run(run_dir):
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
    model = _build(cfg, toks_np.shape[0], C_pad)
    total_chg, wrong_on_chg = 0, 0
    total_pred, total_correct = 0, 0
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
        mask = a == RIGHT
        s, a, n = s[mask], a[mask], n[mask]
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
        preds = (sig > 0.5).astype(np.float32)
        sb_np = np.asarray(sb)[:, :, :lH, :lW]
        nb_np = np.asarray(nb)[:, :, :lH, :lW]
        preds = preds[:, :, :lH, :lW]
        chg = (sb_np != nb_np)
        total_chg += int(chg.sum())
        wrong_on_chg += int(((preds != nb_np) & chg).sum())
        sig_e = sig[:, :, :lH, :lW]
        for i in range(len(s)):
            in_p = np.argwhere(sb_np[i, PLAYER] > 0.5)
            out_p = np.argwhere(nb_np[i, PLAYER] > 0.5)
            if len(in_p) == 0 or len(out_p) == 0:
                continue
            ir, ic = int(in_p[0, 0]), int(in_p[0, 1])
            or_, oc = int(out_p[0, 0]), int(out_p[0, 1])
            if ir != or_ or oc == ic:
                continue
            row_sig = sig_e[i, PLAYER, or_, :]
            am = int(np.argmax(row_sig))
            d = oc - ic
            if d <= 0:
                continue
            total_pred += 1
            if am == oc:
                total_correct += 1
            dist_to_acc.setdefault(d, []).append(am == oc)
    return {
        "right_action_change_err": wrong_on_chg / max(total_chg, 1),
        "argmax_correct": total_correct / max(total_pred, 1),
        "per_distance_argmax_acc": {d: float(np.mean(v)) for d, v in dist_to_acc.items()},
        "per_distance_n": {d: len(v) for d, v in dist_to_acc.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_glob", default="nca_wm/logs_canary/varislide_depth*_s*")
    p.add_argument("--out_dir", default="nca_wm/figures/varislide_depth_sweep")
    args = p.parse_args()

    runs = sorted(glob.glob(args.runs_glob))
    if not runs:
        raise SystemExit(f"no runs match {args.runs_glob}")
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for r in runs:
        m = _TAG_RE.search(os.path.basename(r))
        if not m:
            continue
        depth, seed = int(m.group(1)), int(m.group(2))
        if not os.path.exists(os.path.join(r, "params.pkl")) and \
           not os.path.exists(os.path.join(r, "params_best.pkl")):
            print(f"  skip {r} (no params)")
            continue
        print(f"  evaluating {os.path.basename(r)} (depth={depth}, seed={seed})")
        try:
            res = evaluate_run(r)
        except Exception as e:
            print(f"    FAIL: {e}")
            continue
        rows.append({"depth": depth, "seed": seed,
                     "change_err": res["right_action_change_err"],
                     "argmax_acc": res["argmax_correct"],
                     "per_distance": res["per_distance_argmax_acc"]})

    if not rows:
        raise SystemExit("no successful evals")

    # Save CSV
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # Determine union of distances
        all_d = sorted(set(d for row in rows for d in row["per_distance"].keys()))
        w.writerow(["depth", "seed", "change_err", "argmax_acc"]
                   + [f"d{d}_acc" for d in all_d])
        for row in rows:
            w.writerow([row["depth"], row["seed"], f"{row['change_err']:.4f}",
                        f"{row['argmax_acc']:.4f}"]
                       + [f"{row['per_distance'].get(d, np.nan):.4f}" for d in all_d])
    print(f"  wrote {csv_path}")

    # Aggregate per depth
    depths = sorted(set(r["depth"] for r in rows))
    agg = {}
    for d in depths:
        d_rows = [r for r in rows if r["depth"] == d]
        agg[d] = {
            "change_err_mean": float(np.mean([r["change_err"] for r in d_rows])),
            "change_err_std": float(np.std([r["change_err"] for r in d_rows])),
            "argmax_mean": float(np.mean([r["argmax_acc"] for r in d_rows])),
            "argmax_std": float(np.std([r["argmax_acc"] for r in d_rows])),
            "per_d": {
                dist: ([r["per_distance"].get(dist, np.nan) for r in d_rows])
                for dist in all_d
            },
            "n_seeds": len(d_rows),
        }

    print("\n=== AGGREGATE ===")
    print(f"  depth   n_seeds  change_err (mean ± std)   argmax_acc (mean ± std)")
    for d in depths:
        a = agg[d]
        print(f"  {d:5d}      {a['n_seeds']}      {a['change_err_mean']:.3f} ± {a['change_err_std']:.3f}      "
              f"{a['argmax_mean']:.3f} ± {a['argmax_std']:.3f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "figure.dpi": 120, "savefig.bbox": "tight",
    })

    # Figure 1: argmax accuracy per slide distance vs depth
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for dist in all_d:
        means = [np.nanmean(agg[d]["per_d"][dist]) for d in depths]
        stds = [np.nanstd(agg[d]["per_d"][dist]) for d in depths]
        ax.errorbar(depths, means, yerr=stds, marker="o", label=f"slide_d={dist}",
                     capsize=4, linewidth=2, markersize=6)
    ax.set_xlabel("NCA depth (= n_steps = n_repeats)")
    ax.set_ylabel("Player-position argmax accuracy")
    ax.set_title("Per-slide-distance accuracy vs depth")
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Figure 2: overall change_err and overall argmax vs depth
    ax = axes[1]
    ce_means = [agg[d]["change_err_mean"] for d in depths]
    ce_stds = [agg[d]["change_err_std"] for d in depths]
    am_means = [agg[d]["argmax_mean"] for d in depths]
    am_stds = [agg[d]["argmax_std"] for d in depths]
    ax.errorbar(depths, ce_means, yerr=ce_stds, marker="s", color="tab:red",
                 label="right-action change_err (lower=better)", capsize=4, linewidth=2)
    ax.errorbar(depths, am_means, yerr=am_stds, marker="^", color="tab:green",
                 label="overall argmax_correct (higher=better)", capsize=4, linewidth=2)
    ax.set_xlabel("NCA depth")
    ax.set_ylabel("metric value")
    ax.set_title("Headline metrics vs depth")
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle("Multi-grid varislide depth sweep (shared weights, n_hid=128, 10k steps)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(args.out_dir, f"depth_sweep.{ext}")
        fig.savefig(path)
        print(f"  wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

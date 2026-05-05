"""Combine in-train BFS-oracle eval (eval_multigame.npz) with the held-out
synth + OOD eval (neko_eval.npz) into one summary table for the nekopuzzle
architecture sweep."""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
LOG_DIR = os.path.join(_REPO, "nca_wm/logs_neko_arch")
OUT_DIR = os.path.join(_REPO, "nca_wm/figures/neko_arch")


def _summarize_in_train(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    bfs = []; rnd = []; tf = []
    for li in range(50):
        k = f"nekopuzzle_L{li}_bfs_cell_error_rate"
        if k not in d.files:
            break
        bfs.append(float(d[k][-1]))
        rnd.append(float(d[f"nekopuzzle_L{li}_random_cell_error_rate"][-1]))
        tf.append(float(d[f"nekopuzzle_L{li}_random_tf_cell_error_rate"][-1]))
    return {"bfs_mean": float(np.mean(bfs)),
            "rnd_mean": float(np.mean(rnd)),
            "tf_mean": float(np.mean(tf)),
            "n_levels": len(bfs)}


def _summarize_synth_eval(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    rec = d["summary"].item()
    ho = rec.get("holdout", {})
    ood = rec.get("ood", {})
    ho_mean = float(np.mean([v["cell_err"] for v in ho.values()])) if ho else float("nan")
    ood_mean = float(np.mean([v["cell_err"] for v in ood.values()])) if ood else float("nan")
    return {"ho_mean": ho_mean, "ood_mean": ood_mean,
            "ho_per_size": ho, "ood_per_size": ood}


def main():
    runs = sorted(glob.glob(os.path.join(LOG_DIR, "neko_*_s0")))
    rows = []
    for r in runs:
        in_train = os.path.join(r, "eval_multigame.npz")
        synth_eval = os.path.join(r, "neko_eval.npz")
        if not os.path.exists(in_train):
            continue
        cfg = json.load(open(os.path.join(r, "config.json")))
        depth = cfg["n_nca_steps"]
        share = ("shared" if cfg.get("n_nca_repeats", depth) == depth
                  else "perstep")
        pool = "ON" if cfg.get("axis_pool", True) else "OFF"
        ie = _summarize_in_train(in_train)
        if os.path.exists(synth_eval):
            se = _summarize_synth_eval(synth_eval)
        else:
            se = {"ho_mean": float("nan"), "ood_mean": float("nan")}
        rows.append({
            "run": os.path.basename(r),
            "depth": depth, "share": share, "pool": pool,
            "bfs_authored": ie["bfs_mean"],
            "rnd_authored": ie["rnd_mean"],
            "tf_authored": ie["tf_mean"],
            "holdout_synth": se["ho_mean"],
            "ood_synth": se["ood_mean"],
        })

    rows.sort(key=lambda r: (r["depth"], r["share"], r["pool"]))
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "summary_combined.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {csv_path}")

    md_path = os.path.join(OUT_DIR, "summary_combined.md")

    def fmt(x): return "—" if (isinstance(x, float) and (x != x)) else f"{100*x:.2f}%"

    lines = [
        "# Nekopuzzle synth-arch sweep — combined summary\n",
        "Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, "
        "lr=3e-4, 15k updates, mask_hidden=True default, change_loss_weight=5.0, "
        "balanced_sampling. Synth training data: 64 levels (16 per size) at "
        "multi-grid {5x5, 6x6, 7x7, 8x8} = 2,130 transitions. Authored neko "
        "levels are 8x7, held-out from training.\n",
        "Eval columns:\n"
        "- **BFS authored** = final-step cell-error rate on BFS-solution rollouts on the 10 authored levels (in-train eval, gold standard)\n"
        "- **RAR authored** = final-step cell-error on random-action AR rollouts on authored levels\n"
        "- **TF authored** = final-step cell-error with teacher-forcing on authored levels\n"
        "- **Holdout synth** = single-step cell-error on a fresh synth pool (different seed) at trained sizes (5x5–8x8)\n"
        "- **OOD synth** = single-step cell-error on synth pool at 9x9 (size-OOD, larger than max trained size)\n",
        "| depth | share | pool | BFS authored | RAR authored | TF authored | Holdout synth | OOD synth |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['depth']} | {r['share']} | {r['pool']} | "
            f"{fmt(r['bfs_authored'])} | {fmt(r['rnd_authored'])} | "
            f"{fmt(r['tf_authored'])} | {fmt(r['holdout_synth'])} | "
            f"{fmt(r['ood_synth'])} |"
        )

    # Best-by-metric summary
    lines.append("\n## Best per metric\n")
    for metric, label in [("bfs_authored", "BFS authored"),
                            ("rnd_authored", "Random-AR authored"),
                            ("tf_authored", "TF authored"),
                            ("holdout_synth", "Holdout synth"),
                            ("ood_synth", "OOD synth")]:
        best = min(rows, key=lambda r: r[metric])
        lines.append(f"- **{label}**: d={best['depth']} / {best['share']} / pool={best['pool']} → {fmt(best[metric])}")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-heldout-game scatter for the conditional / unconditional Train-199
matched-recipe checkpoints.

Two-panel figure: left = Train-199 unconditional, right = Train-199
conditional. Each game is a point at (identity TF cell-err, model TF
cell-err). The diagonal is the "ties identity" line; points below it
are wins. Truncated-rule games (rule-token count > model max_seq_len)
are highlighted: the conditional encoder gets only the first 657
tokens of these.

Outputs:
  nca_wm/paper/figures/heldout_truncation/heldout_per_game.{pdf,png}

Usage:
    .venv/bin/python3 nca_wm/scripts/plot_heldout_per_game.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / "nca_wm" / "logs"
OUT_DIR = REPO_ROOT / "nca_wm" / "paper" / "figures" / "heldout_truncation"

TRUNCATED = {
    "angize_by_ali_nikkhah",
    "break_out_of_the_mine_by_jja_i.e._juan,_jose_&_andre",
    "headless_people_problems_by_monakrom",
    "Heroes_of_Sokoban_-_Ancient_Japan",
}

PANELS = [
    ("Train-199 unconditional",
     LOGS_ROOT / "multi_scaling_gallery_v4_uncond_match_s0" / "heldout_v4_n30" / "results.json",
     "#1f77b4"),
    ("Train-199 conditional",
     LOGS_ROOT / "multi_scaling_gallery_v4_cond_match_s0" / "heldout_v4_n30" / "results.json",
     "#d62728"),
]

RC_PARAMS = {
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


def per_game_tf(path: Path):
    if not path.exists():
        return {}, {}
    r = json.loads(path.read_text())
    out_m, out_i = {}, {}
    for g, levels in r.get("heldout", {}).items():
        ms, ids = [], []
        for li, kinds in levels.items():
            tf = kinds.get("random_tf") or {}
            mp = tf.get("model_cell_err_per_step") or []
            ip = tf.get("identity_cell_err_per_step") or []
            if mp and ip:
                ms.append(float(np.mean(mp)))
                ids.append(float(np.mean(ip)))
        if ms:
            out_m[g] = float(np.mean(ms))
            out_i[g] = float(np.mean(ids))
    return out_m, out_i


def _short(name: str) -> str:
    """Trim "_by_<author>" suffix and underscores for nicer labels."""
    n = name
    for suf in ("_by_increpare", "_by_henry-friedman", "_by_jeffjeff123456"):
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    if "_by_" in n:
        n = n.split("_by_")[0]
    return n.replace("_", " ").strip()


def panel(ax, title, path, color):
    m, i = per_game_tf(path)
    if not m:
        ax.set_title(f"{title} (no data)")
        return 0, 0
    games = sorted(m)
    xs = np.array([100 * i[g] for g in games])
    ys = np.array([100 * m[g] for g in games])
    is_trunc = np.array([g in TRUNCATED for g in games])
    wins = int(np.sum(ys < xs))
    n = len(games)

    # Diagonal first (in background).
    lo, hi = 0.05, 100.0
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle=":", linewidth=1.0,
            label="identity")

    # Non-truncated games.
    sel = ~is_trunc
    ax.scatter(xs[sel], ys[sel], s=42, color=color, edgecolor="white",
               linewidth=0.5, label=f"rule fits ($n={int(sel.sum())}$)",
               zorder=3)

    # Truncated games (open marker).
    sel = is_trunc
    if sel.any():
        ax.scatter(xs[sel], ys[sel], s=70, facecolor="none",
                   edgecolor=color, linewidth=1.4, marker="o",
                   label=f"rule truncated ($n={int(sel.sum())}$)", zorder=4)
        # Annotate truncated games inline so readers can see who they are.
        for x, y, g in zip(xs[sel], ys[sel], np.array(games)[sel]):
            ax.annotate(_short(g), (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7,
                        color=color, alpha=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("identity 1-step (TF) cell-error (%)")
    ax.set_ylabel("model 1-step (TF) cell-error (%)")
    ax.set_title(f"{title}: wins {wins}/{n}", fontsize=12)
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    return wins, n


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(RC_PARAMS)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
    info = []
    for ax, (title, path, color) in zip(axes, PANELS):
        info.append((title, *panel(ax, title, path, color)))
    fig.suptitle(
        "Heldout-30: per-game model vs. identity (1-step TF). "
        "Below diagonal = win.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf = OUT_DIR / "heldout_per_game.pdf"
    png = OUT_DIR / "heldout_per_game.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print("Wrote:")
    for p in (pdf, png):
        print(f"  {p}")
    for title, w, n in info:
        print(f"  {title}: {w}/{n} wins")


if __name__ == "__main__":
    main()

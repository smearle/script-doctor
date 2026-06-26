"""Regenerate nca_wm/figures/recurrence_heatmap.{png,pdf}.

Reads OOD held-out-game change_err from the recurrent-NCA run dirs and renders
a (context-length k) x (corpus, in-dist/OOD) heatmap. Cells with no completed
run are drawn white and labeled "not run".

Run dirs (each holds a config.json written by train_recurrent):
  800-game curated corpus (787 games, 18 obj channels, 20k transitions/game):
    nca_wm/logs/ood_memory/k0_800           (k=0)
    nca_wm/logs/ood_recurrent/k16w4_b8_s0   (k=16, from torch)
    nca_wm/logs/ood_recurrent/k32w4_b8_s0   (k=32, from torch)
  diverse 2072-game corpus (52 obj channels, 5k transitions/game):
    nca_wm/logs/ood_memory/k0_full          (k=0)
    (k=16/k=32 not yet completed -- forward-unroll compile bottleneck)

OOD column = best_ood_change_err (model-selection metric).
in-dist column = final_in_dist_change_err (in-dist at best checkpoint is not
stored, so the final value is used; flagged in the title).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(rel):
    p = os.path.join(REPO, rel, "config.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))
    if "best_ood_change_err" not in c:
        return None  # run never finished an eval
    return c


# (row label, [in-dist run dir, OOD run dir]) per (k, corpus) cell.
KS = [0, 16, 32]
CORPORA = [
    ("800\ncurated", {0: "nca_wm/logs/ood_memory/k0_800",
                      16: "nca_wm/logs/ood_recurrent/k16w4_b8_s0",
                      32: "nca_wm/logs/ood_recurrent/k32w4_b8_s0"}),
    ("diverse\n2072", {0: "nca_wm/logs/ood_memory/k0_full",
                       16: None, 32: None}),
]

# Build a (n_k) x (2 corpora * 2 metrics) grid.
col_labels = []
for cname, _ in CORPORA:
    col_labels += [f"{cname}\nin-dist", f"{cname}\nOOD"]

vals = np.full((len(KS), len(col_labels)), np.nan)
texts = [["not run"] * len(col_labels) for _ in KS]

for ci, (cname, dirs) in enumerate(CORPORA):
    for ki, k in enumerate(KS):
        d = dirs.get(k)
        c = _load(d) if d else None
        if c is None:
            continue
        indist = c.get("final_in_dist_change_err")
        ood = c.get("best_ood_change_err")
        vals[ki, 2 * ci] = indist
        vals[ki, 2 * ci + 1] = ood
        texts[ki][2 * ci] = f"{indist:.3f}"
        texts[ki][2 * ci + 1] = f"{ood:.3f}"

# Mark the per-corpus best k (lowest OOD) in bold.
best = {}
for ci in range(len(CORPORA)):
    col = vals[:, 2 * ci + 1]
    if np.isfinite(col).any():
        best[ci] = int(np.nanargmin(col))

fig, ax = plt.subplots(figsize=(8.5, 5.0))
masked = np.ma.masked_invalid(vals)
cmap = plt.cm.RdYlGn_r.copy()
cmap.set_bad("white")
im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=0.7, aspect="auto")

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(KS)))
ax.set_yticklabels([f"k={k}\n(memoryless)" if k == 0 else f"k={k}" for k in KS],
                   fontsize=11)

for ki in range(len(KS)):
    for cj in range(len(col_labels)):
        t = texts[ki][cj]
        is_best = (cj % 2 == 1) and best.get(cj // 2) == ki and np.isfinite(vals[ki, cj])
        color = "0.55" if t == "not run" else "black"
        ax.text(cj, ki, t, ha="center", va="center", fontsize=12,
                color=color, fontweight="bold" if is_best else "normal")

ax.set_title("Recurrent memory (context length k) effect on change_err\n"
             "OOD = best on held-out games never trained on; lower is better",
             fontsize=12)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("change_err", fontsize=11)
fig.tight_layout()

for ext in ("png", "pdf"):
    out = os.path.join(REPO, "nca_wm", "figures", f"recurrence_heatmap.{ext}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

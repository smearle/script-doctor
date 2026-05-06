r"""Regenerate paper artifacts for the within-game-synth result section.

Reads completed runs under nca_wm/logs_per_game_arch_synth/ and writes:

  nca_wm/paper/figures/per_game_arch_synth/
    anchor_microban_d.tex       — single-row table for the Microban-D anchor
    within_game_synth_table.tex — full per-game × per-bucket BFS / TF table
    heatmap_bfs.{pdf,png}       — 2-panel heatmap (BFS / 1-step TF) for inclusion
    summary.csv                 — raw numbers (also useful for sanity checks)

Each .tex file is a self-contained \begin{tabular}…\end{tabular} block (no
\caption, no \label) so the paper can wrap it in a `\begin{table}` and supply
those itself; this keeps the surrounding caption text under the human's
control while the numbers stay scriptable.

Run lookup is by directory name. The anchor run is matched by exact suffix
``Microban__D_d8_w7_k128_30k``. Sweep cells are matched by the bucket-grid
pattern ``<game>__<A|B|C|D>_d<DEPTH>``.

Usage:
    .venv/bin/python3 nca_wm/scripts/make_within_game_synth_artifacts.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
_LOGDIR = os.path.join(_REPO, "nca_wm", "logs_per_game_arch_synth")
_PAPER_FIG_DIR = os.path.join(_REPO, "nca_wm", "paper", "figures", "per_game_arch_synth")

_ANCHOR_RUN = "Microban__D_d8_w7_k128_30k"
_BUCKETS = ["A", "B", "C", "D"]
_GAMES_ORDER = [
    "Microban", "Heroes_of_Sokoban", "Bouncers", "nekopuzzle",
    "Travelling_salesman",
]

_BUCKET_LABELS = {
    "A": "A: pool/shared",
    "B": "B: pool/per-step",
    "C": "C: skip/shared",
    "D": "D: skip/per-step",
}

_RUN_RE = re.compile(r"^(?P<game>.+)__(?P<bucket>[A-D])_d(?P<depth>\d+)$")


def _read_eval(run_dir):
    """Return mean BFS / A* / random / random_tf cell-error across authored
    levels of the (single) game in this run, plus per-level BFS list."""
    p = os.path.join(run_dir, "eval_multigame.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    pat = re.compile(
        r"^(?P<game>.+)_L(?P<lvl>\d+)_(?P<kind>bfs|astar|random|random_tf)_cell_error_rate$"
    )
    by_kind = defaultdict(list)
    for k in z.files:
        m = pat.match(k)
        if not m:
            continue
        v = np.asarray(z[k]).astype(float)
        if v.size == 0 or not np.isfinite(v).any():
            continue
        by_kind[m.group("kind")].append((int(m.group("lvl")), float(np.nanmean(v))))
    out = {}
    for kind in ("bfs", "astar", "random", "random_tf"):
        vals = sorted(by_kind[kind])
        if not vals:
            continue
        out[f"{kind}_mean"] = float(np.mean([v for _, v in vals]))
        out[f"{kind}_per_level"] = [v for _, v in vals]
    return out


def _read_train(run_dir):
    curves = sorted(
        glob.glob(os.path.join(run_dir, "curves_step*.npz")),
        key=lambda p: int(os.path.basename(p)[len("curves_step"):-len(".npz")]),
    )
    if not curves:
        return {}
    z = np.load(curves[-1])
    losses = z.get("losses")
    if losses is None or losses.size == 0:
        return {}
    n = int(losses.size)
    out = {
        "n_steps": n,
        "best_loss": float(losses.min()),
        "final_loss": float(losses[-1]),
    }
    if n >= 200:
        prev = float(np.mean(losses[int(n * 0.50): int(n * 0.75)]))
        tail = float(np.mean(losses[int(n * 0.75):]))
        out["loss_ratio_q3_to_q4"] = prev / tail if tail > 1e-30 else float("nan")
    return out


def _read_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        return json.load(open(cfg_path))
    except Exception:
        return {}


def _is_complete(run_dir):
    return (
        os.path.exists(os.path.join(run_dir, "params.pkl"))
        and os.path.exists(os.path.join(run_dir, "train_meta.json"))
        and os.path.exists(os.path.join(run_dir, "eval_multigame.npz"))
    )


# ---------------------------------------------------------------------------
# Anchor run table
# ---------------------------------------------------------------------------

def write_anchor_tex(anchor_dir, out_tex):
    ev = _read_eval(anchor_dir)
    tr = _read_train(anchor_dir)
    cfg = _read_config(anchor_dir)
    if ev is None:
        with open(out_tex, "w") as f:
            f.write(
                "% Anchor run not present: " + anchor_dir + "\n"
                "% Run nca_wm/scripts/run_per_game_arch_synth_grid.sh first.\n"
            )
        print(f"  [anchor] no eval; wrote stub to {out_tex}")
        return
    n_levels = len(ev.get("bfs_per_level", []))
    sw = int(cfg.get("synthetic_w", 0))
    sh = int(cfg.get("synthetic_h", 0))
    sk = int(cfg.get("synthetic_levels", 0))
    nu = int(tr.get("n_steps", 0))
    body = [
        "% Auto-generated by nca_wm/scripts/make_within_game_synth_artifacts.py",
        "% Anchor run: " + os.path.basename(anchor_dir),
        "\\begin{tabular}{l c c c c}",
        "  \\toprule",
        "  Setting & BFS mean & A* mean & random AR & 1-step TF \\\\",
        "  \\midrule",
        f"  Microban, bucket D, ${sw}{{\\times}}{sh}$ synth, $K{{=}}{sk}$, ${nu/1000:.0f}$k updates "
        f"& ${ev.get('bfs_mean', float('nan'))*100:.2f}\\%$ "
        f"& ${ev.get('astar_mean', float('nan'))*100:.2f}\\%$ "
        f"& ${ev.get('random_mean', float('nan'))*100:.2f}\\%$ "
        f"& $\\mathbf{{{ev.get('random_tf_mean', float('nan'))*100:.2f}\\%}}$ \\\\",
        "  \\bottomrule",
        "\\end{tabular}",
    ]
    with open(out_tex, "w") as f:
        f.write("\n".join(body) + "\n")
    n_zero = sum(1 for v in ev.get("bfs_per_level", []) if v == 0.0)
    print(
        f"  [anchor] wrote {out_tex}: BFS={ev['bfs_mean']*100:.2f}%  "
        f"TF={ev['random_tf_mean']*100:.2f}%  "
        f"({n_zero}/{n_levels} levels at 0% BFS)"
    )


# ---------------------------------------------------------------------------
# Sweep table & heatmap
# ---------------------------------------------------------------------------

def _gather_sweep_cells(include_anchor_as_microban_d=True):
    """Return {(game, bucket): {bfs_mean, random_tf_mean, ...}}.

    The anchor run lives at a non-standard path
    (``Microban__D_d8_w7_k128_30k``) but reflects the validated recipe;
    when ``include_anchor_as_microban_d`` is true and no standard
    ``Microban__D_d8`` cell exists, we slot the anchor into that
    (Microban, D) row of the sweep table.
    """
    rows = {}
    for d in sorted(glob.glob(os.path.join(_LOGDIR, "*"))):
        if not os.path.isdir(d) or not _is_complete(d):
            continue
        name = os.path.basename(d)
        m = _RUN_RE.match(name)
        if not m:
            continue
        ev = _read_eval(d)
        tr = _read_train(d)
        if ev is None:
            continue
        rows[(m.group("game"), m.group("bucket"))] = {
            "run": name,
            "bfs_mean": ev.get("bfs_mean", float("nan")),
            "astar_mean": ev.get("astar_mean", float("nan")),
            "random_mean": ev.get("random_mean", float("nan")),
            "random_tf_mean": ev.get("random_tf_mean", float("nan")),
            "best_loss": tr.get("best_loss", float("nan")),
            "loss_ratio_q3_to_q4": tr.get("loss_ratio_q3_to_q4", float("nan")),
            "n_train_steps": tr.get("n_steps"),
        }
    if include_anchor_as_microban_d and ("Microban", "D") not in rows:
        anchor_dir = os.path.join(_LOGDIR, _ANCHOR_RUN)
        if _is_complete(anchor_dir):
            ev = _read_eval(anchor_dir)
            tr = _read_train(anchor_dir)
            if ev is not None:
                rows[("Microban", "D")] = {
                    "run": _ANCHOR_RUN + " (anchor)",
                    "bfs_mean": ev.get("bfs_mean", float("nan")),
                    "astar_mean": ev.get("astar_mean", float("nan")),
                    "random_mean": ev.get("random_mean", float("nan")),
                    "random_tf_mean": ev.get("random_tf_mean", float("nan")),
                    "best_loss": tr.get("best_loss", float("nan")),
                    "loss_ratio_q3_to_q4": tr.get("loss_ratio_q3_to_q4", float("nan")),
                    "n_train_steps": tr.get("n_steps"),
                }
    return rows


def write_sweep_tex(rows, out_tex):
    games_present = [g for g in _GAMES_ORDER if any((g, b) in rows for b in _BUCKETS)]
    if not games_present:
        with open(out_tex, "w") as f:
            f.write(
                "% No completed sweep cells under nca_wm/logs_per_game_arch_synth/\n"
                "% (anchor run is excluded — it has a non-standard run-name suffix.)\n"
            )
        print(f"  [sweep] no cells; wrote stub to {out_tex}")
        return

    def _fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"${v*100:.2f}\\%$"

    lines = [
        "% Auto-generated by nca_wm/scripts/make_within_game_synth_artifacts.py",
        "\\begin{tabular}{l c c c c}",
        "  \\toprule",
        "  Game & " + " & ".join(_BUCKET_LABELS[b] for b in _BUCKETS) + " \\\\",
        "  \\midrule",
    ]
    # Pick the row metric per cell. We report BFS mean as the primary, with a
    # tiebreaker — bold the lowest BFS in each row.
    for g in games_present:
        cells_bfs = [rows.get((g, b), {}).get("bfs_mean", float("nan")) for b in _BUCKETS]
        finite = [c for c in cells_bfs if np.isfinite(c)]
        bestv = min(finite) if finite else None
        cell_strs = []
        for b, v in zip(_BUCKETS, cells_bfs):
            s = _fmt(v)
            if bestv is not None and np.isfinite(v) and v == bestv:
                s = "$\\mathbf{" + s.strip("$") + "}$"
            cell_strs.append(s)
        nice_g = "\\textsc{" + g.replace("_", "\\_") + "}"
        lines.append(f"  {nice_g} & " + " & ".join(cell_strs) + " \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [sweep] wrote {out_tex} ({len(games_present)} game rows)")


def write_sweep_csv(rows, out_csv):
    fieldnames = ["game", "bucket", "run", "bfs_mean", "astar_mean",
                  "random_mean", "random_tf_mean", "best_loss",
                  "loss_ratio_q3_to_q4", "n_train_steps"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (g, b), r in sorted(rows.items()):
            row = {"game": g, "bucket": b}
            row.update({k: r.get(k) for k in fieldnames if k not in row})
            w.writerow(row)
    print(f"  [csv] wrote {out_csv} ({len(rows)} rows)")


def write_sweep_heatmap(rows, out_pdf, out_png):
    games_present = [g for g in _GAMES_ORDER if any((g, b) in rows for b in _BUCKETS)]
    if not games_present:
        print("  [heatmap] no cells; skipped")
        return
    M_bfs = np.full((len(games_present), len(_BUCKETS)), np.nan)
    M_tf = np.full((len(games_present), len(_BUCKETS)), np.nan)
    for i, g in enumerate(games_present):
        for j, b in enumerate(_BUCKETS):
            r = rows.get((g, b))
            if r is not None:
                M_bfs[i, j] = r.get("bfs_mean", np.nan)
                M_tf[i, j] = r.get("random_tf_mean", np.nan)

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.constrained_layout.use": True,
    })
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#dddddd")
    finite = np.concatenate([M_bfs[np.isfinite(M_bfs)], M_tf[np.isfinite(M_tf)]])
    vmax = max(min(float(np.nanmax(finite)) if finite.size else 0.05, 0.10), 0.02)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9, max(3.0, 0.55 * len(games_present) + 1.5)))

    def _draw(ax, M, title):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                txt = "—" if not np.isfinite(v) else f"{v*100:.2f}%"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
        ax.set_xticks(range(len(_BUCKETS)))
        ax.set_xticklabels([_BUCKET_LABELS[b].replace(": ", ":\n") for b in _BUCKETS])
        ax.set_yticks(range(len(games_present)))
        ax.set_yticklabels(games_present)
        ax.set_title(title)
        return im

    _draw(axL, M_bfs, "BFS rollout cell-error")
    im1 = _draw(axR, M_tf, "1-step teacher-forced cell-error")
    cbar = fig.colorbar(im1, ax=[axL, axR], shrink=0.85, pad=0.02)
    cbar.set_label(f"cell-error (capped at {vmax*100:.0f}%)")
    fig.suptitle("Synth-only $\\to$ all-authored generalization "
                 "(per-game $\\times$ per-architecture, $T{=}8$)")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [heatmap] wrote {out_pdf} (+ .png)  vmax={vmax*100:.0f}%")


def main():
    os.makedirs(_PAPER_FIG_DIR, exist_ok=True)

    print("== Anchor run ==")
    anchor_dir = os.path.join(_LOGDIR, _ANCHOR_RUN)
    write_anchor_tex(
        anchor_dir,
        os.path.join(_PAPER_FIG_DIR, "anchor_microban_d.tex"),
    )

    print("\n== Sweep cells ==")
    rows = _gather_sweep_cells()
    write_sweep_csv(rows, os.path.join(_PAPER_FIG_DIR, "summary.csv"))
    write_sweep_tex(rows, os.path.join(_PAPER_FIG_DIR, "within_game_synth_table.tex"))
    write_sweep_heatmap(
        rows,
        os.path.join(_PAPER_FIG_DIR, "heatmap_bfs.pdf"),
        os.path.join(_PAPER_FIG_DIR, "heatmap_bfs.png"),
    )


if __name__ == "__main__":
    main()

"""Analyze the use_layernorm ablation (logs/ln_ablation/<cfg>/).

For each config it reads:
  - curves_step*.npz  -> val_change_acc trajectory (val_change_err = 1 - acc),
    final train change_acc, best val_change_err, and convergence step
    (first val step at which val_change_err <= EPS).
  - eval_multigame.npz -> per-level bfs / astar / random(-AR) cell error,
    averaged over levels.
  - train_meta.json    -> best_loss.

Emits a markdown comparison table (LN off vs on within each game/depth/regime
group) and paper-ready convergence + final-metric figures (PDF+PNG, big fonts).

Usage: .venv/bin/python3 nca_wm/scripts/analyze_ln_ablation.py
"""
from __future__ import annotations
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGDIR = os.path.join(REPO, "nca_wm", "logs", "ln_ablation")
OUTDIR = os.path.join(REPO, "nca_wm", "figures", "ln_ablation")
EPS = 1e-4  # "perfect" threshold on val change_err

# (group_key, depth, ln) -> cfg name. group_key is game+regime.
GROUPS = {
    "TSM pool-ON+skip": [("tsm_d8_lnoff", 8, False), ("tsm_d8_lnon", 8, True),
                          ("tsm_d32_lnoff", 32, False), ("tsm_d32_lnon", 32, True)],
    "THL pool-ON+skip": [("thl_d8_lnoff", 8, False), ("thl_d8_lnon", 8, True),
                          ("thl_d32_lnoff", 32, False), ("thl_d32_lnon", 32, True)],
    "TSM pool-OFF+skip (d32)": [("tsm_pooloff_skipon_d32_lnoff", 32, False),
                                 ("tsm_pooloff_skipon_d32_lnon", 32, True)],
    "TSM pool-OFF,no-skip (d32)": [("tsm_pooloff_skipoff_d32_lnoff", 32, False),
                                    ("tsm_pooloff_skipoff_d32_lnon", 32, True)],
}


def load_cfg(cfg):
    d = os.path.join(LOGDIR, cfg)
    if not os.path.isdir(d):
        return None
    out = {"cfg": cfg}
    curves = sorted(glob.glob(os.path.join(d, "curves_step*.npz")),
                    key=lambda p: int(p.split("curves_step")[1].split(".npz")[0]))
    if not curves:
        return None
    c = np.load(curves[-1])
    vstep = c["val_step"]
    vcerr = 1.0 - c["val_change_acc"]
    out["vstep"] = vstep
    out["vcerr"] = vcerr
    out["best_val_cerr"] = float(np.nanmin(vcerr)) if len(vcerr) else float("nan")
    out["final_val_cerr"] = float(vcerr[-1]) if len(vcerr) else float("nan")
    out["final_train_cerr"] = float(1.0 - c["change_accs"][-1])
    conv = vstep[vcerr <= EPS]
    out["conv_step"] = int(conv[0]) if len(conv) else None
    meta_p = os.path.join(d, "train_meta.json")
    out["best_loss"] = (json.load(open(meta_p)).get("best_loss")
                        if os.path.isfile(meta_p) else None)
    # rollout eval, averaged over levels
    ev_p = os.path.join(d, "eval_multigame.npz")
    for key, dst in [("bfs_cell_error_rate", "bfs"),
                     ("astar_cell_error_rate", "astar"),
                     ("random_cell_error_rate", "rand_ar")]:
        out[dst] = float("nan")
    if os.path.isfile(ev_p):
        ev = np.load(ev_p, allow_pickle=True)
        for key, dst in [("bfs_cell_error_rate", "bfs"),
                         ("astar_cell_error_rate", "astar"),
                         ("random_cell_error_rate", "rand_ar")]:
            # each key is a per-rollout-step array for one level; reduce to a
            # per-level mean, then average across levels.
            per_level = [float(np.mean(ev[k])) for k in ev.files
                         if k.endswith("_" + key) and ev[k].size]
            if per_level:
                out[dst] = float(np.mean(per_level))
    return out


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.3f}%"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    loaded = {}
    for g, cells in GROUPS.items():
        for cfg, depth, ln in cells:
            loaded[cfg] = load_cfg(cfg)

    # ---- markdown table ----
    lines = ["# use_layernorm ablation — results\n",
             f"(`val change_err` = held-out-transition 1-step changed-cell error; "
             f"`conv@` = first val step with val change_err <= {EPS}; rollout = "
             f"mean over levels of AR cell-error.)\n",
             "| group | depth | LN | best val cerr | final train cerr | conv@ | "
             "best_loss | bfs | astar | rand-AR |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for g, cells in GROUPS.items():
        for cfg, depth, ln in cells:
            r = loaded.get(cfg)
            if r is None:
                lines.append(f"| {g} | {depth} | {'on' if ln else 'off'} | "
                             f"_pending_ | | | | | | |")
                continue
            bl = (f"{r['best_loss']:.2e}" if r['best_loss'] is not None else "n/a")
            lines.append(
                f"| {g} | {depth} | {'**on**' if ln else 'off'} | "
                f"{pct(r['best_val_cerr'])} | {pct(r['final_train_cerr'])} | "
                f"{r['conv_step'] if r['conv_step'] is not None else 'never'} | "
                f"{bl} | {pct(r['bfs'])} | {pct(r['astar'])} | {pct(r['rand_ar'])} |")
    table = "\n".join(lines)
    with open(os.path.join(OUTDIR, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)

    # ---- convergence figure: one panel per group, LN off vs on ----
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16,
                         "axes.titlesize": 16, "legend.fontsize": 13})
    groups = list(GROUPS.keys())
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    floor = 5e-5
    depth_style = {8: "-", 32: "--"}
    ln_color = {False: "#1f77b4", True: "#d62728"}
    for ax, g in zip(axes, groups):
        for cfg, depth, ln in GROUPS[g]:
            r = loaded.get(cfg)
            if r is None or "vstep" not in r:
                continue
            y = np.maximum(r["vcerr"], floor)
            ax.plot(r["vstep"], y, depth_style.get(depth, "-"),
                    color=ln_color[ln], lw=2,
                    label=f"d{depth} LN {'on' if ln else 'off'}")
        ax.set_yscale("log")
        ax.set_title(g)
        ax.set_xlabel("train step")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
    axes[0].set_ylabel("val change_err (held-out transitions)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"convergence.{ext}"), bbox_inches="tight")
    print(f"\nwrote {OUTDIR}/convergence.{{pdf,png}} and summary.md")


if __name__ == "__main__":
    main()

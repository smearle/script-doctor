"""Per-level halt-distribution plot for the within-game adaptive-halt test.

Loads a single trained adaptive-halt run that was trained on multiple
levels of one game (e.g. varislide) and produces:

  - One panel per level showing learned p_k as bars
  - A scatter/line summary of E[halt step] vs level index (paper headline)
  - Train-loss curve as a small inset

Levels are pulled from the per-level cache directories under
`rollout_data/<game>/level_<i>/`. The plot's level ordering follows
numeric level index. If level metadata (e.g. slide distance) is known
from the game definition, pass --level_metric "name1=val1,name2=val2"
to use those as the x-axis instead of raw level index.

Usage:
    python -m nca_wm.scripts.plot_per_level_halt \
        --run nca_wm/logs_halt_arch/varislide_halt_seed0 \
        --out nca_wm/logs_halt_arch/varislide_per_level_figure
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
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
ACTION_NAMES = ["up", "down", "left", "right", "action"]


def _build_model(cfg: dict, gtoks_len: int, n_objs: int):
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg["d_model"], enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg["d_slot"], n_attn_heads=cfg["n_heads"],
        max_seq_len=gtoks_len + 1,
        axis_pool=cfg["axis_pool"], axis_cummax=cfg["axis_cummax"],
        global_pool=cfg["global_pool"],
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=cfg["n_nca_repeats"],
        adaptive_halt=cfg["adaptive_halt"],
    )


def _halt_dist(halt_logits: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam = jax.nn.sigmoid(halt_logits)
    log_omlam = jnp.log(jnp.clip(1.0 - lam, 1e-6, 1.0))
    B = lam.shape[1]
    surv = jnp.exp(jnp.concatenate(
        [jnp.zeros((1, B)), jnp.cumsum(log_omlam, axis=0)[:-1]], axis=0))
    p = lam * surv
    last_surv = jnp.exp(jnp.sum(log_omlam[:-1], axis=0))
    p = p.at[-1].set(last_surv)
    return np.asarray(p.mean(axis=1)), np.asarray(p)


def analyze_run(run_dir: str, action_filter: int | None = None,
                batch_size: int = 64) -> dict:
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p_path = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p_path):
        p_path = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p_path, "rb"))

    g = gi[0]
    game = g["name"]
    level_dirs = sorted(glob.glob(os.path.join(ROLLOUT_CACHE_DIR, game, "level_*")),
                        key=lambda p: int(p.rsplit("_", 1)[-1]))
    print(f"  found {len(level_dirs)} levels for {game}")

    embed_in = params["params"]["embed"]["kernel"].shape[0]
    C_pad = embed_in - N_ACTIONS

    # CRITICAL: pad to the *training* max shape, not an arbitrary floor.
    # Training batches all transitions of a game to the game's native (H, W);
    # padding analysis inputs to a different shape gives the model an OOD
    # input it was never trained on, and predictions degrade catastrophically.
    # game_infos[0]["H"]/["W"] hold the per-game native max from the loader.
    max_H = g["H"]
    max_W = g["W"]

    toks_np = np.asarray(g["token_ids"])
    model = _build_model(cfg, toks_np.shape[0], C_pad)

    per_level = []
    for ld in level_dirs:
        level_i = int(ld.rsplit("_", 1)[-1])
        npzs = sorted(glob.glob(os.path.join(ld, "*transitions*.npz")),
                      key=lambda p: -os.path.getsize(p))
        if not npzs:
            per_level.append(dict(level=level_i, p_marg=None, n=0,
                                  p_marg_per_action=None))
            continue
        data = np.load(npzs[0])
        lW = int(data["W"])
        # Unpack bit-packed cache (states are np.packbits-compressed along W).
        s_np = _unpack_states(data["states"], lW).astype(np.float32)
        n_np = _unpack_states(data["next_states"], lW).astype(np.float32)
        a_np = data["actions"]
        if action_filter is not None:
            mask = a_np == action_filter
            s_np, a_np, n_np = s_np[mask], a_np[mask], n_np[mask]
        if s_np.shape[0] == 0:
            per_level.append(dict(level=level_i, p_marg=None, n=0,
                                  p_marg_per_action=None))
            continue
        B = min(batch_size, s_np.shape[0])
        s_pad = np.pad(
            s_np[:B],
            [(0, 0), (0, C_pad - s_np.shape[1]),
             (0, max_H - s_np.shape[2]), (0, max_W - s_np.shape[3])],
        )
        sb = jnp.asarray(s_pad)
        ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[a_np[:B]])
        gtoks = jnp.asarray(np.tile(toks_np[None], (B, 1)))
        gmask = jnp.ones_like(gtoks, dtype=bool)
        out = model.apply(params, sb, ab, gtoks, gmask)
        halt_aux = out[-1]
        ps_halt = halt_aux[2]
        p_marg, _p_full = _halt_dist(ps_halt)

        # Per-action breakdown when action_filter is None
        per_action = {}
        for act in range(N_ACTIONS):
            sel = (a_np[:B] == act)
            if sel.sum() == 0:
                per_action[act] = None
                continue
            sub_logits = jnp.asarray(np.asarray(ps_halt)[:, sel])
            sub_p, _ = _halt_dist(sub_logits)
            per_action[act] = sub_p
        per_level.append(dict(level=level_i, p_marg=p_marg, n=B,
                              p_marg_per_action=per_action))

    curves = sorted(glob.glob(os.path.join(run_dir, "curves_step*.npz")))
    losses = np.asarray(np.load(curves[-1])["losses"]) if curves else None

    return dict(cfg=cfg, game=game, per_level=per_level, losses=losses,
                run_dir=run_dir)


def plot_figure(result: dict, level_metric: list[float] | None,
                metric_name: str, out_base: str) -> None:
    cfg = result["cfg"]
    levels = result["per_level"]
    valid = [l for l in levels if l["p_marg"] is not None]
    if not valid:
        print("[plot] no valid levels — nothing to render")
        return
    n_lvl = len(valid)
    T = len(valid[0]["p_marg"])
    ks = np.arange(1, T + 1)

    # Two-panel figure: (top row) per-level halt-distribution panels,
    # (bottom) scatter of E[k] vs metric.
    fig = plt.figure(figsize=(min(3.0 * n_lvl, 16), 6.5))
    gs = fig.add_gridspec(2, n_lvl, height_ratios=[1.5, 1.0],
                          hspace=0.45, wspace=0.3)

    # Top: bars per level
    for i, lvl in enumerate(valid):
        ax = fig.add_subplot(gs[0, i])
        ax.bar(ks, lvl["p_marg"], width=0.7, color="#3b6ea8", alpha=0.85,
               label=f"$p_k$ (n={lvl['n']})")
        Ek = float((lvl["p_marg"] * (np.arange(T) + 1)).sum())
        ax.axvline(Ek, color="#cc4444", linestyle="--", alpha=0.7,
                   label=fr"$\mathbb{{E}}[k]={Ek:.2f}$")
        title = f"L{lvl['level']}"
        if level_metric is not None and lvl["level"] < len(level_metric):
            title += f" ({metric_name}={level_metric[lvl['level']]})"
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ks if T <= 12 else ks[::2])
        ax.set_ylim(0, 1.0)
        if i == 0:
            ax.set_ylabel(r"halt prob $p_k$")
        ax.set_xlabel("step $k$")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
        ax.grid(axis="y", alpha=0.25)

    # Bottom: scatter E[k] vs metric (or vs level index)
    ax_summary = fig.add_subplot(gs[1, :])
    xs = []
    ys = []
    labels = []
    for lvl in valid:
        Ek = float((lvl["p_marg"] * (np.arange(T) + 1)).sum())
        if level_metric is not None and lvl["level"] < len(level_metric):
            xs.append(level_metric[lvl["level"]])
        else:
            xs.append(lvl["level"])
        ys.append(Ek)
        labels.append(f"L{lvl['level']}")
    ax_summary.plot(xs, ys, "o-", color="#3b6ea8", markersize=8)
    for x, y, lab in zip(xs, ys, labels):
        ax_summary.annotate(lab, (x, y), xytext=(4, 4),
                            textcoords="offset points", fontsize=8)
    # y=x diagonal as reference (only if metric_name suggests a step count)
    if metric_name in ("distance", "depth_required", "slide_distance"):
        lo = min(xs + [1])
        hi = max(xs + [T])
        ax_summary.plot([lo, hi], [lo, hi], "--", color="#999999",
                        alpha=0.5, label=r"$y=x$ (perfect tracking)")
        ax_summary.legend(fontsize=8)
    ax_summary.set_xlabel(metric_name if level_metric is not None else "level index")
    ax_summary.set_ylabel(r"$\mathbb{E}[\mathrm{halt\ step}]$")
    ax_summary.set_ylim(0, T + 1)
    ax_summary.grid(alpha=0.3)

    fig.suptitle(
        f"Per-level adaptive halt on `{result['game']}` "
        f"(n_steps={cfg['n_nca_steps']}, $\\lambda_p={cfg['halt_prior_p']:g}$, "
        f"$\\lambda_{{KL}}={cfg['halt_kl_weight']:g}$)",
        fontsize=12, y=0.99,
    )

    pdf = out_base + ".pdf"
    png = out_base + ".png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(f"[plot] wrote {pdf}")
    print(f"[plot] wrote {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Trained run directory.")
    p.add_argument("--out", required=True, help="Output base path (no extension).")
    p.add_argument("--level_metric", default=None,
                   help='Optional comma-separated per-level metric values, e.g. "1,2,3,4,6,8,12,16".')
    p.add_argument("--metric_name", default="distance",
                   help="Label for the metric axis (default: 'distance').")
    p.add_argument("--action", type=int, default=None,
                   help="Filter to a single action index (0=up,1=down,2=left,3=right,4=action). Default: all.")
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    metric = None
    if args.level_metric:
        metric = [float(x) for x in args.level_metric.split(",")]

    result = analyze_run(args.run, action_filter=args.action,
                         batch_size=args.batch_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_figure(result, metric, args.metric_name, args.out)


if __name__ == "__main__":
    main()

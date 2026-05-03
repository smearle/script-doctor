"""Render a paper-ready figure of learned adaptive-halt distributions.

Loads a list of trained checkpoints (each one a single-game adaptive-halt
run from `train.py`), runs a forward pass over a sample of cached
transitions, and plots:

  - Per-game halt distribution p_k as bars
  - Geometric prior overlaid as a line (so the deviation is visible)
  - Per-game E[halt step] annotated
  - Best-train-loss curve as a small inset

The output is a single composite figure (PDF + PNG), one panel per game.

Usage:
    python -m nca_wm.scripts.plot_halt_distributions \
        --runs nca_wm/logs_halt_arch/multigame_halt_sokoban_basic_seed0 \
               nca_wm/logs_halt_arch/multigame_halt_sokoban_match3_seed0 \
               nca_wm/logs_halt_arch/multigame_halt_atlas_shrank_seed0 \
        --out  nca_wm/logs_halt_arch/multigame_halt_figure
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

# Add repo root to sys.path so we can import nca_wm.* when invoked via -m
# OR via direct file path.
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from nca_wm.rule_attn_model import RuleAttnNCAWorldModel  # noqa: E402

ROLLOUT_CACHE_DIR = os.path.join(_REPO, "rollout_data")
N_ACTIONS = 5


def _find_cache(game: str, level: int) -> str:
    """Find the largest cached transitions npz for a (game, level)."""
    pat = os.path.join(ROLLOUT_CACHE_DIR, game, f"level_{level}",
                       "*transitions*.npz")
    matches = sorted(glob.glob(pat),
                     key=lambda p: os.path.getsize(p), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No cache matching {pat}")
    return matches[0]


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
    """halt_logits: (T, B). Returns (p_marginal_T, p_per_b_TxB) as numpy."""
    lam = jax.nn.sigmoid(halt_logits)
    log_omlam = jnp.log(jnp.clip(1.0 - lam, 1e-6, 1.0))
    B = lam.shape[1]
    surv = jnp.exp(jnp.concatenate(
        [jnp.zeros((1, B)), jnp.cumsum(log_omlam, axis=0)[:-1]], axis=0))
    p = lam * surv
    last_surv = jnp.exp(jnp.sum(log_omlam[:-1], axis=0))
    p = p.at[-1].set(last_surv)
    return np.asarray(p.mean(axis=1)), np.asarray(p)


def _geom_prior(T: int, p_prior: float) -> np.ndarray:
    ks = np.arange(T)
    log_prior = np.where(
        ks < T - 1,
        ks * np.log1p(-p_prior) + np.log(p_prior),
        (T - 1) * np.log1p(-p_prior),
    )
    return np.exp(log_prior)


def analyze_run(run_dir: str, batch_size: int = 64) -> dict:
    """Returns a dict with halt distribution + train-loss curve for one run."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p_path = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p_path):
        p_path = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p_path, "rb"))

    g = gi[0]
    game_name = g["name"]
    level_i = cfg.get("level", 0) or 0
    cache_path = _find_cache(game_name, level_i)
    data = np.load(cache_path)
    states_np = data["states"]
    nxts_np = data["next_states"]
    acts_np = data["actions"]

    # Compute padded shape from the trained embed weight
    embed_in = params["params"]["embed"]["kernel"].shape[0]  # (C_pad + N_ACTIONS)
    C_pad = embed_in - N_ACTIONS
    H_pad = max(g["H"], 16)
    W_pad = max(g["W"], 64)
    while H_pad % 2: H_pad += 1
    while W_pad % 2: W_pad += 1

    def pad_states(s):
        return np.pad(
            s,
            [(0, 0), (0, C_pad - s.shape[1]),
             (0, H_pad - s.shape[2]), (0, W_pad - s.shape[3])],
        )

    B = min(batch_size, states_np.shape[0])
    sb = jnp.asarray(pad_states(states_np[:B]).astype(np.float32))
    ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[acts_np[:B]])
    toks_np = np.asarray(g["token_ids"])
    gtoks = jnp.asarray(np.tile(toks_np[None], (B, 1)))
    gmask = jnp.ones_like(gtoks, dtype=bool)

    model = _build_model(cfg, toks_np.shape[0], C_pad)
    out = model.apply(params, sb, ab, gtoks, gmask)
    halt_aux = out[-1]
    _ps_logits, _ps_win, ps_halt = halt_aux  # (T, B)

    p_marg, p_full = _halt_dist(ps_halt)
    T = p_marg.shape[0]
    prior = _geom_prior(T, cfg["halt_prior_p"])
    exp_step = float((p_marg * (np.arange(T) + 1)).sum())
    prior_exp = float((prior * (np.arange(T) + 1)).sum())
    log_prior = np.log(np.clip(prior, 1e-8, 1.0))
    kl = float((p_full * (np.log(np.clip(p_full, 1e-8, 1.0))
                          - log_prior[:, None])).sum(axis=0).mean())

    # Train-loss curve from the latest curves npz
    curves = sorted(glob.glob(os.path.join(run_dir, "curves_step*.npz")))
    losses = None
    if curves:
        z = np.load(curves[-1])
        losses = np.asarray(z["losses"])

    return dict(
        game=game_name,
        cfg=cfg,
        p_marg=p_marg,
        prior=prior,
        exp_step=exp_step,
        prior_exp=prior_exp,
        kl=kl,
        losses=losses,
        run_dir=run_dir,
    )


def plot_figure(results: list[dict], out_base: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(
        1, n, figsize=(4.0 * n, 3.6),
        sharey=False, gridspec_kw=dict(wspace=0.3),
    )
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        T = len(r["p_marg"])
        ks = np.arange(1, T + 1)
        ax.bar(ks, r["p_marg"], width=0.7,
               color="#3b6ea8", alpha=0.85, label="learned $p_k$")
        ax.plot(ks, r["prior"], "o-", color="#cc4444",
                label=fr"prior Geom($\lambda_p={r['cfg']['halt_prior_p']:g}$)",
                linewidth=1.5, markersize=4)
        ax.axvline(r["exp_step"], color="#3b6ea8", linestyle="--", alpha=0.6,
                   label=fr"$\mathbb{{E}}[k]={r['exp_step']:.2f}$")
        ax.axvline(r["prior_exp"], color="#cc4444", linestyle="--", alpha=0.5)
        ax.set_xticks(ks)
        ax.set_xlabel("NCA step $k$")
        ax.set_title(r["game"], fontsize=11)
        ax.set_ylim(0, max(r["p_marg"].max(), r["prior"].max()) * 1.15)
        if ax is axes[0]:
            ax.set_ylabel("halt probability $p_k$")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.grid(axis="y", alpha=0.25)
        # Annotation: KL + final train loss
        final_loss = r["losses"][-1] if r["losses"] is not None else float("nan")
        ax.text(0.04, 0.97,
                fr"KL$={r['kl']:.2f}$ nats" + "\n" +
                fr"final loss $={final_loss:.2e}$",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.6", alpha=0.85))

    cfg0 = results[0]["cfg"]
    fig.suptitle(
        f"Adaptive halt across games (rule_attn, n_steps={cfg0['n_nca_steps']}, "
        f"shared body, $\\lambda_{{KL}}={cfg0['halt_kl_weight']:g}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    pdf = out_base + ".pdf"
    png = out_base + ".png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(f"[plot] wrote {pdf}")
    print(f"[plot] wrote {png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run directories (each from a single-game adaptive_halt run).")
    p.add_argument("--out", required=True,
                   help="Output base path (no extension); writes .pdf + .png.")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Number of transitions to push through for halt-dist estimate.")
    args = p.parse_args()

    results = []
    for run in args.runs:
        print(f"[analyze] {run}")
        r = analyze_run(run, batch_size=args.batch_size)
        print(f"  game={r['game']}  E[k]={r['exp_step']:.2f}  "
              f"prior E[k]={r['prior_exp']:.2f}  KL={r['kl']:.3f}")
        results.append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_figure(results, args.out)


if __name__ == "__main__":
    main()

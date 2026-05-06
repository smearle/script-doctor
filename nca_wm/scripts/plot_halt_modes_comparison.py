"""4-mode halt comparison figure.

Loads Collapse-L0 checkpoints for ponder / uniform / argmax_st /
convergence_st (all adaptive modes) plus the `none` baseline, and
renders a two-panel figure:

  Left:  per-step error trajectory for each adaptive mode + `none`
         baseline as a horizontal line.
  Right: err under fixed_T vs convergence-halt (eps=0.01) for each
         adaptive mode, plus none's fixed_T err.

Usage:
    python -m nca_wm.scripts.plot_halt_modes_comparison \
        --runs ponder=nca_wm/logs_halt_arch/halt_collapse_learned_seed0 \
               uniform=nca_wm/logs_halt_arch/halt_collapse_uniform_seed0 \
               argmax_st=nca_wm/logs_halt_arch/halt_collapse_argmax_st_seed0 \
               convergence_st=nca_wm/logs_halt_arch/halt_collapse_convergence_st_seed0 \
               none=nca_wm/logs_halt_arch/halt_collapse_none_seed0 \
        --out nca_wm/logs_halt_arch/halt_modes_comparison_figure
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


def _build_model(cfg, gtoks_len, n_objs):
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
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def analyze(run_dir, batch_size=256):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p): p = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p, "rb"))
    g = gi[0]
    embed_in = params["params"]["embed"]["kernel"].shape[0]
    C_pad = embed_in - N_ACTIONS
    maxH, maxW = g["H"], g["W"]
    toks_np = np.asarray(g["token_ids"])
    cache = sorted(glob.glob(os.path.join(
        ROLLOUT_CACHE_DIR, g["name"], "level_0", "*transitions*.npz")),
        key=lambda p: -os.path.getsize(p))[0]
    z = np.load(cache)
    lW = int(z["W"])
    # IMPORTANT: states are bit-packed along W (np.packbits axis=-1) for cache
    # compactness; reading the packed bytes as raw state gives garbage.
    s = _unpack_states(z["states"], lW).astype(np.float32)
    n = _unpack_states(z["next_states"], lW).astype(np.float32)
    a = z["actions"]
    B = min(batch_size, s.shape[0])
    pad = lambda arr: np.pad(arr[:B], [(0,0),(0,C_pad-arr.shape[1]),(0,maxH-arr.shape[2]),(0,maxW-arr.shape[3])])
    sb = jnp.asarray(pad(s))
    nb = jnp.asarray(pad(n))
    ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[a[:B]])
    toks = jnp.asarray(np.tile(toks_np[None], (B, 1)))
    gm = jnp.ones_like(toks, dtype=bool)
    model = _build_model(cfg, toks_np.shape[0], C_pad)
    out = model.apply(params, sb, ab, toks, gm)
    nb_np = np.asarray(nb)
    if not cfg.get("adaptive_halt", False):
        logits = out[0]
        preds_T = np.asarray((jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32))
        return dict(cfg=cfg, is_adaptive=False,
                    err_T=float((preds_T != nb_np).mean()), preds=None)
    ps_logits = out[-1][0]
    T = ps_logits.shape[0]
    preds = np.asarray((jax.nn.sigmoid(ps_logits) > 0.5).astype(jnp.float32))
    err_per_step = (preds != nb_np[None]).mean(axis=(1, 2, 3, 4))
    return dict(cfg=cfg, is_adaptive=True, T=T, preds=preds,
                err_per_step=err_per_step, nb_np=nb_np)


def conv_halt_steps(preds, eps):
    T, B = preds.shape[0], preds.shape[1]
    cells = preds.shape[2] * preds.shape[3] * preds.shape[4]
    halt = np.full(B, T, dtype=np.int32)
    halted = np.zeros(B, dtype=bool)
    for k in range(1, T):
        diff = (preds[k] != preds[k-1]).reshape(B, -1).sum(axis=1) / cells
        new_halt = (~halted) & (diff < eps)
        halt = np.where(new_halt, k+1, halt)
        halted = halted | new_halt
        if halted.all(): break
    return halt


def plot(results, out_base):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4),
                             gridspec_kw=dict(wspace=0.3))
    ax_traj, ax_compare = axes

    colors = {
        "ponder":         "#3b6ea8",
        "uniform":        "#cc4444",
        "argmax_st":      "#d9a000",
        "convergence_st": "#7a4ec1",
        "none":           "#3a8a3a",
    }
    labels = {
        "ponder":         "ponder (PonderNet)",
        "uniform":        "uniform (mean over k)",
        "argmax_st":      "argmax-ST (learned head, hard sel.)",
        "convergence_st": "convergence-ST (no head, conv. sel.)",
        "none":           "none (single readout)",
    }

    # LEFT: per-step error trajectory
    for tag, r in results.items():
        if not r["is_adaptive"]:
            ax_traj.axhline(r["err_T"], color=colors[tag], linestyle=":",
                            alpha=0.85, linewidth=2,
                            label=f"{labels[tag]} (fixed_T err)")
            continue
        ks = np.arange(1, r["T"] + 1)
        ax_traj.plot(ks, r["err_per_step"], "o-",
                     color=colors[tag], label=labels[tag],
                     markersize=5, linewidth=1.7)
    ax_traj.set_xlabel("NCA step $k$")
    ax_traj.set_ylabel("err (fraction of cells wrong)")
    ax_traj.set_title("Per-step error trajectory")
    ax_traj.legend(fontsize=8, framealpha=0.9, loc="best")
    ax_traj.grid(alpha=0.3)

    # RIGHT: fixed_T vs convergence-halt err per mode
    eps = 0.01
    bar_data = []
    for tag, r in results.items():
        if not r["is_adaptive"]:
            bar_data.append((tag, r["err_T"], None, r["err_T"]))
            continue
        halt_k = conv_halt_steps(r["preds"], eps)
        B = halt_k.shape[0]
        pred_at = r["preds"][halt_k - 1, np.arange(B)]
        err_conv = (pred_at != r["nb_np"]).mean()
        err_T = float(r["err_per_step"][-1])
        bar_data.append((tag, err_T, err_conv, halt_k.mean()))

    n = len(bar_data)
    x = np.arange(n)
    width = 0.36
    fixed_vals = [d[1] for d in bar_data]
    conv_vals = [d[2] if d[2] is not None else d[1] for d in bar_data]
    ax_compare.bar(x - width/2, fixed_vals, width=width,
                   color=[colors[d[0]] for d in bar_data],
                   alpha=0.45, label="fixed_T err")
    ax_compare.bar(x + width/2, conv_vals, width=width,
                   color=[colors[d[0]] for d in bar_data],
                   alpha=0.95, label=f"conv-halt ε={eps:g}")
    # Annotate halt step
    for i, (tag, err_T, err_conv, mean_k) in enumerate(bar_data):
        if err_conv is not None and err_conv != err_T:
            ax_compare.text(i + width/2, err_conv + 0.003,
                            fr"$\bar{{k}}={mean_k:.1f}$",
                            ha="center", fontsize=8)
    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels([d[0] for d in bar_data], rotation=20, ha="right",
                                fontsize=9)
    ax_compare.set_ylabel("err")
    ax_compare.set_title(fr"fixed_T (T=8) vs convergence-halt (ε={eps:g})")
    ax_compare.legend(fontsize=8, loc="upper right")
    ax_compare.grid(axis="y", alpha=0.3)
    ax_compare.set_ylim(0, max(max(fixed_vals), max(conv_vals)) * 1.18)

    cfg0 = next(iter(results.values()))["cfg"]
    fig.suptitle(
        f"Halt-mode comparison on Collapse-L0 (rule_attn, n_steps={cfg0['n_nca_steps']}, "
        f"shared body, h={cfg0['n_hid']}, 5000 updates)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=180, bbox_inches="tight")
    print(f"[plot] wrote {out_base}.pdf")
    print(f"[plot] wrote {out_base}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help='List of "tag=path" pairs.')
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()
    results = {}
    for spec in args.runs:
        tag, path = spec.split("=", 1)
        print(f"[analyze] {tag} = {path}")
        results[tag] = analyze(path, args.batch_size)
    plot(results, args.out)


if __name__ == "__main__":
    main()

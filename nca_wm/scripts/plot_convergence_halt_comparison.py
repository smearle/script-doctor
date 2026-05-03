"""Paper-ready figure: convergence-halting vs learned-halt vs fixed-T.

Loads three trained Collapse-L0 checkpoints (--halt_mode in
{learned, uniform} and a no-adaptive_halt baseline) and produces a
two-panel figure:

  Left:  per-step error trajectory for each model (shows how err evolves
         as the body iterates; uniform should be U-shaped, learned should
         be flat-after-saturation, none has no trajectory).
  Right: err at convergence-halt under several ε thresholds vs fixed_T
         vs oracle-best-step, for the uniform model. The headline
         comparison: convergence-halt picks a near-oracle step with no
         learning needed.

Usage:
    python -m nca_wm.scripts.plot_convergence_halt_comparison \
        --learned nca_wm/logs_halt_arch/halt_collapse_learned_seed0 \
        --uniform nca_wm/logs_halt_arch/halt_collapse_uniform_seed0 \
        --none    nca_wm/logs_halt_arch/halt_collapse_none_seed0 \
        --out     nca_wm/logs_halt_arch/convergence_halt_figure
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


def analyze(run_dir: str, batch_size: int = 256) -> dict:
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
    s_np, a_np, n_np = z["states"], z["actions"], z["next_states"]
    B = min(batch_size, s_np.shape[0])
    pad = lambda arr: np.pad(
        arr[:B],
        [(0, 0), (0, C_pad - arr.shape[1]),
         (0, maxH - arr.shape[2]), (0, maxW - arr.shape[3])],
    )
    sb = jnp.asarray(pad(s_np).astype(np.float32))
    nb = jnp.asarray(pad(n_np).astype(np.float32))
    ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[a_np[:B]])
    toks = jnp.asarray(np.tile(toks_np[None], (B, 1)))
    gm = jnp.ones_like(toks, dtype=bool)

    model = _build_model(cfg, toks_np.shape[0], C_pad)
    out = model.apply(params, sb, ab, toks, gm)

    nb_np = np.asarray(nb)
    is_adaptive = cfg.get("adaptive_halt", False)
    if not is_adaptive:
        logits = out[0]
        preds_T = np.asarray((jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32))
        err_T = float((preds_T != nb_np).mean())
        return dict(cfg=cfg, run=run_dir, is_adaptive=False,
                    err_T=err_T, preds=None)
    halt_aux = out[-1]
    ps_logits = halt_aux[0]
    T = ps_logits.shape[0]
    preds = np.asarray((jax.nn.sigmoid(ps_logits) > 0.5).astype(jnp.float32))
    err_per_step = (preds != nb_np[None]).mean(axis=(1, 2, 3, 4))
    return dict(cfg=cfg, run=run_dir, is_adaptive=True, T=T,
                preds=preds, nb_np=nb_np, err_per_step=err_per_step)


def convergence_halt_steps(preds: np.ndarray, eps: float) -> np.ndarray:
    T, B = preds.shape[0], preds.shape[1]
    cells = preds.shape[2] * preds.shape[3] * preds.shape[4]
    halt = np.full(B, T, dtype=np.int32)
    halted = np.zeros(B, dtype=bool)
    for k in range(1, T):
        diff = (preds[k] != preds[k - 1]).reshape(B, -1).sum(axis=1) / cells
        new_halt = (~halted) & (diff < eps)
        halt = np.where(new_halt, k + 1, halt)
        halted = halted | new_halt
        if halted.all(): break
    return halt


def plot(results: dict[str, dict], out_base: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw=dict(wspace=0.32))
    ax_traj, ax_compare = axes

    # LEFT: per-step error trajectory
    colors = {"learned": "#3b6ea8", "uniform": "#cc4444", "none": "#3a8a3a"}
    for tag, r in results.items():
        if not r["is_adaptive"]:
            ax_traj.axhline(r["err_T"], color=colors.get(tag, "gray"),
                            linestyle=":", alpha=0.7,
                            label=f"{tag} (single readout, fixed_T)")
            continue
        ks = np.arange(1, r["T"] + 1)
        ax_traj.plot(ks, r["err_per_step"], "o-",
                     color=colors.get(tag, "gray"),
                     label=f"{tag} (per-step)", markersize=5, linewidth=1.5)
    ax_traj.set_xlabel("NCA step $k$")
    ax_traj.set_ylabel("err (fraction of cells wrong)")
    ax_traj.set_title("Per-step error trajectory")
    ax_traj.legend(fontsize=8, framealpha=0.9)
    ax_traj.grid(alpha=0.3)

    # RIGHT: err under different stopping criteria for the uniform model
    if "uniform" in results and results["uniform"]["is_adaptive"]:
        r_u = results["uniform"]
        eps_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        labels = []
        ys = []
        ms = []
        # Convergence-halt at multiple eps
        for eps in eps_list:
            halt_k = convergence_halt_steps(r_u["preds"], eps)
            B = halt_k.shape[0]
            pred_at = r_u["preds"][halt_k - 1, np.arange(B)]
            err = (pred_at != r_u["nb_np"]).mean()
            labels.append(fr"conv $\epsilon={eps:g}$")
            ys.append(err)
            ms.append(halt_k.mean())
        # Fixed-T baseline
        labels.append(f"fixed T={r_u['T']}")
        ys.append(r_u["err_per_step"][-1])
        ms.append(r_u["T"])
        # Oracle-min baseline
        kbest = int(r_u["err_per_step"].argmin())
        labels.append(f"oracle min (k={kbest+1})")
        ys.append(r_u["err_per_step"][kbest])
        ms.append(kbest + 1)

        x = np.arange(len(labels))
        bars = ax_compare.bar(x, ys, color=["#cc4444"] * len(eps_list)
                              + ["#999999", "#3a8a3a"],
                              alpha=0.85)
        # Annotate mean halt step
        for i, (b, m) in enumerate(zip(bars, ms)):
            ax_compare.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + 0.003,
                            f"$\\bar{{k}}={m:.1f}$",
                            ha="center", fontsize=8)
        ax_compare.set_xticks(x)
        ax_compare.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax_compare.set_ylabel("err (uniform model)")
        ax_compare.set_title("Convergence-halt vs fixed_T vs oracle\n(uniform-trained body)")
        ax_compare.grid(axis="y", alpha=0.3)
        ax_compare.set_ylim(0, max(ys) * 1.18)

    cfg0 = next(iter(results.values()))["cfg"]
    fig.suptitle(
        f"Convergence-based halting on Collapse-L0 "
        f"(rule_attn, n_steps={cfg0['n_nca_steps']}, shared body, h={cfg0['n_hid']})",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    pdf = out_base + ".pdf"
    png = out_base + ".png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(f"[plot] wrote {pdf}")
    print(f"[plot] wrote {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--learned", required=True)
    p.add_argument("--uniform", required=True)
    p.add_argument("--none", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()
    results = {
        "learned": analyze(args.learned, args.batch_size),
        "uniform": analyze(args.uniform, args.batch_size),
        "none":    analyze(args.none, args.batch_size),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot(results, args.out)


if __name__ == "__main__":
    main()

"""Summarize the post-bitpack-fix varislide sweep.

For each run in nca_wm/logs_canary/varislide_postfix{A,B,C,Cp}_*:
  - parse config (depth, repeats, pool flags, mask flags)
  - read final train metrics from the latest curves_step*.npz
  - compute per-distance argmax accuracy via inspect_varislide_distance
  - report a wide CSV + a short markdown table

Usage:
    .venv/bin/python3 nca_wm/scripts/summarize_varislide_postfix_sweep.py \\
        --out_csv nca_wm/figures/varislide_postfix/summary.csv \\
        --out_md  nca_wm/figures/varislide_postfix/summary.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

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
RIGHT_ACTION = 3
PLAYER = 2


def _build_model(cfg, gtoks_len, n_objs):
    n_steps = cfg.get("n_nca_steps", cfg.get("n_steps"))
    n_repeats = cfg.get("n_nca_repeats", n_steps)
    eff_seq_len = max(gtoks_len, 1) + 1
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=n_steps, n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg.get("d_model", 64),
        enc_n_self_layers=cfg.get("n_enc_layers", 2),
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg.get("d_slot", 64), n_attn_heads=cfg.get("n_heads", 4),
        max_seq_len=eff_seq_len,
        axis_pool=cfg.get("axis_pool", True),
        axis_cummax=cfg.get("axis_cummax", True),
        global_pool=cfg.get("global_pool", True),
        use_vq=cfg.get("vq_codebook", False),
        vq_codebook_size=cfg.get("vq_codebook_size", 512),
        vq_commitment_weight=cfg.get("vq_commitment_weight", 0.25),
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=n_repeats,
        adaptive_halt=cfg.get("adaptive_halt", False),
        mask_hidden=cfg.get("mask_hidden", False),
    )


def _per_distance_argmax(run_dir, train_seed=0):
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

    model = _build_model(cfg, toks_np.shape[0], C_pad)
    pattern = os.path.join(ROLLOUT_CACHE_DIR, "varislide", "synthetic_*x3",
                           f"seed{train_seed}_n*_v9*solv0*.npz")
    dist_to_correct = defaultdict(list)
    dist_to_pred_d = defaultdict(list)
    n_total = 0
    for cache in sorted(glob.glob(pattern)):
        z = np.load(cache)
        s_raw, a, n_raw = z["states"], z["actions"], z["next_states"]
        if "W" in z.files:
            lW = int(z["W"])
            s = _unpack_states(s_raw, lW).astype(np.float32)
            n = _unpack_states(n_raw, lW).astype(np.float32)
        else:
            s = s_raw.astype(np.float32)
            n = n_raw.astype(np.float32)
        mask = (a == RIGHT_ACTION)
        s, a, n = s[mask], a[mask], n[mask]
        if len(s) == 0:
            continue
        delta = (s != n).any(axis=(1, 2, 3))
        s, a, n = s[delta], a[delta], n[delta]
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
        sb_np = np.asarray(sb)[:, :, :lH, :lW]
        nb_np = np.asarray(nb)[:, :, :lH, :lW]
        sig_np = sig[:, :, :lH, :lW]
        for i in range(len(s)):
            in_p = np.argwhere(sb_np[i, PLAYER] > 0.5)
            out_p = np.argwhere(nb_np[i, PLAYER] > 0.5)
            if len(in_p) == 0 or len(out_p) == 0:
                continue
            in_row, in_col = int(in_p[0, 0]), int(in_p[0, 1])
            out_row, out_col = int(out_p[0, 0]), int(out_p[0, 1])
            if in_row != out_row or out_col <= in_col:
                continue
            d = out_col - in_col
            row_sig = sig_np[i, PLAYER, out_row, :]
            argmax = int(np.argmax(row_sig))
            dist_to_correct[d].append(argmax == out_col)
            dist_to_pred_d[d].append(argmax - in_col)
            n_total += 1
    return dist_to_correct, dist_to_pred_d, n_total


def _final_train_metrics(run_dir):
    """Read the last entry of the latest curves_step*.npz."""
    curves = sorted(
        glob.glob(os.path.join(run_dir, "curves_step*.npz")),
        key=lambda p: int(os.path.basename(p)[len("curves_step"):-len(".npz")])
    )
    if not curves:
        return None
    z = np.load(curves[-1])
    keys = z.files
    out = {}
    for k in keys:
        v = z[k]
        if v.ndim == 1 and v.size > 0:
            out[f"final_{k}"] = float(v[-1])
    return out


def _summarize_run(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    name = os.path.basename(run_dir)
    train = _final_train_metrics(run_dir) or {}
    try:
        d2c, d2p, n_total = _per_distance_argmax(run_dir)
    except Exception as e:
        print(f"  per-dist eval failed for {name}: {e}")
        d2c, d2p, n_total = {}, {}, 0
    # overall argmax across all distances
    all_correct = [c for v in d2c.values() for c in v]
    overall = float(np.mean(all_correct)) if all_correct else float("nan")
    # per-distance breakdown (most informative: d=1, 2, 3+)
    d1 = float(np.mean(d2c[1])) if 1 in d2c else float("nan")
    d2_ = float(np.mean(d2c[2])) if 2 in d2c else float("nan")
    d3p = ([c for d, cs in d2c.items() if d >= 3 for c in cs])
    d3 = float(np.mean(d3p)) if d3p else float("nan")
    return {
        "run": name,
        "n_steps": cfg.get("n_nca_steps"),
        "n_repeats": cfg.get("n_nca_repeats"),
        "n_layers": (cfg.get("n_nca_steps", 0) // max(cfg.get("n_nca_repeats", 1), 1)),
        "axis_pool": cfg.get("axis_pool", True),
        "axis_cummax": cfg.get("axis_cummax", True),
        "global_pool": cfg.get("global_pool", True),
        "input_skip": cfg.get("input_skip", False),
        "mask_hidden": cfg.get("mask_hidden", False),
        "seed": cfg.get("seed", 0),
        "n_total": n_total,
        "argmax_overall": overall,
        "argmax_d1": d1,
        "argmax_d2": d2_,
        "argmax_d3p": d3,
        "final_loss": train.get("final_losses"),
        "final_acc": train.get("final_accs"),
        "final_change_acc": train.get("final_change_accs"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="nca_wm/logs_canary")
    p.add_argument("--prefix", default="varislide_postfix")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_md", required=True)
    args = p.parse_args()

    runs = sorted(glob.glob(os.path.join(_REPO, args.logs, f"{args.prefix}[ABC]*")))
    def _is_complete(r):
        meta = os.path.join(r, "train_meta.json")
        cfg = os.path.join(r, "config.json")
        if not (os.path.isdir(r) and os.path.exists(meta) and os.path.exists(cfg)):
            return False
        try:
            m = json.load(open(meta))
            req = m.get("n_updates_requested")
            return req is not None and m.get("total_steps", 0) >= req
        except Exception:
            return False
    runs = [r for r in runs if _is_complete(r)]
    print(f"Found {len(runs)} runs.")
    rows = []
    for r in runs:
        print(f"  {os.path.basename(r)}")
        rows.append(_summarize_run(r))

    out_csv = os.path.join(_REPO, args.out_csv) if not os.path.isabs(args.out_csv) else args.out_csv
    out_md  = os.path.join(_REPO, args.out_md)  if not os.path.isabs(args.out_md)  else args.out_md
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}")

    # Markdown
    by_bucket = {"A": [], "B": [], "C": [], "Cp": []}
    for r in rows:
        for k in by_bucket:
            if r["run"].startswith(f"varislide_postfix{k}_"):
                by_bucket[k].append(r); break

    def fmt_pct(x):
        return "—" if x is None or (isinstance(x, float) and (np.isnan(x))) else f"{x*100:.1f}%"
    def fmt_loss(x):
        return "—" if x is None else f"{x:.2e}"

    lines = [
        "# Varislide post-bitpack-fix sweep summary\n",
        "All runs at h=128, batch=16, lr=3e-4, 10k updates, mask_hidden=True, multi-grid {6,8,10,12,16}x3.",
        "Per-distance argmax-correct: % of right-action transitions where argmax(player-row sigmoid) lands at the correct slide-stop column.",
        "n_total = # of right-action delta transitions evaluated (the same set across configs).\n",
        "## A: depth × seed (fully shared, n_repeats = n_steps)\n",
        "| depth | seed | argmax overall | d=1 | d=2 | d≥3 | final loss | final change_acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    A = sorted(by_bucket["A"], key=lambda r: (r["n_steps"], r["seed"]))
    for r in A:
        lines.append(
            f"| {r['n_steps']} | {r['seed']} | {fmt_pct(r['argmax_overall'])} | "
            f"{fmt_pct(r['argmax_d1'])} | {fmt_pct(r['argmax_d2'])} | "
            f"{fmt_pct(r['argmax_d3p'])} | {fmt_loss(r['final_loss'])} | "
            f"{fmt_pct(r['final_change_acc'])} |"
        )

    lines += [
        "\n## B: L × R factor at total=16\n",
        "| L | R | seed | argmax overall | d=1 | d=2 | d≥3 | final loss | final change_acc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    B = sorted(by_bucket["B"], key=lambda r: (r["n_layers"], r["n_repeats"], r["seed"]))
    for r in B:
        lines.append(
            f"| {r['n_layers']} | {r['n_repeats']} | {r['seed']} | "
            f"{fmt_pct(r['argmax_overall'])} | {fmt_pct(r['argmax_d1'])} | "
            f"{fmt_pct(r['argmax_d2'])} | {fmt_pct(r['argmax_d3p'])} | "
            f"{fmt_loss(r['final_loss'])} | "
            f"{fmt_pct(r['final_change_acc'])} |"
        )

    lines += [
        "\n## C: pool OFF + input_skip\n",
        "| depth | seed | argmax overall | d=1 | d=2 | d≥3 | final loss | final change_acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    C = sorted(by_bucket["C"], key=lambda r: (r["n_steps"], r["seed"]))
    for r in C:
        lines.append(
            f"| {r['n_steps']} | {r['seed']} | {fmt_pct(r['argmax_overall'])} | "
            f"{fmt_pct(r['argmax_d1'])} | {fmt_pct(r['argmax_d2'])} | "
            f"{fmt_pct(r['argmax_d3p'])} | {fmt_loss(r['final_loss'])} | "
            f"{fmt_pct(r['final_change_acc'])} |"
        )

    if by_bucket["Cp"]:
        lines += [
            "\n## C': pool OFF without input_skip control\n",
            "| depth | seed | argmax overall | d=1 | d=2 | d≥3 | final loss | final change_acc |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(by_bucket["Cp"], key=lambda r: (r["n_steps"], r["seed"])):
            lines.append(
                f"| {r['n_steps']} | {r['seed']} | {fmt_pct(r['argmax_overall'])} | "
                f"{fmt_pct(r['argmax_d1'])} | {fmt_pct(r['argmax_d2'])} | "
                f"{fmt_pct(r['argmax_d3p'])} | {fmt_loss(r['final_loss'])} | "
                f"{fmt_pct(r['final_change_acc'])} |"
            )

    out_md_path = out_md
    with open(out_md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_md_path}")


if __name__ == "__main__":
    main()

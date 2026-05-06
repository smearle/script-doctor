"""Re-evaluate trained varislide checkpoints on authored levels L0-L7 using
top-left slicing (matching _pad_state_for_model). The original
eval_multigame.npz files were generated with a centered-slice bug that
inflated cell-error for small levels; this script regenerates per-level
cell-error using the correct slicing and writes `authored_eval.npz`.

For each run, computes:
  - tf_cell_err_per_step: TF rollout per-step cell error (T,)
  - ar_cell_err_per_step: AR (autoregressive) rollout per-step cell error
  - tf_step1, ar_step1, tf_mean, ar_mean (over T steps)

Output: nca_wm/figures/varislide_postfix/authored_eval.csv +
        nca_wm/figures/varislide_postfix/authored_eval.md
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

import jax  # noqa
import jax.numpy as jnp  # noqa

from nca_wm.rule_attn_model import RuleAttnNCAWorldModel  # noqa
from nca_wm.train import _pad_state_for_model, N_ACTIONS  # noqa
from nca_wm.tokenize_game import tokenize_game, get_game_tree_from_js  # noqa
from puzzlescript_cpp import CppPuzzleScriptEnv, CppPuzzleScriptBackend  # noqa
from puzzlescript_jax.utils import init_ps_lark_parser  # noqa


def _build_model(cfg, gtoks_len, n_objs):
    n_steps = cfg["n_nca_steps"]
    n_repeats = cfg.get("n_nca_repeats", n_steps) or n_steps
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=n_steps, n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg.get("d_model", 64),
        enc_n_self_layers=cfg.get("n_enc_layers", 2),
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg.get("d_slot", 64), n_attn_heads=cfg.get("n_heads", 4),
        max_seq_len=max(gtoks_len, 1) + 1,
        axis_pool=cfg.get("axis_pool", True),
        axis_cummax=cfg.get("axis_cummax", True),
        global_pool=cfg.get("global_pool", True),
        use_vq=cfg.get("vq_codebook", False),
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=n_repeats,
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def _eval_run(run_dir, ps_parser, n_steps_eval=30, n_eps=5, seed=0):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p_path = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p_path):
        p_path = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p_path, "rb"))
    g = gi[0]
    embed_in = params["params"]["embed"]["kernel"].shape[0]
    n_objs_model = embed_in - N_ACTIONS
    toks_np = np.asarray(g["token_ids"])
    eff_len = max(toks_np.shape[0], 1)
    train_H, train_W = g["H"], g["W"]

    model = _build_model(cfg, toks_np.shape[0], n_objs_model)
    apply_fn = jax.jit(model.apply)

    # Build authored varislide game
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(ps_parser, "varislide")

    # Per level
    out = {"per_level": {}}
    rng = np.random.RandomState(seed)
    for li in range(8):
        try:
            env = CppPuzzleScriptEnv(json_str, level_i=li,
                                      max_episode_steps=n_steps_eval)
        except Exception as e:
            continue
        real_obs, _ = env.reset()
        n_objs_env, H, W = real_obs.shape
        # Build input padding shape: max of model bucket and real
        pad_H = max(train_H, H)
        pad_W = max(train_W, W)
        toks_padded = np.zeros((1, eff_len), dtype=np.int32)
        if toks_np.shape[0] > 0:
            toks_padded[0, :toks_np.shape[0]] = toks_np
        gmask = np.zeros((1, eff_len), dtype=bool)
        if toks_np.shape[0] > 0:
            gmask[0, :toks_np.shape[0]] = True
        gt = jnp.asarray(toks_padded)
        gm = jnp.asarray(gmask)

        # Run multiple random episodes per level for both TF and AR rollouts
        tf_cell_errs = []
        ar_cell_errs = []
        for ep in range(n_eps):
            actions = [int(rng.randint(N_ACTIONS)) for _ in range(n_steps_eval)]
            for mode in ("tf", "ar"):
                env_ep = CppPuzzleScriptEnv(json_str, level_i=li,
                                             max_episode_steps=n_steps_eval)
                ro, _ = env_ep.reset()
                pred_state = _pad_state_for_model(ro, n_objs_model, pad_H, pad_W)
                per_step_err = []
                for t in range(n_steps_eval):
                    a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[actions[t]][None])
                    logits, _, _ = apply_fn(params, pred_state, a_oh, gt, gm)
                    pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
                    real_next, _, done, trunc, _ = env_ep.step(actions[t])
                    # TOP-LEFT slicing — match training-time padding
                    pred_binary = np.array(
                        pred_next[0, :n_objs_env, :H, :W] > 0.5, dtype=np.uint8,
                    )
                    mismatch = (pred_binary != real_next)
                    n_wrong_cells = int(mismatch.any(axis=0).sum())
                    per_step_err.append(n_wrong_cells / (H * W))
                    pred_state = (
                        _pad_state_for_model(real_next, n_objs_model, pad_H, pad_W)
                        if mode == "tf" else pred_next
                    )
                    if done or trunc:
                        break
                arr = np.array(per_step_err)
                if mode == "tf":
                    tf_cell_errs.append(arr)
                else:
                    ar_cell_errs.append(arr)
        # Aggregate over episodes (pad to longest)
        def _agg(epslist):
            if not epslist: return None
            T = max(len(a) for a in epslist)
            mat = np.full((len(epslist), T), np.nan)
            for i, a in enumerate(epslist):
                mat[i, :len(a)] = a
            return np.nanmean(mat, axis=0)
        tf_curve = _agg(tf_cell_errs)
        ar_curve = _agg(ar_cell_errs)
        out["per_level"][li] = {
            "W": int(W),
            "H": int(H),
            "tf_step1": float(tf_curve[0]) if tf_curve is not None else None,
            "tf_mean": float(np.nanmean(tf_curve)) if tf_curve is not None else None,
            "ar_step1": float(ar_curve[0]) if ar_curve is not None else None,
            "ar_mean": float(np.nanmean(ar_curve)) if ar_curve is not None else None,
        }

    npz_path = os.path.join(run_dir, "authored_eval_tlfix.npz")
    np.savez(npz_path, summary=np.array([out], dtype=object))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="nca_wm/logs_canary")
    ap.add_argument("--prefix", default="varislide_postfix")
    ap.add_argument("--out_csv", default="nca_wm/figures/varislide_postfix/authored_eval.csv")
    ap.add_argument("--out_md",  default="nca_wm/figures/varislide_postfix/authored_eval.md")
    args = ap.parse_args()

    runs = sorted(glob.glob(os.path.join(_REPO, args.logs, f"{args.prefix}[ABC]*")))
    runs = [r for r in runs if os.path.isdir(r)
            and os.path.exists(os.path.join(r, "config.json"))
            and os.path.exists(os.path.join(r, "params.pkl"))
            and os.path.exists(os.path.join(r, "train_meta.json"))]
    print(f"Found {len(runs)} runs.")

    ps_parser = init_ps_lark_parser()
    rows = []
    for r in runs:
        name = os.path.basename(r)
        print(f"  evaluating {name}...")
        try:
            res = _eval_run(r, ps_parser)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        cfg = json.load(open(os.path.join(r, "config.json")))
        for li, rec in res["per_level"].items():
            rows.append({
                "run": name,
                "n_steps": cfg.get("n_nca_steps"),
                "n_repeats": cfg.get("n_nca_repeats"),
                "n_layers": (cfg.get("n_nca_steps", 0) // max(cfg.get("n_nca_repeats", 1), 1)),
                "axis_pool": cfg.get("axis_pool", True),
                "input_skip": cfg.get("input_skip", False),
                "seed": cfg.get("seed", 0),
                "level": li,
                "W": rec["W"],
                "regime": "train" if rec["W"] in {6, 8, 10, 12, 16} else (
                    "OOD" if rec["W"] > 16 else "interp"),
                "tf_step1": rec["tf_step1"], "tf_mean": rec["tf_mean"],
                "ar_step1": rec["ar_step1"], "ar_mean": rec["ar_mean"],
            })

    if not rows:
        print("no rows produced")
        return

    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(_REPO, args.out_csv)
    out_md  = args.out_md  if os.path.isabs(args.out_md)  else os.path.join(_REPO, args.out_md)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {out_csv}")

    # Markdown: per-run aggregates by regime
    def fmt(x): return "—" if x is None else f"{x*100:.2f}%"
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)
    by_bucket = defaultdict(list)
    for run, rs in by_run.items():
        if   run.startswith("varislide_postfixA_"):  bk = "A"
        elif run.startswith("varislide_postfixB_"):  bk = "B"
        elif run.startswith("varislide_postfixC_nopool_"): bk = "C"
        elif run.startswith("varislide_postfixCp_"): bk = "Cp"
        else: continue
        # Aggregate per regime
        agg = {"train": [], "interp": [], "OOD": []}
        for r in rs:
            agg[r["regime"]].append(r)
        first = rs[0]
        by_bucket[bk].append((first, agg))

    lines = ["# Varislide post-bitpack-fix sweep — corrected authored-level eval\n"]
    lines.append(
        "Re-eval of all 34 checkpoints' authored levels L0–L7 with the top-left "
        "slicing fix (the original eval_multigame.npz used centered slicing, which "
        "compares the model's top-left prediction against the centered region of "
        "the bucket → 85% cell-error on small training-width levels). 5 random "
        "episodes × 30 steps per level, both TF and AR.\n"
    )
    lines.append(
        "Authored level widths: L0=6, L1=7, L2=8, L3=9, L4=10, L5=12, L6=15, L7=19.\n"
        "Training synth widths: {6, 8, 10, 12, 16}. "
        "Regimes: train (W∈{6,8,10,12,16}), interp (W=7,9,15), OOD (W=19).\n"
    )
    bucket_titles = {
        "A": "## A: depth × seed (fully shared)",
        "B": "## B: L × R factor at total=16",
        "C": "## C: pool OFF + input_skip",
        "Cp": "## C': pool OFF, no input_skip",
    }

    def regime_field(rs, regime, key):
        vals = [r[key] for r in rs[regime] if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    for bkey in ["A", "B", "C", "Cp"]:
        items = sorted(by_bucket.get(bkey, []),
                       key=lambda x: (x[0]["n_layers"], x[0]["n_steps"],
                                      x[0]["n_repeats"], x[0]["seed"]))
        if not items:
            continue
        lines.append(f"\n{bucket_titles[bkey]}\n")
        if bkey == "B":
            head = "| L | R | seed |"
        else:
            head = "| depth | seed |"
        head += " TF train | TF interp | TF OOD(L7) | AR train | AR interp | AR OOD(L7) |"
        lines.append(head)
        lines.append("|" + "---|" * (head.count("|") - 1))
        for first, agg in items:
            tf_t = regime_field(agg, "train", "tf_mean")
            tf_i = regime_field(agg, "interp", "tf_mean")
            tf_o = regime_field(agg, "OOD", "tf_mean")
            ar_t = regime_field(agg, "train", "ar_mean")
            ar_i = regime_field(agg, "interp", "ar_mean")
            ar_o = regime_field(agg, "OOD", "ar_mean")
            if bkey == "B":
                pre = f"| {first['n_layers']} | {first['n_repeats']} | {first['seed']} |"
            else:
                pre = f"| {first['n_steps']} | {first['seed']} |"
            lines.append(pre + f" {fmt(tf_t)} | {fmt(tf_i)} | {fmt(tf_o)} |"
                              f" {fmt(ar_t)} | {fmt(ar_i)} | {fmt(ar_o)} |")

    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

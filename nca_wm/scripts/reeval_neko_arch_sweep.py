"""Re-evaluate trained nekopuzzle synth-sweep checkpoints on:
  - the authored 8x7 levels (held out — model never saw them in training)
  - a fresh held-out synth pool (different seed) at the trained sizes
  - an OOD synth pool at 9x9 (size larger than max trained size)

Top-left slicing matches training-time padding. Both teacher-forced (TF) and
autoregressive (AR) rollouts are scored at single-step cell-error and over a
30-step rollout.

Outputs:
  - <run_dir>/neko_eval.npz  (per-run detail)
  - <out_csv> + <out_md>     (sweep-level summary)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import sys
import time
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
from puzzlescript_cpp._puzzlescript_cpp import Engine  # noqa
from puzzlescript_jax.utils import init_ps_lark_parser  # noqa

from nca_wm.synthetic_levels import (
    LevelGenerator as _LG,
    search_validate as _SV,
    _bitpacked_to_multihot as _MH,
)


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
        mask_hidden=cfg.get("mask_hidden", False),
    )


def _load(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    gi = pickle.load(open(os.path.join(run_dir, "game_infos.pkl"), "rb"))
    p_path = os.path.join(run_dir, "params_best.pkl")
    if not os.path.exists(p_path):
        p_path = os.path.join(run_dir, "params.pkl")
    params = pickle.load(open(p_path, "rb"))
    g = gi[0]
    embed_in = params["params"]["embed"]["kernel"].shape[0]
    n_objs_model = embed_in - N_ACTIONS
    return cfg, params, g, n_objs_model


def _gen_holdout_pool(json_str, gen, engine, sizes, n_per_size, seed,
                      *, min_states, max_iters, timeout_ms,
                      max_attempts_per_level=2000):
    rng = np.random.default_rng(seed)
    out = []  # list of (w, h, payload)
    for w, h in sizes:
        accepted = 0
        attempts = 0
        seen = set()
        budget = n_per_size * max_attempts_per_level
        while accepted < n_per_size and attempts < budget:
            attempts += 1
            d = gen.random_dat(rng, w, h)
            if d is None or not gen.structurally_valid(d, w, h):
                continue
            k = tuple(d)
            if k in seen:
                continue
            seen.add(k)
            r = _SV(engine, d, w, h, max_iters=max_iters,
                    timeout_ms=timeout_ms, min_states=min_states,
                    require_solvable=True)
            if r is None:
                continue
            out.append((w, h, r))
            accepted += 1
    return out


def _materialize(payload, n_objs, stride_obj, w, h):
    payload["mh_states"] = (
        np.stack([_MH(s, w, h, n_objs, stride_obj) for s in payload["states"]]).astype(np.uint8)
        if payload["states"] else np.zeros((0, n_objs, h, w), dtype=np.uint8)
    )
    payload["mh_next_states"] = (
        np.stack([_MH(s, w, h, n_objs, stride_obj) for s in payload["next_states"]]).astype(np.uint8)
        if payload["next_states"] else np.zeros((0, n_objs, h, w), dtype=np.uint8)
    )
    payload["np_actions"] = np.array(payload["actions"], dtype=np.int32)
    return payload


def _eval_synth_pool(apply_fn, params, n_objs_model, pool, eff_len,
                    toks_padded, gmask, train_H, train_W):
    """Per-level cell-error on (state, action, next_state) triples (single-step,
    no rollout)."""
    # Aggregate buckets: by (W,H)
    by_size: dict[tuple[int, int], dict] = {}
    for (w, h, payload) in pool:
        key = (w, h)
        rec = by_size.setdefault(key, {"err_sum": 0.0, "n": 0,
                                         "perfect_levels": 0,
                                         "n_levels": 0})
        states = payload["mh_states"]
        next_states = payload["mh_next_states"]
        acts = payload["np_actions"]
        if states.shape[0] == 0:
            continue
        n_objs_env, H, W = states.shape[1], states.shape[2], states.shape[3]
        pad_H = max(train_H, H)
        pad_W = max(train_W, W)
        # Process per-transition: pad each (no batching b/c sizes differ across levels)
        per_lev_err = 0
        per_lev_total = 0
        for t in range(states.shape[0]):
            s_in = _pad_state_for_model(states[t], n_objs_model, pad_H, pad_W)
            a = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[acts[t]][None])
            logits, _, _ = apply_fn(params, s_in, a, toks_padded, gmask)
            pred_next_full = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
            pred_binary = np.array(
                pred_next_full[0, :n_objs_env, :H, :W] > 0.5, dtype=np.uint8,
            )
            mismatch = (pred_binary != next_states[t])
            n_wrong_cells = int(mismatch.any(axis=0).sum())
            per_lev_err += n_wrong_cells
            per_lev_total += H * W
        rec["err_sum"] += per_lev_err
        rec["n"] += per_lev_total
        rec["n_levels"] += 1
        if per_lev_err == 0:
            rec["perfect_levels"] += 1

    return {
        f"{w}x{h}": {
            "cell_err": rec["err_sum"] / max(rec["n"], 1),
            "n_levels": rec["n_levels"],
            "perfect_levels": rec["perfect_levels"],
        }
        for (w, h), rec in by_size.items()
    }


def _eval_authored(apply_fn, params, n_objs_model, json_str, train_H, train_W,
                   toks_padded, gmask, *, n_eps=3, n_steps=30, seed=0):
    """Per-authored-level TF + AR rollout cell-error, similar to
    reeval_varislide_authored.py."""
    rng = np.random.RandomState(seed)
    backend = CppPuzzleScriptBackend()
    env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=n_steps)
    n_levels = env0.num_levels
    out = {}
    for li in range(n_levels):
        env = CppPuzzleScriptEnv(json_str, level_i=li,
                                  max_episode_steps=n_steps)
        real_obs, _ = env.reset()
        n_objs_env, H, W = real_obs.shape
        pad_H = max(train_H, H)
        pad_W = max(train_W, W)

        tf_curves = []
        ar_curves = []
        for ep in range(n_eps):
            actions = [int(rng.randint(N_ACTIONS)) for _ in range(n_steps)]
            for mode in ("tf", "ar"):
                env_ep = CppPuzzleScriptEnv(json_str, level_i=li,
                                             max_episode_steps=n_steps)
                ro, _ = env_ep.reset()
                pred_state = _pad_state_for_model(ro, n_objs_model,
                                                   pad_H, pad_W)
                per_step_err = []
                for t in range(n_steps):
                    a_oh = jnp.array(np.eye(N_ACTIONS, dtype=np.float32)[actions[t]][None])
                    logits, _, _ = apply_fn(params, pred_state, a_oh,
                                              toks_padded, gmask)
                    pred_next = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
                    real_next, _, done, trunc, _ = env_ep.step(actions[t])
                    pred_binary = np.array(
                        pred_next[0, :n_objs_env, :H, :W] > 0.5, dtype=np.uint8,
                    )
                    mismatch = (pred_binary != real_next)
                    per_step_err.append(int(mismatch.any(axis=0).sum())
                                         / (H * W))
                    pred_state = (
                        _pad_state_for_model(real_next, n_objs_model,
                                              pad_H, pad_W)
                        if mode == "tf" else pred_next
                    )
                    if done or trunc:
                        break
                arr = np.array(per_step_err)
                if mode == "tf":
                    tf_curves.append(arr)
                else:
                    ar_curves.append(arr)
        def _agg(epslist):
            if not epslist:
                return None
            T = max(len(a) for a in epslist)
            mat = np.full((len(epslist), T), np.nan)
            for i, a in enumerate(epslist):
                mat[i, :len(a)] = a
            return np.nanmean(mat, axis=0)
        tf_curve = _agg(tf_curves)
        ar_curve = _agg(ar_curves)
        out[li] = {
            "W": int(W), "H": int(H),
            "tf_step1": float(tf_curve[0]) if tf_curve is not None else None,
            "tf_mean": float(np.nanmean(tf_curve)) if tf_curve is not None else None,
            "ar_step1": float(ar_curve[0]) if ar_curve is not None else None,
            "ar_mean": float(np.nanmean(ar_curve)) if ar_curve is not None else None,
        }
    return out


def _eval_run(run_dir, ps_parser, *, holdout_seed=4242, ood_seed=4243):
    cfg, params, g, n_objs_model = _load(run_dir)
    toks_np = np.asarray(g["token_ids"])
    eff_len = max(toks_np.shape[0], 1)
    train_H, train_W = g["H"], g["W"]

    model = _build_model(cfg, toks_np.shape[0], n_objs_model)
    apply_fn = jax.jit(model.apply)

    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(ps_parser, "nekopuzzle")
    json_state = json.loads(json_str)

    tp = np.zeros((1, eff_len), dtype=np.int32)
    if toks_np.shape[0] > 0:
        tp[0, :toks_np.shape[0]] = toks_np
    gm = np.zeros((1, eff_len), dtype=bool)
    if toks_np.shape[0] > 0:
        gm[0, :toks_np.shape[0]] = True
    toks_padded = jnp.asarray(tp)
    gmask = jnp.asarray(gm)

    out = {"run": os.path.basename(run_dir), "cfg": cfg, "train_H": train_H,
           "train_W": train_W}

    # Authored eval
    print(f"  authored eval...", flush=True)
    out["authored"] = _eval_authored(
        apply_fn, params, n_objs_model, json_str,
        train_H, train_W, toks_padded, gmask,
    )

    # Held-out synth pool at trained sizes
    print(f"  holdout synth pool (trained sizes)...", flush=True)
    gen = _LG(json_state, mode="tile_pattern_empirical")
    engine = Engine(); engine.load_from_json(json_str); engine.load_level(0)
    holdout = _gen_holdout_pool(
        json_str, gen, engine,
        sizes=[(5, 5), (6, 6), (7, 7), (8, 8)],
        n_per_size=8, seed=holdout_seed,
        min_states=5, max_iters=5000, timeout_ms=2000,
    )
    for (w, h, p) in holdout:
        _materialize(p, gen.n_objs, gen.stride, w, h)
    out["holdout"] = _eval_synth_pool(
        apply_fn, params, n_objs_model, holdout, eff_len,
        toks_padded, gmask, train_H, train_W,
    )

    # OOD synth pool at 9x9
    print(f"  OOD synth pool (9x9)...", flush=True)
    ood = _gen_holdout_pool(
        json_str, gen, engine,
        sizes=[(9, 9)], n_per_size=8, seed=ood_seed,
        min_states=5, max_iters=5000, timeout_ms=2000,
    )
    for (w, h, p) in ood:
        _materialize(p, gen.n_objs, gen.stride, w, h)
    out["ood"] = _eval_synth_pool(
        apply_fn, params, n_objs_model, ood, eff_len,
        toks_padded, gmask, train_H, train_W,
    )

    # Save per-run npz
    np.savez(os.path.join(run_dir, "neko_eval.npz"),
             summary=np.array([out], dtype=object))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="nca_wm/logs_neko_arch")
    ap.add_argument("--out_csv", default="nca_wm/figures/neko_arch/summary.csv")
    ap.add_argument("--out_md",  default="nca_wm/figures/neko_arch/summary.md")
    ap.add_argument("--include_running", action="store_true",
                    help="Also eval runs without train_meta.json (still in progress).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip runs that already have neko_eval.npz")
    args = ap.parse_args()

    log_root = args.logs if os.path.isabs(args.logs) else os.path.join(_REPO, args.logs)
    runs = sorted(glob.glob(os.path.join(log_root, "neko_*_s0")))
    runs = [r for r in runs if os.path.isdir(r)
            and os.path.exists(os.path.join(r, "config.json"))
            and (os.path.exists(os.path.join(r, "params_best.pkl"))
                 or os.path.exists(os.path.join(r, "params.pkl")))
            and (args.include_running or os.path.exists(os.path.join(r, "train_meta.json")))]
    print(f"Found {len(runs)} runs.")
    ps_parser = init_ps_lark_parser()
    rows = []
    for r in runs:
        name = os.path.basename(r)
        print(f"\n=== {name} ===", flush=True)
        existing = os.path.join(r, "neko_eval.npz")
        if args.skip_existing and os.path.exists(existing):
            print(f"  reusing existing {existing}")
            res = np.load(existing, allow_pickle=True)["summary"].item()
            res["cfg"] = json.load(open(os.path.join(r, "config.json")))
        else:
            try:
                res = _eval_run(r, ps_parser)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  FAILED: {e}")
                continue
        cfg = res["cfg"]
        # Aggregate authored: mean over all levels, both TF and AR.
        tf_step1 = np.nanmean([rec["tf_step1"] for rec in res["authored"].values()
                                 if rec["tf_step1"] is not None])
        tf_mean  = np.nanmean([rec["tf_mean"] for rec in res["authored"].values()
                                 if rec["tf_mean"] is not None])
        ar_step1 = np.nanmean([rec["ar_step1"] for rec in res["authored"].values()
                                 if rec["ar_step1"] is not None])
        ar_mean  = np.nanmean([rec["ar_mean"] for rec in res["authored"].values()
                                 if rec["ar_mean"] is not None])
        # Holdout: per-size cell err (mean across sizes)
        ho_per_size = res["holdout"]
        ho_mean = float(np.mean([v["cell_err"] for v in ho_per_size.values()])) if ho_per_size else float("nan")
        ood_per_size = res["ood"]
        ood_mean = float(np.mean([v["cell_err"] for v in ood_per_size.values()])) if ood_per_size else float("nan")
        rows.append({
            "run": name,
            "depth": cfg["n_nca_steps"],
            "n_repeats": cfg.get("n_nca_repeats", cfg["n_nca_steps"]),
            "share": "shared" if (cfg.get("n_nca_repeats", cfg["n_nca_steps"])
                                    == cfg["n_nca_steps"]) else "per-step",
            "pool": ("ON" if cfg.get("axis_pool", True)
                       and cfg.get("axis_cummax", True)
                       and cfg.get("global_pool", True) else "OFF"),
            "input_skip": cfg.get("input_skip", False),
            "tf_step1": tf_step1, "tf_mean": tf_mean,
            "ar_step1": ar_step1, "ar_mean": ar_mean,
            "ho_mean": ho_mean,
            "ood_mean": ood_mean,
        })
        print(f"  authored TF step1 {100*tf_step1:.2f}% / mean {100*tf_mean:.2f}%  "
              f"AR step1 {100*ar_step1:.2f}% / mean {100*ar_mean:.2f}%  "
              f"holdout {100*ho_mean:.2f}%  ood {100*ood_mean:.2f}%")

    if not rows:
        print("no rows produced")
        return

    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(_REPO, args.out_csv)
    out_md  = args.out_md  if os.path.isabs(args.out_md)  else os.path.join(_REPO, args.out_md)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"\nwrote {out_csv}")

    def fmt(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.2f}%"
    rows_sorted = sorted(rows, key=lambda r: (r["depth"], r["share"], r["pool"]))
    lines = ["# Nekopuzzle synth-level architecture sweep — eval summary\n"]
    lines.append(
        "Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4, "
        "15k updates, mask_hidden=True default. Synth: 64 levels at multi-grid "
        "{5x5, 6x6, 7x7, 8x8}. Authored levels are 8x7 — held-out from training.\n"
    )
    lines.append(
        "Columns: TF/AR rollout cell-error on authored 10 levels (step-1 + 30-step mean), "
        "holdout = held-out synth at trained sizes (different seed), "
        "ood = synth at 9x9 (size-OOD).\n"
    )
    lines.append(
        "| depth | share | pool | TF step1 | TF mean | AR step1 | AR mean | holdout | ood |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows_sorted:
        lines.append(
            f"| {r['depth']} | {r['share']} | {r['pool']} | "
            f"{fmt(r['tf_step1'])} | {fmt(r['tf_mean'])} | "
            f"{fmt(r['ar_step1'])} | {fmt(r['ar_mean'])} | "
            f"{fmt(r['ho_mean'])} | {fmt(r['ood_mean'])} |"
        )
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

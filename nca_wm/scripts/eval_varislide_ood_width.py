"""OOD-width eval for varislide trained with synthetic multi-size data.

Generates held-out synthetic varislide levels at widths NOT seen during
training (default: 20, 24), evaluates a trained checkpoint on each, and
reports:

  - Per-width best-step change_err on right-action transitions
  - Best-step k per width (does the model use deeper computation for
    wider levels?)
  - OOD test: re-instantiate the trained body at n_nca_steps > training
    depth, see if predictions stay correct or improve

Usage:
    python -m nca_wm.scripts.eval_varislide_ood_width \
        --run nca_wm/logs_halt_arch/varislide_synth_pool_seed0 \
        --ood_widths "20,24" \
        --extra_repeats "16,24,32"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import subprocess
import sys

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


def _build_model(cfg, gtoks_len, n_objs, n_steps_override=None,
                 n_repeats_override=None):
    n_steps = n_steps_override or cfg["n_nca_steps"]
    n_repeats = n_repeats_override or cfg["n_nca_repeats"]
    if n_steps_override is not None and n_repeats_override is None:
        # When stretching depth, default to fully shared (one body applied
        # n_steps times) — same body weights at every step.
        n_repeats = n_steps
    # Mirror the train-time max_seq_len construction: max(len(tokens), 1) + 1
    # (the +1 is for CLS). Without the floor at 1, games with empty token
    # lists (varislide) get pos_embed of shape (1, d) but training had (2, d).
    eff_seq_len = max(gtoks_len, 1) + 1
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=n_steps, n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg["d_model"], enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg["d_slot"], n_attn_heads=cfg["n_heads"],
        max_seq_len=eff_seq_len,
        axis_pool=cfg["axis_pool"], axis_cummax=cfg["axis_cummax"],
        global_pool=cfg["global_pool"],
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=n_repeats,
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def _ensure_synth_cache(width: int, height: int = 3, n_levels: int = 16,
                         seed: int = 999) -> str:
    """Generate a synthetic varislide cache at (width, height) if missing.

    Uses a distinct seed (default 999) so OOD eval levels never overlap
    with training-seed synth caches.
    """
    cache_dir = os.path.join(ROLLOUT_CACHE_DIR, "varislide",
                              f"synthetic_{width}x{height}")
    pat = os.path.join(cache_dir,
                        f"seed{seed}_n{n_levels}_v*_mode-tile_pattern_empirical*solv0*.npz")
    matches = glob.glob(pat)
    if matches:
        return sorted(matches, key=lambda p: -os.path.getsize(p))[0]
    # Generate via the synthetic_levels CLI.
    print(f"  [synth] generating {n_levels} held-out levels at {width}x{height}")
    cmd = [
        os.path.join(_REPO, ".venv/bin/python3"),
        os.path.join(_REPO, "nca_wm/synthetic_levels.py"),
        "--game", "varislide",
        "--n_levels", str(n_levels),
        "--width", str(width), "--height", str(height),
        "--mode", "tile_pattern_empirical",
        "--max_iters", "10000", "--timeout_ms", "10000",
        "--min_states", "5",
        "--max_attempts_per_level", "200",
        "--seed", str(seed),
    ]
    subprocess.run(cmd, check=True, env={**os.environ,
                                          "JAX_PLATFORMS": "cpu"})
    matches = glob.glob(pat)
    if not matches:
        raise RuntimeError(f"Synth gen produced no cache at {cache_dir}")
    return matches[0]


def evaluate_at_depth(run_dir: str, ood_widths: list[int],
                      extra_repeats: list[int] | None = None,
                      n_levels_eval: int = 16):
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

    # max H from training
    train_H = g["H"]
    train_W = g["W"]

    print(f"=== {run_dir} ===")
    print(f"  trained at n_steps={cfg['n_nca_steps']}, n_repeats={cfg['n_nca_repeats']}")
    print(f"  pool: ap={cfg['axis_pool']} ac={cfg['axis_cummax']} gp={cfg['global_pool']}")
    print(f"  trained max (H, W) = ({train_H}, {train_W})")

    # Combine: trained depth + extra depths (ints)
    depths_to_test = [cfg["n_nca_steps"]] + (extra_repeats or [])
    depths_to_test = sorted(set(depths_to_test))

    for width in ood_widths:
        cache_path = _ensure_synth_cache(width, height=3,
                                          n_levels=n_levels_eval)
        z = np.load(cache_path)
        s_raw = z["states"]
        a = z["actions"]
        n_raw = z["next_states"]
        if len(s_raw) == 0:
            print(f"\n--- OOD width={width}: no cached transitions ---")
            continue
        # BFS caches store states bit-packed along W with a 'W' field;
        # synthetic caches store unpacked states with no 'W' field. Handle
        # both transparently.
        if "W" in z.files:
            lW = int(z["W"])
            s = _unpack_states(s_raw, lW).astype(np.float32)
            n = _unpack_states(n_raw, lW).astype(np.float32)
        else:
            s = s_raw.astype(np.float32)
            n = n_raw.astype(np.float32)
        # Filter to right action only
        mask = a == 3
        s, a, n = s[mask], a[mask], n[mask]
        if len(s) == 0:
            print(f"\n--- OOD width={width}: no right-action transitions ---")
            continue
        B = min(64, len(s))
        # Pad to (C_pad, max(train_H, lH), max(train_W, lW))
        lH = s.shape[2]
        lW = s.shape[3]
        eval_H = max(train_H, lH)
        eval_W = max(train_W, lW)
        pad = lambda arr: np.pad(arr[:B],
            [(0,0), (0, C_pad-arr.shape[1]),
             (0, eval_H-arr.shape[2]), (0, eval_W-arr.shape[3])])
        sb = jnp.asarray(pad(s))
        nb = jnp.asarray(pad(n))
        ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[a[:B]])
        # Match train-time tokens: pad to max(len(tokens), 1) and mask zeros.
        eff_len = max(toks_np.shape[0], 1)
        toks_padded = np.zeros((B, eff_len), dtype=np.int32)
        if toks_np.shape[0] > 0:
            toks_padded[:, :toks_np.shape[0]] = toks_np
        toks = jnp.asarray(toks_padded)
        # Mask is False for the synthetic "empty token" pad cells and True
        # for real tokens; with 0 real tokens this is all False, but the
        # model handles that case internally (slot_xattn over empty keys
        # returns the residual / zero attn output).
        gmask = np.zeros((B, eff_len), dtype=bool)
        if toks_np.shape[0] > 0:
            gmask[:, :toks_np.shape[0]] = True
        gm = jnp.asarray(gmask)

        print(f"\n--- OOD width={width} (lH={lH}, lW={lW}, B={B} right-actions) ---")
        for n_steps in depths_to_test:
            model = _build_model(cfg, toks_np.shape[0], C_pad,
                                  n_steps_override=n_steps)
            try:
                out = model.apply(params, sb, ab, toks, gm)
            except Exception as e:
                print(f"  n_steps={n_steps}: FAILED — {e}")
                continue
            # Extract per-step logits if adaptive_halt; else single readout
            is_adaptive = cfg.get("adaptive_halt", False)
            if is_adaptive:
                ps_logits = out[-1][0]   # (T, B, C, H, W)
                preds = (jax.nn.sigmoid(ps_logits) > 0.5).astype(jnp.float32)
            else:
                logits = out[0]          # (B, C, H, W)
                preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)[None]
            # Per-step change_err
            changed = (sb != nb).astype(jnp.float32)
            n_changed = changed.sum()
            nb_full = jnp.broadcast_to(nb[None], preds.shape)
            changed_full = jnp.broadcast_to(changed[None], preds.shape)
            wrong = ((preds != nb_full).astype(jnp.float32) * changed_full).sum(
                axis=(1, 2, 3, 4))
            err_per_step = wrong / jnp.maximum(n_changed, 1.0)
            best_k = int(np.argmin(np.asarray(err_per_step)))
            best_err = float(err_per_step[best_k])
            last_err = float(err_per_step[-1])
            print(f"  n_steps={n_steps:>3d}  best_k={best_k+1:>3d}  "
                  f"L_best chg={best_err:.4f}  L_T chg={last_err:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ood_widths", default="20,24")
    p.add_argument("--extra_repeats", default="32",
                   help="Comma-separated n_steps values to test beyond training depth.")
    p.add_argument("--n_levels_eval", type=int, default=16)
    args = p.parse_args()
    widths = [int(x) for x in args.ood_widths.split(",")]
    extra = [int(x) for x in args.extra_repeats.split(",") if x.strip()] if args.extra_repeats else []
    evaluate_at_depth(args.run, widths, extra, args.n_levels_eval)


if __name__ == "__main__":
    main()

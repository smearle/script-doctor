"""Failure-mode inspector for varislide multi-grid synth checkpoints.

Goal: pin down *what* the model is predicting when it fails to learn the
slide rule under multi-grid training. The aggregate change_err number
(0.30-0.50) doesn't distinguish:

  (a) identity collapse — model predicts "no change" everywhere
  (b) low-confidence-everywhere — sigmoids near 0.5, no clear prediction
  (c) off-by-N — model predicts a slide of the wrong length
  (d) wrong direction / wrong cells — predicts change in random places

For each of these the right architectural intervention is different.

Reports per grid-width (filtered to action="right"):
  - Mean sigmoid logit at GT-changed cells: should-change-to-1, should-change-to-0
  - Mean sigmoid logit at no-change cells
  - Per-channel confidence histogram on the player channel
  - Visualization of one (state, gt_next, pred) triplet per width

Usage:
    .venv/bin/python3 nca_wm/scripts/inspect_varislide_failure.py \\
        --run nca_wm/logs_canary/varislide_reproduce_h128
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
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
RIGHT_ACTION = 3  # JAX action mapping per project memory reference_action_mappings


def _build_model(cfg, gtoks_len, n_objs):
    eff_seq_len = max(gtoks_len, 1) + 1
    return RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"], n_steps=cfg["n_nca_steps"], n_out=n_objs,
        vocab_size=cfg["vocab_size"] + 1,
        enc_d_model=cfg["d_model"], enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"], n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg["d_slot"], n_attn_heads=cfg["n_heads"],
        max_seq_len=eff_seq_len,
        axis_pool=cfg["axis_pool"], axis_cummax=cfg["axis_cummax"],
        global_pool=cfg["global_pool"],
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        n_repeats=cfg["n_nca_repeats"],
        adaptive_halt=cfg.get("adaptive_halt", False),
    )


def _list_synth_caches(game: str, train_seed: int = 0):
    """Find all synthetic_{w}x{h} caches for this game from training seed."""
    pat = os.path.join(ROLLOUT_CACHE_DIR, game,
                       f"synthetic_*x*",
                       f"seed{train_seed}_n*_v*_mode-*solv0*.npz")
    return sorted(glob.glob(pat))


def _ascii_render_state(state: np.ndarray, channel_names: list[str] | None = None) -> str:
    """state: (C, H, W) one-hot. Render the top-priority channel per cell."""
    C, H, W = state.shape
    glyphs = ["."] * C
    if channel_names:
        for i, n in enumerate(channel_names):
            ln = n.lower()
            if "background" in ln: glyphs[i] = "."
            elif "wall" in ln: glyphs[i] = "#"
            elif "player" in ln: glyphs[i] = "P"
            else: glyphs[i] = chr(ord("a") + (i % 26))
    else:
        glyphs = [str(i % 10) for i in range(C)]
    out_lines = []
    for r in range(H):
        row = []
        for c in range(W):
            # take topmost (highest-channel) cell that is "on"
            on = np.where(state[:, r, c] > 0.5)[0]
            if len(on) == 0:
                row.append(" ")
            else:
                # last-listed-channel-wins (matches CollisionLayers semantics
                # for varislide: Player > Wall > Background)
                row.append(glyphs[int(on[-1])])
        out_lines.append("".join(row))
    return "\n".join(out_lines)


def _ascii_render_pred(pred_sig: np.ndarray, channel_names: list[str] | None = None) -> str:
    """pred_sig: (C, H, W) sigmoid output. Render with confidence markers."""
    C, H, W = pred_sig.shape
    out_lines = []
    for r in range(H):
        row = []
        for c in range(W):
            cell_probs = pred_sig[:, r, c]
            top = int(np.argmax(cell_probs))
            top_p = float(cell_probs[top])
            on_count = int(np.sum(cell_probs > 0.5))
            glyph = "."
            if channel_names and top < len(channel_names):
                ln = channel_names[top].lower()
                if "wall" in ln: glyph = "#"
                elif "player" in ln: glyph = "P"
                elif "background" in ln: glyph = "."
                else: glyph = chr(ord("a") + (top % 26))
            # confidence marker
            if on_count == 0:
                glyph = "?"  # nothing above 0.5 — unconfident
            elif on_count > 1:
                glyph = "*"  # multiple channels above 0.5 — overlapped
            elif top_p < 0.7:
                glyph = glyph.lower() if glyph.isupper() else "~"
            row.append(glyph)
        out_lines.append("".join(row))
    return "\n".join(out_lines)


def inspect_run(run_dir: str, max_examples_per_width: int = 3,
                channel_names: list[str] | None = None):
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
    train_H = g["H"]
    train_W = g["W"]

    # Channel names from the game_info if available (object names in the
    # spec's order). Fall back to numeric.
    if channel_names is None:
        channel_names = [
            o.get("name", f"obj{i}") if isinstance(o, dict) else str(o)
            for i, o in enumerate(g.get("objects", []))
        ] or None

    print(f"=== inspecting {run_dir} ===")
    print(f"  cfg: n_hid={cfg['n_hid']} n_steps={cfg['n_nca_steps']} "
          f"n_repeats={cfg['n_nca_repeats']} n_slots={cfg['n_slots']}")
    print(f"  pool: ap={cfg['axis_pool']} ac={cfg['axis_cummax']} gp={cfg['global_pool']}")
    print(f"  trained max (H, W) = ({train_H}, {train_W})")
    print(f"  channel names: {channel_names}")
    print()

    model = _build_model(cfg, toks_np.shape[0], C_pad)

    caches = _list_synth_caches(g["name"])
    if not caches:
        print(f"  NO synth caches found for {g['name']} — was training run?")
        return
    print(f"  Found {len(caches)} synth caches:")
    for c in caches:
        print(f"    {os.path.relpath(c, _REPO)}")
    print()

    # Aggregate stats across all widths
    agg_cells = {"to_on": [], "to_off": [], "no_change": []}

    for cache_path in caches:
        z = np.load(cache_path)
        s_raw, a, n_raw = z["states"], z["actions"], z["next_states"]
        if "W" in z.files:
            lW = int(z["W"])
            s = _unpack_states(s_raw, lW).astype(np.float32)
            n = _unpack_states(n_raw, lW).astype(np.float32)
        else:
            s = s_raw.astype(np.float32)
            n = n_raw.astype(np.float32)
        mask = a == RIGHT_ACTION
        s, a, n = s[mask], a[mask], n[mask]
        if len(s) == 0:
            continue
        # only keep transitions that actually change something (slide cases)
        delta = (s != n).any(axis=(1, 2, 3))
        s_changing = s[delta]
        n_changing = n[delta]
        a_changing = a[delta]
        if len(s_changing) == 0:
            continue

        B = min(max_examples_per_width, len(s_changing))
        lH = s.shape[2]
        lW = s.shape[3]
        eval_H = max(train_H, lH)
        eval_W = max(train_W, lW)
        pad = lambda arr: np.pad(
            arr[:B],
            [(0,0), (0, C_pad-arr.shape[1]),
             (0, eval_H-arr.shape[2]), (0, eval_W-arr.shape[3])])
        sb = jnp.asarray(pad(s_changing))
        nb = jnp.asarray(pad(n_changing))
        ab = jnp.asarray(np.eye(N_ACTIONS, dtype=np.float32)[a_changing[:B]])
        eff_len = max(toks_np.shape[0], 1)
        toks_padded = np.zeros((B, eff_len), dtype=np.int32)
        if toks_np.shape[0] > 0:
            toks_padded[:, :toks_np.shape[0]] = toks_np
        toks = jnp.asarray(toks_padded)
        gmask = np.zeros((B, eff_len), dtype=bool)
        if toks_np.shape[0] > 0:
            gmask[:, :toks_np.shape[0]] = True
        gm = jnp.asarray(gmask)

        out = model.apply(params, sb, ab, toks, gm)
        logits = out[0]  # (B, C, H, W)
        sig = np.asarray(jax.nn.sigmoid(logits))

        # Scope to active region (no padding)
        sb_np = np.asarray(sb)[:, :, :lH, :lW]
        nb_np = np.asarray(nb)[:, :, :lH, :lW]
        sig_np = sig[:, :, :lH, :lW]

        # Per-cell categorization
        gt_change = (sb_np != nb_np)  # bool (B, C, H, W)
        gt_to_on = gt_change & (nb_np > 0.5)   # cells GT goes 0->1
        gt_to_off = gt_change & (nb_np < 0.5)  # cells GT goes 1->0

        if gt_to_on.sum() > 0:
            agg_cells["to_on"].extend(sig_np[gt_to_on].tolist())
        if gt_to_off.sum() > 0:
            agg_cells["to_off"].extend(sig_np[gt_to_off].tolist())
        no_change = (~gt_change)
        if no_change.sum() > 0:
            # subsample to keep memory bounded
            samp = sig_np[no_change]
            if len(samp) > 1000:
                idx = np.random.choice(len(samp), 1000, replace=False)
                samp = samp[idx]
            agg_cells["no_change"].extend(samp.tolist())

        # Visual examples
        print(f"--- width={lW} (cache: {os.path.basename(os.path.dirname(cache_path))}) ---")
        print(f"  total right-action transitions in cache: {mask.sum()}; "
              f"changing: {len(s_changing)}; visualizing first {B}")
        for i in range(B):
            print(f"\n  example {i}: state -> gt_next  // pred (?=low-conf, *=overlapped, lowercase=conf<0.7)")
            # Compose side-by-side rendering
            cn = channel_names
            s_render = _ascii_render_state(sb_np[i], cn).split("\n")
            n_render = _ascii_render_state(nb_np[i], cn).split("\n")
            p_render = _ascii_render_pred(sig_np[i], cn).split("\n")
            for sr, nr, pr in zip(s_render, n_render, p_render):
                print(f"    {sr}    {nr}    {pr}")

            # Per-cell logit at the changing cells, for this example
            ic_to_on = (gt_change[i] & (nb_np[i] > 0.5))
            ic_to_off = (gt_change[i] & (nb_np[i] < 0.5))
            if ic_to_on.sum() > 0:
                v = sig_np[i][ic_to_on]
                print(f"    sigmoid at GT-to-on cells: mean={v.mean():.3f} min={v.min():.3f} max={v.max():.3f} (want >> 0.5)")
            if ic_to_off.sum() > 0:
                v = sig_np[i][ic_to_off]
                print(f"    sigmoid at GT-to-off cells: mean={v.mean():.3f} min={v.min():.3f} max={v.max():.3f} (want << 0.5)")
        print()

    # Aggregate report
    print("=== AGGREGATE (all widths, all changing transitions, action=right) ===")
    for label, arr in agg_cells.items():
        if not arr:
            continue
        a = np.array(arr)
        # Bins: "predicting on" vs "predicting off" vs "uncertain"
        on_frac = float((a > 0.7).mean())
        off_frac = float((a < 0.3).mean())
        unc_frac = float(((a >= 0.3) & (a <= 0.7)).mean())
        print(f"  {label:10s} (n={len(a):6d}): mean={a.mean():.3f}  "
              f">0.7: {on_frac:.2%}  <0.3: {off_frac:.2%}  "
              f"in [0.3,0.7] (low-conf): {unc_frac:.2%}")

    # Diagnostic interpretation
    on_arr = np.array(agg_cells["to_on"]) if agg_cells["to_on"] else np.array([])
    off_arr = np.array(agg_cells["to_off"]) if agg_cells["to_off"] else np.array([])
    nc_arr = np.array(agg_cells["no_change"]) if agg_cells["no_change"] else np.array([])
    print()
    print("=== INTERPRETATION ===")
    if len(on_arr) and on_arr.mean() < 0.4 and len(off_arr) and off_arr.mean() > 0.6:
        print("  >>> IDENTITY COLLAPSE: model predicts no change at GT-changing cells.")
    elif (len(on_arr) and 0.3 <= on_arr.mean() <= 0.7
          and len(off_arr) and 0.3 <= off_arr.mean() <= 0.7):
        print("  >>> LOW CONFIDENCE: model is unsure where to predict change.")
    elif len(on_arr) and on_arr.mean() > 0.6 and len(off_arr) and off_arr.mean() < 0.4:
        print("  >>> CONFIDENT BUT WRONG: predicting change with confidence — likely off-by-N or wrong cell.")
    else:
        print("  >>> MIXED — see per-example renderings above.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--max_examples", type=int, default=2)
    args = p.parse_args()
    inspect_run(args.run, max_examples_per_width=args.max_examples)


if __name__ == "__main__":
    main()

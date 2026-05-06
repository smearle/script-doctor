"""Sample novel games from a v3+decoder rule_attn checkpoint.

Loads the joint encoder+NCA+SlotTokenDecoder trained with
`--token_decoder_loss_weight > 0`, then:
  1. Reconstructs each training game's slots → tokens (sanity check;
     reports per-game token-accuracy).
  2. Samples slot matrices from an empirical distribution fit to the
     training slots and decodes them.
  3. Interpolates linearly between two named training games' slots and
     decodes at intermediate points.
  4. For each decoded token sequence, runs `detokenize` → PuzzleScript
     source string and attempts a JS-engine compile. Reports the
     compile-success rate (the ``engine-load success rate'' metric the
     paper TODO asks for).

This addresses two papertodos:
  * `results.tex` line 21 — symbolic autoencoder results
    (reconstruction accuracy, grammar-valid decode rate, engine-load
    success rate, examples).
  * `intro.tex` line 72 — Contribution 3 ``novel latent games'' is
    aspirational; this tool demonstrates the loop end-to-end.

Usage:
    .venv/bin/python3 -m nca_wm.sample_latent_games_rule_attn \
        --load nca_wm/logs/multi_scaling_gallery_v3_decoder \
        --n_random 16 --n_interp 9 \
        --interp_pair "sokoban_basic,Travelling_salesman" \
        --out_dir nca_wm/logs/multi_scaling_gallery_v3_decoder/sampled_games
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_run(save_dir: str, prefer_best: bool = True):
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    pp = "params_best.pkl" if (
        prefer_best and os.path.exists(os.path.join(save_dir, "params_best.pkl"))
    ) else "params.pkl"
    with open(os.path.join(save_dir, pp), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(save_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    print(f"Loaded {pp} from {save_dir}", file=sys.stderr)
    return cfg, params, game_infos


def _derive_vocab_size(cfg, game_infos):
    """Re-derive vocab_size from game_infos when missing from config (some
    checkpoints don't stash it). Mirrors `train.py:5012`."""
    if cfg.get("vocab_size") is not None:
        return cfg["vocab_size"]
    from nca_wm.tokenize_game import VOCAB_SIZE_EXT
    max_id = 0
    for g in game_infos:
        tids = g.get("token_ids") or []
        if tids:
            max_id = max(max_id, int(max(tids)))
    return max(max_id + 1, VOCAB_SIZE_EXT + 1)


def _max_seq_lens_from_params(params):
    """Find encoder's and decoder's max_seq_len from their pos_embed shapes.

    Returns (enc_max, dec_max) or (None, None) if not found. Joint params
    layout: {"wm": <world_model>, "dec": <decoder>}. Encoder's pos_embed is
    deep inside wm (under params/game_encoder/pos_embed). Decoder's pos_embed
    is under dec/params/pos_embed.
    """
    def _walk(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if "pos_embed" in str(k) and isinstance(v, dict) and "embedding" in v:
                    return v["embedding"].shape[0]
                r = _walk(v)
                if r is not None:
                    return r
        return None
    enc_max = _walk(params.get("wm", params))
    dec_max = _walk(params.get("dec", {}))
    return enc_max, dec_max


def build_modules(cfg, game_infos, params=None):
    """Build RuleSlotEncoder and SlotTokenDecoder matching the saved cfg."""
    from nca_wm.rule_attn_model import RuleSlotEncoder
    from nca_wm.token_decoder import SlotTokenDecoder

    vocab_size = _derive_vocab_size(cfg, game_infos)
    # Encoder uses max_seq_len = max_tok_len + 1 (per train.py line 5022).
    # Decoder uses max_seq_len = max_tok_len directly (per train.py line 2762).
    enc_max_seq_len, dec_max_seq_len = (None, None)
    if params is not None:
        enc_max_seq_len, dec_max_seq_len = _max_seq_lens_from_params(params)
    if enc_max_seq_len:
        max_tok_len = enc_max_seq_len - 1
    else:
        max_tok_len = max(
            max((len(g.get("token_ids", [])) for g in game_infos), default=1), 1
        )
        enc_max_seq_len = max_tok_len + 1
    if not dec_max_seq_len:
        dec_max_seq_len = max_tok_len  # joint training default
    enc = RuleSlotEncoder(
        vocab_size=vocab_size + 1,
        max_seq_len=enc_max_seq_len,
        d_model=cfg["d_model"],
        n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"],
        d_slot=cfg["d_slot"],
        n_heads=cfg["n_heads"],
    )
    dec = SlotTokenDecoder(
        vocab_size=vocab_size + 1,
        max_seq_len=dec_max_seq_len,
        d_model=cfg.get("decoder_d_model", 128),
        n_heads=cfg.get("decoder_n_heads", 4),
        n_layers=cfg.get("decoder_n_layers", 4),
        d_slot=cfg["d_slot"],
    )
    return enc, dec, max_tok_len


def pad_tokens(token_ids, max_seq_len):
    pad = np.zeros(max_seq_len, dtype=np.int32)
    mask = np.zeros(max_seq_len, dtype=np.bool_)
    L = min(len(token_ids), max_seq_len)
    pad[:L] = token_ids[:L]
    mask[:L] = True
    return pad, mask, L


def decode_slots_greedy(decoder, dec_params, slots, max_len, bos_id=0,
                         eos_id=None):
    """Greedy decode from slots → token sequence (B, max_len)."""
    from nca_wm.token_decoder import sample_tokens_from_slots
    return sample_tokens_from_slots(
        decoder, dec_params, slots, max_len=max_len,
        bos_id=bos_id, eos_id=eos_id, temperature=0.0,
    )


def try_compile(text: str, ps_parser, name: str) -> tuple[bool, str]:
    """Save `text` to a tempfile + try JS-engine compile. Returns (ok, reason)."""
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.preprocessing import add_extra_games_dir
    import tempfile

    # We need the parser to find this game's .txt by name. Write to a
    # tempdir and register it so get_tree_from_txt picks it up.
    tmp = tempfile.mkdtemp(prefix="latent_decode_")
    safe_name = name.replace("/", "_")
    fp = os.path.join(tmp, safe_name + ".txt")
    with open(fp, "w") as f:
        f.write(text)
    add_extra_games_dir(tmp)
    try:
        backend = CppPuzzleScriptBackend()
        backend.compile_and_serialize(ps_parser, safe_name)
        return True, "ok"
    except Exception as e:
        # truncate long error messages
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return False, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--n_random", type=int, default=16,
                    help="N random samples drawn from the empirical "
                         "slot distribution.")
    ap.add_argument("--n_interp", type=int, default=9,
                    help="N interpolation points between two games.")
    ap.add_argument("--interp_pair", default=None,
                    help="Comma-separated pair of training-game names. "
                         "Default: pick two extremes by ||z|| from the "
                         "encoded training slots.")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_compile", action="store_true",
                    help="Skip JS-engine compile validation (faster).")
    args = ap.parse_args()

    cfg, params, game_infos = load_run(args.load)
    encoder, decoder, max_tok_len = build_modules(cfg, game_infos, params=params)
    dec_max_seq_len = decoder.max_seq_len
    enc_max_seq_len = encoder.max_seq_len

    # The joint params layout is {"wm": <world model>, "dec": <decoder>}
    if isinstance(params, dict) and "wm" in params and "dec" in params:
        wm_params = params["wm"]
        dec_params = params["dec"]
    else:
        # Fallback for non-joint runs
        wm_params = params
        dec_params = None
    enc_params = {"params": wm_params["params"]["game_encoder"]}

    out_dir = args.out_dir or os.path.join(args.load, "sampled_games")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing outputs to {out_dir}", file=sys.stderr)

    if dec_params is None:
        print("ERROR: this checkpoint has no joint decoder. Run with "
              "--token_decoder_loss_weight > 0 to train one.",
              file=sys.stderr)
        sys.exit(1)

    # Encode all training games → slot matrices
    print(f"\nEncoding {len(game_infos)} training games...",
          file=sys.stderr)
    @jax.jit
    def _enc(tids, mask):
        return encoder.apply(enc_params, tids[None], mask[None])  # (1, K, d)

    slots_list = []
    tokens_list = []
    masks_list = []
    names = []
    for info in game_infos:
        pad, mask, L = pad_tokens(info.get("token_ids", []), max_tok_len + 1)
        slots = np.array(_enc(jnp.array(pad), jnp.array(mask))[0])
        slots_list.append(slots)
        tokens_list.append(pad)
        masks_list.append(mask)
        names.append(info["name"])
    Z = np.stack(slots_list)  # (N, K, d_slot)
    print(f"  encoded slots: {Z.shape}", file=sys.stderr)

    # ---- Mode 1: teacher-forced reconstruction for each training game ----
    # Single forward pass per game (much faster than autoregressive).
    print(f"\n[Mode 1] Teacher-forced reconstruction across all "
          f"{len(names)} games...", file=sys.stderr)

    from nca_wm.token_decoder import shift_right

    @jax.jit
    def _decoder_logits(slots, tokens_in):
        return decoder.apply(dec_params, tokens_in, slots, deterministic=True)

    # EOS is unconditional in tokenization, so the decoder always learns it
    # and we always stop AR generation at the first emitted EOS.
    from nca_wm.tokenize_game import EOS_ID
    eos_id_for_sampling = EOS_ID

    def _decode_batch(slots_batch, max_len):
        # Autoregressive sampling for novel-sample mode (slow, only used on
        # a handful of samples).
        return decode_slots_greedy(decoder, dec_params, slots_batch, max_len,
                                    eos_id=eos_id_for_sampling)

    recon_acc = []
    for i, name in enumerate(names):
        slots_b = jnp.array(Z[i:i+1])
        # Match the training feed: shifted = [BOS, target[:-1]], decoder
        # predicts target[t] from shifted[:t+1].
        target = tokens_list[i][:dec_max_seq_len]
        valid = masks_list[i][:dec_max_seq_len]
        shifted = np.array(shift_right(jnp.array(target[None])))[0]
        logits = np.array(_decoder_logits(slots_b, jnp.array(shifted[None]))[0])
        preds = logits.argmax(axis=-1)
        match = (preds == target) & valid
        n_cmp = int(valid.sum())
        acc = float(match.sum() / n_cmp) if n_cmp > 0 else 1.0
        recon_acc.append(acc)
    recon_acc = np.array(recon_acc)
    print(f"  reconstruction accuracy: mean={recon_acc.mean():.3f} "
          f"median={np.median(recon_acc):.3f} min={recon_acc.min():.3f}",
          file=sys.stderr)
    n_perfect = int((recon_acc == 1.0).sum())
    print(f"  perfect-reconstruction games: {n_perfect}/{len(names)}",
          file=sys.stderr)

    # ---- Mode 2: random samples from empirical slot distribution ----
    rng = np.random.default_rng(args.seed)
    print(f"\n[Mode 2] Random samples ({args.n_random} from per-slot "
          f"empirical Gaussian)...", file=sys.stderr)
    # Per-slot mean/std (treat each slot dim independently for simplicity)
    mu = Z.mean(axis=0)            # (K, d)
    sigma = Z.std(axis=0) + 1e-6   # (K, d)
    K, d = mu.shape
    sampled_slots = mu[None] + sigma[None] * rng.standard_normal(
        (args.n_random, K, d)
    ).astype(np.float32)

    sampled_decoded = []
    for i in range(args.n_random):
        slots_b = jnp.array(sampled_slots[i:i+1])
        decoded = np.array(_decode_batch(slots_b, max_tok_len))[0]
        sampled_decoded.append(decoded)

    # ---- Mode 3: interpolation ----
    print(f"\n[Mode 3] Interpolation ({args.n_interp} points)...",
          file=sys.stderr)
    if args.interp_pair:
        a_name, b_name = args.interp_pair.split(",")
        a_name = a_name.strip()
        b_name = b_name.strip()
    else:
        # Pick two games that are far apart in the slot space (max ||z_a -
        # z_b||). Cheap: take the two most-extreme on PC1.
        flat = Z.reshape(len(names), -1)
        Xc = flat - flat.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[0]
        a_idx = int(np.argmin(proj))
        b_idx = int(np.argmax(proj))
        a_name = names[a_idx]
        b_name = names[b_idx]
    a_idx = names.index(a_name)
    b_idx = names.index(b_name)
    z_a = Z[a_idx]
    z_b = Z[b_idx]
    ts = np.linspace(0.0, 1.0, args.n_interp).astype(np.float32)
    interp_slots = (1 - ts[:, None, None]) * z_a[None] + ts[:, None, None] * z_b[None]
    interp_decoded = []
    for i in range(args.n_interp):
        slots_b = jnp.array(interp_slots[i:i+1])
        decoded = np.array(_decode_batch(slots_b, max_tok_len))[0]
        interp_decoded.append(decoded)
    print(f"  interpolating: {a_name} → {b_name}", file=sys.stderr)

    # ---- Detokenize + (optional) compile ----
    if args.no_compile:
        ps_parser = None
    else:
        from puzzlescript_jax.utils import init_ps_lark_parser
        print(f"\nInitializing JS parser for compile validation...",
              file=sys.stderr)
        ps_parser = init_ps_lark_parser()

    from nca_wm.detokenize_game import detokenize

    summary = {
        "n_train_games": len(names),
        "reconstruction": {
            "mean_acc": float(recon_acc.mean()),
            "median_acc": float(np.median(recon_acc)),
            "min_acc": float(recon_acc.min()),
            "n_perfect": n_perfect,
        },
        "random": [],
        "interp": {"a": a_name, "b": b_name, "points": []},
    }

    def _detok_and_save(decoded_ids, label):
        ids = [int(t) for t in decoded_ids if int(t) > 0]  # strip BOS/PAD=0
        try:
            text = detokenize(ids, title=label)
        except Exception as e:
            return None, f"detokenize: {e}"
        out_fp = os.path.join(out_dir, label + ".txt")
        with open(out_fp, "w") as f:
            f.write(text)
        if ps_parser is None:
            return out_fp, "skipped"
        try:
            ok, reason = try_compile(text, ps_parser, label)
            return out_fp, ("compile_ok" if ok else f"compile_fail: {reason}")
        except Exception as e:
            # Don't let a buggy compile-check kill the rest of the loop.
            return out_fp, f"compile_error: {type(e).__name__}: {str(e)[:120]}"

    print(f"\nDecoding random samples + compile-check...",
          file=sys.stderr)
    for i, decoded in enumerate(sampled_decoded):
        label = f"random_{i:02d}"
        path, status = _detok_and_save(decoded, label)
        summary["random"].append({"label": label, "file": path,
                                   "status": status})
        print(f"  {label}: {status}", file=sys.stderr)

    print(f"\nDecoding interpolation points + compile-check...",
          file=sys.stderr)
    for i, decoded in enumerate(interp_decoded):
        label = f"interp_{i:02d}_{ts[i]:.2f}"
        path, status = _detok_and_save(decoded, label)
        summary["interp"]["points"].append({"t": float(ts[i]), "label": label,
                                             "file": path, "status": status})
        print(f"  {label} (t={ts[i]:.2f}): {status}", file=sys.stderr)

    # Aggregate compile-success rate
    n_random = len(summary["random"])
    n_random_ok = sum(
        1 for r in summary["random"] if r["status"] == "compile_ok"
    )
    n_interp = len(summary["interp"]["points"])
    n_interp_ok = sum(
        1 for r in summary["interp"]["points"] if r["status"] == "compile_ok"
    )
    summary["compile_success_random"] = n_random_ok / max(n_random, 1)
    summary["compile_success_interp"] = n_interp_ok / max(n_interp, 1)
    print(f"\nCompile success: random {n_random_ok}/{n_random}, "
          f"interp {n_interp_ok}/{n_interp}", file=sys.stderr)

    summary_fp = os.path.join(out_dir, "summary.json")
    with open(summary_fp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_fp}", file=sys.stderr)


if __name__ == "__main__":
    main()

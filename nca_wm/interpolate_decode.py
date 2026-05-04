"""Interpolate game latents and decode the resulting PuzzleScript source.

Loads a joint-trained checkpoint (rule_attn WM + SlotTokenDecoder), encodes
two endpoint games into K slot vectors, then for each t in [0, 1] interpolates
the slots and runs the decoder autoregressively to reconstruct a token
sequence. The token sequence is written to disk verbatim (numbers; the
human-readable detokenizer is a follow-up).

Two interpolation modes are supported:
  - ``--mode linear``: smoothly interpolate ALL slots together,
    z(t) = (1-t) * z_A + t * z_B. Default.
  - ``--mode swap``: for each k in [0, K), produce one decode where slot k
    is taken from B and the rest from A (and vice versa). Tests slot
    disentanglement: a single-slot swap should change a focal piece of the
    grammar while leaving the rest of the source intact.

Usage:
    python nca_wm/interpolate_decode.py \\
        --load nca_wm/logs/scaling_2_joint_v1 \\
        --game_a sokoban_basic --game_b nekopuzzle \\
        --n_interp 7 --mode linear
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.rule_attn_model import RuleAttnNCAWorldModel
from nca_wm.token_decoder import SlotTokenDecoder, sample_tokens_from_slots
from nca_wm.tokenize_game import tokens_to_str
from nca_wm.detokenize_game import detokenize


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_run(save_dir: str):
    with open(os.path.join(save_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(save_dir, "params.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(save_dir, "game_infos.pkl"), "rb") as f:
        game_infos = pickle.load(f)
    return cfg, params, game_infos


def split_params(params):
    """Joint-checkpoint params is {"wm": ..., "dec": ...}; raise otherwise."""
    if not (isinstance(params, dict) and {"wm", "dec"} <= set(params.keys())):
        raise ValueError(
            "interpolate_decode requires a joint checkpoint "
            "(token_decoder_loss_weight > 0)."
        )
    return params["wm"], params["dec"]


def build_models(cfg, game_infos):
    if cfg.get("architecture") != "rule_attn":
        raise NotImplementedError(
            f"Only --architecture rule_attn is supported here; got "
            f"{cfg.get('architecture')!r}."
        )
    max_C = max(g["n_objs"] for g in game_infos)
    max_tok_len = max(max((len(g.get("token_ids", [])) for g in game_infos), default=1), 1)
    vocab_size = int(cfg["vocab_size"])

    wm = RuleAttnNCAWorldModel(
        n_hid=cfg["n_hid"],
        n_steps=cfg["n_nca_steps"],
        n_out=max_C,
        vocab_size=vocab_size + 1,
        enc_d_model=cfg["d_model"],
        enc_n_self_layers=cfg["n_enc_layers"],
        n_slots=cfg["n_slots"],
        n_app_slots=cfg.get("n_app_slots", 0),
        d_slot=cfg["d_slot"],
        n_attn_heads=cfg["n_heads"],
        max_seq_len=max_tok_len + 1,
        axis_pool=cfg.get("axis_pool", False),
        axis_cummax=cfg.get("axis_cummax", False),
        global_pool=cfg.get("global_pool", False),
        use_vq=cfg.get("vq_codebook", False),
        vq_codebook_size=cfg.get("vq_codebook_size", 512),
        vq_commitment_weight=cfg.get("vq_commitment_weight", 0.25),
        n_repeats=cfg.get("n_nca_repeats", 1),
        mask_hidden=cfg.get("mask_hidden", False),
        use_layernorm=cfg.get("use_layernorm", False),
        input_skip=cfg.get("input_skip", False),
        adaptive_halt=cfg.get("adaptive_halt", False),
    )
    dec = SlotTokenDecoder(
        vocab_size=vocab_size + 1,
        max_seq_len=max_tok_len,
        d_model=cfg["decoder_d_model"],
        n_heads=cfg["decoder_n_heads"],
        n_layers=cfg["decoder_n_layers"],
        d_slot=cfg["d_slot"],
    )
    return wm, dec, max_tok_len


def find_game(game_infos, name):
    for i, info in enumerate(game_infos):
        if info["name"] == name:
            return i, info
    names = [g["name"] for g in game_infos]
    raise ValueError(f"Game {name!r} not in checkpoint; available: {names}")


def pad_tokens(token_ids, max_seq_len):
    pad = np.zeros(max_seq_len, dtype=np.int32)
    mask = np.zeros(max_seq_len, dtype=np.bool_)
    L = min(len(token_ids), max_seq_len)
    pad[:L] = token_ids[:L]
    mask[:L] = True
    return pad, mask


# ----------------------------------------------------------------------
# Encoding via the WM module (slots-only forward)
# ----------------------------------------------------------------------

def encode_slots(wm, wm_params, info, max_tok_len, max_C, max_H, max_W):
    """Run the encoder by calling the WM with dummy state/action and harvest slots."""
    pad, mask = pad_tokens(info.get("token_ids", []), max_tok_len)
    dummy_state = jnp.zeros((1, max_C, max_H, max_W), dtype=jnp.float32)
    dummy_action = jnp.zeros((1, 5), dtype=jnp.float32)
    _, _, _, slots = wm.apply(
        wm_params, dummy_state, dummy_action,
        jnp.array(pad[None]), jnp.array(mask[None]),
        return_slots=True,
    )
    return np.array(slots[0])  # (n_slots, d_slot)


# ----------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------

def decode_slots(dec, dec_params, slots, max_len, eos_token_id=0):
    """Greedy autoregressive decode from a (1, K, d_slot) slot tensor.

    sample_tokens_from_slots returns the *input* sequence, which is
    [BOS, pred_0, pred_1, ..., pred_{max_len-2}] — the final prediction is
    computed but discarded. We strip the BOS and append a final greedy step
    so the returned array contains exactly max_len predicted tokens.
    """
    raw = sample_tokens_from_slots(
        dec, dec_params,
        jnp.array(slots[None].astype(np.float32)),
        max_len=max_len, bos_id=0, temperature=0.0,
    )
    # Recover the final prediction (logits at position max_len-1).
    final_logits = dec.apply(dec_params, raw, jnp.array(slots[None].astype(np.float32)),
                              deterministic=True)
    final_pred = int(jnp.argmax(final_logits[0, -1]))
    seq = np.array(raw[0])
    return np.concatenate([seq[1:], [final_pred]]).astype(np.int32)


def stop_at_pad(token_seq, pad_id=0):
    """Truncate trailing padding (treat the first pad_id after pos 0 as EOS).

    Token id 0 = PAD per tokenize_game.py."""
    if len(token_seq) == 0:
        return token_seq
    # First pad after position 0 ends the meaningful output.
    for i in range(1, len(token_seq)):
        if token_seq[i] == pad_id:
            return token_seq[:i]
    return token_seq


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True,
                   help="Joint-trained run dir (with params.pkl shaped "
                        "{wm, dec}).")
    p.add_argument("--game_a", required=True)
    p.add_argument("--game_b", required=True)
    p.add_argument("--n_interp", type=int, default=7,
                   help="Number of t values across [0, 1] inclusive.")
    p.add_argument("--mode", default="linear", choices=["linear", "swap"],
                   help="linear = whole-slot interp; swap = one slot taken "
                        "from B at a time.")
    p.add_argument("--save_dir", default=None,
                   help="Output dir (default: <load>/interp_decode).")
    args = p.parse_args()

    cfg, params, game_infos = load_run(args.load)
    wm_params, dec_params = split_params(params)
    wm, dec, max_tok_len = build_models(cfg, game_infos)
    save_dir = args.save_dir or os.path.join(args.load, "interp_decode")
    os.makedirs(save_dir, exist_ok=True)

    max_C = max(g["n_objs"] for g in game_infos)
    max_H = max(g["H"] for g in game_infos)
    max_W = max(g["W"] for g in game_infos)

    _, info_a = find_game(game_infos, args.game_a)
    _, info_b = find_game(game_infos, args.game_b)
    sl_a = encode_slots(wm, wm_params, info_a, max_tok_len, max_C, max_H, max_W)
    sl_b = encode_slots(wm, wm_params, info_b, max_tok_len, max_C, max_H, max_W)
    print(f"Encoded slots: A={sl_a.shape}, B={sl_b.shape}")
    K, d_slot = sl_a.shape
    n_dyn = cfg["n_slots"] - cfg.get("n_app_slots", 0)
    print(f"  n_dyn={n_dyn}, n_app={K - n_dyn}")

    # Sanity-check round-trip recon on each endpoint.
    for tag, info, sl in [("A", info_a, sl_a), ("B", info_b, sl_b)]:
        out = decode_slots(dec, dec_params, sl, max_tok_len)
        out = stop_at_pad(out)
        gt = np.asarray(info.get("token_ids", []), dtype=np.int32)
        # Compare prefix overlap.
        n_match = 0
        for i in range(min(len(out), len(gt))):
            if int(out[i]) == int(gt[i]):
                n_match += 1
            else:
                break
        print(f"  recon[{tag} = {info['name']}]: prefix_match={n_match}/{len(gt)}")

    if args.mode == "linear":
        ts = np.linspace(0.0, 1.0, args.n_interp)
        for t_idx, t in enumerate(ts):
            sl_t = (1 - t) * sl_a + t * sl_b
            out = decode_slots(dec, dec_params, sl_t, max_tok_len)
            out = stop_at_pad(out)
            stem = os.path.join(save_dir, f"linear_t{t_idx:02d}_{t:.2f}")
            header = (f"# linear interp: t={t:.4f} from {args.game_a} -> {args.game_b}\n"
                      f"# n_tokens={len(out)}\n")
            with open(stem + ".txt", "w") as f:
                f.write(header)
                f.write(" ".join(str(int(x)) for x in out) + "\n")
            with open(stem + ".names.txt", "w") as f:
                f.write(header)
                f.write(tokens_to_str([int(x) for x in out]) + "\n")
            with open(stem + ".ps.txt", "w") as f:
                f.write(detokenize(
                    [int(x) for x in out],
                    title=f"interp_t{t:.2f}_{args.game_a}_to_{args.game_b}",
                ))
            print(f"  saved {stem}.{{txt,names.txt,ps.txt}} ({len(out)} tokens)")
    elif args.mode == "swap":
        # For each slot k, produce two outputs: only-k-from-B (in A's source) and
        # only-k-from-A (in B's source). Useful to find which slot owns what.
        def _save_pair(stem: str, header: str, out, title: str):
            with open(stem + ".txt", "w") as f:
                f.write(header)
                f.write(" ".join(str(int(x)) for x in out) + "\n")
            with open(stem + ".names.txt", "w") as f:
                f.write(header)
                f.write(tokens_to_str([int(x) for x in out]) + "\n")
            with open(stem + ".ps.txt", "w") as f:
                f.write(detokenize([int(x) for x in out], title=title))

        for k in range(K):
            tag = "dyn" if k < n_dyn else "app"

            sl_a_with_kb = sl_a.copy()
            sl_a_with_kb[k] = sl_b[k]
            out = decode_slots(dec, dec_params, sl_a_with_kb, max_tok_len)
            out = stop_at_pad(out)
            header = (f"# swap: A's source w/ slot{k} ({tag}) replaced by B's slot{k}\n"
                      f"# n_tokens={len(out)}\n")
            _save_pair(os.path.join(save_dir, f"swap_A_with_k{k:02d}_{tag}_fromB"),
                       header, out, title=f"swap_A_k{k}_fromB")

            sl_b_with_ka = sl_b.copy()
            sl_b_with_ka[k] = sl_a[k]
            out = decode_slots(dec, dec_params, sl_b_with_ka, max_tok_len)
            out = stop_at_pad(out)
            header = (f"# swap: B's source w/ slot{k} ({tag}) replaced by A's slot{k}\n"
                      f"# n_tokens={len(out)}\n")
            _save_pair(os.path.join(save_dir, f"swap_B_with_k{k:02d}_{tag}_fromA"),
                       header, out, title=f"swap_B_k{k}_fromA")
            print(f"  swapped slot {k} ({tag})")


if __name__ == "__main__":
    main()

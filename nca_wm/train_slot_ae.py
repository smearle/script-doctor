"""Train a slot-based game-token autoencoder.

Companion to train_token_ae.py — but here the encoder is `RuleSlotEncoder`
(from rule_attn_model.py, the architecture used for the cosine_v2 19-game
and gallery world models) and the decoder is `SlotTokenDecoder` (cross-
attention to the K-slot game latent).

Two modes:
  --init_from <ckpt_dir>   : initialize encoder params from a trained
        RuleAttnNCAWorldModel checkpoint.
  (default)                : fresh random encoder.

The slot encoder produces (N_games, K, d_slot) — much richer than the
single-z FiLM encoder. The decoder uses cross-attention at every layer to
attend into those K slots, mirroring the per-step cross-attention used in
the world model.

Usage:
    python -m nca_wm.train_slot_ae \\
        --init_from nca_wm/logs/multi_gallery_..._rule_attn_..._s-0 \\
        --freeze_encoder \\
        --save_dir nca_wm/logs/slot_ae_gallery
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.rule_attn_model import RuleSlotEncoder
from nca_wm.token_decoder import SlotTokenDecoder, decoder_loss, shift_right
from nca_wm.tokenize_game import VOCAB_SIZE_BASE


def _load_game_infos(ckpt_dir: str):
    infos_path = os.path.join(ckpt_dir, "game_infos.pkl")
    if not os.path.isfile(infos_path):
        raise FileNotFoundError(f"No game_infos.pkl at {infos_path}")
    with open(infos_path, "rb") as f:
        return pickle.load(f)


def _tokens_to_arrays(game_infos, max_len):
    """Pad each game's token_ids to (N_games, max_len) int32 + bool mask."""
    N = len(game_infos)
    out = np.zeros((N, max_len), dtype=np.int32)
    mask = np.zeros((N, max_len), dtype=np.bool_)
    for i, info in enumerate(game_infos):
        tids = info.get("token_ids", [])
        L = min(len(tids), max_len)
        if L > 0:
            out[i, :L] = tids[:L]
            mask[i, :L] = True
    return out, mask


def _load_encoder_params(ckpt_dir: str):
    """Extract the `game_encoder` subtree from a trained RuleAttn world model."""
    params_path = os.path.join(ckpt_dir, "params.pkl")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"No params.pkl at {params_path}")
    with open(params_path, "rb") as f:
        full_params = pickle.load(f)
    enc_params = full_params.get("params", full_params)
    for key in ("game_encoder", "encoder"):
        if key in enc_params:
            return {"params": enc_params[key]}
    raise KeyError(
        f"No game_encoder subtree in {params_path}; "
        f"top-level keys: {list(enc_params.keys())}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", type=str, default=None,
                   help="Trained rule-attn world-model dir.")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder params; train only decoder.")
    p.add_argument("--vocab_size", type=int, default=VOCAB_SIZE_BASE)
    p.add_argument("--max_seq_len", type=int, default=192)
    # Encoder shape — must match the world model's encoder if --init_from
    p.add_argument("--enc_d_model", type=int, default=64)
    p.add_argument("--enc_n_self_layers", type=int, default=2)
    p.add_argument("--n_slots", type=int, default=16)
    p.add_argument("--d_slot", type=int, default=64)
    p.add_argument("--enc_n_heads", type=int, default=4)
    # Decoder shape (independent — newly trained)
    p.add_argument("--dec_d_model", type=int, default=128)
    p.add_argument("--dec_n_layers", type=int, default=4)
    p.add_argument("--dec_n_heads", type=int, default=4)
    # Optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_updates", type=int, default=20000)
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--game_infos", type=str, default=None,
                   help="Path to game_infos.pkl (defaults to --init_from dir).")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    infos_dir = args.game_infos or args.init_from
    if infos_dir is None:
        raise ValueError("Need --init_from or --game_infos for game_infos.pkl")
    if os.path.isdir(infos_dir):
        game_infos = _load_game_infos(infos_dir)
    else:
        with open(infos_dir, "rb") as f:
            game_infos = pickle.load(f)
    N_games = len(game_infos)
    print(f"Loaded {N_games} games from {infos_dir}")

    # Match train.py's encoder shape: max_seq_len = max_tok_len + 1.
    max_tok_len = max(len(info.get("token_ids", [])) for info in game_infos)
    enc_max_seq_len = max(max_tok_len + 1, 2)
    print(f"Encoder max_seq_len = {enc_max_seq_len} (max token sequence + 1)")

    tokens_np, mask_np = _tokens_to_arrays(game_infos, enc_max_seq_len)
    tok_lens = mask_np.sum(axis=1)
    print(f"Token lengths: min={tok_lens.min()}, max={tok_lens.max()}, "
          f"mean={tok_lens.mean():.1f}")

    # Models
    encoder = RuleSlotEncoder(
        vocab_size=args.vocab_size + 1,
        max_seq_len=enc_max_seq_len,
        d_model=args.enc_d_model,
        n_self_layers=args.enc_n_self_layers,
        n_slots=args.n_slots,
        d_slot=args.d_slot,
        n_heads=args.enc_n_heads,
    )
    decoder = SlotTokenDecoder(
        vocab_size=args.vocab_size,
        max_seq_len=enc_max_seq_len,
        d_model=args.dec_d_model,
        n_layers=args.dec_n_layers,
        n_heads=args.dec_n_heads,
        d_slot=args.d_slot,
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, enc_rng, dec_rng = jax.random.split(rng, 3)
    toks_j = jnp.array(tokens_np[:1])
    mask_j = jnp.array(mask_np[:1])
    enc_params = encoder.init(enc_rng, toks_j, mask_j, deterministic=True)
    slots_dummy = encoder.apply(enc_params, toks_j, mask_j, deterministic=True)
    dec_params = decoder.init(dec_rng, toks_j, slots_dummy, deterministic=True)

    if args.init_from:
        loaded = _load_encoder_params(args.init_from)
        try:
            _ = encoder.apply(loaded, toks_j, mask_j, deterministic=True)
            enc_params = loaded
            print(f"Loaded encoder from {args.init_from}")
        except Exception as e:
            print(f"WARNING: encoder shape mismatch ({e}); using random init.")

    n_enc = sum(p.size for p in jax.tree_util.tree_leaves(enc_params))
    n_dec = sum(p.size for p in jax.tree_util.tree_leaves(dec_params))
    print(f"Encoder params: {n_enc:,}   Decoder params: {n_dec:,}")

    if args.freeze_encoder:
        trainable = {"dec": dec_params}
        frozen = {"enc": enc_params}
    else:
        trainable = {"enc": enc_params, "dec": dec_params}
        frozen = {}

    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(trainable)

    def forward(enc_p, dec_p, tokens, mask):
        slots = encoder.apply(enc_p, tokens, mask, deterministic=True)
        inputs = shift_right(tokens, bos_id=0)
        logits = decoder.apply(dec_p, inputs, slots, deterministic=True)
        loss, acc = decoder_loss(logits, tokens, mask)
        return loss, (acc, logits)

    def loss_fn(trainable_params, frozen_params, tokens, mask):
        enc_p = trainable_params.get("enc", frozen_params.get("enc"))
        dec_p = trainable_params["dec"]
        loss, (acc, _) = forward(enc_p, dec_p, tokens, mask)
        return loss, acc

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def update(trainable_params, opt_state, frozen_params, tokens, mask):
        (loss, acc), grads = grad_fn(trainable_params, frozen_params, tokens, mask)
        updates, new_opt = optimizer.update(grads, opt_state, trainable_params)
        new_params = optax.apply_updates(trainable_params, updates)
        return new_params, new_opt, loss, acc

    tokens_j = jnp.array(tokens_np)
    mask_j = jnp.array(mask_np)
    t0 = time.time()
    for step in range(args.n_updates):
        trainable, opt_state, loss, acc = update(
            trainable, opt_state, frozen, tokens_j, mask_j
        )
        if step % args.log_interval == 0 or step == args.n_updates - 1:
            print(f"step {step:6d}/{args.n_updates}  loss={float(loss):.4e}  "
                  f"acc={float(acc):.4f}  ({time.time()-t0:.0f}s)")

    enc_p = trainable.get("enc", frozen.get("enc"))
    dec_p = trainable["dec"]
    loss_final, (acc_final, _) = forward(enc_p, dec_p, tokens_j, mask_j)
    print(f"\nFinal: loss={float(loss_final):.4e}  acc={float(acc_final):.4f}")

    # Per-game reconstruction
    slots_all = encoder.apply(enc_p, tokens_j, mask_j, deterministic=True)
    inputs = shift_right(tokens_j, bos_id=0)
    logits_all = decoder.apply(dec_p, inputs, slots_all, deterministic=True)
    preds = jnp.argmax(logits_all, axis=-1)
    print("\nPer-game reconstruction accuracy:")
    per_game_acc = []
    for i, info in enumerate(game_infos):
        m = mask_np[i]
        if m.sum() == 0:
            continue
        correct = int(np.array((preds[i] == tokens_j[i]) & jnp.array(m)).sum())
        a = correct / int(m.sum())
        per_game_acc.append((info["name"], float(a), int(m.sum())))
    for name, a, n in sorted(per_game_acc, key=lambda x: x[1])[:20]:
        print(f"  {name:35s}  acc={a:.3f}  ({n} tokens)")
    if len(per_game_acc) > 20:
        mean_acc = np.mean([a for _, a, _ in per_game_acc])
        print(f"  ... ({len(per_game_acc)} games total, mean acc={mean_acc:.3f})")

    out = {
        "enc_params": enc_p,
        "dec_params": dec_p,
        "game_names": [info["name"] for info in game_infos],
        "slots_all": np.array(slots_all),
        "tokens": tokens_np,
        "mask": mask_np,
        "final_acc": float(acc_final),
        "per_game_acc": per_game_acc,
        "args": vars(args),
    }
    out_path = os.path.join(args.save_dir, "slot_ae.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

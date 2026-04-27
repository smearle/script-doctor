"""Train a game-token autoencoder: encoder(tokens)=z, decoder(z)=tokens.

Reuses `GameSpecEncoder` from train.py (same architecture used by the
conditional NCA world model) and the new `TokenDecoder` from
token_decoder.py. Standalone trainer, does NOT touch the NCA training path.

Two modes:
  --init_from <ckpt_dir>   : initialize encoder params from a trained
        ConditionalNCAWorldModel checkpoint (behavior-grounded z).
  (default)                : fresh random encoder (pure token AE).

Loads the game-info tokens directly from <ckpt_dir>/game_infos.pkl (built
during multi-game dataset collection). Each game has a single (tokens, mask)
pair; the autoencoder is trained to reconstruct each.

This is a tiny dataset (N_games <= 50 for any preset we run), so the model
will overfit — that's the point of the reconstruction test. Later we scale
up via the full gallery (hundreds of games) to get a prior worth sampling.
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

from nca_wm.train import GameSpecEncoder
from nca_wm.token_decoder import TokenDecoder, decoder_loss, shift_right
from nca_wm.tokenize_game import VOCAB_SIZE_BASE


def _load_game_infos(ckpt_dir: str):
    infos_path = os.path.join(ckpt_dir, "game_infos.pkl")
    if not os.path.isfile(infos_path):
        raise FileNotFoundError(f"No game_infos.pkl at {infos_path}")
    with open(infos_path, "rb") as f:
        return pickle.load(f)


def _tokens_to_arrays(game_infos: list[dict], max_len: int):
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
    """Extract just the `encoder` subtree from a trained NCA world model."""
    params_path = os.path.join(ckpt_dir, "params.pkl")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"No params.pkl at {params_path}")
    with open(params_path, "rb") as f:
        full_params = pickle.load(f)
    # ConditionalNCAWorldModel exposes the encoder under key 'encoder'.
    enc_params = full_params.get("params", full_params)
    # The encoder submodule is named 'game_encoder' (Flax @nn.compact names it
    # from its class attribute or CLI-set name). Keep a small fallback list.
    for key in ("game_encoder", "encoder", "game_spec_encoder"):
        if key in enc_params:
            return {"params": enc_params[key]}
    raise KeyError(
        f"No encoder submodule ('game_encoder'/'encoder'/'game_spec_encoder') "
        f"in params at {params_path}; top-level keys: {list(enc_params.keys())}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", type=str, default=None,
                   help="Path to a trained multi-game checkpoint dir. "
                        "If given, initializes the encoder from its params.")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder params; train only decoder.")
    p.add_argument("--vocab_size", type=int, default=VOCAB_SIZE_BASE,
                   help=f"Token vocab size (default {VOCAB_SIZE_BASE} = VOCAB_SIZE_BASE).")
    p.add_argument("--max_seq_len", type=int, default=192)
    p.add_argument("--d_z", type=int, default=64)
    p.add_argument("--d_model_enc", type=int, default=64)
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--enc_heads", type=int, default=4)
    p.add_argument("--d_model_dec", type=int, default=128)
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--dec_heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--n_updates", type=int, default=20000)
    p.add_argument("--log_interval", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=0,
                   help="If 0, use full batch (all games at once) — usual "
                        "for tiny N_games.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, required=True)
    # Dataset source: either --init_from (which also has game_infos.pkl) or
    # --game_infos overriding.
    p.add_argument("--game_infos", type=str, default=None,
                   help="Path to game_infos.pkl (defaults to --init_from dir).")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Resolve game_infos source
    infos_dir = args.game_infos or args.init_from
    if infos_dir is None:
        raise ValueError("Need --init_from or --game_infos to locate game_infos.pkl")
    if os.path.isdir(infos_dir):
        game_infos = _load_game_infos(infos_dir)
    else:
        with open(infos_dir, "rb") as f:
            game_infos = pickle.load(f)
    N_games = len(game_infos)
    print(f"Loaded {N_games} games from {infos_dir}")

    tokens_np, mask_np = _tokens_to_arrays(game_infos, args.max_seq_len)
    tok_lens = mask_np.sum(axis=1)
    print(f"Token-seq lengths: min={tok_lens.min()}, max={tok_lens.max()}, "
          f"mean={tok_lens.mean():.1f}")

    # Models
    encoder = GameSpecEncoder(
        vocab_size=args.vocab_size + 1,  # +1 for CLS
        d_model=args.d_model_enc, n_heads=args.enc_heads, n_layers=args.enc_layers,
        d_z=args.d_z, max_seq_len=args.max_seq_len + 1,
    )
    decoder = TokenDecoder(
        vocab_size=args.vocab_size, max_seq_len=args.max_seq_len,
        d_model=args.d_model_dec, n_heads=args.dec_heads, n_layers=args.dec_layers,
    )

    # Init params
    rng = jax.random.PRNGKey(args.seed)
    rng, enc_rng, dec_rng = jax.random.split(rng, 3)
    toks_j = jnp.array(tokens_np[:1])
    mask_j = jnp.array(mask_np[:1])
    enc_params = encoder.init(enc_rng, toks_j, mask_j, deterministic=True)
    z_dummy = encoder.apply(enc_params, toks_j, mask_j, deterministic=True)
    dec_params = decoder.init(dec_rng, toks_j, z_dummy, deterministic=True)

    if args.init_from:
        loaded_enc = _load_encoder_params(args.init_from)
        # Basic shape compatibility: ensure the loaded encoder architecture
        # matches. If not, warn and fall back to random init.
        try:
            # Try apply with loaded params — will raise if mismatch
            _ = encoder.apply(loaded_enc, toks_j, mask_j, deterministic=True)
            enc_params = loaded_enc
            print(f"Loaded encoder params from {args.init_from}")
        except Exception as e:
            print(f"WARNING: encoder shape mismatch ({e}); using random init.")

    n_enc = sum(p.size for p in jax.tree_util.tree_leaves(enc_params))
    n_dec = sum(p.size for p in jax.tree_util.tree_leaves(dec_params))
    print(f"Encoder params: {n_enc:,}   Decoder params: {n_dec:,}")

    # Partitioning: optionally freeze encoder params.
    if args.freeze_encoder:
        trainable = {"dec": dec_params}
        frozen = {"enc": enc_params}
    else:
        trainable = {"enc": enc_params, "dec": dec_params}
        frozen = {}

    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(trainable)

    # Training step
    def forward(enc_p, dec_p, tokens, mask):
        z = encoder.apply(enc_p, tokens, mask, deterministic=True)
        inputs = shift_right(tokens, bos_id=0)  # PAD=0 used as BOS stand-in
        logits = decoder.apply(dec_p, inputs, z, deterministic=True)
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

    # Full-batch loop (typical N_games ≤ 20)
    tokens_j = jnp.array(tokens_np)
    mask_j = jnp.array(mask_np)
    t0 = time.time()
    for step in range(args.n_updates):
        trainable, opt_state, loss, acc = update(
            trainable, opt_state, frozen, tokens_j, mask_j
        )
        if step % args.log_interval == 0 or step == args.n_updates - 1:
            print(f"step {step:6d}/{args.n_updates}  loss={float(loss):.4e}  "
                  f"per-tok acc={float(acc):.4f}  ({time.time()-t0:.0f}s)")

    # Final per-game reconstruction accuracy
    enc_p = trainable.get("enc", frozen.get("enc"))
    dec_p = trainable["dec"]
    loss_final, (acc_final, _) = forward(enc_p, dec_p, tokens_j, mask_j)
    print(f"\nFinal: loss={float(loss_final):.4e}  acc={float(acc_final):.4f}")

    # Per-game breakdown
    z_all = encoder.apply(enc_p, tokens_j, mask_j, deterministic=True)
    inputs = shift_right(tokens_j, bos_id=0)
    logits_all = decoder.apply(dec_p, inputs, z_all, deterministic=True)
    preds = jnp.argmax(logits_all, axis=-1)
    print("\nPer-game reconstruction accuracy (unmasked positions):")
    per_game_acc = []
    for i, info in enumerate(game_infos):
        m = mask_np[i]
        if m.sum() == 0:
            continue
        correct = np.array((preds[i] == tokens_j[i]) & jnp.array(m)).sum()
        a = correct / int(m.sum())
        per_game_acc.append((info["name"], float(a), int(m.sum())))
    for name, a, n in sorted(per_game_acc, key=lambda x: x[1]):
        print(f"  {name:35s}  acc={a:.3f}  ({n} tokens)")

    # Save results + z's
    out = {
        "enc_params": enc_p,
        "dec_params": dec_p,
        "game_names": [info["name"] for info in game_infos],
        "z_all": np.array(z_all),
        "tokens": tokens_np,
        "mask": mask_np,
        "final_acc": float(acc_final),
        "per_game_acc": per_game_acc,
        "args": vars(args),
    }
    out_path = os.path.join(args.save_dir, "token_ae.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

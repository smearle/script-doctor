"""Demo: sample / interpolate in the game-latent space and decode to tokens.

Loads a trained token_ae.pkl (produced by train_token_ae.py), then:
  - Decodes each training game's z deterministically (sanity check)
  - Interpolates linearly between two training games' z's and decodes at
    several steps — if the latent is smooth, we should get plausible
    hybrid token sequences.
  - Samples from an empirical Gaussian fit to training z's and decodes.

No PS-engine integration yet — this is purely token-level output. Next step
(detokenizer) converts decoded token sequences back to .puzzlescript source.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nca_wm.train import GameSpecEncoder
from nca_wm.token_decoder import TokenDecoder, sample_tokens


def _rebuild_models(args_dict):
    """Reconstruct encoder + decoder modules from the saved training args."""
    a = args_dict
    encoder = GameSpecEncoder(
        vocab_size=a["vocab_size"] + 1,
        d_model=a["d_model_enc"],
        n_heads=a["enc_heads"],
        n_layers=a["enc_layers"],
        d_z=a["d_z"],
        max_seq_len=a["max_seq_len"] + 1,
    )
    decoder = TokenDecoder(
        vocab_size=a["vocab_size"],
        max_seq_len=a["max_seq_len"],
        d_model=a["d_model_dec"],
        n_heads=a["dec_heads"],
        n_layers=a["dec_layers"],
    )
    return encoder, decoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae_dir", type=str, required=True,
                   help="Directory containing token_ae.pkl.")
    p.add_argument("--n_interp_steps", type=int, default=9,
                   help="Number of interpolation steps between two game z's.")
    p.add_argument("--n_random_samples", type=int, default=5,
                   help="Number of z's to draw from the empirical Gaussian.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0=greedy).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ae_path = os.path.join(args.ae_dir, "token_ae.pkl")
    with open(ae_path, "rb") as f:
        ae = pickle.load(f)

    encoder, decoder = _rebuild_models(ae["args"])
    enc_p = ae["enc_params"]
    dec_p = ae["dec_params"]

    z_all = jnp.array(ae["z_all"])            # (N_games, d_z)
    tokens_all = ae["tokens"]                  # (N_games, L) np.int32
    mask_all = ae["mask"]                      # (N_games, L) np.bool
    names = ae["game_names"]
    max_len = tokens_all.shape[1]

    rng = jax.random.PRNGKey(args.seed)

    print("=" * 70)
    print("1) Reconstruction (teacher-forced, fast) — per-game token accuracy")
    print("=" * 70)
    # Teacher-forced reconstruction: one batched forward pass.
    from nca_wm.token_decoder import shift_right
    inputs = shift_right(jnp.array(tokens_all), bos_id=0)
    logits = decoder.apply(dec_p, inputs, z_all, deterministic=True)
    preds = np.array(jnp.argmax(logits, axis=-1))
    for i, n in enumerate(names):
        m = mask_all[i]
        if m.sum() == 0:
            continue
        correct = ((preds[i] == tokens_all[i]) & m).sum()
        acc = correct / m.sum()
        print(f"  {n:30s}  acc={acc:.3f}   "
              f"tokens[:15]: real={tokens_all[i, :15].tolist()}  "
              f"dec={preds[i, :15].tolist()}")

    print()
    print("=" * 70)
    print(f"2) Latent interpolation — {names[0]} <-> {names[-1]} in "
          f"{args.n_interp_steps} steps")
    print("=" * 70)
    z_a, z_b = z_all[0], z_all[-1]
    # Cap AR sampling length to keep CPU demo under a minute.
    gen_len = min(max_len, 60)
    for t in np.linspace(0, 1, args.n_interp_steps):
        z_mix = ((1 - t) * z_a + t * z_b)[None, :]
        decoded = sample_tokens(
            decoder, dec_p, encoder, enc_p,
            z_mix, max_len=gen_len, bos_id=0, temperature=0.0, rng=None,
        )
        decoded = np.array(decoded[0])
        # truncate at first PAD token (0) after position 0
        nonpad = np.where(decoded == 0)[0]
        cut = nonpad[nonpad > 0][0] if len(nonpad[nonpad > 0]) > 0 else len(decoded)
        print(f"  t={t:.2f}  len={cut:3d}  tokens[:30]={decoded[:30].tolist()}")

    print()
    print("=" * 70)
    print(f"3) Gaussian-prior samples — {args.n_random_samples} draws from "
          f"N(mean, cov) fit to training z's")
    print("=" * 70)
    z_np = np.array(z_all)
    mu = z_np.mean(axis=0)
    cov = np.cov(z_np, rowvar=False) + 1e-4 * np.eye(z_np.shape[1])
    L = np.linalg.cholesky(cov)
    rs = np.random.RandomState(args.seed)
    gen_len_s = min(max_len, 60)
    for i in range(args.n_random_samples):
        eps = rs.randn(z_np.shape[1])
        z_sample = mu + L @ eps
        z_j = jnp.array(z_sample[None, :])
        decoded = sample_tokens(
            decoder, dec_p, encoder, enc_p,
            z_j, max_len=gen_len_s, bos_id=0,
            temperature=args.temperature,
            rng=jax.random.PRNGKey(args.seed + 100 + i) if args.temperature > 0 else None,
        )
        decoded = np.array(decoded[0])
        nonpad = np.where(decoded == 0)[0]
        cut = nonpad[nonpad > 0][0] if len(nonpad[nonpad > 0]) > 0 else len(decoded)
        print(f"  sample {i}: len={cut:3d}  tokens[:30]={decoded[:30].tolist()}")


if __name__ == "__main__":
    main()

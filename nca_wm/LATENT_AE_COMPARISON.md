# Latent AE / VAE / VQ-VAE Comparison

Last updated: 2026-06-17

## What The Current VQ Model Is

The world-model `--vq_codebook` path is VQ-VAE-style, not a full probabilistic
VAE. It quantizes rule slots through a learned codebook and trains with
codebook + commitment losses, plus the optional soft usage-entropy regularizer.
It does not learn a Gaussian posterior, optimize a KL term to a continuous
prior, or sample a latent from that prior during world-model training.

In this repo's terminology:

- `train_token_ae.py` and deterministic `train_slot_ae.py` are AEs.
- `train.py --vq_codebook` and `train_slot_ae.py --latent_model vqvae` are
  VQ-VAE-style discrete bottlenecks.
- `train_slot_ae.py --latent_model vae` is the Gaussian VAE comparison.

## Implemented Comparison

`train_slot_ae.py` now supports three latent bottlenecks:

- `--latent_model ae`: deterministic rule slots, equivalent to the previous
  slot AE.
- `--latent_model vae`: per-slot Gaussian posterior, residual mean initialized
  from the deterministic slots, reparameterized sampling during training, and
  KL-to-standard-normal loss.
- `--latent_model vqvae`: learned codebook quantization using the same
  `VectorQuantizer` as the world model, with codebook, commitment, and optional
  usage-entropy losses.

Each run writes:

- `slot_ae.pkl`
- `summary.json`
- `history.npz`

The helper launcher runs all three with matched settings:

```bash
SAVE_ROOT=nca_wm/refine-logs/slot_latent_compare_w0p01_2k \
DEC_D_MODEL=64 DEC_N_LAYERS=2 DEC_N_HEADS=4 \
nca_wm/scripts/run_slot_latent_compare.sh
```

The comparison freezes the `w0p01` rule-slot encoder from:

```text
nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01
```

and trains only the latent bottleneck plus token decoder. This isolates how
much reconstruction quality the bottleneck preserves.

## 2k-Step Result

Output root:

```text
nca_wm/refine-logs/slot_latent_compare_w0p01_2k
```

| model | final recon loss | token acc | mean game acc | min game acc | worst game | KL/dim | sigma | hard util | soft perp |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| AE | 0.007311 | 0.9973 | 0.9969 | 0.9931 | `notsnake` | - | - | - | - |
| VAE | 0.007422 | 0.9970 | 0.9964 | 0.9874 | `blocks` | 0.663 | 0.334 | - | - |
| VQ-VAE | 0.010636 | 0.9960 | 0.9959 | 0.9912 | `Love_and_Pieces` | - | - | 2.0 | 1021.8 |

Extra loss terms at the end:

- VAE: `kl_total=679.4`, `loss_with_kl=0.07536` at `vae_kl_weight=1e-4`.
- VQ-VAE: `vq_codebook_loss=0.00166`, `vq_commit_loss=0.00166`,
  `vq_usage_loss=0.00211`, `loss_with_vq=0.01273`.

## Interpretation

For token reconstruction from a frozen, already-trained rule-slot encoder, all
three bottlenecks preserve most information. The deterministic AE is still the
cleanest reconstruction baseline. The Gaussian VAE reaches nearly the same
reconstruction accuracy, but its posterior is not close to the unit Gaussian
prior after this short run; it behaves more like a lightly-noised AE than a
good sampling model. The VQ-VAE reaches similar reconstruction accuracy and
high soft perplexity, but hard code usage remains collapsed at two active
codes, matching the world-model-side finding that soft usage regularization
does not solve hard code collapse.

This comparison supports using VAE/VQ-VAE diagnostics as latent-space probes,
but it does not yet show that either bottleneck is a better generative prior
than the deterministic slot AE.

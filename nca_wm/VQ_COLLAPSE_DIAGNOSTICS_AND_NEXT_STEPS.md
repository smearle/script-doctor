# VQ Collapse Diagnostics And Next Steps

Last updated: 2026-06-19

This is the handoff document for the VQ hard-code collapse investigation in
`nca_wm`. It supersedes the old `LATENT_AE_COMPARISON.md` name because the
track is no longer just an AE/VAE/VQ-VAE reconstruction comparison. The current
question is whether the learned rule-slot latent can become a robust discrete
adaptation interface for novel games.

## Read This First

The main finding is negative but useful: the original random VQ setup has a
real hard-code collapse, and the later fixes improve reconstruction or soft
assignment statistics without producing a backend-stable, high-utilization
hard codebook.

Most important results:

- Random VQ collapses to about two active hard codes.
- `kmeans + layernorm` was the strongest 500-step fast pilot, reaching 70
  active hard codes with token accuracy `0.9963`, but its assignment margin was
  almost zero.
- A longer CPU 5k `kmeans + layernorm` validation kept higher utilization
  than random VQ, but the near-zero margin remained.
- Matching GPU runs showed the high-utilization result is not backend-stable:
  saved hard usage fell to roughly 5 active codes, while CPU recomputation
  found many alternative nearest codes from the same checkpoints.
- Direct margin loss, straight-through hard-balance loss, soft-to-hard
  annealing, and Gumbel-ST annealing all failed to create a robust diverse
  hard nearest-code partition.
- Residual VQ and product VQ also failed to produce a robust diverse saved
  hard codebook on GPU. Residual VQ slightly improved reconstruction and joint
  hard utilization, but still used only `[5, 6]` active codes across its two
  stages.

Recommended next model direction:

```text
explicit discrete code assignment search
```

Do not spend more time only increasing soft usage entropy, hard-balance weight,
assignment annealing duration, or simple residual/product capacity unless
there is a new diagnostic explaining why those objectives should overcome the
saved hard-code collapse.

Key artifacts:

- Diagnostics: `nca_wm/scripts/diagnose_slot_vq.py`
- Latent ablations: `nca_wm/scripts/eval_slot_latent_ablation.py`
- Fast pilots: `nca_wm/scripts/run_vq_collapse_pilots.sh`
- Main pilot root: `nca_wm/refine-logs/vq_collapse_pilots_w0p01_500`
- GPU soft/Gumbel roots:
  `nca_wm/refine-logs/vq_softst_anneal_kmeans_layernorm_w0p01_20k_t2_to_0p05_s0`,
  `nca_wm/refine-logs/vq_soft_anneal_kmeans_layernorm_w0p01_20k_t2_to_0p05_s0`,
  and
  `nca_wm/refine-logs/vq_gumbelst_anneal_kmeans_layernorm_w0p01_20k_t2_to_0p05_s0`.
- Residual/product roots:
  `nca_wm/refine-logs/vq_residual2_kmeans_layernorm_w0p01_5k_s0_gpu` and
  `nca_wm/refine-logs/vq_product4_kmeans_layernorm_w0p01_5k_s0_gpu`.

## Reproduction Guide

Run all commands from the repository root with the repo virtualenv available:

```bash
source .venv/bin/activate
test -d nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01
```

The source checkpoint above must contain `config.json`, `game_infos.pkl`, and
the encoder weights used by `train_slot_ae.py --init_from`. To reproduce GPU
results, set both `JAX_PLATFORMS=cuda` and `CUDA_VISIBLE_DEVICES=<gpu_id>`.
To reproduce CPU results, either omit those variables or set
`JAX_PLATFORMS=cpu`. CPU and GPU hard argmin results are intentionally tracked
separately because near-tie assignments are backend-sensitive.

Static checks:

```bash
.venv/bin/python -m py_compile \
    nca_wm/train_slot_ae.py \
    nca_wm/scripts/diagnose_slot_vq.py \
    nca_wm/scripts/eval_slot_latent_ablation.py
bash -n nca_wm/scripts/run_vq_collapse_pilots.sh
```

Recompute diagnostics for any existing run:

```bash
RUN=nca_wm/refine-logs/vq_kmeans_layernorm_w0p01_5k_s0_gpu
.venv/bin/python -m nca_wm.scripts.diagnose_slot_vq "$RUN/slot_ae.pkl"
.venv/bin/python -m nca_wm.scripts.eval_slot_latent_ablation "$RUN" --out_dir "$RUN"
```

Reproduce the 2k AE/VAE/VQ-VAE baseline:

```bash
SAVE_ROOT=nca_wm/refine-logs/slot_latent_compare_w0p01_2k \
N_UPDATES=2000 \
DEC_D_MODEL=64 DEC_N_LAYERS=2 DEC_N_HEADS=4 \
nca_wm/scripts/run_slot_latent_compare.sh
```

Reproduce the six 500-step VQ collapse pilots:

```bash
SAVE_ROOT=nca_wm/refine-logs/vq_collapse_pilots_w0p01_500 \
N_UPDATES=500 \
DEC_D_MODEL=64 DEC_N_LAYERS=2 DEC_N_HEADS=4 \
nca_wm/scripts/run_vq_collapse_pilots.sh
```

Reproduce the GPU 5k `kmeans + layernorm` no-margin baseline:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
.venv/bin/python -m nca_wm.train_slot_ae \
    --init_from nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01 \
    --freeze_encoder \
    --latent_model vqvae \
    --vq_init kmeans \
    --slot_pre_norm layernorm \
    --vq_usage_loss_weight 0.01 \
    --n_updates 5000 \
    --log_interval 500 \
    --dec_d_model 64 \
    --dec_n_layers 2 \
    --dec_n_heads 4 \
    --seed 0 \
    --save_dir nca_wm/refine-logs/vq_kmeans_layernorm_w0p01_5k_s0_gpu
```

Reproduce the GPU 5k margin run:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
.venv/bin/python -m nca_wm.train_slot_ae \
    --init_from nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01 \
    --freeze_encoder \
    --latent_model vqvae \
    --vq_init kmeans \
    --slot_pre_norm layernorm \
    --vq_usage_loss_weight 0.01 \
    --vq_margin_loss_weight 0.1 \
    --vq_margin_target 0.01 \
    --n_updates 5000 \
    --log_interval 500 \
    --dec_d_model 64 \
    --dec_n_layers 2 \
    --dec_n_heads 4 \
    --seed 0 \
    --save_dir nca_wm/refine-logs/vq_margin_kmeans_layernorm_w0p01_5k_w0p1_m0p01_s0
```

Reproduce the GPU 5k hard-balance runs:

```bash
COMMON_ARGS=(
  --init_from nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01
  --freeze_encoder
  --latent_model vqvae
  --vq_init kmeans
  --slot_pre_norm layernorm
  --vq_usage_loss_weight 0.01
  --n_updates 5000
  --log_interval 500
  --dec_d_model 64
  --dec_n_layers 2
  --dec_n_heads 4
  --seed 0
)

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
.venv/bin/python -m nca_wm.train_slot_ae "${COMMON_ARGS[@]}" \
    --vq_hard_balance_loss_weight 0.1 \
    --vq_hard_balance_target 64 \
    --save_dir nca_wm/refine-logs/vq_balance_kmeans_layernorm_w0p01_5k_bal0p1_t64_s0

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
.venv/bin/python -m nca_wm.train_slot_ae "${COMMON_ARGS[@]}" \
    --vq_hard_balance_loss_weight 1.0 \
    --vq_hard_balance_target 64 \
    --save_dir nca_wm/refine-logs/vq_balance_kmeans_layernorm_w0p01_5k_bal1p0_t64_s0

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
.venv/bin/python -m nca_wm.train_slot_ae "${COMMON_ARGS[@]}" \
    --vq_hard_balance_loss_weight 0.1 \
    --vq_hard_balance_target 64 \
    --vq_margin_loss_weight 0.1 \
    --vq_margin_target 0.01 \
    --save_dir nca_wm/refine-logs/vq_balance_margin_kmeans_layernorm_w0p01_5k_bal0p1_t64_margin0p1_m0p01_s0
```

Reproduce the GPU 20k soft-to-hard and Gumbel-ST annealing runs:

```bash
for MODE in soft_st soft gumbel_st; do
  case "$MODE" in
    soft_st) NAME=softst ;;
    gumbel_st) NAME=gumbelst ;;
    *) NAME="$MODE" ;;
  esac
  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  .venv/bin/python -m nca_wm.train_slot_ae \
      --init_from nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01 \
      --freeze_encoder \
      --latent_model vqvae \
      --vq_init kmeans \
      --slot_pre_norm layernorm \
      --vq_usage_loss_weight 0.01 \
      --vq_assign_mode "$MODE" \
      --vq_assign_temp_start 2.0 \
      --vq_assign_temp_end 0.05 \
      --vq_assign_anneal_steps 20000 \
      --n_updates 20000 \
      --log_interval 2000 \
      --dec_d_model 64 \
      --dec_n_layers 2 \
      --dec_n_heads 4 \
      --seed 0 \
      --save_dir "nca_wm/refine-logs/vq_${NAME}_anneal_kmeans_layernorm_w0p01_20k_t2_to_0p05_s0"
done
```

After every reproduced training run, regenerate diagnostics and latent
ablations with the `RUN=...` commands above before comparing against the
tables in this document.

Reproduce the GPU residual and product VQ pilots:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
.venv/bin/python -m nca_wm.train_slot_ae \
    --init_from nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01 \
    --freeze_encoder \
    --latent_model vqvae \
    --vq_quantizer residual \
    --vq_num_quantizers 2 \
    --vq_residual_codebook_size 256 \
    --vq_init kmeans \
    --slot_pre_norm layernorm \
    --vq_usage_loss_weight 0.01 \
    --n_updates 5000 \
    --log_interval 500 \
    --dec_d_model 64 \
    --dec_n_layers 2 \
    --dec_n_heads 4 \
    --seed 0 \
    --save_dir nca_wm/refine-logs/vq_residual2_kmeans_layernorm_w0p01_5k_s0_gpu

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
.venv/bin/python -m nca_wm.train_slot_ae \
    --init_from nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01 \
    --freeze_encoder \
    --latent_model vqvae \
    --vq_quantizer product \
    --vq_num_quantizers 4 \
    --vq_residual_codebook_size 256 \
    --vq_init kmeans \
    --slot_pre_norm layernorm \
    --vq_usage_loss_weight 0.01 \
    --n_updates 5000 \
    --log_interval 500 \
    --dec_d_model 64 \
    --dec_n_layers 2 \
    --dec_n_heads 4 \
    --seed 0 \
    --save_dir nca_wm/refine-logs/vq_product4_kmeans_layernorm_w0p01_5k_s0_gpu
```

## Background: What The Current VQ Model Is

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

## Background: AE/VAE/VQ-VAE Comparison

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

## Completed Work

This diagnostic pass added the VQ hard-collapse tools needed to separate
three failure modes: collapsed slot geometry, bad codebook initialization, and
a token decoder that reconstructs mostly from teacher-forced context.

Implemented diagnostics:

- `diagnose_slot_vq.py` reads a saved `slot_ae.pkl` and writes
  `vq_diagnostic.json` plus `vq_diagnostic.md`, including per-game and
  per-slot hard code indices, active-code counts, nearest-code distance,
  nearest/second-nearest distance margin, slot norm, and soft entropy.
- `eval_slot_latent_ablation.py` reuses saved decoders and evaluates
  `original`, `zero_slots`, `mean_slots`, `shuffled_across_games`, and
  `shuffled_within_game`.
- `run_vq_collapse_pilots.sh` runs the six 500-step VQ collapse pilots used
  below and records diagnostics after each run.

Implemented VQ-VAE training knobs:

- `--vq_init {random,slot_sample,kmeans}` controls codebook initialization.
- `--slot_pre_norm {none,layernorm,l2}` applies a parameter-free slot
  normalization before VQ only.
- `--vq_margin_loss_weight` and `--vq_margin_target` add an optional
  nearest-vs-second-nearest assignment margin objective.
- `--vq_hard_balance_loss_weight`, `--vq_hard_balance_target`, and
  `--vq_hard_balance_temp` add an optional straight-through hard-assignment
  balance objective.
- `--vq_assign_mode {hard,soft,soft_st,gumbel_st}` and the
  `--vq_assign_temp_*` flags add soft-to-hard and Gumbel-ST assignment
  annealing.
- `--vq_quantizer {single,residual,product}`, `--vq_num_quantizers`, and
  `--vq_residual_codebook_size` add residual VQ and fixed-subspace product VQ
  for the slot-token AE diagnostic path.
- VQ-VAE summaries now include the final hard assignment histogram, and saved
  checkpoints include the final hard VQ indices for reproducible diagnostics.
  Margin and balance runs also record `vq_margin_loss`, `vq_mean_margin`,
  `vq_hard_balance_loss`, and `vq_hard_balance_perp`.
  Residual/product runs also record per-stage hard utilization, per-stage
  margins, and joint code-tuple histograms.

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

## Latent Ablation Diagnostic

Output root:

```text
nca_wm/refine-logs/slot_latent_compare_w0p01_2k
```

`eval_slot_latent_ablation.py` reuses each saved decoder and evaluates
`original`, `zero_slots`, `mean_slots`, `shuffled_across_games`, and
`shuffled_within_game`.

| model | original acc | zero drop | mean drop | shuffle-game drop | shuffle-slot drop |
|---|---:|---:|---:|---:|---:|
| AE | 0.9973 | 0.2742 | 0.0074 | 0.0172 | 0.0000 |
| VAE | 0.9970 | 0.0051 | 0.0017 | 0.0034 | 0.0000 |
| VQ-VAE | 0.9960 | 0.2307 | 0.0010 | 0.0040 | 0.0000 |

The deterministic AE and collapsed VQ-VAE do use the latent: replacing all
slots with zeros causes a large accuracy drop. However, replacing slots with a
global mean or slots from another game causes only a small drop, and shuffling
slots within a game causes no drop. The within-game result is expected because
the decoder cross-attends to slots without slot-position semantics. The
mean/shuffle result means token reconstruction is still a weak proof of
fine-grained latent usage; it mostly verifies that the decoder needs a coarse
conditioning signal.

The Gaussian VAE is the outlier: zeroing its slots barely hurts accuracy. That
run should not be interpreted as a useful latent bottleneck yet.

## VQ Collapse Fast Pilots

Output root:

```text
nca_wm/refine-logs/vq_collapse_pilots_w0p01_500
```

All pilots use the same frozen source checkpoint unless noted:

```text
nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01
```

| pilot | init | pre-norm | frozen enc | token acc | mean game acc | min game acc | hard util | soft perp | mean margin | zero drop | mean drop | shuffle-game drop |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq_random` | random | none | yes | 0.9960 | 0.9959 | 0.9912 | 2 | 1014.2 | 0.188853 | 0.1381 | 0.0013 | 0.0047 |
| `vq_kmeans` | kmeans | none | yes | 0.9966 | 0.9964 | 0.9912 | 15 | 1024.0 | 0.003325 | 0.2203 | 0.0057 | 0.0135 |
| `vq_slot_sample` | slot_sample | none | yes | 0.9966 | 0.9964 | 0.9912 | 17 | 1024.0 | 0.003136 | 0.2139 | 0.0057 | 0.0131 |
| `vq_layernorm` | random | layernorm | yes | 0.9956 | 0.9953 | 0.9912 | 1 | 460.0 | 3.376551 | 0.1671 | 0.0000 | 0.0000 |
| `vq_kmeans_layernorm` | kmeans | layernorm | yes | 0.9963 | 0.9960 | 0.9908 | 70 | 1021.8 | 0.000007 | 0.5581 | 0.0044 | 0.0155 |
| `vq_unfrozen` | random | none | no | 0.9953 | 0.9949 | 0.9908 | 2 | 1017.2 | 0.275577 | 0.0687 | 0.0003 | 0.0010 |

The main signal is that codebook initialization matters. Random VQ reproduces
the hard collapse at 2 codes. K-means and slot-sampling both raise hard usage
to roughly one active code per game. LayerNorm alone makes collapse worse, but
K-means plus LayerNorm raises hard usage to 70 active codes without hurting
token accuracy.

The best pilot still has a warning sign: its mean nearest/second-nearest
distance margin is almost zero. This means many assignments are fragile ties,
not a robust discrete partition. The next longer run should use
`--vq_init kmeans --slot_pre_norm layernorm`, while tracking hard histograms,
assignment margins, and latent ablations. If the margin stays near zero after
longer training, the next model change should be a stronger discrete
assignment mechanism rather than more soft usage entropy.

## 5k K-Means + LayerNorm Validation

Output root:

```text
nca_wm/refine-logs/vq_kmeans_layernorm_w0p01_5k_s0
```

This run uses the recommended fast-pilot setting for a longer single-seed
validation:

```bash
.venv/bin/python -m nca_wm.train_slot_ae \
    --init_from nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01 \
    --freeze_encoder \
    --latent_model vqvae \
    --vq_init kmeans \
    --slot_pre_norm layernorm \
    --vq_usage_loss_weight 0.01 \
    --n_updates 5000 \
    --dec_d_model 64 \
    --dec_n_layers 2 \
    --dec_n_heads 4 \
    --seed 0 \
    --save_dir nca_wm/refine-logs/vq_kmeans_layernorm_w0p01_5k_s0
```

| run | token acc | mean game acc | min game acc | hard util | soft perp | mean margin | zero drop | mean drop | shuffle-game drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500-step pilot | 0.9963 | 0.9960 | 0.9908 | 70 | 1021.8 | 0.000007 | 0.5581 | 0.0044 | 0.0155 |
| 5k validation | 0.9976 | 0.9972 | 0.9912 | 52 | 1021.9 | 0.000012 | 0.4392 | 0.0054 | 0.0145 |

The longer run improves reconstruction accuracy and keeps hard utilization far
above the random-init VQ baseline, so `kmeans + layernorm` is a real
improvement over the original collapse mode. However, the assignment margin
remains near zero. The active codes are therefore still fragile nearest-code
ties rather than a robust discrete partition.

There was a transient training spike around step 1500, but the run recovered
by step 2000 and finished with lower loss than the 500-step pilot.

## Assignment Margin Objective

The first stronger assignment objective directly penalizes fragile nearest-code
ties:

```text
mean(relu(vq_margin_target - (second_nearest_dist - nearest_dist)))
```

Two 500-step pilots used the same `kmeans + layernorm` setting as above and
added `--vq_margin_loss_weight 0.1` with two target margins.

| run | token acc | mean game acc | hard util | soft perp | mean margin | zero drop | shuffle-game drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| 500-step no-margin | 0.9963 | 0.9960 | 70 | 1021.8 | 0.000007 | 0.5581 | 0.0155 |
| 5k no-margin | 0.9976 | 0.9972 | 52 | 1021.9 | 0.000012 | 0.4392 | 0.0145 |
| margin target 0.05 | 0.9966 | 0.9964 | 11 | 1021.7 | 0.017642 | 0.5588 | 0.0158 |
| margin target 0.01 | 0.9966 | 0.9964 | 11 | 1021.7 | 0.005957 | 0.5557 | 0.0155 |

Output roots:

```text
nca_wm/refine-logs/vq_margin_kmeans_layernorm_w0p01_500_w0p1_m0p05_s0
nca_wm/refine-logs/vq_margin_kmeans_layernorm_w0p01_500_w0p1_m0p01_s0
```

The direct margin objective works in the narrow sense: it raises the
nearest/second-nearest gap by roughly three orders of magnitude and preserves
token reconstruction accuracy. However, it also collapses hard utilization
from 52-70 active codes down to 11 active codes. This is not the desired final
discrete bottleneck; it trades fragile high-utilization assignments for a much
smaller number of stable assignments.

### GPU 5k Follow-Up

The long margin run was rerun on an RTX 4090 with the same seed and setting:

```text
nca_wm/refine-logs/vq_margin_kmeans_layernorm_w0p01_5k_w0p1_m0p01_s0
```

Because the CPU and GPU backends made very different hard argmin choices under
near-ties, a matching no-margin GPU baseline was also run:

```text
nca_wm/refine-logs/vq_kmeans_layernorm_w0p01_5k_s0_gpu
```

| run | token acc | mean game acc | hard util | soft perp | train margin | recomputed util | diagnostic margin | zero drop | shuffle-game drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU 5k no-margin | 0.9976 | 0.9972 | 52 | 1021.9 | - | 51 | 0.000012 | 0.4392 | 0.0145 |
| GPU 5k no-margin | 0.9973 | 0.9967 | 5 | 1021.7 | 0.003181 | 41 | -0.001040 | 0.3658 | 0.0081 |
| CPU 500 margin target 0.01 | 0.9966 | 0.9964 | 11 | 1021.7 | 0.005954 | 11 | 0.005957 | 0.5557 | 0.0155 |
| GPU 5k margin target 0.01 | 0.9973 | 0.9967 | 5 | 1021.7 | 0.007576 | 33 | -0.000958 | 0.4052 | 0.0054 |

The GPU result changes the diagnosis. The high hard utilization seen in the
CPU `kmeans + layernorm` run is not backend-stable. On GPU, the same
initialization and seed already uses only a handful of saved hard codes, while
the CPU diagnostic can recompute many alternative nearest codes from the same
checkpoint. Negative diagnostic margins mean that CPU recomputation sometimes
finds a different nearest code than the GPU-saved hard index.

This confirms that the bottleneck is dominated by numerical near-ties rather
than a robust discrete partition. The direct margin loss does not solve the
core problem on GPU: it slightly raises the training margin but leaves saved
hard utilization at 5 active codes.

### Hard-Assignment Balance Objective

The next objective added a straight-through hard-assignment balance loss. Its
forward value uses hard one-hot assignments, while gradients flow through a
softmax over negative distances. The target was hard-assignment perplexity 64.

GPU 5k runs:

```text
nca_wm/refine-logs/vq_balance_kmeans_layernorm_w0p01_5k_bal0p1_t64_s0
nca_wm/refine-logs/vq_balance_margin_kmeans_layernorm_w0p01_5k_bal0p1_t64_margin0p1_m0p01_s0
nca_wm/refine-logs/vq_balance_kmeans_layernorm_w0p01_5k_bal1p0_t64_s0
```

| run | token acc | mean game acc | saved hard util | train hard-balance perp | train margin | recomputed util | diagnostic margin | zero drop | shuffle-game drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU no-margin | 0.9973 | 0.9967 | 5 | - | 0.003181 | 41 | -0.001040 | 0.3658 | 0.0081 |
| GPU margin | 0.9973 | 0.9967 | 5 | - | 0.007576 | 33 | -0.000958 | 0.4052 | 0.0054 |
| GPU balance 0.1 | 0.9970 | 0.9968 | 6 | 4.3 | 0.003437 | 61 | -0.001125 | 0.4119 | 0.0108 |
| GPU balance+margin 0.1 | 0.9973 | 0.9968 | 7 | 4.5 | 0.005765 | 51 | -0.001265 | 0.5682 | 0.0391 |
| GPU balance 1.0 | 0.9973 | 0.9968 | 6 | 4.4 | 0.001982 | 84 | -0.001582 | 0.3789 | 0.0074 |

This objective also does not solve the saved hard-assignment collapse. Even at
10x higher weight, saved hard utilization remains 6-7 active codes and
training hard-balance perplexity remains around 4-5, far below the target 64.
The CPU diagnostic can recompute many alternative nearest codes from these
checkpoints, but the saved GPU hard indices remain collapsed. This again
points to unstable near-ties rather than a robust discrete code partition.

### Soft-To-Hard And Gumbel-ST Annealing

The next experiment tested annealed assignment modes on GPU for 20k updates,
using temperature `2.0 -> 0.05`:

- `soft_st`: hard forward assignments with softmax gradients.
- `soft`: soft forward assignments during training/eval at the annealed
  temperature.
- `gumbel_st`: sampled hard forward assignments during training with
  Gumbel-softmax gradients; deterministic nearest-code evaluation at the end.

Output roots:

```text
nca_wm/refine-logs/vq_softst_anneal_kmeans_layernorm_w0p01_20k_t2_to_0p05_s0
nca_wm/refine-logs/vq_soft_anneal_kmeans_layernorm_w0p01_20k_t2_to_0p05_s0
nca_wm/refine-logs/vq_gumbelst_anneal_kmeans_layernorm_w0p01_20k_t2_to_0p05_s0
```

| run | token acc | mean game acc | saved hard util | final temp | assign perplexity | train margin | recomputed util | diagnostic margin | zero drop | shuffle-game drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU hard 5k | 0.9973 | 0.9967 | 5 | - | - | 0.003181 | 41 | -0.001040 | 0.3658 | 0.0081 |
| GPU soft-ST 20k | 0.9973 | 0.9968 | 5 | 0.050 | 1001.1 | 0.004917 | 52 | -0.000731 | 0.1384 | 0.0098 |
| GPU soft 20k | 0.9997 | 0.9996 | 5 | 0.050 | 753.5 | 0.072637 | 8 | 0.065270 | 0.1516 | 0.0152 |
| GPU Gumbel-ST 20k | 0.9970 | 0.9965 | 5 | 0.050 | 974.2 | 0.001331 | 51 | -0.000634 | 0.2711 | 0.0071 |

Annealing improves some secondary metrics but not the main target. The soft
forward run reaches the best reconstruction accuracy and a positive diagnostic
margin, but it still saves only 5 hard codes. Soft-ST and Gumbel-ST keep high
assignment perplexity during training, but deterministic hard utilization at
the end is still 5 active codes. This means the soft assignment distribution
can remain broad without creating a stable diverse nearest-code partition.

### Residual And Product VQ Capacity

The next follow-up changed the discrete representation capacity rather than
only the assignment loss:

- `residual`: two 256-entry codebooks applied sequentially to slot residuals.
- `product`: four 256-entry codebooks applied to fixed 16-dimensional slot
  subspaces and concatenated back to 64 dimensions.

Output roots:

```text
nca_wm/refine-logs/vq_residual2_kmeans_layernorm_w0p01_5k_s0_gpu
nca_wm/refine-logs/vq_product4_kmeans_layernorm_w0p01_5k_s0_gpu
```

| run | token acc | mean game acc | saved hard util | stage hard utils | soft perp | train margin | diagnostic hard util | recomputed util | diagnostic margin | zero drop | shuffle-game drop |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| GPU single 5k | 0.9973 | 0.9967 | 5 | - | 1021.7 | 0.003181 | 5 | 41 | -0.001040 | 0.3658 | 0.0081 |
| GPU residual-2 5k | 0.9980 | 0.9974 | 11 | `[5, 6]` | 256.0 | 0.005487 | 11 | 44 | -0.000006 | 0.2839 | 0.0115 |
| GPU product-4 5k | 0.9973 | 0.9968 | 2 | `[5, 5, 5, 5]` | 255.4 | 0.005637 | 5 | 5 | -0.000077 | 0.3014 | 0.0104 |

Residual VQ improves token reconstruction slightly and raises the saved joint
code-tuple count from 5 to 11, but the per-stage hard usage remains collapsed:
the two stages use only 5 and 6 active codes. The diagnostic margin is still
near zero, and CPU recomputation still finds many alternative tuples. This is
not enough to justify a seed sweep.

Product VQ does not help. Each subspace uses only about 5 active codes, and
the saved joint tuple count is still tiny. The final summary and diagnostic
even disagree on the exact joint count (`2` vs `5`), which is another symptom
of fragile near-tie assignments rather than a stable partition.

The representation-capacity variants therefore do not solve the hard-collapse
failure. The next change should stop relying on smooth nearest-code training.

## State Coverage Question

The current world-model training states do not come from an RL gameplay agent
trajectory. They come from C++ engine search over each authored level:

- `collect_unique_transitions` initializes the engine with `load_level`, then
  calls either `collect_transitions_astar` or `collect_transitions_bfs`.
- The C++ collectors start from `engine.backupLevel()` after the level is
  loaded, expand a frontier of changed successor states, and record every
  `(state, action, next_state)` triple for each expanded state, including
  no-op identity transitions.
- `collect_multigame_dataset` repeats that per game and per selected level,
  then samples minibatches from the cached `per_game_states`,
  `per_game_actions`, and `per_game_next_states` arrays.

So the training distribution is broader than "states visited by one gameplay
agent", but it is not the full set of arbitrary valid PuzzleScript states. It
is the set of states reachable from authored level starts under the search
budget, timeout, cache cap, and any per-game train budget. If a valid state is
not reachable from a level start, or if it is reachable but not explored before
the budget/cap, the current training loop does not train on it.

For the long-term goal of replacing the engine, this matters. A true engine
replacement should be correct for any valid state in the game state space, not
only the reachable subset sampled by A*/BFS from authored starts. The current
setup is closer to a learned transition model over a search-collected reachable
state distribution. To test engine replacement quality, add an explicit
out-of-distribution state eval:

- random-walk starts that are held out from training;
- synthetic valid states generated from the game object's type constraints;
- full BFS reachable-state evaluation for small levels where exhaustive
  coverage is tractable;
- per-state membership checks distinguishing train-cache states, reachable
  but held-out states, and generated valid-but-unseen states.

## Accuracy Question

The current evidence does not show that fixing hard-code collapse will
materially improve token reconstruction accuracy by itself. The token decoder
is already near saturation:

- GPU single-codebook 5k reaches token accuracy `0.9973` with saved hard util
  `5`.
- GPU residual-2 5k reaches `0.9980`, but still has collapsed per-stage hard
  usage `[5, 6]`.
- GPU product-4 5k returns to `0.9973` and also remains collapsed.
- The earlier soft-forward 20k run reached `0.9997` while still saving only 5
  deterministic hard codes.

That pattern means token reconstruction accuracy can improve without solving
hard collapse, and solving hard collapse is not currently proven to be the
limiting factor for reconstruction. The more plausible reason to fix collapse
is not raw token-AE accuracy; it is to create a stable discrete adaptation
interface for downstream world-model fitting, inverse fitting, and few-shot
novel-game adaptation.

To answer the accuracy question rigorously, the next useful evaluation is not
another token-AE table alone. It should compare downstream transition metrics
after using the same encoder source with different latent treatments:

- teacher-forced transition error and changed-cell error;
- autoregressive rollout error;
- token/slot ablation sensitivity;
- few-shot inverse-fit performance on held-out games;
- reachable-state versus held-out/random-valid-state accuracy.

If a non-collapsed discrete latent improves those downstream metrics while
token reconstruction changes little, that still counts as a success for this
research direction.

## Next Steps

Stop treating the bottleneck as a smooth nearest-code training problem. The
next experiment should be alternating discrete assignment search:

- freeze the encoder slots;
- assign per-game/per-slot discrete codes by nearest code or local search;
- train the token decoder and codebook against those fixed assignments;
- periodically refresh assignments.

Acceptance criteria for that next experiment:

- saved hard utilization should be high by construction and remain stable
  after saving/reloading;
- token accuracy should stay at least comparable to the GPU single-codebook
  baseline;
- zero-slot and shuffled-across-games ablations should still show that the
  decoder relies on the latent;
- assignments should not change under CPU diagnostic recomputation unless the
  search objective explicitly allows reassignment.

If explicit assignment search also fails to improve latent usefulness, stop
optimizing token-AE reconstruction and move the VQ question back into the
downstream world-model/adaptation objective.

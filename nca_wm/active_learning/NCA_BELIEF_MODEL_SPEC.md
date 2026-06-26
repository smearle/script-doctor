# Spec: 2-D NCA-belief world model for information-gain active learning

Replaces the 1-D token transformer. The token model identifies a world by
**autoregressively decoding every cell** of each frame — O(H*W) forward passes per
sample, which made the sokoban probe take ~30 min and makes IG-planner collection
intractable at real grid sizes. The 2-D model predicts a **whole frame in one
pass** and carries a **recurrent spatial belief state** across frames — the NCA's
persistent hidden channels *are* the in-context posterior over the hidden dynamics
`theta`. This removes the decode bottleneck, respects grid geometry, scales to many
objects, and is the bridge to a large multi-game dataset.

## 0. What must be preserved from the token model

The token model works because it **directly learns the posterior predictive**
`p(o' | h, a, o)` from `haoo'` data — it never represents `theta` or separates
epistemic uncertainty from noise; the information-gain behavior falls out. Two
properties we must not break:

1. **Joint coherence of a predicted frame.** When the variant is unknown, the next
   frame is a *mixture* over variant-outcomes (box at x+1 OR at the wall OR
   unmoved OR random). A per-cell-independent head cannot represent this — a sample
   could place the box in several cells or none. The autoregressive decoder gave
   coherence for free (later cells condition on earlier ones). We must restore it.
2. **Correct resample conditional.** IG needs `q1(o'|h,a,o)` = "having seen o,
   how likely is an i.i.d. re-roll o'". For unknown-deterministic, seeing o
   identifies the variant so `q1` sharpens onto o (high IG). For **known-random
   (chaos)**, the re-roll is independent of o, so `q1` must stay spread (IG ~0,
   not mesmerized). The token model learns this `o' ⊥ o | known-noise` structure
   from haoo' data. A model that instead *derives* `q1` by conditioning a latent
   on o will WRONGLY sharpen on the observed random direction and report
   known-chaos as informative. So `q1` must be **learned from haoo' data**, not
   computed from a posterior over a noise latent. (This is the main design trap.)

## 1. Architecture

State is an object-channel grid `o_t ∈ {0,1}^{C×H×W}` (C = #objects, multihot).

```
 encoder  E:  o_t            -> e_t         (C,H,W) -> (d,H,W)   conv stack
 belief   R:  B_{t-1},e_t,a_t-> B_t         (d_b,H,W) NCA update (the ICL belief)
 prior0   π0: B_t,a          -> Cat(K)      sample-mode for q0
 dec0     D0: B_t,a,z0       -> logits o    (C,H,W) one pass  -> q0(o|h,a)
 prior1   π1: B_t,a,e(o)     -> Cat(K)      sample-mode for q1 (sees first obs o)
 dec1     D1: B_t,a,e(o),z1  -> logits o'   (C,H,W) one pass  -> q1(o'|h,a,o)
```

- **Encoder E**: 2-3 conv layers (3x3), C->d (d≈96). Shared everywhere a frame is read.
- **Belief recurrence R (the NCA)**: `B_t` is a `(d_b,H,W)` spatial hidden state
  (d_b≈128). Update = concat[`B_{t-1}`, `e_t`, action-plane(`a_t`)] -> a few
  (2-4) shared-weight NCA steps (3x3 conv + ReLU residual). `B_t` accumulates
  evidence about `theta` from the observed transition `(a_t, o_t)` — local rule
  structure lands in local channels (cf. project_nca_arch_global_rules: add an
  axis/global pool op for long-range rules). `B_0` = encode(`o_0`) lifted to d_b.
- **Latent heads**: a *small discrete* latent `z ∈ {1..K}` (K≈16) per frame gives
  joint coherence with tractable likelihoods (the marginal is an exact K-sum).
  `prior` predicts `Cat(K)` from pooled belief+action; `decoder` is FiLM-modulated
  by `(a, z)` and emits per-cell object logits in ONE conv pass.
- **Two decoder heads, q0 and q1**: `q0` predicts `o` from `(B,a)`; `q1` predicts a
  re-roll `o'` from `(B,a,e(o))` — i.e. it additionally sees the first observation.
  Training `q1` on haoo' pairs is what teaches the correct resample conditional
  (point 0.2). Heads share E and B; differ only in whether they condition on `e(o)`.

## 2. Information-gain readout (tractable, one-pass)

Per-cell likelihood under a decoder sample-mode k:
`P(x | B,a,k) = Π_{c,i,j} softmax(D(B,a,k))[c,i,j]^{x[c,i,j]}` (masked to valid cells).

```
 q0(o | B,a)      = Σ_k π0(k|B,a)     · P(o  | D0,B,a,k)          # exact K-sum
 q1(o'| B,a,o)    = Σ_k π1(k|B,a,e(o))· P(o' | D1,B,a,e(o),k)     # exact K-sum
 IG(B,a) = E_{o~q0}[ log q1(o|B,a,o) − log q0(o|B,a) ]            # MC over o
```

Sampling `o ~ q0` is coherent: draw `k ~ π0`, then sample cells from `D0(B,a,k)`
(coherent because conditioned on k). Cost per IG estimate ≈ `n_samples × (1 q0
sample + K decoder evals for each of q0,q1)` conv passes — **no per-cell
autoregression, no O(H*W) sequential decode**. The expectimax planner reuses this
IG exactly as today, but each evaluation is conv passes not 56 transformer decodes.

Expected behavior (must reproduce): unknown variant -> `q1` sharpens on the
identifying outcome -> IG high; known-deterministic -> `q0≈q1≈1` -> IG~0;
known-chaos -> `q1` learned to stay spread (o'⊥o) -> IG~0 (not mesmerized).

## 3. Training

- Roll trajectories with the existing engine + mixed policy (reuse `worlds.py`,
  `collect.py`); feed **grids**, not tokens. Each step provides `(B_{t-1}, a_t,
  o_t, o'_t)` where `o'_t` is the haoo' resample (same pre-action snapshot, fresh
  seed — already produced by `data.sample_training_tokens`'s engine layer).
- Loss per step = two conditional-VAE ELBOs (one per head), summed, with the belief
  rolled forward (BPTT through R so ICL is learned):
  - q0: `E_{z~enc0(z|B,a,o)}[log P(o|D0,B,a,z)] − β·KL(enc0 || π0)`
  - q1: `E_{z~enc1(z|B,a,o,o')}[log P(o'|D1,B,a,e(o),z)] − β·KL(enc1 || π1)`
  - (posterior encoders `enc0/enc1` are train-only; discrete-z via
    Gumbel-softmax or exact K-marginalization since K is small — prefer exact:
    `log Σ_k π(k)P(·|k)`, no sampling, no Gumbel.)
- Per-cell loss masked to the valid (non-padded) region (cf. feedback_padding_always_masked).
- Truncated BPTT over windows of ~8-12 steps; batch of trajectories.

With exact K-marginalization the ELBO collapses to a clean mixture NLL
`−log Σ_k π(k)·P(o|k)` + an entropy/usage regularizer on `π` (to avoid latent
collapse, cf. the VQ-usage work) — no reparameterization needed.

## 4. Config + cost

| | value |
|---|---|
| channels d / d_b | 96 / 128 |
| NCA steps per belief update | 3 |
| latent K | 16 |
| conv kernel | 3x3 (+ optional axis/global pool for long-range rules) |
| params | ~1-3M (conv, grid-agnostic) |
| forward predict | 1 conv pass (all cells jointly) |
| IG estimate | n_samples × ~2K conv passes (vs token: n_samples × H*W transformer decodes) |

Grid-agnostic params -> the SAME model handles 5x5..bigger and varying object
counts (pad C to a max), which is what enables multi-game.

## 5. Validation gate, then scale

1. **Gate**: train on the existing sokoban-variant family and reproduce the token
   model's tables — per-variant held-out NLL (deterministic ≈0, chaos floor) AND
   the push-IG identification (fresh high, known ~0, **chaos known ~0**). If the
   not-mesmerized check fails, the q1/coherence design is wrong — fix before scaling.
2. **Speed**: confirm IG/probe is >>faster than the token model (target: probe in
   ~1 min vs ~30); confirms IG-planner collection (step 4) is now tractable.
3. **Scale**: bigger grids; then multi-game — many objects (pad C), a coherent
   game distribution, held-out games. This is option A from the Beam-Islands
   discussion, now reachable because (a) frame prediction is one pass and (b) the
   spatial belief generalizes across object layouts.

## Results — GATE PASSED (`nca_belief_train.py`)

Implemented (`nca_belief_model.py` + `grid_data.py`), 1.28M params, trained 6000
updates on the sokoban-variant family (loss 382 -> 0.27). Reproduces the token
model's tables predicting the whole frame in ONE conv pass:

Per-variant held-out NLL (q0 at the identified final transition):
classic 0.0000, slide 0.0000, swap 0.0000, inert 0.0515, chaos 0.9606.
(Deterministic ~0; chaos floors at ~0.96 = irreducible full-frame cost of a 1-of-4
random push direction — correctly higher than the token model's per-token-diluted
0.016.)

Push-IG identification (fresh = unknown, known = after one push):
classic 1.244->0.000, inert 1.514->0.293, slide 1.278->0.000, swap 1.395->0.000,
chaos 1.487->**-0.008**. All variants: high IG to investigate, ~0 once identified;
**chaos-known ~0 = NOT MESMERIZED** (the q1/coherence design canary holds — the
discrete latent + haoo'-trained q1 learned o'⊥o for noise).

Speed: IG probe finished in <1 min (vs ~30 min for the token model's autoregressive
56-cell decode) -> step 4 (IG-planner collection) is now tractable. Ckpt:
ckpts/nca_belief_sokoban.pt.

## Step 4 — online-vs-offline on sokoban (`sokoban_collect_compare.py`)

Fresh NCA-belief WMs trained on random- vs navigate-collected sokoban data at
matched budget, eval = held-out NLL at the identified final transition (2 seeds):

| budget | random | navigate |
|---:|---:|---:|
| 64 | 23.49 ± 12.0 | 4.25 ± 0.47 |
| 256 | 12.78 ± 1.9 | 2.09 ± 0.04 |
| 1024 | 2.96 ± 0.24 | 0.44 ± 0.05 |
| 4096 | 1.53 ± 0.34 | 0.17 ± 0.0002 |

Active >= passive at every budget by ~6-9x, far lower variance, gap sustained
(navigate@256 ≈ random@4096 -> ~8-16x sample efficiency). Fig
`nca_wm/figures/sokoban_collect_compare.{png,pdf}`.

PLANNING-COST FINDING: expectimax active *collection* does not scale (depth-3
lookahead × thousands of action selections = hours), though `belief_planner.py`
DOES navigate to the informative box qualitatively (RIGHT:+1.46 at step 0 in an
unknown world). Greedy-IG active *selection* is cheap but cannot navigate (no IG
until adjacent) — the learned-active-collection result was shown cleanly with the
token model on the directional (adjacent-action) family. So at scale, collection
wants an amortized/homing policy (doc's homing-policy idea), not raw expectimax.

## 6. Code mapping + risks

- **Reuse**: `worlds.py` (all families incl. sokoban), `collect.py` policies,
  `data.py` engine/haoo' rollout layer (swap the token serializer for a grid
  tensor adapter), the expectimax `planner.py` (IG interface unchanged), eval
  harnesses.
- **New**: `nca_belief_model.py` (E, R, heads, IG readout), a grid data adapter,
  a trainer + the gate eval. Multihot grids come straight from
  `read_obs`/`get_objects_2d` (skip the bitmask-token step).
- **Risks / open questions**:
  - *Latent capacity*: a single global K-way latent may be too coarse if outcomes
    vary in independent spatial regions (multiple boxes). Fallback: a small grid of
    local latents, or K-autoregressive latent tokens (few, not per-cell).
  - *Latent collapse*: regularize `π` usage entropy (reuse the VQ-usage approach).
  - *The chaos/known trap (0.2)*: `q1` MUST be trained on haoo' `o'`, never derived
    by conditioning a noise-latent on `o`. The gate's chaos-known≈0 check is the
    canary; treat a non-zero chaos-known as a correctness bug, not a tuning issue.
  - *Long-range rules*: pure 3x3 conv won't capture ellipsis/`[X][Y]` dynamics; add
    the axis/global pool ops (project_nca_arch_global_rules) when moving past
    sokoban.
```

# NCA World-Model Architecture Report

Living architectural reference for `nca_wm/`. This doc takes a stand on what
the right architecture is and *why* — the place to look when designing a new
ablation. Chronological run notes live in `RUNNING_REPORT.md`; promote a
finding here once other experiments depend on it. Last refresh: 2026-05-03.

## What an "NCA world model" is doing here

Given `(state_t, action_t)` for a PuzzleScript game and that game's tokenized
spec, predict `state_{t+1}` cell-wise. The model is a Neural Cellular
Automaton: a learned local update rule applied for `n_steps` iterations
(shared or per-step weights) on a grid of hidden vectors. Conditioning on
the game spec is via a perceiver-style encoder that emits K rule slots; the
NCA cross-attends to those slots at every step.

The PuzzleScript engine itself is fundamentally iterative: each tick applies
its rules **and `again` rules until the state stops changing**. So the NCA's
`n_steps` must cover whatever sequential reasoning the engine does for the
hardest rule chain in the game. Games with looping dynamics (gravity,
projectile motion, propagation chains) demand more passes than games with a
single-shot rule set.

## Module map

| Module | What it is | Where |
|---|---|---|
| `NCAWorldModel` | Unconditional shared-weight NCA. Has input-skip, residual+ReLU updates, optional shared LN. | `train.py` ~L1183 |
| `ConditionalNCAWorldModel` | FiLM-conditioned variant of `NCAWorldModel`. Same NCA body; modulated by a single pooled `z` from `GameSpecEncoder`. | `train.py` ~L1371 |
| `RuleAttnNCAWorldModel` | Current default. Cells cross-attend to K rule slots from a perceiver encoder; `--n_nca_repeats=n_nca_steps` (all-shared) recommended. | `rule_attn_model.py` |
| `GameSpecEncoder` (FiLM) / `RuleSlotEncoder` (rule_attn) | Two flavors of game-spec → conditioning. CLS+pool vs K-query slots. | `train.py` / `rule_attn_model.py` |
| `_pool_features` | Shared global-context primitive used by all three NCA variants. | `train.py` ~L1146 |

## Major findings

The findings below are load-bearing — every subsequent ablation assumes
them. Change with care, and prefer a controlled re-test over silently
toggling a flag.

### F1. Global-pool features are load-bearing, not optional

A pure 3×3 conv can only propagate one cell of context per NCA step. Several
PuzzleScript rule patterns reference cells that are arbitrarily far apart in
the same tick: `[ X | ... | Y ]` (same row/col, any distance) and `[X] [Y]`
(any positions on the level). Without pooled features, the NCA needs
`n_steps ≥ max_grid_dim` just to *see* across the grid. With them, one step
suffices.

`_pool_features` exposes three independent flags (any combination):

| Flag | Cost (per step) | Adds | Necessary for |
|---|---|---|---|
| `axis_pool` | 2 broadcasted features | row-max, col-max | `[X \| ... \| Y]` (existence in same axis) |
| `axis_cummax` | 4 broadcasted features | directional prefix-max (L→R, R→L, T→B, B→T) | directional `…` rules; "is X to my left/right" |
| `global_pool` | 1 broadcasted feature | grid-wide max | `[X] [Y]` multi-bracket |

**Evidence:**

- 19-game `scaling_large` rule_attn FiLM ablation: turning the pool stack
  off costs convergence speed and final loss.
- Collapse-L0 no-pool runs (per-step n=4 and n=16+stab) sit at ~1% train
  loss / 30-40% change-error — identity-collapse — even with 4× the depth
  and the stability patch. Pooling carries information that depth in a
  bare 3×3 conv stack does not recover within 15k optimizer steps.

**Consequence:** all three flags default on. Turn them off only as a
controlled ablation, document, and turn back on. See Q1 below for the open
question of whether pooling is "hacking" specific games' dynamics.

### F2. Shared NCA-body weights are the right inductive bias (and the engine has *two* loop levels)

The PuzzleScript engine has two natural levels of iteration:

1. **Inner loop:** an ordered list of distinct rules, applied once each
   per tick. Each rule is a different "rule" — different weights are
   appropriate.
2. **Outer loop:** the engine re-runs the *same* rule list while any
   `again` rule fired or a chain rule had cascading effects. Same rule
   set, applied repeatedly until the state stops changing.

A model that matches this prior should mirror the two-level structure:
distinct weights for distinct rules within an inner block, **shared**
weights across outer-loop repeats. 2026-05-02 added `--shared_weights`
(boolean: all-or-nothing across `n_nca_steps`). 2026-05-03 generalized it
to `--n_nca_repeats` (the current flag), which factors `n_nca_steps` into
`n_layers × n_repeats`:

- `n_layers = n_nca_steps // n_nca_repeats` distinct layers in the inner
  block (each with its own conv / pool_proj / attn_ln / slot_ln /
  cell_slot_xattn / out).
- `n_repeats` applications of the inner block, sharing weights across
  repeats.

Special cases:

- `--n_nca_repeats=1` (default, back-compat): n_layers = n_nca_steps. Old
  per-step body bit-identically.
- `--n_nca_repeats=n_nca_steps`: n_layers = 1. One layer, applied
  n_nca_steps times — equivalent to the old `--shared_weights`.
- `--n_nca_repeats=k` with `1 < k < n_nca_steps`: hierarchical. E.g.
  n_steps=8, n_repeats=2 gives a 4-layer inner block applied twice.

Constraint: `n_nca_steps` must be divisible by `n_nca_repeats`.

The all-shared regime (`n_repeats=n_steps`) addresses the original
adaptive-halting blocker: with one shared layer, "halt at step k" asks
the same rule set to converge in k iterations, not selecting between k
different models. The hierarchical regime is a free generalization that
opens a new ablation axis ("how many distinct rule layers does the inner
block need?") not yet measured.

Empirical evidence below uses the all-shared variant (`n_repeats=n_steps`)
since that's what we have for now; the hierarchical regime is in the
experiments table (E10).

**Evidence — Collapse-L0 single-game depth sweep (h=256, K=16, 15k steps):**

Shared-weights body:

| n_nca_steps | params | best loss | L0 rand_tf | L0 random | L0 bfs | L0 astar |
|---|---|---|---|---|---|---|
| 2  | 1.39M | 1.59e-6   | 3  | 209 | 61  | 71  |
| 4  | 1.39M | 5.42e-7   | 4  | 165 | 111 | **60** |
| 8  | 1.39M | **2.79e-7** | 17 | 207 | 204 | 119 |
| 16 | 1.39M | 5.43e-7   | 24 | **153** | 75  | 100 |
| 16 + stab | 1.98M | 1.61e-5 | 4 | 155 | 212 | 124 |

Per-step (un-shared) body, same setup, for direct comparison:

| n_nca_steps | params | best loss | rand_tf | random | bfs | astar |
|---|---|---|---|---|---|---|
| 2          | 2.6M  | 1.07e-3 | 2 | 196 | 117 | 112 |
| 4          | 5.0M  | 1.70e-4 | 3 | 102 | 169 | 200 |
| 8          | 9.9M  | 1.55e-6 | 5 | 207 | 198 | 208 |
| 16         | ~19M  | 3.71e-6 | 8 | 206 | 179 | 171 |
| 16 + stab  | ~19M  | 1.36e-5 | **1** | **67**  | **53**  | **56**  |

(Wrong-tile counts mean per-state over 50-step rollouts on level 0; divisor
3,040 cells. `random_tf` = teacher-forced 1-step under random actions;
`random` / `bfs` / `astar` = autoregressive under random or oracle policies.
"Stab" = `--use_layernorm --input_skip` — see F4.)

**Quantitative claims:**

- Shared n=8 reaches **5.5× lower train loss with 7× fewer params** than
  per-step n=8 (2.79e-7 @ 1.39M vs 1.55e-6 @ 9.9M).
- Shared n=2 (1.39M, 14× smaller than the best per-step row) ties or beats
  per-step n=16-stab on bfs (61 vs 53) and astar (71 vs 56).
- On random-action rollouts the shared body trails per-step+stab (best
  shared random=153 vs 67) — so per-step+stab still wins under
  distribution shift; see F3 for why this gap is real and per-step-specific.

**Sanity check on a different game (Microban L0, 2026-05-03):** Same
setup ported to Microban level 0 (a tiny 6×7 single-box puzzle).
Shared-body train losses across n ∈ {2, 4, 8, 16}: 5.79e-7, 1.78e-7,
1.58e-7, 3.10e-7 — all much lower than the per-step Collapse numbers.
All four runs hit zero wrong-tile rollouts on L0
(random/random_tf/bfs/astar) at every depth, so depth was invisible
here too. The interesting datum: shared weights ports off Collapse
without re-tuning. (Microban is **not** a Q1 test — chain pushing is
axis-aligned single-rule iteration that pool can fully substitute for.
See E9 in the experiments table.)

### F3. Train loss decouples from autoregressive rollout error

A model with the lowest train (per-step) loss is not the model with the
lowest autoregressive rollout error, especially in the over-capacity regime.
The clearest demonstrations:

- Per-step n=8 has the lowest per-step body's train loss (1.55e-6) but
  random/bfs/astar all ~200. Per-step n=16-stab has 9× *higher* train loss
  (1.36e-5) yet 3-4× *lower* rollout error (random=67, bfs=53, astar=56).
- Shared n=2 reaches train loss 1.59e-6 (≈ per-step n=8's 1.55e-6) but
  with bfs=61 (≈ per-step n=16-stab's 53) — i.e. shared n=2 matches
  per-step n=8 on train loss while matching per-step n=16-stab on rollout.

**Mechanism:** high-capacity per-step bodies overfit to per-step prediction
in a way that produces small but systematic errors compounding destructively
under autoregressive rollout. Teacher-forced 1-step error tracks train loss;
multi-step error doesn't.

**Consequence:** Reporting only `best_loss` (as `RUNNING_REPORT` does for
most experiments) hides the rollout-drift cost of going deeper or wider.
New rule_attn experiments should report **both** the train metric and at
least the 50-step autoregressive eval — they're cheap (the eval already
runs at end-of-train) and they're often pointing in opposite directions.

### F4. The "stab patch" is just `--input_skip`; `--use_layernorm` is at best neutral, at worst harmful

Two flags were added 2026-05-01 to make deep per-step rule_attn unrolls
trainable, originally treated as a single bundle:

| Flag | Effect |
|---|---|
| `--use_layernorm` | shared pre-norm `LayerNorm` on `h` at start of every NCA step |
| `--input_skip` | re-inject the embedded `(state, action)` into the conv input at every step |

Both default off.

**Disaggregating the bundle (Bouncers no-pool, 2026-05-03):** the bundle's
benefit comes entirely from `input_skip`. Adding LN on top is at best
neutral and in the per-step regime degrades rollouts.

| (L, R) | bare bfs | LN-only | **skip-only** | bundled |
|---|---|---|---|---|
| (1, 16) max-shared | 3.59% | 2.80% | **0.88%** | 0.88% |
| (16, 1) per-step deep | (n/a) | 1.83% | **0.88%** | 1.06% |

Earlier (2026-05-02) shared-weights Collapse evidence said the bundle was
*anti*-helpful for shared bodies (raised train loss 30×, degraded rollouts
75→212). The 2026-05-03 disaggregation explains that result: the offending
component was LN. With pool ON and shared weights, LN raised train loss
30×; with pool OFF and shared weights, LN partially helps (3.59 → 2.80) but
input_skip alone fully recovers (3.59 → 0.88). **LN's effect is regime-
dependent and unreliable; input_skip's effect is consistent and large.**

**Recommendation:** default `--input_skip` on for any deep config (≥8
iterations), pool or no-pool, shared or per-step. Drop `--use_layernorm`
entirely from the recommended recipe. The earlier-cited rollout-error
3-4× recovery on per-step n=16 was almost certainly the input_skip half
of the patch; the LN half was carrying along for the ride.

### F5. `change_loss_weight` is what actually breaks identity-collapse

In multi-game training, most cells don't change between `t` and `t+1`. Plain
BCE has the model sit on the "predict input = input" identity minimum for
many thousands of steps; patience-on-uniform-BCE was firing mid-recovery in
earlier experiments. `--change_loss_weight 5.0` makes each changed cell
count 6× — the patience metric tracks something the researcher cares about,
and the model gets gradient signal to leave identity. Not strictly an
architectural decision but part of the canonical recipe; should not be
omitted.

### F6. Rule_attn dominates FiLM at the same conditioning capacity

On the 19-game `scaling_large` set (h=256), rule_attn K=16 reaches 9.4%
mean change-error vs FiLM (d_z=256, d_model=128, encL=4) at 14.5%, with
about 3.4× faster convergence. K=32 (over-parameterized) hits 10.4%; K=16
sits on the saturating end of the diminishing-returns curve. Rule_attn is
the current default for this reason.

### F7. (Resolved 2026-04-18) Encoder mask bug invalidated pre-fix checkpoints

Flax's `MultiHeadDotProductAttention` interprets the `mask` argument as
**boolean** (True = keep). Earlier code passed a float mask with inverted
semantics. Pre-fix, every game's slot tensor cosine-similarity was
0.77–0.99 — the encoder was effectively shared-conditioning across all
games. Post-fix, distinct games have cos-sim around 0.31, and rule_attn
saturates `scaling_6` at 1.5e-5 best-loss. **Any pre-2026-04-18 saved
checkpoint should be considered suspect.** Kept here so that anyone reading
old logs knows why pre-fix numbers don't reproduce.

### F8. (2026-05-03) Multi-grid varislide is an init-determined basin failure, not depth/compute/sharing/clw

> **2026-05-04 INVALIDATED.** All five claims below were artifacts of the
> 2026-05-01 bitpack regression (commit 1fa557d): `collect_multigame_dataset_synthetic`
> appended raw uint8 multihot arrays where the bucket loader expected bit-packed
> data, so every synth-trained model since May 1 saw all-zero `states` /
> `next_states`. Train loss looked like it converged (BCE on all-zero targets
> with logits → -∞ underflows to ~1e-42); meanwhile the model learned to predict
> 0 everywhere, producing the 86% right-action change_err and the per-distance
> "fire-once" failure mode reported here.
>
> Post-fix (1fa557d) re-runs at h=128, batch=16, lr=3e-4, 10k updates,
> mask_hidden=True default: depth ∈ {8, 16} fully shared all reach 100%
> argmax across slide distances {1, 2, 3+} on the same multi-grid synth set.
> The full re-run sweep (depth, L×R factor, pool on/off) is in
> `nca_wm/logs_canary/varislide_postfix{A,B,C,Cp}_*` and summarized in
> `nca_wm/figures/varislide_postfix/`. F8's specific architectural claims
> (depth flat, sharing flat, basin-init dominant, "rule-conditioned NCAs do
> not reliably learn iterative rule application") should be considered
> withdrawn pending that sweep's full results.

Multi-seed verification (E20-E22) of the architecture-report E15-E19
results pins down what's actually happening on multi-grid varislide
(custom 1-rule slide-until-wall game, widths {6,8,10,12,16}):

**Headline (3 seeds × 4 depths × multiple recipes, all measured by
right-action change_err and player-position argmax accuracy):**

| recipe | change_err (mean ± std) | argmax_acc (mean ± std) |
|---|---|---|
| shared, depth=8, 10k | 0.443 ± 0.041 | 0.227 ± 0.067 |
| shared, depth=16, 10k | 0.443 ± 0.055 | 0.231 ± 0.105 |
| shared, depth=32, 10k | 0.420 ± 0.109 | 0.174 ± 0.095 |
| shared, depth=64, 10k | 0.429 ± 0.049 | 0.296 ± 0.164 |
| shared, depth=16, 50k | 0.466 ± 0.026 | 0.194 ± 0.114 |
| **per-step**, depth=16, 10k | 0.418 ± 0.075 | 0.250 ± 0.041 |
| `--input_skip`, depth=16, 10k (n=1) | 0.474 | 0.196 |
| `--change_loss_weight 50`, depth=16, 10k (n=1) | 0.445 | 0.532 |

**Five claims established by these data:**

1. **Depth doesn't help.** 8 ↔ 64 with shared weights: no monotonic
   trend, all means within each other's std. The E19 "n=20 outlier at
   0.29" was single-seed noise (E20).
2. **5× compute doesn't help.** Long-training (50k) at depth=16 is no
   better than 10k. Per-seed *ordering* is preserved across compute
   budgets (best seed at 10k is best at 50k) — i.e., the basin is
   determined at init and stable under further optimization.
3. **Sharing isn't the active variable.** Per-step weights (~11× more
   params, n_repeats=1) give essentially identical aggregate metrics.
   Faint hint that per-step gets d≥2 right more often (~0.4 vs ~0.2),
   but overall it's a wash.
4. **Per-seed variance dwarfs the recipe signal.** Within shared-depth=64,
   seed 0 → argmax 0.07, seed 1 → 0.44 — same hyperparams, ~6× different
   solution quality. The optimization landscape has multiple basins of
   very different rule-learning quality reachable from random init.
5. **Best-loss is essentially identical (1.6–1.8e-4) across every
   recipe and every seed of every depth.** F3's train-rollout decoupling
   is amplified here: the train loss is *blind* to a 5× swing in actual
   rule-learning quality.

**Diagnosis (per-distance breakdown, baseline seed=0):** the failure is
*asymmetric*. The model perfectly suppresses the player at the cleared
cell (sigmoid 0.000 on Player at GT 1→0 cells) and at no-change cells,
but fails to place the player at the destination (sigmoid 0.090, only
6.4% above 0.5). For slide_d=1 the model gets the right cell ~64% of the
time at low confidence; for slide_d≥2 it collapses to chance (~33%) and
mean predicted slide distance is ~2 regardless of true distance. The
model has converged to a "fire the rule once" mode and never iterates.

**Mechanism (working hypothesis):** the per-state lookup attractor is
much easier to find than the iterative-rule-application attractor under
final-step BCE. Multi-grid prevents per-state memorization (forces SOME
rule-learning), but the model gets stuck in the lowest-effort partial
solution — fire the rule once, get d=1 right ~60% of the time on
average, plateau loss at ~1.7e-4. None of the tested architectural
variants (depth, sharing, capacity, input_skip, change_loss_weight)
reliably escape this attractor. The "lucky" seeds are in a different
basin that learned more iteration; they're not reproducible from a
distinct init.

**Consequence — paper-relevant:** rule-conditioned NCAs with the
current architecture do not reliably learn iterative rule application
on multi-grid synth data, regardless of depth budget, compute budget,
weight sharing, capacity, or change-loss weighting. The remaining
candidate fixes that have NOT been ruled out are (a) hard / sparse
rule-slot routing (one slot active per step) so the model can't smear
"fire-once" across all slots, (b) per-cell halt mechanism so cells in
their fixed point stop updating, (c) curriculum / replay-buffer
training that selectively exposes the model to slide_d ≥ 2 instances,
(d) auxiliary "is-changing" mask head factoring "where to fire" from
"what to write." Hand-coded engine-aligned per-step supervision (E18-ish)
is technically ruled in but is single-game and doesn't generalize.

**Operational note:** train change_err for multi-grid varislide can
read 0% in the log while right-action change_err is 86%, because most
training batches have zero changing transitions and the per-batch
formula returns 1.0 when n_changed=0. Always re-evaluate per relevant
action with the inspector at `nca_wm/scripts/inspect_varislide_distance.py`.

## Open questions

### Q1. Does NCA depth help games with genuine looping dynamics?

Collapse depth (shared body) shows essentially flat best-loss across
n ∈ {2, 4, 8, 16} — 1.59e-6 to 5.43e-7 — and rollout error is mixed across
depths with no monotonic trend. So *on Collapse*, depth past n=2 doesn't
add much. The honest interpretation: Collapse-style "move horizontally
until hitting a wall, then fall to ground" is exactly the regime where
`axis_cummax` already gives a one-step lookup ("is there a Wall to my
right?"), so the canonical pool stack masks any depth-helps signal.

Need a game where pool provably can't substitute for depth. Criteria:

1. The looping rule's reach is **not axis-aligned** (so pool/cummax can't
   express it as a one-step lookup).
2. The looping is **long** (mean number of `again` iterations per tick
   significantly > 1).
3. Ideally **sequential-state-dependent** — each iteration sees a state
   modified by the prior one, so the model can't precompute the answer
   from the initial state alone.

Candidate audit (gallery games, by how cleanly they meet the criteria):

- **Bouncers** — `again`-driven ball-trajectory simulation. Ball moves
  one cell per `again`; bouncers redirect it; trajectory ends when ball
  hits something. **Hits all three criteria cleanly** (non-axis-aligned
  because the ball turns; long because trajectories cross most of the
  level; sequential-state-dependent because each `again` step sees the
  ball at a new position). 18 `again` mentions in rules. **Picked as the
  current Q1 test (E1 below).**
- **(unnamed) flood-fill parliament/pathfinding game** — the user
  recalls a gallery game that simulates a parliament using flood-based
  pathfinding to compute relative distances. Flood fill is the
  textbook example of "depth must equal max path length" propagation
  that pool can't substitute for. Title not yet located in
  `scraped_games/` (searches for parliament/political/government/vote
  titles found nothing matching). If the title surfaces later, a
  flood-fill game would be a stronger Q1 test than Bouncers and should
  replace E1.
- **Mirror Isles / Mirror_Bounce** — light-beam reflection looks
  non-axis-aligned in spirit, but the rules turn out to be `[X | ... | Y]`
  patterns (axis-aligned, pool-hackable).
- **Heroes of Sokoban** — multi-character control with chained
  Wizard-Temp propagation.
- **Sokoboros** — N-deep chained body-following along the snake; non-axis
  because the body curves. Big game (90 rules).
- **Microban** — classic Sokoban chain pushing. **Considered then
  rejected** as a Q1 test: chain pushing is a single rule iterated
  axis-aligned along one direction, exactly the regime
  `axis_cummax` was built for.

Active follow-up experiments — see "In-progress" section below.

### Q2. Adaptive pass count — PonderNet-style halt (v1 implemented 2026-05-03)

A fixed `n_steps` is wasteful for easy transitions and insufficient for
hard ones. The natural extension: let the model halt — analogous to
PonderNet / Adaptive Computation Time / Universal Transformers' halting.
F2 made this well-posed: with one shared body
(`--n_nca_repeats=n_nca_steps`), "halt at step k" asks the same rule set
to converge in k iterations, not selecting between k different models.

#### What's now in the code

`RuleAttnNCAWorldModel` gained an `adaptive_halt: bool = False` flag.
When True:

- A small head (one shared `LayerNorm` + `Dense(n_hid → 1)`, ~97 params
  at h=32) reads pooled `h` at every NCA step and emits a halt logit.
- The model returns per-step `(logits_k, win_logit_k, halt_logit_k)`
  stacked along a leading T axis as a trailing `halt_aux` element.

Training side (`make_train_step` / `_ponder_loss` in `train.py`):

- Halt distribution: `λ_k = sigmoid(halt_logit_k)` for `k < T`; the last
  step is forced halt (`λ_T := 1`) so `Σ_k p_k = 1` exactly, where
  `p_k = λ_k · Π_{j<k}(1 − λ_j)`.
- Loss `= L_rec + halt_kl_weight · KL(p ‖ Geom(halt_prior_p))` where
  `L_rec = Σ_k p_k · L_k` and `L_k` is the same state+win loss
  `_heads_loss` computes (with the same `change_loss_weight`).

CLI flags: `--adaptive_halt`, `--halt_prior_p` (geometric prior
parameter, default 0.1 → expected ~10 steps), `--halt_kl_weight` (KL
weight, default 0.01).

Validation gates in `train.py` enforce the prerequisites:
`--n_nca_repeats == --n_nca_steps`, `--conditional`, no VQ, no joint
token decoder. Smoke-tested on Collapse-L0 (300 updates, h=64): training
loss decreased monotonically; eval ran end-to-end.

#### v1 → v2 ponder-loss fix (2026-05-03)

The v1 ponder loss did `L_rec = (mean_batch(p_k)) · (mean_batch(L_k))`,
i.e. averaged the halt distribution and the per-step loss separately
across the batch and then took their outer product. This is the
"average-difficulty model" — it can't reward per-input adaptive
halting, because each batch element's halt distribution gets paired
with the batch *mean* per-step loss rather than its own.

The bug surfaced cleanly on the no-pool varislide test (E13): with
fixed code the model should halt deeper for longer slide distances
(more iteration needed); with the v1 loss it instead halted *shallower*
for harder examples (an artifact of the wrong gradient).

**v2 fix**: `L_rec = mean_b(Σ_k p_k(b) · L_k(b))` — per-batch-element
halt distribution paired with per-batch-element per-step loss. State
loss and win loss both keep their batch axis until *after* the
multiplication by p. Reporting metrics (`acc`, `change_acc`, `win_acc`,
`E[k]`, etc.) still use the batch-marginal halt distribution since
they're scalars; the loss now correctly couples per-input p_k(b) with
per-input L_k(b).

All E11/E12/E13 numbers were collected under v1; E13 was re-run under
v2. Per-instance E[k] decreased very slightly (1.85 → 1.62 across the
distance range) but the headline pattern was unchanged, see below.

#### E13 (v2) result: halt collapses to k=1 + model never learns slide dynamics

Same setup as E13 v1 but with the per-batch ponder-loss fix. Per-level
halt distribution at eval (filtering to "right" action):

| level | slide_distance | E[halt step] | argmax_k | L_1 chg_err | L_8 chg_err | L_16 chg_err |
|---|---|---|---|---|---|---|
| L0 | 1  | 1.85 | 1 | 1.000 | 1.000 | 1.000 |
| L1 | 2  | 1.84 | 1 | 1.000 | 1.000 | 1.000 |
| L2 | 3  | 1.84 | 1 | 1.000 | 1.000 | 1.000 |
| L3 | 4  | 1.84 | 1 | 1.000 | 1.000 | 1.000 |
| L4 | 6  | 1.74 | 1 | 0.667 | 0.889 | 0.889 |
| L5 | 8  | 1.68 | 1 | 0.950 | 0.750 | 0.750 |
| L6 | 12 | 1.65 | 1 | 0.962 | 0.808 | 0.808 |
| L7 | 16 | 1.62 | 1 | 0.946 | 0.730 | 0.730 |

Halt mass concentrated at k=1 (~0.6) for every level. **The model never
learned to predict the slide correctly** — change_err on changed cells
is 65-100% across all NCA depths. For long slides (L5-L7), depth helps
a little (L_16 chg_err ≈ 0.73 vs L_1 ≈ 0.95) but the gap is small
enough that the KL prior toward early halting wins.

**Why this happened — the F1 ↔ Q2 interaction.** The no-pool regime
caused per-F1 identity-collapse on the dynamics: the model fits L_k ≈
"predict identity at all k" rather than learning depth-dependent
predictions. Per-step losses then look uniformly mediocre, the
optimizer takes the easiest gradient path (halt early to satisfy KL),
and the halt distribution loses the per-instance signal it needs.

**What the experiment was actually trying to do, and where the
substrate failed:**

- **Goal**: show E[k] tracks instance difficulty within a single
  trained model.
- **Required**: (a) dynamics tractable for the model so L_k can be
  meaningfully low for some k, and (b) the *required* k differs by
  instance.
- **What happened**: pool-on (E12) made k irrelevant (L_k uniformly
  low ⇒ halt is KL-dominated near prior); pool-off (E13) made k useless
  (L_k uniformly high ⇒ halt is KL-dominated near k=1).
- **Open**: the middle regime — dynamics tractable enough to learn
  cleanly but with depth genuinely binding — needs a different
  substrate. Likely candidates: pool with one channel but not all
  three, or a Q1-positive game where pool *cannot* substitute even
  when on (flood-fill / non-axis-aligned propagation).

Saved figures (both with corrected padding): `varislide_per_level_figure.{pdf,png}`
(pool-on) and `varislide_per_level_nopool_v2_fixed_figure.{pdf,png}`
(pool-off, v2 loss).

#### v1 limitations (worth knowing before reading numbers)

- **Inference uses final-step logits, not the halt-aware prediction.**
  `apply_fn` returns the (logits, win, sprite) at step T. A real
  inference path would either pick `argmax_k p_k` or use the expected
  prediction `Σ_k p_k · y_k`. Eval rollouts under v1 are thus *the
  fixed-depth-T equivalent* — useful only as a sanity check, not as a
  measurement of what halting buys at inference.
- **Logged metrics are the expected-step state/win loss, not raw `L_k`
  or KL.** The KL term enters `loss` but isn't surfaced separately in
  the log line. Hard to tell from a log alone whether the halt prior
  is too tight.
- **No expected-step trace.** `E_p[k]` is computed inside `_ponder_loss`
  but discarded; should be logged so we can watch the model decide how
  many steps it actually wants.
- **Only the conditional / non-VQ / non-joint-decoder path is wired.**
  FiLM and unconditional bodies don't expose per-step intermediates.
- **Halt head is global (one halt prob per batch element).** A per-cell
  halt — closer to NCA semantics — wasn't implemented; cells all halt
  together.

#### Smoke-test result: halt head learns sensible depth on Collapse-L0

300 updates at h=64 (small), `halt_prior_p=0.2`, `halt_kl_weight=0.01`,
batch of 64 transitions:

| step k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `p_k` (learned) | 0.235 | **0.302** | 0.239 | 0.142 | 0.059 | 0.018 | 0.004 | 0.001 |
| Geom prior | 0.200 | 0.160 | 0.128 | 0.102 | 0.082 | 0.066 | 0.052 | 0.210 |

`E[halt step] = 2.55` (prior would give 4.16); `KL(p ‖ prior) = 0.38`
nats. **56/64** batch elements pick k=2 as their argmax. Matches F2:
Collapse-L0 saturates at n_steps=2, and the halt head independently
discovered the same.

#### KL-weight sensitivity sweep (3000 updates, h=128, Collapse-L0)

| `halt_kl_weight` | best loss | E[k] | p_1 | p_2 | p_3 | p_4 | p_5..8 |
|---|---|---|---|---|---|---|---|
| 0 (no KL)        | 6.09e-3 | **1.08** | 0.92 | 0.07 | 0.00 | 0.00 | 0.00 |
| 0.01 (default)   | 6.67e-3 | 4.03 | 0.01 | 0.21 | 0.27 | 0.20 | 0.31 |
| 0.1 (strong)     | 6.60e-3 | 3.72 | 0.16 | 0.21 | 0.18 | 0.13 | 0.32 |

**Findings:**

- **`kl=0` collapses the halt distribution to step 1.** This is the
  standard PonderNet failure mode: with no regularizer, the gradient
  takes the path of least resistance and forces λ_1 ≈ 1, so all loss
  mass is on L_1. The halt head learns "always halt now"; deep iteration
  is unused.
- **`kl=0.01` (default) is a clean working point.** The distribution is
  data-driven (peak at k=3, nontrivial mass through k=8) and not
  dominated by the prior.
- **`kl=0.1` (strong)** pulls the distribution closer to the prior shape
  (head and truncation both visible) at modest train-loss cost.
- **Train loss is ~constant (6.0e-3 to 6.7e-3) across all three.**
  Once Collapse's dynamics are learned, *which step* the readout
  happens at barely affects accuracy — confirming F2 (Collapse
  saturates fast). The KL term is essentially free on this game and
  necessary to avoid halt collapse.

#### Cross-game gradient (E11 results, 2026-05-03) — muddier than hoped

5000-update runs at h=128, identical recipe across three games chosen
for an a-priori complexity gradient (sokoban_basic → sokoban_match3 →
Atlas_Shrank, the last with 4× `again` rules and gravity propagation).
Halt distribution and final-step state loss:

| game | E[k] | KL | final state_loss | change_err |
|---|---|---|---|---|
| sokoban_basic   | 3.98 | 0.29 | 1.5e-3 | low |
| sokoban_match3  | 4.48 | **0.01** | 5.0e-4 | low |
| Atlas_Shrank    | 3.09 | 0.20 | 7.4e-4 | **0.29** |

The hypothesized "complexity → deeper halt" doesn't show up as a clean
monotone gradient. Three things going on:

- **sokoban_match3 KL ≈ 0**: the halt head essentially didn't learn
  anything game-specific — distribution = the geometric prior. The
  per-step state loss is low across all k, so there's no gradient
  signal pulling the halt away from prior.
- **Atlas_Shrank's change_err is 0.29**: 5000 updates at h=128 wasn't
  enough to learn its dynamics. The halt distribution reflects an
  under-trained model, not a saturated one.
- **All three with full pool features**: pool/cummax/global make the
  *predictive* problem easy enough that depth isn't binding, so the
  halt head's training signal is weak. The proper test (Q1) needs
  pool ablated off, or a game where pool can't substitute.

Saved figure: `nca_wm/logs_halt_arch/multigame_halt_figure.{pdf,png}`.
Useful as a "what halt distributions look like across games" reference,
*not* as evidence of complexity-tracked halting. The within-game
varislide test (E12) is the cleaner demonstration.

#### Within-game per-instance test (E12, pool-on, 2026-05-03) — flat

Custom `varislide` game: one rule (`right [ > Player | no Wall ] -> [ |
> Player ] again`), 8 levels with the player at distance ∈ {1, 2, 3,
4, 6, 8, 12, 16} from the right wall. Trained jointly on all 8 levels,
n_steps=16, n_repeats=16, halt_kl_weight=0.01, halt_prior_p=0.2,
5000 updates. Per-level halt distribution at eval (filtering to
"right" action only):

| level | slide_distance | E[halt step] |
|---|---|---|
| L0 | 1  | 4.75 |
| L1 | 2  | 4.78 |
| L2 | 3  | 4.79 |
| L3 | 4  | 4.78 |
| L4 | 6  | 4.82 |
| L5 | 8  | 4.72 |
| L6 | 12 | 4.70 |
| L7 | 16 | 4.68 |

**E[k] is essentially constant (4.7-4.8) across the entire distance
range.** The model didn't learn to adapt computation per instance.
Train state_loss ended at 5e-5 (perfectly fit) and change_err ≈ 0, so
the model *did* learn the dynamics — just without iterating per-input.

This is **F1 in disguise**: with `axis_cummax + global_pool` on, the
NCA can resolve "where's the wall?" in one step via cummax, so the
slide is a 1-step prediction problem regardless of distance. Depth
isn't binding, halt training has no per-input gradient signal, and the
distribution stabilizes near the prior shape.

**Direct implication for showing per-instance adaptive computation:**
the test substrate has to be one where pool cannot substitute for
depth. This collapses the question back into Q1: we need a Q1-positive
game (no axis-aligned `[X | ... | Y]` shortcut, e.g. flood-fill, light
beams, multi-character coordination) for adaptive halt to have anything
interesting to do.

E13 (no-pool varislide, currently running bg `bit1zr59b`) is the
controlled version: same game, all pool features off. The model now
needs n_steps ≥ slide_distance to even *see* the wall, so the halt
head has a real per-input difficulty gradient to fit. If it still
doesn't track distance after that, we need to question the loss
formulation; if it does, the pool-vs-depth interaction with halting
is the load-bearing finding.

Saved figure (pool-on): `nca_wm/logs_halt_arch/varislide_per_level_figure.{pdf,png}`.

#### Beyond v1

- **Halt-aware inference.** Add an inference path that thresholds `p_k`
  or returns `Σ_k p_k · y_k`. Without this, "what does halting buy at
  inference time?" can't be answered.
- **Per-cell halting.** PuzzleScript convergence is a per-tick global
  property, but per-cell halting would let cells stop early once their
  local dynamics converged (closer to engine semantics where a rule
  doesn't fire on cells where its precondition is false).
- **KL prior shape.** Geometric is the natural default; uniform-over-{1..T}
  or a learned prior are obvious alternatives.

#### Convergence-based stopping (uniform-halt + ε-threshold)

Implemented 2026-05-03 as an alternative to the learned halt head. The
core idea: instead of training a halt-prediction head (which collapsed
in E13 because of the F1 ↔ Q2 interaction), train the body so its
readout is good at *every* step, then halt at inference when consecutive
predictions stop changing. This matches the PuzzleScript engine's actual
termination criterion ("stop when state stops changing").

**New CLI flag**: `--halt_mode {ponder, uniform, argmax_st}` (only
meaningful with `--adaptive_halt`):

- `ponder` (default): existing PonderNet ponder loss with learned halt
  head + KL prior. Body gradient at every k weighted by p_k → rewards
  shortcuts.
- `uniform`: `L = mean_k(L_k)` (every step weighted equally). The halt
  head's gradients still come from the KL term (so set
  `--halt_kl_weight=0` for a clean uniform run); the halt distribution
  is no longer used for the loss. Body must be good at every depth →
  even more shortcut pressure than ponder.
- `argmax_st` (added 2026-05-03 in response to a "shortcut pressure"
  observation): straight-through estimator on the argmax of p. Forward
  computes `L_{k*}` only at the per-example selected step
  `k*(b) = argmax_k p_k(b)`; backward gradient on halt logits flows via
  soft p so halt can still learn. **Removes the per-step shortcut
  pressure** — body sees gradient only through L at its selected
  depth, so it isn't penalized for "wrong at k=1" on examples that
  should compute longer. Keep KL prior on (`--halt_kl_weight=0.01`) to
  prevent halt collapsing to k=1.

**Inference helper**: `nca_wm/scripts/eval_convergence_halt.py` takes a
trained checkpoint, computes per-step predictions, and reports the
effective halt step at multiple ε thresholds (`fraction of cells
changing < ε between consecutive steps → halt`), plus the prediction
quality at the convergence step vs at fixed `n_steps` vs at the
oracle-best step.

**Diagnostic result on existing learned-halt models**: per-step error
trajectories and Δ_disc (fraction of cells whose binary prediction
changed between consecutive steps) on three checkpoints:

| model | err_per_step trajectory | Δ_disc trajectory | usable convergence signal? |
|---|---|---|---|
| smoke (300u, h=64) | 0.31 → 0.20 monotone↓ | 0.05 → 0.005 monotone↓ | yes |
| kl=0.01 (3000u, h=128) | 0.19 → 0.13 (k=4 min) → 0.25 | non-monotone, min at k=4 | partially — model *diverges* after k=4 |
| multigame sokoban_basic | 0.002 → 0.024 monotone↑ | 0.002–0.007 small | no (best at k=1; later steps worse) |

Headline finding: **the learned-halt models drift after their
expected-halt step** — they only optimize the readout at steps where
`p_k` is large. For convergence-stopping to work cleanly, the body
needs to be uniformly good at every step, which is what `--halt_mode
uniform` is designed to provide.

#### E6c result (2026-05-03) — convergence-halt is a strict win on uniform-trained bodies

Three matched runs on Collapse-L0 (h=128, n_steps=8, n_repeats=8,
5000 updates), differing only in halt mode:

| mode | per-step err trajectory (k=1..8) | err @ fixed_T=8 | err @ conv-halt (ε=0.01) | mean halt k |
|---|---|---|---|---|
| `learned` (PonderNet) | 0.14, 0.07, 0.06, 0.06, 0.06, **0.06**, 0.06, 0.07 | 0.074 | 0.064 | 3.1 |
| `uniform` (mean-over-k) | 0.13, 0.06, 0.06, **0.06**, 0.06, 0.09, 0.12, 0.15 | **0.149** | **0.063** | 3.0 |
| `none` (final-step only) | (no per-step output) | 0.017 | n/a | 8 |

(Eval batch = 256 transitions from `Collapse/level_0` cache, mixed
across the 4 directional actions.)

**Three findings:**

1. **Convergence stopping recovers near-oracle performance on the
   uniform model.** `uniform` mode's body trains to be good at k=2-5
   (err ≈ 0.063) but **drifts catastrophically past k=5** (err climbs to
   0.149 by k=8). Naively reading out at the trained `n_steps=8` is
   2.4× worse than reading out at the oracle-best step. Convergence
   halting at ε ∈ {0.005, 0.01, 0.05} picks step 3 automatically with
   err = 0.063 — **matches the oracle minimum (err 0.062 at k=4) without
   any learned halt mechanism**.

2. **The drift is inherent to splitting gradient across many readouts.**
   `learned` mode (which weights early steps more heavily under the
   ponder loss) drifts much less (k=1..8 err: 0.14, 0.07, 0.06, ..., 0.07).
   `uniform` mode (which weights all steps equally) sees the drift
   strongly because it's actively asking the body to be good at all 8
   steps simultaneously — and the residual stack can't, so later steps
   become noisy. The net effect: convergence stopping isn't useful for
   `learned` (no drift to fix) but is essential for `uniform`.

3. **`none` mode wins on absolute error but loses on adaptability.**
   The single-readout baseline reaches err=0.017 — 4× lower than either
   adaptive mode at their best step. This isn't surprising: with only
   one readout, all 8 NCA steps' gradients flow through it; with 8
   readouts, gradient is divided. The tradeoff: `none` mode gives no
   per-instance compute control, can't halt early on easy inputs, and
   has no inference-time mechanism to drop computation. For tasks where
   adaptive computation is the goal, `uniform` + convergence-halt is
   the cleanest mechanism we have; for tasks where it isn't, single
   readout is strictly better.

**Implication for adaptive halt overall**: convergence stopping
sidesteps the halt-collapse failure mode entirely (no learned head, no
KL prior to tune, no F1 ↔ Q2 interaction). The remaining work is
closing the absolute-err gap between uniform-mode and single-readout
mode — which is an *optimization* problem (how to train a body to be
uniformly good at all depths without sacrificing peak accuracy), not a
halting problem. Likely interventions: stochastic unroll length,
geometric weighting on per-step loss, longer training.

Saved figure: `nca_wm/logs_halt_arch/convergence_halt_figure.{pdf,png}`.

#### Four halt-mode comparison on Collapse-L0 (2026-05-03, **corrected**)

**IMPORTANT correction**: an earlier version of this section had eval
numbers that were systematically wrong because the analysis script
read bit-packed cache bytes as raw state (cache stores
`np.packbits(states, axis=-1)` for ~8× compression). The corrected
numbers below are dramatically lower across the board, and reverse
the headline conclusion about `none` mode:

| mode | best per-step err | err @ fixed_T=8 | err @ conv-halt (ε=0.01) | conv-halt picks k≈ |
|---|---|---|---|---|
| `ponder`         | **0.005** (k=8) | 0.005 | 0.005 | 2.2 |
| `uniform`        | 0.005 (k=2)     | 0.045 | 0.005 | 2.0 |
| `argmax_st`      | 0.005 (k=2)     | 0.075 | 0.005 | 2.5 |
| `convergence_st` | 0.005 (k=2)     | 0.050 | 0.005 | 2.4 |
| `none`           | (n/a) — single readout | **0.013** | n/a | 8 |

(per-step err = whole-cell error; full-dataset change_err = error
restricted to cells that change, on right-action transitions only.
Eval batch = 256 transitions.)

**Findings (corrected):**

1. **All four adaptive modes reach essentially perfect peak err
   (~0.005).** The body learns the dynamics in ~1-2 NCA steps; any
   step beyond that is "extra" computation.
2. **Drift severity ranks: `argmax_st` > `convergence_st` > `uniform` > `ponder`.**
   `ponder` mode is essentially flat (no drift at any depth) because
   its per-step loss weighting reflects the KL-prior'd halt
   distribution — since halt converges to small k, only later-step
   weights are tiny and the body stays near its fixed point.
   `uniform`/`argmax_st`/`convergence_st` have weaker pressure on
   later steps' usefulness, so they drift past their effective depth.
3. **Convergence-stopping fully recovers the drift.** All three
   drifting modes hit the ~0.005 floor under conv-halt at ε=0.01.
   The mechanism does exactly what it was designed to do.
4. **`none` mode is actually 2-3× WORSE** than the best adaptive
   mode (0.013 vs 0.005). The earlier inversion of this conclusion
   was an analysis artifact (raw bytes read instead of unpacked
   states). The story is now: **adaptive computation is a strict win
   on this task** — it gives both lower err *and* the option to halt
   early at inference.
5. **The dynamics ARE learned in ~1 NCA step** — see varislide check
   below. Cells encode (state, action) at embed time; the slot
   encoder + 1 conv pass + cross-attention seems to be enough to
   memorize the per-state next-state mapping for the small reachable
   state spaces of these games. This explains why halt prefers k=1-2
   so aggressively across all modes.

Saved figure: `nca_wm/logs_halt_arch/halt_modes_comparison_figure.{pdf,png}`.

#### Varislide check: dynamics ARE learned (corrected 2026-05-03)

Per-level best-step change_err (action=3, right) on the full cached
right-action set, with proper bit-unpacking:

| mode | L0 (d=1) | L3 (d=4) | L5 (d=8) | L7 (d=16) | best_k typical |
|---|---|---|---|---|---|
| **pool-on ponder**       | 0.00 | 0.00 | 0.00 | 0.00 | 1 |
| **pool-off ponder v2**   | 0.07 | 0.00 | 0.00 | 0.01 | 1-2 |
| **pool-off argmax_st**   | 0.07 | 0.00 | 0.00 | 0.00 | 1 |
| **pool-off convergence_st** | 0.03 | 0.03 | 0.02 | 0.01 | 2 |

**The model perfectly fits varislide in 1 NCA step — even without
pool features.** This is genuinely surprising given the no-pool 3×3
receptive field shouldn't be enough to predict a 16-cell slide.

**Most likely explanation**: each level has only ~30-60 unique
BFS-reachable states. Each state's encoded `(state, action)` tensor
is quite distinct (different player positions, different wall
layouts in the level-conditioning slots). The shared-weight body —
via cross-attention to the slot-encoder output and a single 3×3
conv that sees the cell's local neighborhood — effectively
memorizes a per-state lookup table. Adaptive depth is unnecessary;
the model isn't doing iterative computation, it's pattern-matching.

This dissolves the F1 ↔ Q2 framing for varislide: the model
*can* learn the dynamics in 1 step regardless of pool. The earlier
"halt collapse" findings were correct in describing the halt
distribution (peaks at k=1-3) but wrong in interpreting them as
shortcut-pressure failure — the dynamics genuinely complete in 1
step on this task. Halt converging to k=1 is the *right* answer.

**The relevant question for adaptive halting becomes**: do we have a
task where the model genuinely *can't* fit the dynamics in 1 step?
Likely answer: yes for true Q1-positive games (flood-fill, beam
tracing, multi-character coordination), no for varislide and
Collapse despite their `again`-driven design. The 1-rule games like
varislide collapse to "memorize one tick's outcome per starting
state" with enough training and modest model capacity.

**Multi-size synthetic-data correction (2026-05-03, E15-E17)**:
The "memorization-via-cross-attention-to-slots" hypothesis was tested
by training varislide on 64 random levels at each of widths
{6, 8, 10, 12, 16}. Observations:

1. **Memorization broke as predicted**: train change_err climbed
   from single-level 5e-5 → multi-grid 0.05-0.10 (1000×), confirming
   that with diverse levels the model can no longer 1-shot-memorize
   per-state outcomes.
2. **The model failed to learn the rule instead**: eval change_err on
   right-action transitions stays at 0.44-0.50 across all widths
   (in-distribution AND OOD). The model's predictions become
   low-confidence (many cells with no channel above 0.5).
3. **More compute doesn't help**: 50k updates at h=256 (5× more
   updates, 2× wider) plateaus at the same 0.05-0.08 train change_err.
   Not an under-training problem — an architectural ceiling.

The implication is sobering: the rule_attn architecture seems to have
a "memorize-or-fail" character on this task, with no in-between regime
that learns the underlying slide rule. With the slot encoder being
level-independent (slots only depend on the game's rule tokens, not on
the level's spatial state), the body has to do all level-conditional
work via the conv + pool features alone — and it apparently can't,
even with rich pool features available. This is a stronger version of
F1 ("pool is load-bearing"): pool can substitute for depth on a
single level (where memorization works), but cannot substitute for the
rule-learning capacity needed across diverse levels.

Saved figures:
- `halt_modes_comparison_figure.{pdf,png}` (Collapse 4-mode comparison, corrected)
- varislide per-level figures show the *unused* learned halt
  distribution; the actual model behavior (per-level best step) is
  reported in the table above.

### Q3. Cross-game generalization of shared weights

The shared-weights win on Collapse is a single-game / single-level result.
Open: does shared also dominate per-step on the 19-game `scaling_large`
preset, or does multi-game training need the per-step capacity? The naive
prediction (from F2) is that shared wins by even more in the multi-game
regime, since the per-step body would be storing 19 different rule sets in
each step's weights, but this hasn't been measured.

Right experiment: rerun `scaling_large` at h=256, K=16 with
`--n_nca_repeats=4` (all-shared at n_nca_steps=4), compare to the
per-step baseline (`--n_nca_repeats=1`) at the same compute budget. See
E3 in the experiments table.

## Experiments planned / in progress

Numbered for reference; status updates land here as runs complete.

| ID | Experiment | Tests | Status |
|---|---|---|---|
| E1 | Bouncers L0 shared depth sweep (n∈{2,4,8,16}, full pool) | Q1 — does NCA depth help when looping is non-axis-aligned and pool can't substitute? | running (bg `bzxjlfefb`); script `run_bouncers_shared_depth_sweep.sh` |
| E2 | Collapse no-pool shared sweep (n=4, n=16) | Q1; F1 vs depth interaction — does shared bias unlock depth-without-pool? | running (bg `bzxjlfefb`); script `run_collapse_nopool_shared_sweep.sh` |
| E3 | Multi-game shared-weights validation on `scaling_large` | Q3 — does the shared inductive bias hold under multi-game training? | not started; script `run_scaling_large_shared_validation.sh` ready |
| E4 | Param-matched shared-vs-per-step (shared n=16 vs per-step n=2 at ~equal params) | F2 — is the win param efficiency or the inductive bias? | not started |
| E5 | Shared-weights, no-LN attn variant | Decoupling whether the per-step `attn_ln_{i}` LNs were doing meaningful per-step work | not started |
| E6 | Adaptive-halt sweep — halt_kl_weight ∈ {0, 1e-3, 1e-2, 1e-1}, halt_prior_p ∈ {0.05, 0.1, 0.2} on the Q1-positive game | Q2; whether halting actually concentrates probability on a sensible step distribution and whether L_rec beats the equivalent fixed-depth model | implementation landed 2026-05-03 (`--adaptive_halt`); awaits Q1-positive game pick. Smoke-test passed on Collapse-L0 (300 updates). |
| E6b | Halt-aware inference | Q2 v1 limitation — measure what halting buys at inference if you actually use the halt distribution at predict time | not started; needs apply_fn extension |
| E6c | Heuristic stopping baseline (`||y_k − y_{k−1}|| < ε`) | Q2; free baseline against learned halting | **done — strict win on uniform-trained bodies** (matches oracle min at ~step 3 vs 2.4× worse at fixed_T=8). See E6c result section above. |
| E6d | Close uniform vs single-readout absolute-err gap | Optimization side: uniform mode err is 4× worse than single readout. Try stochastic unroll length, geometric per-step weighting, longer training. | not started |
| E6e | Argmax-ST halt mode — addresses the "every aggregator rewards shortcuts" observation by computing loss at one selected step per example | Q2 — does removing per-step shortcut pressure let the body learn depth-required dynamics in the no-pool regime? | **done — partial**: same halt collapse as ponder on varislide-nopool (E[k] = 1.07-1.25); on Collapse, drift signature differs (sharp k=4 jump) but body has same peak err as other modes. |
| E6f | Convergence-ST halt mode — symmetric to E6e but step selection comes from body's own convergence pattern (no learned head) | Q2 — does the body learn cleaner depth-required dynamics when the halt rule is mechanically the same at training and inference? | **done**: doesn't fix the no-pool dynamics-learning problem (still F1) but is the only mode that doesn't collapse halt to k=1 (mean k=4-9 on varislide). On Collapse, drifts most severely past k=2 → benefits most from convergence-stopping at inference. |
| E7 | Cross-game depth seed-variance | F2 / F3 — is the rollout-error noise we see across n one-seed noise or systematic? | not started |
| E8 | Locate flood-fill parliament/pathfinding game and re-run E1 there | Q1 with the strongest possible substrate (flood-fill ≡ depth-bound propagation) | not started — title not yet located in gallery |
| E9 (sanity) | Microban L0 shared depth sweep | Out-of-scope for Q1 (axis-aligned chains), kept as a "shared weights port off Collapse" sanity check | L0 done; folds into F2 evidence, not Q1. L0 was too easy (shared n=2 → 0 wrong tiles); all-levels variant ready as `run_microban_alllevels_shared_sweep.sh` if needed |
| E10 | Hierarchical n_repeats sweep (e.g. n=16 with repeats∈{1,2,4,8,16}) | F2 — what's the right rule-layer granularity? Pure inner-block-only and pure outer-loop-only are extremes; intermediate could win | not started; needs the new `--n_nca_repeats` factorization (added 2026-05-03) |
| E11 | Multi-game adaptive-halt sweep across a complexity gradient (`sokoban_basic` → `sokoban_match3` → `Atlas_Shrank`) + paper-ready figure | Q2 — does E[halt step] track game complexity? Same shared body, same KL/prior, only the game spec varies. | running (bg `bkykzibu0`); scripts `run_multigame_halt_sweep.sh` + `plot_halt_distributions.py` |
| E12 | Within-game per-instance halt: custom `varislide` game (1-rule slide-until-wall) trained jointly on levels with player at distances ∈ {1,2,3,4,6,8,12,16}; per-level halt-distribution plot. | Q2 — does the *same* model adapt computation per input instance? Cleaner than E11 because the encoder output is constant across instances; only the spatial state varies. | **done with pool-on, negative result** — E[k] ≈ 4.75 flat across all distances; pool features substitute for depth. See F1+Q2 cross-ref above. |
| E13 | varislide with all pool features OFF | Force depth to be binding. Hypothesis: E[k] tracks slide_distance. | **done — negative**: model collapses to halt-at-k=1 + identity-prediction (F1 ↔ Q2 interaction). See section above. |
| E14 | Adaptive halt on a Q1-positive substrate | Show per-instance computation in the regime where pool can't substitute even when on (flood-fill, non-axis propagation). | not started — needs Q1-positive game first |
| E15 | Varislide × multi-size synthetic levels (widths 6,8,10,12,16) — pool-on vs pool-off | F1 / Q2 — does data diversity break the per-state memorization? | **done — partial**: data diversity DID break memorization (train change_err climbed from 5e-5 single-level to 0.05-0.10 multi-grid pool, 0.12-0.17 multi-grid no-pool; eval change_err 0.30-0.50 on right-action transitions for both). Model attempts to learn the rule but is under-trained at 10k updates / h=128. Visual inspection of predictions shows low-confidence outputs (many cells with no channel > 0.5). |
| E16 | OOD-width varislide eval (widths 20, 24, never seen during training) at trained depth + at extended n_repeats | Q1+Q2 — does the model generalize the slide rule to wider levels? Does extending n_repeats at inference help on harder OOD examples? | **done — preliminary**: change_err ~0.40-0.50 at OOD widths, but ALSO ~0.30-0.40 at IN-distribution widths (with held-out seed). Increasing n_repeats {16, 32, 48} doesn't help (in fact slight degradation). Conclusion is moot until in-distribution training succeeds; needs E17. |
| E17 | Varislide synth multi-grid with longer training (50k updates, h=256) | Determine if data diversity + sufficient compute lets the model actually learn the slide rule. | **done — negative**: train change_err plateaued at 0.05-0.08 (basically same as 10k/h=128 run). Eval right-action change_err still 0.44-0.50 at all widths. **5× compute + 2× width did not help**. The architecture appears to have a real ceiling on this task once memorization is unavailable. |
| E18 | Architectural workarounds for the multi-size synth ceiling | Try: (a) per-cell halting, (b) input_skip + LN at depth, (c) higher change_loss_weight, (d) larger n_slots, (e) asynchronous/stochastic NCA updates (Mordvintsev's NCA training tricks) | not started — exploratory |
| E19 | Depth sweep on varislide synth multi-grid (n_steps ∈ {8, 16, 20, 24, 32}, all fully shared, h=256, 50k updates) | Q1 — does depth genuinely help on the diverse-data regime where memorization is unavailable? | **done — non-monotone**: n=8/16/24/32 all sit at eval change_err ≈ 0.45 (basically the architectural ceiling); only **n=20 is a clear outlier at 0.29**, including OOD widths {20, 24}. Train metric is identical across depths (vacuous-batch averaging masks the difference). The non-monotone shape argues this is **single-seed optimization variance** more than a true depth-effect signal. Needs multi-seed verification. |
| E20 | Multi-seed verification of depth sweep on multi-grid varislide | Disambiguate "n=20 is lucky" vs "depth genuinely helps". Shared weights, n_hid=128, depth ∈ {8, 16, 32, 64} × 3 seeds, 10k steps. | **done — n=20 was noise**: change_err 0.42–0.44 mean across all depths (within each other's std); argmax_acc 0.17–0.30. Per-seed argmax variance is huge (~5× within a single depth), best_loss is identical across all 12 runs. See F8. Script: `run_varislide_depth_seedsweep.sh`. |
| E21 | Long-training depth=16 multi-seed (50k steps × 3 seeds) | Optimization-vs-architecture disambiguation: does longer training find the iteration basin? | **done — negative**: 5× compute does not move the needle (10k argmax 0.231±0.105; 50k argmax 0.194±0.114). Per-seed ordering is preserved (best seed at 10k is still best at 50k), confirming the basin is init-determined and stable under more optimization. Script: `run_varislide_long_seedsweep.sh`. |
| E22 | Per-step (un-shared) weights at depth=16 × 3 seeds | F2 cross-check: does removing the sharing constraint find a different basin? Per-step has ~11× more params (5.1M vs 447K). | **done — wash**: per-step change_err 0.418±0.075 vs shared 0.443±0.055 — within noise. Hint that per-step gets d=2,3 right more often (~0.4 vs ~0.2 shared) at the cost of d=1, but overall metrics indistinguishable. Sharing wasn't the active variable. Script: `run_varislide_perstep_seedsweep.sh`. |
| E23 | Multi-grid microban cross-check (single-grid w=12 vs multi-grid widths 8-16) | Does the multi-grid synth failure recur on a sokoban-class chain-push game with depth-bound dynamics? Tests whether the canary generalizes beyond varislide. | not started; script `run_microban_multigrid.sh` ready |

Sweep scripts live in `nca_wm/scripts/`; templates:
`run_collapse_shared_weights_sweep.sh` (the F2 sweep),
`run_bouncers_shared_depth_sweep.sh` (E1),
`run_collapse_nopool_shared_sweep.sh` (E2),
`run_scaling_large_shared_validation.sh` (E3),
`run_microban_shared_depth_sweep.sh` and
`run_microban_alllevels_shared_sweep.sh` (E9 sanity).
All shared-mode scripts use `--n_nca_repeats $n_steps` (the
fully-shared special case of the new factorization).

## Operational guidance

### Default recipe for new experiments

```
--architecture rule_attn \
--n_hid 256 --n_nca_steps 4 --n_nca_repeats 4 --n_slots 16 \
--axis_pool --axis_cummax --global_pool \
--input_skip \
--change_loss_weight 5.0 --grad_clip 0.5 \
--balanced_sampling --kernel_sep \
--token_decoder_loss_weight 0.1 \
--patience 200 --min_delta 1e-6
```

`--n_nca_repeats=n_nca_steps` (the all-shared special case) is the new
default per F2. It cuts NCA-body params ~`n_steps`× without hurting
train loss or oracle-action rollout, and is a prerequisite for adaptive
halting (Q2). Set `--n_nca_repeats=1` to fall back to the per-step body
(needed for back-compat with pre-2026-05-02 checkpoints; intermediate
values give a hierarchy not yet swept — E10).

### When to deviate

- **Looping-dynamics game / `again`-heavy**: try `--n_nca_steps {8, 16}`
  with shared weights and `--input_skip` on. Compare to `n_steps=4` with
  cummax-off for the pooling-as-substitute control. Do **not** add
  `--use_layernorm` — input_skip alone is the active component (F4).
- **Single-game runs on small games**: `--no-balanced_sampling` is fine.
- **Synthetic data**: see the synth section in `RUNNING_REPORT`. The
  per-game-size + `--synthetic_no_a_count_max 5` recipe is current.
- **Reproducing pre-2026-05-02 numbers**: set `--n_nca_repeats=1`
  (the default). You'll pay the per-step weight cost but the un-shared
  default is what those numbers were computed against (F7 caveat
  applies for pre-2026-04-18). Note: 2026-05-02 runs that used the
  short-lived `--shared_weights` boolean correspond to
  `--n_nca_repeats=n_nca_steps` under the new factorization, but layer
  names changed (`conv` → `conv_0`) so old shared-mode checkpoints will
  not load.

### Reporting expectations

Per F3, report **both** `best_loss` and at least the 50-step autoregressive
eval (random + bfs + astar). Reporting only one hides systematic
disagreements between the two metrics.

### What this report is for

Settled architectural findings live here. Active experiments and
chronological run notes live in `RUNNING_REPORT.md`. When a finding becomes
load-bearing (other experiments now assume it), promote it from RUNNING to
this doc and trim the chronology.

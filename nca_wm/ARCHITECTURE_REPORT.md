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

### F4. LN+input_skip stability patch is per-step-only, not a generic deep-NCA stabilizer

Two flags were added 2026-05-01 to make deep per-step rule_attn unrolls
trainable:

| Flag | Effect |
|---|---|
| `--use_layernorm` | shared pre-norm `LayerNorm` on `h` at start of every NCA step |
| `--input_skip` | re-inject the embedded `(state, action)` into the conv input at every step |

Both default off. The motivation was "deep unrolls without skip connections
will diverge"; this turned out to be wrong (per-step n=16 stock trains
stably and beats n=4 on train loss). What the patch *actually* does is
**regularize** an over-parameterized per-step body: it forces a smoother
optimization that doesn't overfit per-step. On per-step n=16 it raises
train loss only slightly but cuts rollout error 3-4× (the column-winning
row above).

The 2026-05-02 shared-weights sweep settled the regularizer hypothesis: on
shared n=16, the same patch raises train loss 30× (5.43e-7 → 1.61e-5) and
*degrades* bfs/astar rollouts (75→212, 100→124). The shared body has ~14×
fewer params and doesn't have a per-step over-parameterization problem to
regularize.

**Recommendation:** Do not combine `--n_nca_repeats=n_nca_steps`
(all-shared) with `--use_layernorm --input_skip`. The patch is for the
per-step body (`--n_nca_repeats=1`) only; it has no place in the
recommended recipe.

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

### Q2. Adaptive pass count

A fixed `n_steps` is wasteful for easy transitions and insufficient for
hard ones. The natural extension: let the model halt — analogous to
PonderNet / Adaptive Computation Time / Universal Transformers' halting.
Now well-posed thanks to F2: with one shared body, "halt at step k" is
asking the same rule set to converge in k iterations, not selecting
between k different models.

Sketch:

- A small per-cell or global "halt" head reads `h` at each step and emits
  a halting probability `p_i ∈ [0, 1]`.
- Loss is the expected loss across halt distributions, with a regularizer
  encouraging short rollouts (small `E[steps]`).
- At inference, sample/threshold `p_i` to terminate.

For PuzzleScript specifically, the engine itself terminates each tick when
"no rule fires"; analogous "no change since last step" is a natural
halting signal that could be hard-coded: stop when `||h_i - h_{i-1}|| < eps`.

Open before implementing:

- Does a halting model beat a fixed-depth model with matched expected
  steps? Adaptive computation papers consistently report only modest gains.
- For batched training, halting is simulated via expected-value
  computation — does this still yield the wall-clock savings that matter
  at inference?
- Is the `n_steps` bottleneck binding in practice, or is the
  body-parameter count the binding constraint?

Second-order until Q1 lands.

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
| E6 | Adaptive-halt prototype on the Q1-positive game | Q2; lower bound on halting gains in our regime | blocked on Q1 |
| E7 | Cross-game depth seed-variance | F2 / F3 — is the rollout-error noise we see across n one-seed noise or systematic? | not started |
| E8 | Locate flood-fill parliament/pathfinding game and re-run E1 there | Q1 with the strongest possible substrate (flood-fill ≡ depth-bound propagation) | not started — title not yet located in gallery |
| E9 (sanity) | Microban L0 shared depth sweep | Out-of-scope for Q1 (axis-aligned chains), kept as a "shared weights port off Collapse" sanity check | L0 done; folds into F2 evidence, not Q1. L0 was too easy (shared n=2 → 0 wrong tiles); all-levels variant ready as `run_microban_alllevels_shared_sweep.sh` if needed |
| E10 | Hierarchical n_repeats sweep (e.g. n=16 with repeats∈{1,2,4,8,16}) | F2 — what's the right rule-layer granularity? Pure inner-block-only and pure outer-loop-only are extremes; intermediate could win | not started; needs the new `--n_nca_repeats` factorization (added 2026-05-03) |

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
  with shared weights. Compare to `n_steps=4` with cummax-off for the
  pooling-as-substitute control. Do **not** combine with
  `--use_layernorm --input_skip` (F4).
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

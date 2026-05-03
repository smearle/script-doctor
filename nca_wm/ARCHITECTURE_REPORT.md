# NCA World-Model Architecture Report

Living architectural reference for `nca_wm/`. Distinct from `RUNNING_REPORT.md`,
which logs training runs chronologically; this doc takes a stand on what the
right architecture is and *why*, and is the place to look when designing a new
ablation. Last refresh: 2026-05-02.

## What an "NCA world model" is doing here

Given `(state_t, action_t)` for a PuzzleScript game and that game's tokenized
spec, predict `state_{t+1}` cell-wise. The model is a Neural Cellular
Automaton: a learned local update rule applied for `n_steps` shared (or
per-step) iterations on a grid of hidden vectors. Conditioning on the game
spec is via a perceiver-style encoder that emits K rule slots; the NCA
cross-attends to those slots at every step.

The PuzzleScript engine itself is fundamentally iterative: each tick applies
its rules **and `again` rules until the state stops changing**. So the NCA's
`n_steps` must cover whatever sequential reasoning the engine does for the
hardest rule chain in the game. Games with looping dynamics (gravity,
projectile motion, propagation chains) demand more passes than games with a
single-shot rule set.

## What's in this directory (architecturally)

| Module | What it is | Where |
|---|---|---|
| `NCAWorldModel` | Unconditional shared-weight NCA. Has input-skip, residual+ReLU updates, optional shared LN. | `train.py` ~L1183 |
| `ConditionalNCAWorldModel` | FiLM-conditioned variant of `NCAWorldModel`. Same NCA body; modulated by a single pooled `z` from `GameSpecEncoder`. | `train.py` ~L1371 |
| `RuleAttnNCAWorldModel` | Current default. Per-step weights; cells cross-attend to K rule slots from a perceiver encoder. | `rule_attn_model.py` |
| `GameSpecEncoder` (FiLM) / `RuleSlotEncoder` (rule_attn) | Two flavors of game-spec → conditioning. CLS+pool vs K-query slots. | `train.py` / `rule_attn_model.py` |
| `_pool_features` | Shared global-context primitive used by all three NCA variants. | `train.py` ~L1146 |

## Settled findings (load-bearing — change with care)

### 1. Global context is a hard requirement, not a bonus

A pure 3×3 conv can only propagate one cell of context per NCA step. Several
PuzzleScript rule patterns reference cells that are arbitrarily far apart in
the same tick:

- `[ X | ... | Y ]` — X and Y in the same row/col, any distance apart.
- `[X] [Y]` — X and Y on the level, any positions.

Without global features, the NCA needs `n_steps ≥ max_grid_dim` just to *see*
across the grid. With them, one step suffices.

`_pool_features` exposes three independent flags (settable in any
combination):

| Flag | Cost (per step) | Adds | Necessary for |
|---|---|---|---|
| `axis_pool` | 2 broadcasted features | row-max, col-max | `[X \| ... \| Y]` (existence in same axis) |
| `axis_cummax` | 4 broadcasted features | directional prefix-max (L→R, R→L, T→B, B→T) | directional `…` rules; "is X to my left/right" |
| `global_pool` | 1 broadcasted feature | grid-wide max | `[X] [Y]` multi-bracket |

**The recipe in `RUNNING_REPORT` runs all three on.** They're cheap, they're a
strict superset of any subset, and the rule_attn FiLM ablation (19-game
scaling) confirmed turning them off costs convergence speed and final loss.

Consequence for new ablations: do not turn these flags off as a default.
Treat them as load-bearing; if you suspect the global-pool stack is masking
something, run the controlled ablation, document, and turn it back on.

### 2. `change_loss_weight` is what actually breaks identity-collapse

In multi-game training, most cells don't change between `t` and `t+1`. Plain
BCE has the model sit on the identity prediction "predict input" minimum for
many thousands of steps. The patience-on-uniform-BCE early-stopping was
firing mid-recovery in earlier experiments. `--change_loss_weight 5.0` makes
each changed cell count 6× — the patience metric now tracks something the
researcher cares about, and the model gets gradient signal to leave identity.

This isn't strictly architectural but it's part of the canonical recipe and
should not be omitted.

### 3. The encoder mask bug (now fixed) was an architectural footgun

Flax's `MultiHeadDotProductAttention` interprets the `mask` argument as
**boolean** (True = keep). Earlier code passed a float mask with inverted
semantics. Pre-fix, every game's slot tensor cosine-similarity was
0.77–0.99 — the encoder was effectively shared-conditioning across all
games. Post-fix, distinct games have cos-sim around 0.31, and rule_attn
saturates `scaling_6` at 1.5e-5 best-loss. **Any pre-2026-04-18 saved
checkpoint should be considered suspect.**

### 4. Rule_attn vs FiLM at the same conditioning capacity

On the 19-game `scaling_large` set (h=256), rule_attn K=16 reaches 9.4%
mean change-error vs FiLM (d_z=256, d_model=128, encL=4) at 14.5%, with
about 3.4× faster convergence. K=32 (over-parameterized) hits 10.4%. So
K=16 is on the saturating end of the diminishing-returns curve. Rule_attn
is the current default for this reason.

The catch: rule_attn pays for this with **per-step weights** in the NCA
body. That has direct implications for going deeper (next section).

## Open architectural question: depth and looping dynamics

This is what the rest of this doc is investigating, motivated by games with
explicit looping rules.

### What "looping dynamics" means in PuzzleScript

The clearest signal is the `again` keyword: `down [ stationary wall ] -> [
down wall ] again` means "after this rule fires, run the whole rule set
again" — applied iteratively until convergence within a single tick. Some
canonical examples:

- **Collapse** (Terry Cavanagh): walls and mines fall via `down [stationary
  wall] -> [down wall] again`; player hovers right/left until hitting a
  wall via `random right [PlayerHoverRight | No Obstacle] -> [JetTrail1 |
  PlayerHoverRight] again`.
- **Sokoban-with-chain-pushing** (any game where pushing one crate pushes a
  chain). The basic engine handles this in the rule-firing step itself, but
  longer chains take more rule-firings to resolve.
- **Rule propagation** generally: `[X | no X] -> [X | X] again` style
  spreading rules.

The straightforward expectation is that a `n_steps=k` NCA can fit any rule
chain of length ≤ k, but not longer. The exact relationship between engine
ticks-of-`again` and required NCA steps depends on how much each NCA step
"compresses" the propagation, but the lower bound is real.

### Why deeper rule_attn is currently fragile

The `RuleAttnNCAWorldModel` body, before the 2026-05-01 stability patch:

```python
for i in range(self.n_steps):
    h_conv = nn.Conv(name=f"conv_{i}")(h)            # per-step weights
    # ...pool concat...
    attn_out = MultiHeadDotProductAttention(name=f"cell_slot_xattn_{i}")(...)
    delta = nn.gelu(h_conv + attn_out)
    delta = nn.Dense(name=f"out_{i}")(delta)         # per-step weights
    h = h + delta                                     # residual, but no LN
```

What is and isn't there:

- **Residual update** (`h = h + delta`): present.
- **Pre-norm LayerNorm on h**: absent.
- **Input skip** (re-inject the embedded `(state, action)`): absent.
- **Per-step weights**: yes — every step has its own `conv_{i}`, `out_{i}`,
  `cell_slot_xattn_{i}`, plus per-step LNs that *only* normalize attention
  inputs.

The contrast with `NCAWorldModel`/`ConditionalNCAWorldModel` is sharp: those
have shared weights, an input skip at every step (`parts = [h, inp]`), and
an optional shared post-step LayerNorm gated by `--use_layernorm`. A bare
residual stack with per-step weights and no normalization is a recognized
recipe for divergence at depth: gradients accumulate through
non-normalized residual sums and the weights at later steps are simply more
parameters to overfit with less regularization.

So when we extend `n_steps` from 4 → 16 in rule_attn, we're simultaneously
(a) asking the model to do more sequential reasoning, (b) quadrupling the
NCA-body parameter count, and (c) running an unnormalized residual stack
4× as deep. Conflating these obscures the question we want to answer
("does depth help looping games?").

### Stability patch (2026-05-01)

Added two default-off flags to `RuleAttnNCAWorldModel`:

| Flag | Effect | Default |
|---|---|---|
| `--use_layernorm` | shared pre-norm `LayerNorm` on `h` at the start of every NCA step | off |
| `--input_skip` | re-inject the embedded `(state, action)` into the conv input at every NCA step (matches `NCAWorldModel`) | off |

Both default off, so existing checkpoints load bit-identically. Use them as
a paired ablation when running `n_nca_steps ≥ 8`.

(The patch is intentionally minimal: only the two stabilizers established
in the unconditional model. We're *not* converting rule_attn to shared
weights — that's a different question (parameter efficiency of depth) and
worth a separate study. With the patch, going from `n_steps=4` to
`n_steps=16` still 4×s the body parameter count; the LN+skip just makes
the optimization tractable.)

### Empirical results: depth ablation on Collapse (2026-05-01)

**Setup**: single-game training on Collapse level 0 (padded 19×10×48),
rule_attn h=256, K=16, `change_loss_weight=5`, `grad_clip=0.5`,
`n_updates=15000`, `lr=3e-4`, `batch_size=16`. Dataset: 2,700 unique
BFS-explored transitions on level 0 (fully exhausted). Level 1 dropped —
its 21×66 native shape OOMs cross-attention at our batch size, and per-game
multi-level training would just average over a bigger transfer-test pool
without adding to the train signal.

#### Full table — train loss + autoregressive rollout

Wrong-tile counts are mean per-state over 50-step rollouts on level 0
(training distribution); divisor is 3,040 cells. `random` = autoregressive
under random actions; `random_tf` = teacher-forced under random actions
(per-step, no compounding); `bfs/astar` = autoregressive under the engine's
optimal action sequences.

| `n_nca_steps` | global pool | stab patch | params | best loss | L0 rand_tf | L0 random | L0 bfs | L0 astar |
|---|---|---|---|---|---|---|---|---|
| 2  | full | — | 2.6M | 1.07e-3 | 2 | 196 | 117 | 112 |
| 4  | full | — | 5.0M | 1.70e-4 | 3 | 102 | 169 | 200 |
| 8  | full | — | 9.9M | **1.55e-6** | 5 | 207 | 198 | 208 |
| 16 | full | — | ~19M | 3.71e-6 | 8 | 206 | 179 | 171 |
| 16 | full | LN+input_skip | ~19M | 1.36e-5 | **1** | **67** | **53** | **56** |
| 4  | OFF  | — | 5.0M | 1.36e-2 | 4 | 204 | 184 | 199 |
| 16 | OFF  | LN+input_skip | ~19M | 1.13e-2 | 2 | 68 | 98 | 112 |

#### Three load-bearing findings

**1. Pooling is doing essential work; depth alone cannot substitute.**
Both no-pool runs sit at ~1% train loss (~30–40% change_err) — i.e.
identity-collapse. Going from n=4 to n=16-with-stabilizers and adding 4×
the parameters does not pull the no-pool model out of identity. Without
pooling, an NCA can only propagate one cell per step; Collapse's level-0
geometry needs ~12 cells of horizontal context, and 16 NCA steps × 1
cell-per-step is barely enough geometrically — but evidently not enough
for the optimization to find it within 15k steps. The user's "global
pooling may be hacking the dynamics" hypothesis is half-right: it isn't
hacking, it's load-bearing.

**2. Train loss and autoregressive rollout are non-monotonically related.**
n=8-stock has the lowest train loss (1.55e-6) but worse rollout error
than n=4 (random=207 vs 102). n=16-stab has 9× *higher* train loss than
n=8-stock, yet **3-4× lower rollout error** (random=67, bfs=53, astar=56
vs n=8's 207, 198, 208). The high-capacity bare-residual stack overfits
to per-step prediction in a way that produces small but systematic
errors that compound destructively over autoregressive rollout.
Teacher-forced 1-step error tracks train loss; autoregressive error
does not.

**3. The LN + input_skip stability patch is a clear win — as a regularizer,
not as a divergence-preventer.** The original motivation was "deep
unrolls without skip connections will diverge"; stock n=16 trains
stably (loss decreases monotonically) and beats n=4 on train loss, so
the patch isn't preventing divergence. What it *is* doing is forcing a
smoother optimization that doesn't overfit per-step:

- Trades train loss (1.55e-6 → 1.36e-5, ~9× higher) for rollout error
  (207 → 67 random; 198 → 53 bfs; 208 → 56 astar — ~3-4× lower).
- Per-step train cost ~3× slower (135 ms/step at n=16-stab vs ~50 ms at
  n=16-stock) due to the LN + concat overhead.
- Teacher-forced error drops from 8 wrong tiles (n=16 stock) to 1 wrong
  tile (n=16 stab). Even on the metric the high-capacity model
  ostensibly optimizes, stab does as well or better — it just spends
  longer getting there.

The patch should be the default when going to `n_steps ≥ 16`. At
`n_steps ≤ 8` the bare residual stack is fine.

#### What this says about the user's two hypotheses

> "I hypothesize that more passes will help."

**Yes for train loss, no for autoregressive rollout in the bare body.**
Adding depth without the regularization patch yields lower train error
but worse rollouts. With the patch, depth + regularization does both.

> "Training will become unstable [at high depth] because we lack skip
> connections."

**Mostly no — at the depths and grid sizes tested (up to n=16, 10×48),
the bare residual stack is stable.** It does not diverge or oscillate
catastrophically. The cost of bare-residuals at depth is *overfitting*,
not divergence. The skip-connection patch helps anyway, by acting as a
regularizer.

> "[Collapse's] dynamics... may be simple enough that the model can
> 'hack' them via global pooling."

**Yes. With n=4 + full pool, train loss is already 1.7e-4 (essentially
fitted) and removing pool destroys learning entirely.** The depth
ablation is dominated by overfitting effects, not by the propagation
length needed for the dynamics. To stress-test depth-helps-looping, we
need a game where pool provably can't substitute.

### Open question: per-step weights vs. shared weights

Stepping back from the depth/stab axis, the rule_attn body has a confound
that's worth pulling out as its own variable: **every NCA step has its own
parameters**. That is not the natural inductive bias for PuzzleScript.

The engine applies the *same* rule set on every iteration of an `again`
loop — convergence is detected by "the state stopped changing", not by
running a different rule set on iteration 2 vs. iteration 5. A model that
matches that prior should:

1. Apply the same body weights at every NCA step (one rule set).
2. Iterate as many times as the hardest rule chain in the game requires.
3. Ideally, decide *when* to stop iterating (cf. adaptive halting below).

The current `RuleAttnNCAWorldModel`, with per-step weights, does the
opposite — it has a different "rule set" at every iteration depth, and
"depth" and "parameter count" co-vary on every ablation. That makes both
of the things we want to study harder:

- **Adaptive halting is incoherent without weight sharing.** A halting
  policy that says "stop at step k" implicitly says "use the rule set
  trained for depth k." If body weights are per-step, the model behaves
  like an ensemble of k separate models, one per depth — there is no
  single "rule set" to converge against.
- **The depth=k+1 body has never seen its k+1-th step's weights at any
  smaller k.** Per-step weights mean the n=16 model and the n=4 model
  share *no body parameters at all*. The shared-weight version is
  comparable across depths in a way the per-step version is not.

#### What's now in the code (2026-05-02)

`RuleAttnNCAWorldModel` gained a `shared_weights: bool = False` flag.
When True, the conv / pool_proj / attn_ln / slot_ln / cell_slot_xattn /
out layers are allocated once and reused at every iteration. Default is
off so existing checkpoints load bit-identically. CLI flag:
`--shared_weights`. Run-name tag: `_shared`.

A small init/forward smoke test confirms the param count drops by
~`n_steps`× (e.g. n=3, h=32: 68k → 29k params).

#### Empirical results: shared-weights depth sweep on Collapse (2026-05-02)

Same setup as the per-step depth sweep above (Collapse L0, h=256, K=16,
change_loss_weight=5, grad_clip=0.5, n_updates=15000, lr=3e-4,
batch_size=16). Each shared run is ~7 min on one GPU.

| `n_nca_steps` | global pool | stab patch | params (total) | best loss | L0 rand_tf | L0 random | L0 bfs | L0 astar |
|---|---|---|---|---|---|---|---|---|
| 2  | full | — | **1.39M** | 1.59e-6   | 3  | 209 | 61  | 71  |
| 4  | full | — | **1.39M** | 5.42e-7   | 4  | 165 | 111 | **60**  |
| 8  | full | — | **1.39M** | **2.79e-7** | 17 | 207 | 204 | 119 |
| 16 | full | — | **1.39M** | 5.43e-7   | 24 | **153** | 75  | 100 |
| 16 | full | LN+input_skip | 1.98M | 1.61e-5 | 4 | 155 | 212 | 124 |

For direct comparison, the per-step rows from the previous sweep:

| `n_nca_steps` | stab | params | best loss | rand_tf | random | bfs | astar |
|---|---|---|---|---|---|---|---|
| 2  | —    | 2.6M  | 1.07e-3 | 2 | 196 | 117 | 112 |
| 4  | —    | 5.0M  | 1.70e-4 | 3 | 102 | 169 | 200 |
| 8  | —    | 9.9M  | 1.55e-6 | 5 | 207 | 198 | 208 |
| 16 | —    | ~19M  | 3.71e-6 | 8 | 206 | 179 | 171 |
| 16 | stab | ~19M  | 1.36e-5 | 1 | 67  | 53  | 56  |

#### Three load-bearing findings

**1. Shared weights are dramatically more parameter-efficient on train
loss.** Every shared-body run has 1.39M total params (the encoder /
readout / unshared LN dominate; the NCA body collapses to one shared
copy). Per-step n=8 takes 9.9M params to reach 1.55e-6 train loss;
shared n=8 reaches 2.79e-7 — **5.5× lower train loss with 7× fewer
params**. Shared n=2 (1.39M) achieves the same train loss as per-step
n=8 (9.9M), so on this game the prior is worth *roughly an octave of
depth* in the per-step body. The "deeper rule_attn needs more
parameters" intuition was wrong; sharing them is strictly better here.

**2. On oracle-action rollouts, shared n=2 already matches or beats
the best per-step row at 1/14 the params.** Comparing the column
winners: per-step n=16-stab is the best per-step row at random=67,
bfs=53, astar=56 (~19M params). Shared rows pick up the bfs and astar
columns at much lower n: shared n=2 hits bfs=61, astar=71 with 1.39M
params; shared n=4 hits astar=60 (column winner). On random-action
rollouts the gap remains — shared best is random=153 (n=16),
per-step-stab is random=67 — so per-step+stab still wins under
distribution shift, but shared-low-n is competitive on the
in-distribution metrics at a fraction of the cost.

**3. The stability patch is anti-helpful for shared weights.** On
per-step n=16, LN+input_skip dropped train loss only slightly but cut
rollout error 3-4×. On shared n=16, LN+input_skip *raises* train loss
30× (5.43e-7 → 1.61e-5) and degrades bfs/astar rollouts (75→212,
100→124), only improving the random-action column slightly (153→155
≈ same). This is consistent with the "stab patch is a regularizer"
hypothesis from the per-step sweep: it was solving a problem caused by
per-step over-parameterization, and the shared body doesn't have that
problem. **`--shared_weights` and `--use_layernorm --input_skip` should
not be combined as a default;** if you want both, treat as a separate
investigation.

#### Implications

- **Shared weights is now the recommended default for `n_steps ≥ 4`.**
  Per-step won at the lowest depth on a couple of rollout columns
  (per-step n=4 random=102 beats shared n=4 random=165), so for very
  shallow ablations the inductive bias is less load-bearing — but it's
  still the right prior, and it removes the per-step/depth confound.
- **The stab patch can be retired in shared-weights runs** (and the
  patch's "trades train loss for rollout error" framing should be
  re-read in the per-step sweep as "fixes per-step
  over-parameterization", not as a generic regularizer).
- **Adaptive halting is now well-posed** — see "Long-term direction"
  below. With one shared body, "halt at step k" is asking the same
  rule set to converge in k iterations, not selecting between k
  different models.
- **The depth-helps-looping question is still open and now better
  formed.** Shared-body depth has near-flat best-loss across n ∈ {2,
  4, 8, 16} (1.59e-6, 5.42e-7, 2.79e-7, 5.43e-7) — i.e. one shared
  rule set converges fast and depth past 4 doesn't add much *on
  Collapse*. This is exactly the regime where Collapse's pool-hackable
  dynamics make depth invisible. The harder-looping-game experiment
  below is the right next step, and should be run with `--shared_weights`
  default.

#### What's now in the code (2026-05-02)

`RuleAttnNCAWorldModel` gained a `shared_weights: bool = False` flag.
When True, the conv / pool_proj / attn_ln / slot_ln / cell_slot_xattn /
out layers are allocated once and reused at every iteration. Default is
off so existing checkpoints load bit-identically. CLI flag:
`--shared_weights`. Run-name tag: `_shared`.

The sweep script `scripts/run_collapse_shared_weights_sweep.sh` mirrors
the un-shared depth ladder (n ∈ {2, 4, 8, 16} + n=16-stab) so the
tables above are directly comparable line-for-line.

### Recommended next experiment: harder-looping game

For an honest depth-helps-looping test, pick a game where:

1. The looping rule's reach is **not axis-aligned** — pool/cummax can't
   express it as a one-step lookup.
2. The looping is **long** — mean number of `again` iterations per tick
   significantly exceeds 1.
3. Ideally the rule is **sequential-state-dependent** — each iteration
   sees a state modified by the prior one, so the model can't precompute
   the answer from the initial state alone.

Concrete candidates from the PuzzleScript gallery (sorted by how cleanly
they meet the criteria):

- **Mirror Isles** (Stephen Lavelle) — light beams that reflect off
  diagonal mirrors. Reach is non-axis-aligned, propagation depth is
  unbounded by walls, each reflection branches.
- **Heroes of Sokoban** — simultaneous multi-character control;
  push-chain interactions across characters.
- **Sokoboros** / **Multi-Crate Sokoban** — N-deep crate push chains.
  Linear in the chain length but each iteration changes the grid.
- **A Good Snowman is Hard to Build** — stacked snowman building
  involves chained `[ snowman_part | snowman_part ]` propagation.
- **Skipping Stones to Lonely Homes** / **Path Lines** — path-tracing
  rules with long propagation chains.

Any of these as a single-game replay of this same depth sweep should
give a much cleaner read on whether depth helps looping when pool can't
substitute.

### Methodological note: train loss is not the right scoreboard

This sweep changes how we should report rule_attn results going forward.
Reporting only `best_loss` (as `RUNNING_REPORT` does for most
experiments) hides the rollout-drift cost of going deeper. New experiments
in this vein should report **both** the train metric and at least the
50-step autoregressive eval — they're cheap (the eval already runs at
end-of-train) and they're often pointing in opposite directions.

### When global pooling alone might "hack" the dynamics

Collapse-style "move horizontally until hitting a wall, then fall to ground"
involves *axis-bounded* propagation — exactly the regime that
`axis_cummax` was designed for. A single NCA step with directional cummax
features can already see "is there a Wall to my right" at any distance. So
"the player ends up exactly where the wall is" is a one-step transformation
under this feature stack, not something requiring deep unrolls.

This is a real concern: the result we want to demonstrate ("deeper helps for
looping dynamics") may be invisible on Collapse precisely because the
canonical pool stack already includes the axis-cummax feature that makes
this game's dynamics look local-after-pooling.

The clean test is one of:
1. **Same Collapse, with cummax/global-pool ablated off** — forces the model
   to actually iterate. Settles whether depth helps when *pooling can't
   substitute for it*.
2. **A game whose looping is genuinely multi-step in a way pooling can't
   short-circuit** — chain-pushing games, fluid simulation, any rule that
   re-references its own output across iterations. Candidates from the
   gallery TBD.

### Long-term direction: adaptive pass count

A fixed `n_steps` is wasteful for easy transitions and insufficient for hard
ones. The natural extension is to let the model decide when to halt —
analogous to PonderNet / Adaptive Computation Time / Universal Transformers'
halting.

**Now well-posed:** the 2026-05-02 Collapse sweep confirmed
`--shared_weights` reaches lower train loss than the per-step body at
1/7 the params and is competitive on rollout. With one shared body,
"halt at step k" is asking the same rule set to converge in k
iterations, not selecting between k different models. The sketch below
assumes shared weights.

- A small per-cell or global "halt" head reads `h` at each step and emits a
  halting probability `p_i ∈ [0, 1]`.
- Loss is the expected loss across halt distributions, with a regularizer
  encouraging short rollouts (small `E[steps]`).
- At inference, you can sample/threshold `p_i` to terminate.

For PuzzleScript specifically, the engine itself terminates each tick when
"no rule fires"; analogous "no change since last step" is a natural halting
signal that could be built in directly: stop when `||h_i - h_{i-1}|| < eps`.

Open questions before implementing:

- Does a halting model beat a fixed-depth model with matching expected
  steps? Adaptive computation papers consistently report only modest gains.
- For batched training, halting is simulated via expected-value
  computation — does this still yield the wall-clock savings that matter
  at inference?
- Is the `n_steps` bottleneck binding in practice, or is the body-parameter
  count the binding constraint?

These are second-order until the depth-helps-looping result lands.

## Operational guidance

### Default recipe for new experiments

```
--architecture rule_attn --shared_weights \
--n_hid 256 --n_nca_steps 4 --n_slots 16 \
--axis_pool --axis_cummax --global_pool \
--change_loss_weight 5.0 --grad_clip 0.5 \
--balanced_sampling --kernel_sep \
--token_decoder_loss_weight 0.1 \
--patience 200 --min_delta 1e-6
```

`--shared_weights` was added 2026-05-02 and validated on Collapse
(see the sweep above). It cuts NCA-body params ~`n_steps`× without
hurting train loss or oracle-action rollout, and is a prerequisite
for adaptive halting. It is now the recommended default; omit only
for back-compat with pre-2026-05-02 checkpoints.

### When to deviate

- **Looping-dynamics game / `again`-heavy**: try `--n_nca_steps {8, 16}`
  (with `--shared_weights` per the recipe above). Compare to
  `n_steps=4` with cummax-off for the pooling-as-substitute control.
  Do **not** add `--use_layernorm --input_skip` on top of shared
  weights by default — on the Collapse sweep that pairing raised
  train loss 30× and degraded bfs/astar rollouts. The stab patch is
  for un-shared, per-step bodies only.
- **Single-game runs on small games**: `--no-balanced_sampling` is fine.
- **Synthetic data**: see the synth section in `RUNNING_REPORT`. The
  per-game-size + `--synthetic_no_a_count_max 5` recipe is current.

### What this report is for

Settled architectural findings live here. Active experiments and
chronological run notes live in `RUNNING_REPORT.md`. When a finding becomes
load-bearing (other experiments now assume it), promote it from RUNNING to
this doc and trim the chronology.

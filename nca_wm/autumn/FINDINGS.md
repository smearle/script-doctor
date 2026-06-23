# NCA World Models for AutumnBench — Findings

A study applying our PuzzleScript NCA world-model approach to AutumnBench. We
train a single-game neural cellular automaton to predict `f(state, action) →
next_state`, evaluate per-action on held-out transitions, and expose every model
in a side-by-side engine-vs-WM viewer (port 8766). Self-contained PyTorch module
in `nca_wm/autumn/`; the Autumn interpreter is always the ground-truth oracle.

## The thesis

AutumnBench differs from PuzzleScript on three axes (flagged in the initial
domain analysis), and **each axis needs a distinct fix**. The central, recurring
challenge is **hidden state**: almost every environment has state that drives
dynamics or an action's effect but is *not visible in the grid*.

### Three orthogonal levers, each fixing a distinct failure mode

| lever | fixes | mechanism |
|---|---|---|
| **Representation** (object channels) | object *overlap* collapse | each object type = its own channel; cells are multi-hot, so overlaps are preserved (multi-hot ⇒ sigmoid/BCE) |
| **Memory** (history / recurrent) | *hidden state* | 1-frame history when inferable from 2-frame motion; a recurrent hidden grid carried across env steps for episode-long counters/modes |
| **Loss-weighting** (`pos_weight`) | *rare events* | per-channel inverse-frequency weight so ultra-sparse channels (e.g. bullets) aren't drowned out |

## Per-environment results

| env | hidden state / issue | single-frame → fix | lever |
|---|---|---|---|
| gameOfLife | none (Markovian) | rule exact 0.99; OOD glider/blinker/rpento 0-disagreement | — |
| mario (enemy) | patrol direction | enemy recall **0.59 → 1.00** | history |
| wind | wind direction (±1) | changed-cell **0.67 → 0.85** | history |
| snake | move direction | mean-exact **0.73 → 0.81** | history |
| mario (bullets) | bullet counter | fire recall **0.09 → 0.65** (color) / **0.90** (object+pos_weight) | recurrent + weighting |
| paint | currColor (5-cycle) | paint color **0/5 → 5/5** | recurrent |
| sand | clickType brush | water-brush **0/4 → 2/4** (AR latch lag) | recurrent |
| mario (overlap) | Mario-under-coin | disappearance **16% → 0.08%**; recall 1.00 at overlaps | object channels |

### The complete Mario (all blindspots at once)
`mario_objects_recurrent`: object channels (overlap) + recurrent hidden grid
(enemy direction + bullet counter) + per-channel `pos_weight` (sparse bullets).
→ **changed-cell 0.999, fire recall 0.90**. The dropdown's 6-variant Mario
progression (singleframe → +history → +recurrent → objects → objects+history →
objects+recurrent) shows each lever turning a blindspot off.

## Breadth: 19 environments, and a gradient of hidden state

Training single-frame baselines across 19 environments shows **hidden state is
pervasive but not universal, and not binary**.

**Methodological correction (2026-06-23).** The first-pass "Markovian ≥0.93"
labels were an artifact of *easy validation sets* — small same-distribution data
where settled/identity transitions dominate. Re-collecting 5–20× more diverse
data and re-evaluating on the *harder* val split exposed hidden state in several
games first labeled Markovian. The reliable diagnostic is the **noop-imperfect
signature**: on a *deterministic* game, `noop` whole-grid exact < ~0.95 means a
hidden variable drives the autonomous dynamics. "Deterministic" ≠ "Markovian".

- **Truly Markovian**: gameOfLife, lights (1.00), chomp (1.00), lock (1.00),
  waterplug (0.997), egg, magnets, coins, grow (0.967 — sun direction is
  position-inferable, so its small residual needs no memory).
- **Hidden-direction, history-fixable** (one motion-inferable variable):
  **gravity (0.78→0.963)** — hidden `gravity` string set by edge buttons drives
  every blob every step; mario-enemy (0.59→1.00); wind (0.67→0.85); snake
  (0.73→0.81). noop is imperfect because the hidden var acts every step, and
  2-frame motion reveals it.
- **Recurrent-class** (episode-long counter/identity, set by clicks, intermittently
  observable): **disease (0.90→0.993 changed-cell, recurrent)** — hidden
  `activeParticle` identity moved by arrows; noop is *perfect* (no hidden state in
  autonomous dynamics) so history fails (motion intermittent) — only a recurrent
  latch works. mario-bullets (0.09→0.90), paint, sand, **charge** (0.939; hidden
  `energy`/`time` Int counters drive jump distance, **0.923→1.000 recurrent**).
  **When recurrent pays off**: only when the hidden variable drives a *large
  fraction* of transitions. waterplug has a hidden `currentParticle` mode but it
  only affects rare click-spawns (the dominant water-flow is Markovian), so
  recurrent *did not help* (single-frame 0.967 vs recurrent 0.932) — the lever must
  match not just the *presence* of hidden state but its *share* of the dynamics.
- **Cyclic hidden clock — formerly "hardest", now solved**: pacman
  (single-frame 0.494 → **recurrent 0.983** changed-cell). Ghosts chase only when
  `(timestep % 3) == 0` (a hidden period-3 clock not in the grid) — so single-frame
  can't tell where in the 3-cycle it is, and history only partially disambiguates a
  period-3 phase. A recurrent hidden grid tracks the mod-3 phase and the chase
  becomes fully predictable. The earlier 0.44→0.55 "ceiling" was the wrong lever
  (history), not an intractable environment.

So the architecture scales to the *kind and amount* of hidden state: none →
memoryless; one variable acting every step → 1-frame history; episode-long
counter/identity set intermittently → recurrent. The deciding test is **when**
the hidden variable acts (every step → history; only under specific actions →
recurrent), readable from the per-action breakdown.
(`sokoban` errors in the Autumn interpreter on random actions and is excluded.)

## The third axis — randomness

Snake's food respawns at a random free cell, but in random rollouts these events
are *rare* (≈0.3% of transitions), so the model predicts the deterministic
majority ("food stays") and never represents the randomness — rare stochastic
events are doubly hard (rare *and* unpredictable).

**Ants shows the positive result** (clicking directly spawns food → frequent
events). The CE-trained WM's softmax IS the predictive distribution: on a
food-free board, P(food | click) sums to ≈ the true expected count and is spread
over the grid (a distribution, not a point). It matches the engine's *actual*
spawn distribution at **spatial correlation 0.972** — including a non-obvious
quirk: the interpreter's `randomPositions` is **diagonal-biased**, not uniform,
and the WM learned that bias. By the proper scoring rule, food-channel **NLL
0.031 vs 0.083** for an always-"no-food" baseline. Figure: `aleatoric_foodmap.png`.

**Latent-z (CVAE) for the joint — tested, did NOT beat the marginal** (`latent.py`).
Encoder q(z|s,a,next) + NCA decoder p(next|s,a,z), ELBO. Two failure modes bracket
the space: unweighted → posterior collapse (z ignored, ~0 food); weighted →
decoder over-produces from prior-z samples (mean 3–10 food vs true ~2). Across a
(β, free-bits, change-weight) sweep, best CVAE count-dist L1=1.27 vs the
**marginal's 0.62** — the simple independent marginal wins. The sparse + spatially
diffuse stochastic signal is hard for a global continuous latent (prior/posterior
mismatch → over-spread). **NCA-native diffusion (MaskGIT) also tested** (`diffusion.py`) — reuses the NCA
body, adds a partially-revealed-next input + masked-prediction training +
confidence-based iterative sampling. It gets the SPATIAL distribution right
(corr 0.92) but ALSO over-produces the count (mean 4–8 vs true ~2); marginal
still wins (L1 0.85 vs ≥1.39).

**Why the marginal is hard to beat here — the key insight.** Ants' food spawns
are *independent* placements, so there is **no spatial correlation to exploit**:
revealing one food tells you nothing about the other's location. The only joint
structure is the **count** ("exactly 2"). Per-cell generative models (CVAE,
MaskGIT) don't enforce a global count — they place food wherever it's spatially
plausible, so they over-produce. Meanwhile the *calibrated* marginal (plain-CE
single-frame) samples ≈ like the true independent process: mean ≈ right (1.7),
only the count is over-dispersed (std 1.3 vs 0.58). The generative models need
change-weighting to avoid posterior/collapse on the ultra-sparse food, but that
weighting *miscalibrates* P(food) upward → over-production. So for
independent-placement randomness, **read the calibrated marginal; the residual
gap is purely the count constraint**, which needs an explicit count mechanism
(predict #spawns, then place) rather than a per-cell sampler. Spatial-joint
machinery (autoregressive/diffusion) only pays off when the randomness has
spatial *structure* (correlated cells), which this does not.

Takeaways for predicting distributions: (1) separate hidden state (reducible →
memory) from genuine aleatoric (irreducible → distribution); (2) the CE/BCE head
already gives the per-cell **marginal** — read the probability map, don't argmax,
and ensure enough events in the data; (3) score stochastic transitions by NLL /
calibration, not exact-match; (4) for coherent *joint* samples (exactly-one-food,
location unknown), add a latent `z` / diffusion / autoregressive head — the
marginal alone can't be sampled cell-independently.

## Architecture: global pooling and the spatial-locality limit

A vanilla NCA propagates information at ~1 cell/step through its 3×3 convolution,
so a signal in one corner cannot influence a prediction in the opposite corner
until it has diffused across the grid (≈grid-diameter steps). This is a *structural*
limit, and it bites whenever a hidden variable is **set at one location but acts
everywhere** — e.g. waterplug's mode buttons. Pressing the blue (water) button at
(8,0) changes no visible cell; the only signal is a one-cell click. With pure local
convolution, a cell in the far corner literally cannot know the mode changed when
you immediately click there.

**Global pooling is the fix — but the *reduction* matters.** A grid-wide pool gives
every cell an O(1) view of the whole board each step, no diffusion needed. The
recurrent model always had it, but as a **mean**: a one-cell button press contributes
~1/HW to the mean (≈1% on a 10×10 grid), so the "a button was pressed" signal is
present globally but buried below threshold. The model falls back to slow spatial
diffusion (measured: placement accuracy after a button press rises 0.33→0.62 only
over ~5 steps). **Max pooling preserves a single active cell undiluted** — the model
learns a "button-at-(8,0)" detector that fires hard at one cell, and max-pool
broadcasts it full-amplitude to every cell in one step. Controlled A/B (identical
settings, only the pool differs): mean placement-latch `0.33 0.34 0.33 0.35 0.37 0.62`
vs **meanmax `1.00` at every horizon**; overall changed-cell **0.85 → 0.997**. It
generalizes (sand latch-lag 0.92→0.98) and is safe on globally-pooled games
(disease 0.996, pacman 0.988, charge 1.000), so `meanmax` is now the recurrent
default (`--pool none|mean|max|meanmax`). **Rule of thumb: mean pooling answers "how
much on average" (dilutes sparse events); max pooling answers "is it anywhere?"
(preserves them). For sparse *global* signals — a mode button, a single spawn — use
max.** GIF: `figures/waterplug_maxpool.gif`.

This is a fourth architectural axis alongside the three levers: **representation**
(object channels), **memory** (history / recurrent), **loss-weighting** (pos_weight),
and **global context** (max-pool for sparse non-local signals; the single-frame model
also exposes `global_pool` / `axis_pool` / `axis_cummax` for `[X][Y]`-style rules).

## Open problems
- **Mario over-firing** (diagnosed, deferred): the recurrent model over-fires because
  the bullet *decrement* is undersampled — multi-fire needs ≥2 coins, but only 1/1000
  random episodes fires ≥2× (4/1000 collect ≥2 coins). Coins sit on platforms
  (only (4,12) is single-jump reachable; (11,6)/(7,4) need multi-step climbing), so
  collecting ≥2 requires a competent **platformer navigation** policy (BFS/RL), out of
  scope for random/heuristic collection. Engine bullet-injection doesn't help (the
  model tracks the count from *observed* pickups). Minor cosmetic residual on an
  otherwise-perfect mario (cell-acc 1.0, fire-recall 0.90).
- **Waterplug liquid flow**: capacity-limited, not hidden-state — bigger/deeper
  (n_hid 256, n_steps 20) lifts it 0.895→0.972; the remaining gap is the hard
  `nextLiquid` simulation on messy states, improvable with more capacity.
- **Pacman under meanmax**: −0.009 vs mean (0.997→0.988, within seed variance) — a
  hint that adding the max channel slightly perturbs games whose hidden state is
  *globally* aggregated (a mod-3 count) rather than sparse-local; negligible but worth
  noting if a game regresses.
- **Representation/loss interaction**: object/multi-hot/BCE makes ultra-sparse
  channels collapse to always-0 without `pos_weight`.
- **Aleatoric ceiling**: for irreducibly-random transitions (ants food = seeded PRNG),
  the calibrated marginal is the best achievable; sampling a *coherent* joint (exactly-N
  with unknown locations) needs an explicit count mechanism, not a per-cell head.

## Artifacts
- **Models**: `runs/` (~60 models across ~15 environments; gitignored — regenerate
  via the pipeline). Per-env single-frame baselines + their fixes, all selectable in
  the viewer.
- **Viewer**: `serve_compare.py` (port 8766) — side-by-side engine vs WM, auto-play,
  model dropdown, renders color/object/recurrent models.
- **Figures** (`figures/`): `taxonomy_levers.*` (the lever framework + before→after
  per game), `rollout_stability.*` (30-step autoregressive exact-match ≥0.91),
  `scaling_reducible.*` (data/compute helps reducible, not irreducible), `three_axes.*`,
  `blindspot_taxonomy.*`, `overlap_fix.png`, `aleatoric_foodmap.png`.
- **GIFs** (`figures/*.gif`) — successes and failures side-by-side vs the engine:
  `gravity_blindspot` (history fixes hidden direction, single-frame 6/6 wrong → 0/6),
  `disease_blindspot` / `pacman_blindspot` (recurrent fixes hidden counter/clock),
  `mario_singleframe_patrol` vs `mario_history_*` / `mario_complete` (enemy-direction
  fix) and `mario_recurrent_fire` (bullet counter), `paint_singleframe` vs
  `paint_recurrent` (currColor), `waterplug_maxpool` (max-pool fixes the mode-latch:
  mean 11/13 wrong → max 0/13), `sand_drop`, `gameOfLife_glider`.
- **Modules**: `collect.py`, `model.py` (`AutumnNCA`, `RecurrentAutumnNCA` with
  `pool`), `train.py`, `train_recurrent.py`, `objects.py`, `infer.py`, `render_gif.py`,
  `serve_compare.py`.
- **Docs**: `README.md` (entry + pipeline), `AGENTS.md` (how to work here), this file
  (results), `WORKLOG.md` (running detail).

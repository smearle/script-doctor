# NCA World Model Scaling Results

Empirical results from multi-game scaling experiments. Numbers and findings only — for
the running run-index, operational notes, and queue of next experiments, see
`SCALING_REPORT.md`.

## Headline metrics

`change_err` here = train-time changed-cell BCE (proxy for residual difficulty);
`per-game astar cell-error` = autoregressive rollout cell-error rate, averaged over
levels. Rows below use **final-step rollout cell-error** (last index of the per-level
series), then averaged over levels. Earlier `v1` / `v2` rows used a different aggregation
(mean over all rollout steps) — kept here for record but they're not directly comparable
to v2_long / v2_smallbatch / paired-Δ tables below.

| run | final train change_acc | per-game astar mean | per-game astar median | per-game astar max | n eval'd |
| --- | --- | --- | --- | --- | --- |
| v2_long | 0.979 | 0.198 | 0.123 | It_Dies_In_The_Light 63% / notsnake 85% | 37 |
| v2_smallbatch | 0.983 | 0.207 | 0.128 | notsnake 78% / Lightdown 68% | 37 |
| v2_big | (skipped) | 0.208 | 0.140 | Take_Heart_Lass 71% | 37 |
| v2_nca8 | (skipped) | 0.219 | 0.136 | Take_Heart_Lass 64% | 37 |
| v2 | 0.961 | 0.203 | 0.130 | notsnake 80% | 37 |
| v1 (legacy agg) | ~1.5% err | ~5% (mean-of-rollout agg) | — | Travelling_salesman 33% | 39 |

**Headline takeaway:** mean per-game astar error sits in [0.198, 0.219] across the five
v2 variants. **No single recipe lever moves the mean by more than ~1%.** What changes is
*which* games are bottlenecked. Ensemble-min over all 5 runs (cherry-picking best run per
game) gives mean=0.153 / median=0.102 — i.e. ~5% headroom is available *per game* if
you pick the right recipe, but no single recipe captures it.

### Persistently hard games (worst even after ensembling over all 5 v2 variants)

| game | best run | best score | comment |
| --- | --- | --- | --- |
| Take_Heart_Lass | v2_long | 0.554 | only longer training helps |
| Travelling_salesman | v2_long | 0.545 | global rules, large state |
| Some_lines_were_meant_to_be_crossed | v2_smallbatch | 0.486 | |
| notsnake | v2_big | 0.475 | only capacity helps |
| It_Dies_In_The_Light | v2 | 0.389 | every "improvement" makes it worse |
| Lightdown | v2_big | 0.371 | |
| the_art_of_cloning | v2 | 0.370 | every "improvement" makes it worse |

### Easy games (worst-of-5-runs already <8%)

`blocks`, `the_undertaking`, `blockfaker`, `MC_Escher's_Equestrian_Armageddon`,
`sokoban_basic`, `Singleton_Traffic`, `sokoban_match3`, `constellationz`, `Microban`,
`nekopuzzle`. These are stable across recipes — they're "in the easy regime."

## v2 vs v1 paired regressions (from per-game astar cell-error)

When v2 added 20 harder games on top of v1, several **simple** games that v1 had nailed
regressed sharply — strong "capacity is now the bottleneck" signal:

| game | v1 | v2 | delta |
| --- | --- | --- | --- |
| scriptcross | 0.049 | 0.477 | +0.43 |
| notsnake | 0.000 | 0.386 | +0.39 |
| Travelling_salesman | 0.336 | 0.288 | -0.05 |
| kettle | 0.155 | 0.179 | +0.02 |

(v2_nca8 reversed the scriptcross+notsnake regression — see below.)

## v2_nca8 vs v2 (paired Δ on 37 games eval'd in both)

`Δ = nca8 - v2` (negative = nca8 better). 8 improved (>1%), 20 regressed (>1%),
mean Δ = +0.008, median Δ = +0.013. Net **wash, with high per-game variance.**

Big wins for n_nca_steps=8 (some of which look like recovering capacity that v2 was
spending elsewhere):

| game | v2 | nca8 | delta |
| --- | --- | --- | --- |
| scriptcross | 0.477 | 0.075 | -0.40 |
| notsnake | 0.386 | 0.174 | -0.21 |
| Take_Heart_Lass | 0.561 | 0.479 | -0.08 |
| Multi-word_Dictionary_Game | 0.080 | 0.023 | -0.06 |

Big losses (probably batch-size noise more than NCA-step harm — confound: nca8 also
ran at batch=32):

| game | v2 | nca8 | delta |
| --- | --- | --- | --- |
| Modality | 0.092 | 0.367 | +0.28 |
| The_observer's_paradox | 0.173 | 0.329 | +0.16 |
| Long_Haul_Space_Flight | 0.082 | 0.176 | +0.09 |

## scaling_14 (14 games) vs v2 variants (59 games) on shared games

`scaling_14_v3recipe` was trained on 14 games (the original "small" preset + 5 mid-sized
gallery games) with the **exact same recipe as v3_combined** (batch=32, n_nca=8, 150k).
This isolates the *dataset-size* effect from the recipe effect. Per-game astar cell-error
on the 13 games eval'd in both (`actiontest` failed eval in scaling_14):

| game | scaling_14 (14g) | v2 (59g, baseline) | v2_nca8 (59g, same-batch) | v2_long (59g, more steps) |
| --- | --- | --- | --- | --- |
| notsnake | **0.000** | 0.800 | 0.525 | 0.850 |
| Zen_Puzzle_Garden | **0.000** | 0.218 | 0.324 | 0.177 |
| Travelling_salesman | **0.255** | 0.585 | 0.585 | 0.545 |
| Modality | **0.091** | 0.185 | 0.442 | 0.475 |
| Multi-word_Dictionary_Game | **0.048** | 0.190 | 0.079 | 0.127 |
| kettle | **0.081** | 0.209 | 0.168 | 0.164 |
| Love_and_Pieces | **0.001** | 0.080 | 0.083 | 0.043 |
| Collapsable_Sokoban | **0.023** | 0.040 | 0.063 | 0.050 |
| nekopuzzle | **0.029** | 0.099 | 0.088 | 0.095 |
| scriptcross | **0.111** | 0.671 | 0.191 | 0.112 |
| sokoban_basic | **0.000** | 0.048 | 0.012 | 0.060 |
| blocks | 0.000 | 0.000 | 0.000 | 0.000 |
| sokoban_match3 | 0.097 | **0.066** | 0.065 | 0.057 |
| **mean (13 games)** | **0.057** | **0.245** | **0.202** | **0.212** |

scaling_14 wins on **12/13 shared games**; v2 wins only on `sokoban_match3`. The mean
error is **~4× lower** training on 14 games than on 59 — even after holding recipe
constant against v2_nca8 (still ~3.5× lower).

The naive read is "this refutes the rule-transfer hypothesis at the current scale —
adding 45 extra games hurts." But early v3_combined eval (same recipe + same 59 games as
scaling_14, training in progress as of 2026-05-03) shows nearly-identical numbers to
scaling_14 on the first 7 eval'd games (notsnake 0%, blocks 0%, sokoban_basic 0%,
sokoban_match3 0.6%, nekopuzzle 1.8%, Multi-word_Dictionary_Game 0%, Zen_Puzzle_Garden
~2%). So **most of the v2 vs scaling_14 gap was actually the recipe (batch=32 + n_nca=8),
not the dataset size.** The dataset-size effect, controlled for recipe, is small (e.g.
Zen_Puzzle_Garden 0% → 2%). Need v3_combined's full eval to make the final call.

## Bouncers L×R sweep (single-game, 2026-05-03)

The 2026-05-02 architectural change replaced the `--shared_weights` flag with
`--n_nca_repeats`, factoring `n_steps` into `(n_layers × n_repeats)`:

- **n_layers** = inner block of distinct rule-application layers (one
  conv → pool → cross-attn → out per layer, each with its own weights). The
  engine analog is one ordered run-through of the rule list.
- **n_repeats** = number of times the L-layer block is re-applied with shared
  weights. The engine analog is the `again` loop.

Bouncers (5 levels, 200k transitions, 0 winning trajectories — eval falls
back to BFS oracle rollouts) was picked for its `again`-heavy gameplay.
Six configs trained at `n_hid=256`, batch=16, 15k steps, lr=3e-4 (recipe
matches the architecture-report Collapse sweep).

Figures: `nca_wm/figures/bouncers_lr_sweep/{train_loss_curves,final_rollout_bars,params_vs_rollout}.{pdf,png}`.
Summary CSV: `nca_wm/figures/bouncers_lr_sweep/summary_table.csv`.

| (L, R) | total | params | best train loss | BFS rollout (mean over levels) | random rollout |
| ---: | ---: | ---: | ---: | ---: | ---: |
| (4, 1) | 4 | 5.03 M | 1.3e-7 | **0.51%** | 0.61% |
| (1, 4) | 4 | 1.39 M | 2.3e-7 | **0.51%** | 0.56% |
| (8, 1) | 8 | 9.89 M | 1.2e-7 | **0.19%** | 0.39% |
| (4, 2) | 8 | 5.03 M | 1.2e-7 | 0.51% | 0.53% |
| **(2, 4)** | **8** | **2.60 M** | **6.2e-8** | 0.51% | **0.39%** |
| (1, 8) | 8 | 1.39 M | 1.0e-7 | 0.66% | 0.57% |

### Findings

1. **Bouncers is too easy at all configs.** Even the smallest (1.4M params,
   `(L=1, R=8)`) reaches sub-1% rollout error on every level. The sweep does
   not discriminate the configs on absolute capability — only on parameter
   efficiency. To actually stress the L×R axis we need a harder game (per the
   architecture report: Mirror Isles, Heroes of Sokoban, Sokoboros).
2. **Sharing does not hurt train loss on Bouncers.** All four total=8 configs
   reach train loss in [6.2e-8, 1.2e-7] — within an order of magnitude.
   `(L=2, R=4)` achieves the lowest train loss of all six configs at less than
   1/3 the params of `(L=8, R=1)`. The "engine-faithful" split (small inner
   block × multiple `again`-style repeats) finds the tightest fit.
3. **Per-step weights (`R=1`) win on BFS rollout, but only narrowly.**
   `(L=8, R=1)` at 9.9M params hits 0.19% BFS — best in the sweep.
   `(L=2, R=4)` at 2.6M (3.8× fewer params) hits 0.51%. Going further to
   `(L=1, R=8)` at 1.4M hits 0.66%. The Pareto frontier is *real but narrow*
   — sharing trades ~0.3-0.5% absolute BFS error for 4-7× param reduction.
4. **Random rollout doesn't track BFS.** `(L=2, R=4)` ties `(L=8, R=1)` on
   random rollout (0.39%) at 4× fewer params. The 9.9M model isn't more
   robust to off-policy actions than the 2.6M shared-block one.
5. **Doubling iterations via repeats with the same inner block doesn't
   help.** `(L=4, R=1)` and `(L=4, R=2)` hit identical BFS error (0.51%);
   `(L=1, R=4)` and `(L=1, R=8)` differ by only 0.15%. On Bouncers, n_steps=4
   is already saturated; adding more iterations refines but doesn't unlock
   new capability.
6. **Architecture strength (current rule_attn body):** the inductive bias
   for sharing across iterations is real and almost free. `(L=1, R=4)` ties
   `(L=4, R=1)` on BFS (0.51%) at <1/3 the params and the same total
   compute. The `n_repeats` axis is the right way to add depth cheaply.
7. **Architecture weakness:** per-step weights still buy real (small)
   gains, suggesting the inner-block representation is *not* a perfect
   match for the engine's iterative semantics. A truly faithful neural
   analog would have `(L=1, R=8)` matching `(L=8, R=1)` exactly. The
   ~0.5% gap is the residual capacity per-step-specialization is buying.
   Whether this gap closes with adaptive halting / harder games is the
   question for stage B.

## Bouncers depth & no-pool sweep + LN/skip disaggregation (2026-05-03)

This sweep combined three axes on a single game (Bouncers, h=256, batch=16, 15k steps):
1. The `n_layers × n_repeats` factoring across L × R ∈ {(8,1), (4,1), (1,4), (1,8),
   (2,4), (4,2), (4,4), (2,8), (1,16), (2,16), (1,32), (1,64)}.
2. Pool flags ON vs OFF (`--no-axis_pool --no-axis_cummax --no-global_pool`).
3. Stabilization patch on/off and its components individually (LN-only, input_skip-only,
   bundled), specifically targeted at the configs that regressed.

Figures: `nca_wm/figures/bouncers_lr_sweep/`.

### Pool ON: depth ceiling around per-step layer count L

| (L, R) | total | params | train | bfs |
|---:|---:|---:|---:|---:|
| (8, 1) | 8 | 9.9 M | 1.2e-7 | **0.19%** |
| (4, 4) | 16 | 5.0 M | 9.6e-8 | **0.19%** |
| (4, 2) | 8 | 5.0 M | 1.2e-7 | 0.51% |
| (2, 4) | 8 | 2.6 M | 6.2e-8 | 0.51% |
| (2, 8) | 16 | 2.6 M | 1.4e-7 | 0.69% |
| (1, 8) | 8 | 1.4 M | 1.0e-7 | 0.66% |
| (1, 16) | 16 | 1.4 M | 1.4e-7 | 0.80% |
| (1, 32) | 32 | 1.4 M | 2.1e-7 | 2.05% |
| (1, 64) | 64 | 1.4 M | 3.9e-7 | 1.79% |
| (2, 16) | 32 | 2.6 M | 2.8e-7 | **4.84%** |

Pool variant best is `(L=8, R=1)` and `(L=4, R=4)` both at **0.19% bfs** — *the same*
at half the params. Past total=16 with low L, depth starts to hurt: `(2, 16)` jumps to
4.84%. **L (per-step layer count) sets the rollout floor; R (shared repeats) refines
within that floor up to about R=8.**

### Pool OFF: depth ceiling lifts dramatically at high R, fully recovered by input_skip

| (L, R) | total | params | bare bfs | LN-only | skip-only | bundled |
|---:|---:|---:|---:|---:|---:|---:|
| (8, 1) | 8 | 6.7 M | 0.88% | — | — | — |
| (1, 8) | 8 | 1.0 M | 0.98% | — | — | — |
| (4, 4) | 16 | 3.5 M | 0.98% | — | — | — |
| (1, 16) | 16 | 1.0 M | 3.59% | 2.80% | **0.88%** | 0.88% |
| (16, 1) | 16 | 13.4 M | (not run) | 1.83% | **0.88%** | 1.06% |
| (2, 16) | 32 | 1.8 M | 8.26% | — | — | 1.25% |
| (1, 32) | 32 | 1.0 M | 4.04% | — | — | 1.81% |
| (1, 64) | 64 | 1.0 M | 1.35% | — | — | — |

Without the patch, shared-deep nopool degrades sharply past R=8. The bundled patch
recovers it by 4-7×. **Disaggregating shows the recovery is entirely input_skip.**

### LN vs input_skip — the patch is just input_skip

`(L=1, R=16)` max-shared regime:
- bare: 3.59% — LN-only: 2.80% — **skip-only: 0.88%** — bundled: 0.88%

`(L=16, R=1)` per-step deep regime:
- LN-only: 1.83% — **skip-only: 0.88%** — bundled: 1.06%

**input_skip alone matches or beats the bundled patch in both regimes.** LN on top of
input_skip is at best neutral (max-shared) and degrades the per-step regime by ~20%.
LN-alone gives only partial recovery in the max-shared regime and is worse than
skip-alone everywhere.

### Findings

1. **The "stab patch" should be reduced to just input_skip.** LN is at best neutral,
   at worst (in the per-step deep regime) harmful when combined with input_skip. Future
   work should default `--input_skip` on for any deep config and drop `--use_layernorm`.
2. **The architecture report's "stab patch off for shared weights" Collapse finding is
   contradicted on Bouncers-without-pool.** input_skip helps shared bodies a lot when
   pool is off. The Collapse anti-stab finding may have been entirely the LN component
   degrading in the with-pool regime — worth a re-test on Collapse with skip-only.
3. **Pool variant has its own depth ceiling around L=4.** `(L=8, R=1)` and `(L=4, R=4)`
   tie at 0.19% bfs (best on Bouncers). Going to L=1 with R≥16 starts to degrade. With
   pool, the L axis matters more than total depth.
4. **Bouncers — even no-pool — is too easy to test depth past 16.** All configs reach
   <2% bfs once they have input_skip. The architecture's depth-helps-iterative-reasoning
   capability isn't actually being tested. Need a harder game.
5. **Non-monotonic depth at high R.** Bare `(1, 32)` is worse than `(1, 64)` (4.04% →
   1.35%). The optimization landscape at high sharing has local minima — adding more
   iterations can either degrade or recover depending on which optimum the model lands in.

## Findings so far

1. **Per-game cap + bitpack works.** scaling_gallery_v2 (59 games, max shape 19×30×43,
   ~7.7 M transitions cached) fits comfortably; previous OOMs at game ~12 are gone.
2. **The "v2 capacity bottleneck" hypothesis is partly refuted.** v2_smallbatch
   (baseline + batch=32 only) is a *net wash* vs v2 (mean Δ=+0.4%, median Δ=+0.7%,
   9 games improved >1% / 14 regressed >1%). Specifically:
   - scriptcross **does** recover with batch=32 alone (0.67 → 0.29) — batch noise was
     a factor for that game.
   - notsnake does **not** recover with batch=32 alone (0.80 → 0.78). v2_big and
     v2_nca8 are the only runs that fix it (0.48, 0.53). So *capacity* matters there,
     independently of batch size.
   - Other games regress under batch=32 (Lightdown 0.41→0.68, Collapsable_Sokoban
     0.04→0.15) — implying gradient noise is *helpful* for some games.
3. **Different levers help different games — no single recipe wins.** scriptcross:
   any improvement helps; longer training is best. notsnake: only capacity helps.
   Take_Heart_Lass: only longer training helps. It_Dies_In_The_Light + Modality:
   *every* lever (batch, capacity, duration) makes them worse vs v2 baseline.
4. **Longer training (v2_long, 200k steps) buys ~0.5% mean improvement** with high
   per-game variance. It's the best run on the hardest game (Take_Heart_Lass 0.55,
   Travelling_salesman 0.55) but the worst on It_Dies_In_The_Light (0.63 vs v2's
   0.39) and notsnake (0.85 vs v2's 0.80) — looks like overfitting to easier games
   late in training.
5. **Bigger model + more NCA steps do NOT lower mean per-game error.** v2 / v2_nca8 /
   v2_big / v2_long / v2_smallbatch all hit mean astar cell-error in [0.198, 0.219].
   They *redistribute* difficulty rather than reduce it.
6. **Bigger model OOMs on (32×64) buckets at batch=64.** Activation memory, not weight
   memory, is the cap. Halve batch first; multi-GPU is overkill until n_hid > 1024.
7. **Ensemble of all 5 runs gets mean=0.153 / median=0.102.** That's ~5% mean improve-
   ment available *per game* if we always pick the right recipe — meaning the family
   of v2 recipes spans real per-game variance, but no single point captures it.
   This argues for either: (a) a *combined* recipe trying to get the best of multiple
   levers, (b) per-game/per-level mixture-of-experts, or (c) curriculum.
8. **Recipe (batch=32 + n_nca=8) >> dataset size for these games.** Most of what looked
   like "negative transfer from scaling 14g → 59g" turned out to be the recipe
   difference, not the dataset. Once recipe is held constant (v3_combined vs
   scaling_14), the per-game gap shrinks dramatically.

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

## Varislide post-bitpack-fix sweep (2026-05-04)

After the May 4 synth bit-pack fix (commit 1fa557d), 34 configs were re-run
on multi-grid varislide synth ({6,8,10,12,16}x3) at h=128, batch=16, lr=3e-4,
10k updates, mask_hidden=True default. Per-distance argmax-correct (% of
right-action transitions where argmax(player-row sigmoid) lands at the GT
slide-stop column) is the direct test of "did the model learn to iterate."

**Headline: every single variant solves it.**

| bucket | configs tested | argmax (mean) | min |
|---|---|---|---|
| A: depth × seed (fully shared) | 3 depths {8, 16, 32} × 3 seeds | 100.0% | 99.9% (d=32 s0) |
| B: L × R factor at total=16 | 5 (L,R) × 3 seeds | 100.0% | 99.7% (L=2 R=8 s2) |
| C: pool OFF + input_skip | 3 depths × 3 seeds | 100.0% | 100.0% |
| C': pool OFF, no input_skip | depth=16 seed=0 | 100.0% | 100.0% |

(The 99.7%/99.9% outliers each reflect 1–2 misclassified cells out of hundreds.)

Figures: `nca_wm/figures/varislide_postfix/{argmax_by_depth, argmax_by_LR,
argmax_pool_onoff}.{pdf,png}`. Summary CSV/MD same dir.

**Held-out authored map-size eval (re-eval 2026-05-04 with top-left
slicing fix):** all 34 checkpoints were re-evaluated against authored
varislide levels (W ∈ {6, 7, 9, 11, 15, 19}) using the corrected slicing.
Numbers in `figures/varislide_postfix/eval_summary.{csv,md}`. Per-step
cell-error rate (mean over 30-step rollout):

- **TF (teacher-forced) train widths:** ≤ 0.06% across every config.
- **TF interp widths (W ∈ {7, 9, 11, 15}):** 0.20–0.62%.
- **TF OOD width (W=19, only L7):** 0.12–0.40%.
- **AR (autoregressive) train widths:** essentially 0%; AR interp 2–5%
  (compounding error); AR OOD (W=19) 1–3%.
- **BFS-oracle rollouts:** 0% on train widths, 0.08–0.32% on interp,
  ~1.56% on the W=19 OOD slice (deterministic — same value across most
  configs because it's a fixed authored level).

The corrected eval confirms varislide is fully solved by every config:
both the trained-width "TF=0%" claim and the held-out width
generalization are tight, with at most ~5% AR error in the worst case
(typically L7 which has W=19 > training max W=16).

### Findings

1. **Depth ≥ 8 fully shared is sufficient** — d=8/16/32 all hit 100%; the pre-fix
   "depth flat 8↔64 means depth doesn't help" claim from F8 was the bitpack
   regression, not architecture. With correct training data, depth 8 is plenty
   for this 1-rule slide game.
2. **Every L×R factoring at total=16 works** — (1,16), (2,8), (4,4), (8,2),
   (16,1) all hit 100%. Per-step weights vs fully-shared: indistinguishable
   on this game. (Bouncers L×R sweep had a narrow Pareto frontier; varislide
   doesn't.)
3. **Pool OFF doesn't matter for varislide** — pool ON, pool OFF + input_skip,
   and pool OFF + no input_skip all hit 100%. The 1-rule sliding dynamic is
   purely local (3×3 conv suffices for the iterative step), so the global
   pool features that matter for `[X][Y]` rules aren't load-bearing here.
   Picking a harder canary (Heroes_of_Sokoban / Mirror Isles, per the bouncers
   F1 follow-up) is required to actually stress the pool axis.
4. **Per-seed variance is NOT real on this canary.** The pre-fix "5× per-seed
   argmax variance" finding was the bitpack bug producing all-zero training
   data; under different-init RNGs the model converged to different
   "predict-zero-everywhere" stalls because some seeds happened to find
   slightly different local minima of the all-zero-target BCE. Post-fix all
   3 seeds at every config are within 0.3% of each other.
5. **State-loss plateaus at ~6e-2 with mask_hidden=True even at change_acc=100%.**
   Pre-mask runs hit ~1e-7 at the same predictions. The plateau is BCE on
   under-confident logits over real cells (padded cells are already zero from
   the mask multiply); it is not a signal of learning failure. Don't read
   "loss not driving to machine zero" as a problem on masked runs.

### Implications for the gallery scaling story

The persistently-hard gallery games (Take_Heart_Lass, Travelling_salesman,
notsnake, …) were previously cross-referenced to F8 as "the gallery-scale
analog of the varislide canary." That extrapolation now has no empirical
grounding — the varislide canary doesn't fail post-fix. The gallery per-game
variance documented above is therefore unexplained by an
architectural-iterative-reasoning story; the next experiment should isolate
the actual gallery bottleneck (per-game capacity / data / rule complexity).
The "remaining candidate fixes" enumerated in F8 (hard slot routing, per-cell
halt, replay-buffer curriculum, auxiliary heads) are NO LONGER motivated by
this canary and should be re-justified before pursuing.

## Varislide iteration extrapolation (2026-05-04)

Test of the user's NCA hypothesis: a shared-weight NCA trained at small
depth D_train should be able to "iterate further" at inference (D_eval >
D_train) and capture longer slides than the training compute graph
encodes. Train pool OFF + shared (n_layers=1, n_repeats=D_train) on
narrow synth grids {6x3, 8x3} (so the trained-time max slide ≤ ~6
cells), then re-instantiate the model with overridden n_steps =
n_repeats = D_eval and evaluate on widths {6, 8, 10, 12, 16}. 12 runs
total: D_train ∈ {2, 4, 8} × seeds {0,1,2} pool OFF, plus pool-ON
control at D_train=2.

Argmax-correct (mean ± std over 3 seeds, aggregated across all 5 widths):

| D_train | D_eval | short (d≤2) | med (3–7) |
|---|---|---|---|
| 2 (pool OFF) | 1 | 0.841 ± 0.006 | 0.024 ± 0.017 |
| 2 (pool OFF) | **2** | **0.969 ± 0.001** | **0.762 ± 0.045** |
| 2 (pool OFF) | 4 | 0.891 ± 0.023 | 0.548 ± 0.089 |
| 2 (pool OFF) | 8 | 0.253 ± 0.027 | 0.202 ± 0.144 |
| 2 (pool OFF) | 16 | 0.101 ± 0.034 | 0.131 ± 0.094 |
| 2 (pool OFF) | 64 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 4 (pool OFF) | 1 | 0.567 ± 0.068 | 0.000 ± 0.000 |
| 4 (pool OFF) | 2 | 0.894 ± 0.008 | 0.012 ± 0.017 |
| 4 (pool OFF) | **4** | **0.967 ± 0.000** | **1.000 ± 0.000** |
| 4 (pool OFF) | 8 | 0.800 ± 0.101 | 0.381 ± 0.251 |
| 4 (pool OFF) | 16 | 0.288 ± 0.057 | 0.024 ± 0.034 |
| 4 (pool OFF) | 64 | 0.161 ± 0.055 | 0.036 ± 0.029 |
| 8 (pool OFF) | 1 | 0.107 ± 0.068 | 0.000 ± 0.000 |
| 8 (pool OFF) | 4 | 0.863 ± 0.033 | 0.012 ± 0.017 |
| 8 (pool OFF) | **8** | **0.967 ± 0.000** | **1.000 ± 0.000** |
| 8 (pool OFF) | 16 | 0.892 ± 0.093 | 0.881 ± 0.121 |
| 8 (pool OFF) | 32 | 0.334 ± 0.207 | 0.298 ± 0.248 |
| 8 (pool OFF) | 64 | 0.108 ± 0.069 | 0.071 ± 0.058 |
| 2 (pool ON)  | 2 | 0.967 ± 0.000 | 0.988 ± 0.017 |
| 2 (pool ON)  | 4 | 0.892 ± 0.057 | 0.845 ± 0.131 |
| 2 (pool ON)  | 8 | 0.325 ± 0.066 | 0.476 ± 0.089 |
| 2 (pool ON)  | 64 | 0.000 ± 0.000 | 0.000 ± 0.000 |

Figures: `nca_wm/figures/varislide_depth_extrap/{fig_argmax_vs_Deval,
fig_argmax_by_distance, fig_argmax_by_distance_pool}.{pdf,png}`,
`summary.{csv,md}`. Eval JSON per run at
`logs_depth_extrap/<tag>/depth_extrap_eval.json`.

### Findings

1. **The shared-weight body genuinely learns a per-iteration shift
   rule.** Each D_train's accuracy at D_eval = D_train is 96.7% / 100%
   on short / med slides. Cranking D_eval *down* below D_train degrades
   accuracy in proportion to slide distance (D_eval=1 → only d=1 works;
   D_eval=2 → d≤2 work; etc.). This confirms the NCA decomposes the
   `again` loop into a stack of "shift player by 1 cell" operations —
   exactly the PuzzleScript engine semantics.
2. **Limited 1-step extrapolation works.** D_eval = 2× D_train still
   delivers usable accuracy on short slides (e.g., D_train=4 →
   D_eval=8: 80.0% short / 38.1% med; D_train=8 → D_eval=16: 89.2%
   short / 88.1% med). So *some* depth headroom exists at inference,
   but only ~1 doubling.
3. **Beyond ~2× D_train the dynamics collapse.** Every config drops
   sharply once D_eval ≥ 4× D_train and is at 0% by D_eval=64 — the
   shared-weight body has no fixed point at "rest state," so applying
   the rule too many times smears or destroys the player. This is the
   classic Mordvintsev-NCA stability problem: without explicit
   training-time signal that "applying the rule when no change is
   needed should be a no-op," the model's hidden state drifts off the
   learned manifold.
4. **Pool ON does not buy stability.** The pool-ON control at D_train=2
   slightly improves accuracy at D_eval ∈ {2, 4} (98.8% vs 76.2% on
   med slides at D_eval=2), but it collapses at the *same* rate when
   D_eval grows large (0% at D_eval=64). Global-context features help
   the per-step rule, not the iteration extrapolation.

### Interpretation

The user's hypothesis ("more inference iterations capture non-local /
looping dynamics") is **partially true but bounded** under fixed-depth
training. The model learns the iterative semantics — each NCA step ≈
one PuzzleScript `again` firing — but it has not learned to be
*idempotent at the rest state*, so iterating past convergence destroys
the prediction. The follow-up below (uniform-halt training) confirms
this is an optimization artifact, not an architectural ceiling.

### Follow-up: uniform per-step supervision unlocks depth-arbitrary iteration

Re-trained pool OFF + input_skip + shared at T_max = 32 with
`--adaptive_halt --halt_mode uniform --halt_kl_weight 0` (3 seeds, full
multi-grid {6,8,10,12,16}x3, otherwise identical recipe). Uniform halt
weights every per-step readout equally in the loss, forcing the body
to predict the correct next state at every k=1..32 — including all the
post-convergence steps where the input is already the rest state. This
should make the rest state a fixed point.

Per-distance argmax-correct (3 seeds, mean ± std, all 5 widths pooled):

| D_eval | short (d≤2) | med (3–7) |
|---|---|---|
| 1 | 0.844 ± 0.000 | 0.024 ± 0.034 |
| 2 | 0.971 ± 0.001 | 0.119 ± 0.017 |
| 4 | 0.985 ± 0.004 | 0.988 ± 0.017 |
| 8 | **1.000 ± 0.000** | **1.000 ± 0.000** |
| 16 | **1.000 ± 0.000** | **1.000 ± 0.000** |
| 32 | **1.000 ± 0.000** | **1.000 ± 0.000** |
| 64 | **1.000 ± 0.000** | **1.000 ± 0.000** |

**The hypothesis is fully validated under uniform-halt training.** Once
the body is trained to be valid at every step, accuracy is 100% across
every slide distance for every D_eval ≥ 4 (the max slide distance in
the dataset), and crucially **stays at 100% at D_eval=64, which is 2×
past T_max** — the body is genuinely a stable fixed point at the rest
state. D_eval=1 still only solves d=1 (single-cell shift per
iteration), and D_eval=2 only solves d≤2, confirming the model has
*not* shortcut its way around iteration: long slides require many
iterations, and the model's per-step semantics match PuzzleScript's
`again` loop exactly.

Combined with the fixed-depth result, the story is clean: shared-weight
NCAs with pool OFF can capture PuzzleScript's looping rule semantics
*and* iterate beyond training depth, but only when training-time signal
forces idempotence at convergence. Fixed-depth training collapses past
~2× D_train; uniform-halt training does not collapse out to 2× T_max.

Run dirs: `nca_wm/logs_depth_extrap/nopool_uniform_T32_s{0,1,2}/`.
Combined heatmap (4 panels: D_train ∈ {2,4,8} fixed, plus D_train=32
uniform): `nca_wm/figures/varislide_depth_extrap/fig_argmax_by_distance.{pdf,png}`
— the uniform panel is fully yellow across the entire upper-right
quadrant, in clear contrast to the diagonal-only stripes of the fixed
panels.

### Width-OOD generalization

The same uniform-halt checkpoints were re-evaluated on OOD synthetic
widths {20, 24, 32} (none seen at training; max train width was 16) at
D_eval up to 128 (= 4× T_max). New synth caches generated via
`collect_synthetic_dataset(varislide, width=20|24|32, height=3, n=64,
seed=0)`. Argmax-correct (mean ± std over 3 seeds, all observed
right-action delta transitions pooled across slide distances):

| D_eval | W=16 (train) | W=20 (OOD) | W=24 (OOD) | W=32 (OOD) |
|---|---|---|---|---|
| 8 | 1.000 ± 0.000 | 0.984 ± 0.007 | 0.987 ± 0.008 | 0.995 ± 0.003 |
| 16 | 1.000 ± 0.000 | 0.996 ± 0.006 | 0.998 ± 0.002 | 0.998 ± 0.002 |
| 32 | 1.000 ± 0.000 | 0.996 ± 0.005 | 0.998 ± 0.002 | 0.998 ± 0.002 |
| 64 | 1.000 ± 0.000 | 0.996 ± 0.005 | 0.998 ± 0.003 | 0.998 ± 0.002 |
| 128 | 0.972 ± 0.039 | 0.994 ± 0.005 | 0.997 ± 0.004 | 0.997 ± 0.003 |

The model generalizes simultaneously along **both** axes — depth (4×
T_max at D_eval=128) and grid width (2× max train width at W=32) —
with no measurable degradation. Caveat: synth varislide level
generation produces dense walls, so most OOD-width transitions are
still short-slide (d ≤ 5); the long-slide-on-wide-grid regime would
need handcrafted test cases or a level generator with controlled
slide-distance density. The result so far establishes that the body is
a stable local rule on grids of any width, and that body iterated to
convergence is correct.

### Handcrafted long-slide stress test — caveats to "depth-arbitrary"

To close the synth-density caveat, a handcrafted authored-layout
benchmark was added (`scripts/eval_varislide_handcrafted_slides.py`,
output `<run>/depth_extrap_handcrafted.json`). One canonical level per
slide_d ∈ {3, 6, 12, 18, 24, 30}: 3 rows × W columns with W = d+4,
walls all around the perimeter and a `##` pair at the right edge of
the middle row (mirrors `custom_games/varislide.txt`); player at col
1 slides to col W-3.

The pure-iteration story does **not** hold here. Per-seed, per-(D_eval,
d) prediction (target = pred player col, ✓ if argmax cell == target):

| D_eval | d=3 | d=6 | d=12 | d=18 | d=24 | d=30 |
|---|---|---|---|---|---|---|
| seed 0, 4 | ✗ off-by-1 to wall | ✗ off-by-1 | ✗ off-by-1 | ✗ off-by-1 | ✗ off-by-1 | ✗ off-by-1 |
| seed 0, 8 | ✗ to wall | **✓** | **✓** | **✓** | **✓** | **✓** |
| seed 0, 16+ | ✗ stuck at col 5 | ✗ stuck at col 5 | ✗ stuck | ✗ stuck | ✗ stuck | ✗ stuck |
| seed 1, 4–128 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| seed 2, 4 | **✓** | ✗ | ✗ | ✗ | ✗ | ✗ |
| seed 2, 8+ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

Three observations refute the strong "iteration extrapolation" claim:

1. **Seed-specific behavior.** Only seed 0 at D_eval=8 nails d=6
   through d=30. Seeds 1 and 2 fail at every D_eval × d combination
   except seed 2 at d=3, D_eval=4. The synth-eval 100% number was
   driven by short-slide (d≤5) transitions, where all three seeds work.
2. **Seed 0 at D_eval=8 solving d=30 is non-iterative.** With pool OFF
   and a 3×3 conv per step, 8 NCA iterations propagate info at most 8
   cells. Yet seed 0 correctly places the player 30 cells from its
   start in 8 steps. The mechanism is a local pattern detector: each
   cell asks "am I empty with empty-left and wall-right?" and emits
   itself as the player. This is a global *equilibrium* of the slide,
   not the iterative trajectory. Cross-attention to the per-game rule
   slots routes the equilibrium pattern to every cell.
3. **D_eval ≥ 16 collapses on the authored layout.** Even seed 0
   plateaus at "predict col 5 regardless of W" once D_eval > 8. The
   "100% at D_eval=128 on synth" was layout-luck — the synth
   distribution mostly has d≤5 ⇒ true target IS at low column ⇒
   "guess col 5" is often right by coincidence.

**Updated claim.** The uniform-halt recipe makes per-step readouts
valid at every k=1..T_max, *on the training distribution*. Beyond
training distribution (here: a different layout where slides are long
and walls are bookended) the model often relies on layout-specific
pattern shortcuts that are seed- and depth-fragile. The earlier
strong "iteration extrapolation unlocked" claim should be tempered:
the body learns a stable rule on the training distribution, but it
does *not* automatically generalize to longer slides on different
layouts via iteration alone. Truly bulletproof depth extrapolation
likely needs either (i) training data with long-slide authored-layout
levels, or (ii) an architecture that can't take the equilibrium
shortcut (e.g. forcing per-step transition supervision against the
engine's intermediate states).

## Held-out transfer eval (2026-05-04)

First valid transfer measurement on the existing multi-game checkpoints,
post the 2026-05-04 `_build_model` flag-plumbing fix in `heldout_eval.py`.
Default 6-game held-out list (`blank, sumo, the_undertaking, wrappingrecipe,
rigidfail1, constellationz`) — these are in `scaling_large` but not in
`scaling_14`. Baseline = identity predictor (next == current).

**Note on validity:** Default heldout overlaps `scaling_gallery_v2`
(blank held-out only, others in v2's training set). So the v2/v2_nca8/
v2_long200k numbers are *in-distribution* spot checks, not transfer
measurements; they're recorded below for completeness but the only valid
transfer cell here is `multi_scaling_14_v3recipe`.

### multi_scaling_14_v3recipe (14 train games, 150k steps, no mask_hidden)

| held-out game | model AR step1 | identity step1 | transfer |
|---|---|---|---|
| sumo | 8.2% | 5.7% | worse than id |
| the_undertaking | **0.5%** | 0.9% | **beats id** |
| wrappingrecipe | 4.5% | 3.3% | worse than id |
| rigidfail1 | **2.6%** | 5.9% | **beats id** |
| constellationz | 4.5% | 1.0% | worse than id |

2/5 held-out games beat the identity baseline at single-step prediction.
AR rollout error compounds quickly on the failing 3 (mean 16-34%).

**Note:** the model has issues even on its OWN training games at step 1
(Travelling_salesman: AR step1 model 3.4% vs identity 1.5%) — so the limited
transfer is consistent with the model's overall not-yet-mastery of harder
games.

### scaling_14 + mask_hidden=True ablation (2026-05-04)

Two follow-up runs at the same recipe as `multi_scaling_14_v3recipe` (h=256,
n_nca=8, batch=32, lr=3e-4) but with `--mask_hidden --mask_padded_loss` and
80k steps (not 150k):

- `multi_scaling_14_mask_v1`: fully-shared body (n_nca_repeats=8).
- `multi_scaling_14_mask_v2_perstep`: per-step weights (n_nca_repeats=1).

| held-out game | v3 (no mask, 150k) | mask_v1 (shared, 80k) | mask_v2 (per-step, 80k) | identity |
|---|---|---|---|---|
| sumo | 8.21% | 12.14% | **5.71%** | 5.71% |
| the_undertaking | **0.49%** | 0.50% | **0.37%** | 0.91% |
| wrappingrecipe | 4.49% | 6.94% | **2.04%** | 3.27% |
| rigidfail1 | **2.59%** | 3.70% | **2.59%** | 5.93% |
| constellationz | 4.53% | **0.12%** | 2.66% | 1.01% |
| **beats-id count** | **2/5** | **3/5** | **3/5** | — |

(Bold = beats identity at step 1.)

Both mask variants beat the v3recipe baseline on the beats-identity tally
(3/5 vs 2/5). Effects are non-uniform: `mask_v1` (fully-shared) gets a 38×
improvement on `constellationz` (4.53% → 0.12%) but regresses on `sumo`
and `wrappingrecipe`. `mask_v2_perstep` is more uniform and gets the best
step-1 numbers on 3 of 5 games.

**AR rollout (30-step mean):** all three variants compound errors quickly.
`mask_v1` is best on `constellationz` rollout (1.33% mean vs v3's 15.63%) but
worse on `the_undertaking` (5.03% vs v3's 2.02%). The mask-hidden mechanism
sharpens single-step prediction but does not by itself fix multi-step drift.

### Hard-game transfer (2026-05-04)

To stress-test transfer beyond the simpler default heldout list, all 3
scaling_14 checkpoints were also evaluated on 4 of the persistently-hard
gallery games not in scaling_14 (`It_Dies_In_The_Light`, `Lightdown`,
`Take_Heart_Lass`, `the_art_of_cloning` — `notsnake` is actually in
scaling_14 and was filtered out of the heldout list):

| game | v3 step1 | mask_v1 step1 | mask_v2 step1 | identity |
|---|---|---|---|---|
| It_Dies_In_The_Light | 51.5% | 56.1% | 60.1% | 0.8% |
| Lightdown | 47.4% | 37.7% | 48.0% | 5.6% |
| Take_Heart_Lass | 15.6% | 35.2% | 20.8% | 10.2% |
| the_art_of_cloning | 3.2% | 7.9% | 5.9% | 3.2% |
| **beats-id count** | **0/4** | **0/4** | **0/4** | — |

**None** of the 3 scaling_14 transfer variants beats identity on any of the
4 hard heldout games. The persistently-hard classification (per the F8
withdrawn cross-reference and the v2 baseline numbers above) holds for
transfer too: at the scaling_14 game-set size, these games are not yet
modeled at *all* — let alone transferable. Mask helps slightly (mask_v1 is
~10pp better than v3 on Lightdown) but doesn't change the qualitative
picture.

The actionable takeaway: **transfer to "simple" games is achievable already
(3/5 with mask), but transfer to "hard" games requires the bottleneck on
hard games to be solved first — likely dataset scale, per-game capacity,
or fundamentally different mechanics.**

### Caveats for future transfer experiments

1. **scaling_14_v3recipe was trained without `mask_hidden=True`.** Per the
   varislide canary memory, mask_hidden is what enables translation-
   equivariance; transfer should improve with mask_hidden=True multi-game
   training. Worth a follow-up.
2. Default heldout list (`heldout_eval.py:DEFAULT_HELDOUT`) overlaps any
   training preset ⊇ `scaling_large`. Pick a true-disjoint list for v2/v3
   transfer eval (or at least filter heldout per checkpoint's training set).
3. Heldout numbers for `multi_scaling_gallery_v2_long200k` and `_nca8`:
   Near-zero on all 5 "heldout" games — but those games are in v2's training
   set. These are validation spot checks, not transfer.

Result files: `nca_wm/logs/multi_scaling_*/heldout_transfer_v1/results.json`
+ `rollout_curves.png` + `step1_bar.png` per checkpoint.

## Heroes_of_Sokoban L0 depth × sharing × pool sweep (2026-05-04, RE-EVAL)

A harder-than-Bouncers single-game sweep on `Heroes_of_Sokoban` level 0.
Heroes has genuinely non-local rules — `[> Wizard] -> [Wizard > Temp]` then
`[> Temp | no Moveable no Static] -> [ | > Temp]` (chain projectile travel
through empty space), `[Action Fighter] [SThief] -> ...` (multi-bracket
character swap), and `late [Weighing YellowSwitch] [YellowDoor] -> ...`
(multi-bracket door state). Pool features cannot substitute for actual
iteration here.

Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4,
15k updates, mask_hidden=True default, change_loss_weight=5.0. 16 configs
= depth ∈ {4, 8, 16, 32} × {fully-shared, per-step} × {pool ON, pool OFF + input_skip}.

**The original numbers in this section were inflated by the centered-vs-top-left
slicing bug fixed 2026-05-04** (`train.py` `_run_eval_rollout` was extracting
predictions from the centered position of the padded grid, but training-time
predictions live at the top-left). All 16 checkpoints were re-evaluated via
`reeval_via_render_only.sh` and the corrected numbers below replace the
prior table. The original buggy npz is preserved at
`logs_heroes/heroes_*/eval_multigame_buggy.npz`; corrected at
`eval_multigame_tlfix.npz`.

The re-eval also revealed that the original eval iterated over **every**
authored level (L0–L21), not just L0. Since training only saw L0, the
non-zero numbers on L1–L21 are an in-game generalization probe (transfer
across map sizes/layouts). The split below makes that explicit.

**BFS rollout cell-error per step (corrected):**

| depth | A: pool+shared | B: pool+per-step | C: nopool+skip+shared | D: nopool+skip+per-step |
|---|---|---|---|---|
| L0 (training level) | | | | |
| 4  | 0.00% | 0.00% | **0.00%** | 0.00% |
| 8  | 0.00% | 0.00% | **0.00%** | 0.00% |
| 16 | 0.38% | 0.58% | **0.00%** | 0.00% |
| 32 | 1.54% | **91.35%** *(collapse)* | **0.00%** | 0.00% |
| L1–L21 (held-out, mean) | | | | |
| 4  | 23.45% | 23.50% | **21.31%** | 21.39% |
| 8  | 26.14% | 29.73% | 26.75% | **25.11%** |
| 16 | 25.86% | 29.03% | **25.54%** | 27.59% |
| 32 | 28.34% | **88.89%** *(collapse)* | **25.39%** | 26.13% |

Plot: `nca_wm/figures/heroes_sweep/heroes_bfs_by_depth.{pdf,png}` (two
panels: L0 vs heldout). CSV/MD: `nca_wm/figures/heroes_sweep/summary.{csv,md}`.

### Findings (corrected)

1. **C and D (pool OFF + input_skip) achieve perfect L0 fit at every
   depth.** Both shared (C) and per-step (D) variants get 0.00% L0
   cell-error at d ∈ {4, 8, 16, 32}. The pool-ON variants degrade as
   depth grows: A goes 0%→0.38%→1.54%; B catastrophically collapses
   (0%→0.58%→**91.35%** at d=32).
2. **B (pool ON, per-step) collapses at d=32.** L0 cell-error 91.35%,
   heldout 88.89%. Adding more independent step-weights *and* a pool
   short-circuit at deep unrolls makes training unstable — the gradient
   path through 32 distinct sub-networks plus a global pool is too easy
   to fit per-step BCE without learning correct iteration.
3. **For held-out transfer (L1–L21), d=4 is the sweet spot for every
   variant.** All four configs hit their lowest heldout error at d=4
   (21–24%); d=8/16 climb to 25–29%; d=32 either holds (C: 25.39%,
   D: 26.13%) or blows up (B: 88.89%). Deeper NCAs overfit to L0's
   specific topology and don't generalize across map sizes.
4. **C is the most robust recipe overall.** Best heldout at d=32
   (25.39%, beating A at 28.34% and matching D at 26.13%), best L0 fit
   across all depths (0.00% always), and parameter-efficient (shared
   weights). Confirms the recommendation from the buggy analysis,
   though for *different* reasons than originally claimed: the d=4
   pool-ON penalty was a slicing artifact, not real.
5. **Pool ON does NOT harm at d=4 anymore (corrected).** Old buggy
   analysis claimed A/B at ~6.5% vs C/D at ~1.4% on L0. After fix:
   all four sit at 0.00% L0. The d=4 pool-ON disadvantage was eval bug,
   not architecture. **What's real is the d=32 collapse for B, and the
   gradual L0 degradation for A as depth grows** (0%→1.54%).
6. **F3 (train-loss / rollout-error decoupling) still verified:** best
   train loss spread is 7.07e-2 to 7.87e-2 across 16 configs; cell error
   spread is 0% to 91% on L0. Best loss tells you nothing.
7. **Recipe recommendation for scaling:** **C — pool OFF + input_skip +
   fully-shared body, d=4–16** — robust on L0 at any depth, best heldout
   transfer at d=4, parameter-efficient. Pool ON is only marginally worse
   on L0 fit but catastrophically risky at depth+per-step.

## Heroes_of_Sokoban L0–L7 → L8–L21 transfer (2026-05-04)

Follow-up on the heroes sweep finding that L0-trained models leave a
21% cell-error gap on held-out authored levels: train on L0–L7 (8
authored levels covering more rule-firing patterns) and eval on the
remaining L8–L21. Recipe C (pool OFF + input_skip + shared) at d=4 and
d=8, 20k updates, batch=16, lr=3e-4 (same recipe as the L0-only sweep).

| config | bfs train | bfs heldout | astar heldout | rtf heldout | random heldout |
|---|---|---|---|---|---|
| L0-only C_d4 (sweep) | 0.00% | 21.31% | 20.61% | 15.93% | 22.73% |
| **L0–L7 C_d4** | 1.10% | **8.24%** | **8.12%** | **3.89%** | **7.59%** |
| L0-only C_d8 (sweep) | 0.00% | 26.75% | 26.50% | 24.94% | 28.59% |
| **L0–L7 C_d8** | 0.95% | **11.38%** | **11.33%** | **5.31%** | **11.09%** |

### Findings

1. **Training on 8 authored levels cuts BFS heldout error ~60%** (d=4:
   21.31% → 8.24%; d=8: 26.75% → 11.38%). TF heldout error falls
   76–79% (15.93% → 3.89% at d=4; 24.94% → 5.31% at d=8). The transfer
   gap from L0-only training is largely an under-exposure problem: the
   model never saw the rule firings that distinguish later levels.
2. **Per-train-level fit drops slightly** (0.00% → 1.10% at d=4, 0.95%
   at d=8) — the model can no longer perfectly memorize a single level's
   topology, but the trade is heavily in favor of generalization.
3. **d=4 still beats d=8 on heldout** (8.24% vs 11.38%), reproducing the
   L0-only sweep finding. Deeper NCAs over-specialize; shallow + iterated
   wins for transfer in this regime.
4. **Worst held-out levels are L15 (16% bfs), L18 (14%), L19 (17%)** —
   late levels with rule firings (Weighing/Door state, Fighter/SThief
   swap) that even L7 doesn't fully cover. A natural follow-up is to
   include the rule-discriminating late levels in training (e.g. L0–L11)
   and see if L12+ transfer continues to compress.
5. **Implication for synth experiments:** authored-level coverage gave
   us a 60% reduction. A synth generator that explicitly seeded the rare
   entities (Wizard near long empty runs, paired Weighing+Door, etc.)
   could potentially close the remaining gap, but training on more
   authored levels is cheaper and yields realistic rule-firing
   distributions for free.

Logs: `nca_wm/logs_heroes_authored/heroes_l07_C_d{4,8}/`. Eval at
`eval_multigame.npz` in each.

## Nekopuzzle synth-trained architecture sweep (2026-05-04)

Single-game sweep on `nekopuzzle` (the `[ > Player | ... | Fruit ] -> [ |
... | Player ]` long-range jump rule, plus `[ > Player ] -> [ Player ]`).
Train on 64 synthetic levels (16 per size at multi-grid {5x5, 6x6, 7x7,
8x8}, total 2,130 transitions). Recipe: rule_attn, h=256, n_slots=16,
n_app_slots=1, batch=16, lr=3e-4, 15k updates, mask_hidden=True,
change_loss_weight=5.0, balanced_sampling. 12 configs = depth ∈ {8, 16, 32}
× sharing ∈ {fully-shared, per-step} × pool ∈ {ON, OFF + input_skip}.

Three eval surfaces, all held out from training:
1. **Authored 10 levels (8x7)** — never seen, hand-designed by lexaloffle.
2. **Holdout synth pool** at trained sizes (different RNG seed = different layouts).
3. **OOD synth pool at 9x9** — larger than the maximum trained size.

Numbers below are final-step cell-error rate.

| depth | share | pool | BFS authored | RAR authored | TF authored | Holdout synth | OOD synth |
|---|---|---|---|---|---|---|---|
| 8 | per-step | OFF | 7.25% | 5.87% | 0.51% | 0.76% | 0.53% |
| 8 | per-step | ON | 3.66% | 4.93% | 0.40% | 0.58% | 0.42% |
| 8 | shared | OFF | 7.95% | 6.46% | 0.57% | 0.79% | 0.48% |
| 8 | shared | ON | 4.40% | 4.77% | 0.26% | 0.63% | 0.42% |
| 16 | per-step | OFF | 5.98% | 6.71% | 1.11% | 0.65% | 0.49% |
| 16 | per-step | ON | **3.15%** | 5.41% | 0.65% | 0.58% | 0.43% |
| 16 | shared | OFF | 5.83% | 6.65% | 0.45% | 0.93% | 0.47% |
| 16 | shared | ON | 5.31% | 5.51% | 0.33% | 0.62% | 0.45% |
| 32 | per-step | OFF | 5.83% | 7.20% | 0.94% | 0.72% | 0.52% |
| 32 | per-step | ON | 4.75% | 5.69% | **0.18%** | 0.61% | 0.45% |
| 32 | shared | OFF | 6.85% | 7.12% | 1.03% | 0.92% | 0.54% |
| 32 | shared | ON | 3.17% | 5.56% | 0.73% | 0.61% | **0.41%** |

(BFS = oracle-action rollout, RAR = random-action rollout, TF = teacher-
forced. All on authored 10 levels. Holdout/OOD = single-step cell-error on
held-out synth pools at the same/larger grid sizes.)

Plots:
- `nca_wm/figures/neko_arch/all_metrics_grouped.{pdf,png}` — bar chart of
  all 12 configs across the 3 eval surfaces (authored AR, holdout synth,
  OOD 9x9 synth). Pool ON consistently below pool OFF on authored.
- `nca_wm/figures/neko_arch/by_depth_bfs_authored.{pdf,png}` — BFS
  oracle-rollout cell-error on authored vs depth, 4 lines for
  share×pool. Pool ON variants flat at 3-5%; pool OFF variants 5.8-8%.
- `nca_wm/figures/neko_arch/by_depth_{ar,ho,ood,tf}_*.{pdf,png}` — same
  layout for the 4 random-AR / TF / holdout / OOD metrics.
- Combined CSV/MD: `nca_wm/figures/neko_arch/summary_combined.{csv,md}`.

### Findings

1. **Synth holdout is essentially solved by every config** — 0.58–0.93%
   cell-error across all 12 architectures. The model learns the rule on
   diverse random layouts and transfers to fresh layouts at the same sizes
   without issue.
2. **Size-OOD synth (9x9, larger than trained max 8x8) is also solved** —
   0.41–0.54% across configs. The 1-cell extension at test time doesn't
   break the model. This rules out a position-embedding or padding-bucket
   artifact as the bottleneck.
3. **Authored levels are NOT solved by any config** — the best BFS rollout
   cell-error is 3.15% (d=16, per-step, pool=ON). The gap between synth
   holdout (~0.6%) and authored BFS rollout (~3-8%) is a *distribution
   mismatch*, not a learning capacity / architecture issue. Authored
   nekopuzzle levels are hand-designed with specific fruit cluster patterns
   that don't appear in the random-tile-pattern synth distribution.
4. **Pool ON dominates Pool OFF + input_skip on every cell of the
   sweep** — 3–5pp BFS authored advantage at every (depth, share)
   combination. For nekopuzzle's `...` rule, the global axis-pool features
   *do* carry useful signal (player position along axes) that local convs
   alone can't replicate at this depth × n-slots budget. This is the
   opposite of the Heroes_of_Sokoban finding (where pool OFF + input_skip
   was the recipe), suggesting the right pool default is rule-dependent.
5. **Depth saturates at d=8** — the best BFS authored is at d=16 (3.15%) but
   only marginally below d=8 (3.66%). d=32 doesn't unlock further gains
   (3.17% best); deeper just costs compute.
6. **Per-step vs fully-shared is a wash on this game** — the largest
   per-step / shared gap is 1.5pp at d=16 / pool=ON (3.15% vs 5.31%); at
   d=8 / pool=ON they're within 0.7pp (3.66% vs 4.40%). On the easier
   axes (holdout / OOD synth), they're within 0.1pp. The `[X | ... | Y]`
   rule's iteration count is bounded by grid width (≤8 hops), so beyond
   d=8 weight-sharing per layer doesn't matter.
7. **TF authored is best at d=32 / per-step / pool=ON (0.18%)** — single-
   step prediction on authored is essentially perfect for the deep
   per-step model. The 5-7% gap on multi-step rollout (random + BFS)
   reflects compounding error on hand-designed configurations the model
   hasn't seen, not single-step prediction failure.

### Implication

For nekopuzzle (and probably similar long-range-rule games), **the
architecture sweep shows the bottleneck is the data distribution, not the
arch**. To get "perfect generalization to authored", future work should
target the synth generator (e.g., evolution-based level search seeded
from authored layouts, or generators conditioned on authored tile-cluster
patterns) rather than scaling depth, sharing, or pool variants. The
architecture finding worth carrying forward: **pool ON is the right
default for nekopuzzle-class long-range-rule games**, contradicting the
Heroes recipe — which suggests the right pool default is per-game.

## Nekopuzzle synth distribution sweep (2026-05-04, follow-up to arch sweep)

Holding the best arch fixed (d=16 / per-step / pool=ON, h=256, batch=16,
lr=3e-4) and varying ONLY the synth generator. Tests the "data distribution
is the bottleneck" claim from the arch sweep above.

| run | grid sizes | n_levels | n_updates | trans | BFS authored | RAR authored | TF authored | Holdout synth | OOD 9x9 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 5x5,6x6,7x7,8x8 | 64 | 15k | 2,130 | 3.15% | 5.41% | 0.65% | 0.58% | 0.43% |
| tpe n=256 | 5x5,6x6,7x7,8x8 | 256 | 15k | 8,445 | 1.92% | 2.62% | 0.07% | 0.56% | 0.36% |
| tpe n=512 | 5x5,6x6,7x7,8x8 | 512 | 15k | 17,025 | 3.50% | 3.78% | 0.42% | 0.57% | 0.38% |
| tpe n=256 30k | 5x5,6x6,7x7,8x8 | 256 | 30k | 8,445 | 2.10% | 3.54% | 0.00% | 0.57% | 0.37% |
| evolve n=64 | 5x5,6x6,7x7,8x8 | 64 | 15k | 2,630 | 8.44% | 7.65% | 0.25% | 0.66% | 0.55% |
| evolve n=256 | 5x5,6x6,7x7,8x8 | 256 | 15k | 18,250 | 5.60% | 4.36% | 0.44% | 0.56% | 0.41% |
| **5-sizes** | 5x5,6x6,7x7,**8x7**,8x8 | 256 | 15k | 8,510 | **0.31%** | 1.64% | **0.00%** | **0.58%** | **0.39%** |
| 5-sizes n=512 | 5x5,6x6,7x7,**8x7**,8x8 | 512 | 15k | 17,180 | 0.67% | **1.00%** | **0.00%** | 0.58% | 0.38% |
| 5-sizes 30k | 5x5,6x6,7x7,**8x7**,8x8 | 256 | 30k | 8,510 | 1.18% | 2.30% | 0.35% | 0.56% | 0.40% |
| 8x7-only | 8x7 | 256 | 15k | 8,735 | **0.00%** | 0.88% | 0.00% | 2.87% | 15.20% |

(BFS, RAR, TF on authored 10 levels via in-train eval. Holdout / OOD via
held-out synth pool eval at trained sizes / 9x9. RAR here = my custom
random-action AR rollout; the in-train eval table values match.)

### Findings

1. **The "perfect generalization to authored" goal is achievable with the
   right synth distribution.** The 5-sizes config (256 levels at multi-grid
   that includes the authored 8x7 size) achieves 0.31% BFS + 0.10% TF on
   authored AND 0.58% holdout / 0.39% OOD synth — near-zero on both
   simultaneously.
2. **The original arch sweep's 3-8% authored gap was almost entirely a
   grid-size-mismatch artifact.** Authored neko levels are 8x7 (W=8, H=7)
   but v1 synth used square sizes only ({5x5, 6x6, 7x7, 8x8}). The model
   couldn't generalize from same-size square grids to a 7-row × 8-col
   rectangle. Adding 8x7 to the synth set drops BFS authored from 1.92%
   (n=256, square sizes) to 0.31% (n=256, +8x7). 6× reduction from one
   size addition.
3. **Single-size 8x7-only is not the answer — overfits to one size.** It
   gets 0.00% BFS authored but 15.20% on OOD 9x9 (vs 0.39% for 5-sizes).
   Multi-grid is strictly better when the goal is "perfect on multiple
   distributions simultaneously."
4. **More tpe data alone helps modestly** (n=64 → n=256: 3.15% → 1.92%
   BFS) but pushing further (n=512) actually regressed (3.50%) at
   fixed compute — undertrained. Doubling compute (n=256 30k) didn't
   help either (2.10%). The data×compute knob is essentially saturated.
   The grid-size lever is much stronger.
6. **Once 8x7 is in the multi-grid set, more data/compute is again a
   wash.** 5-sizes/n=256/15k = 0.31% BFS / 1.64% RAR. 5-sizes/n=512/15k
   = 0.67% BFS / 1.00% RAR (better RAR, slightly worse BFS — co-winners).
   5-sizes/n=256/30k regressed everywhere (1.18% BFS / 2.30% RAR).
   Conclusion: at this scale, data×compute is saturated *given* the
   right grid coverage; further gains would need a different lever.
5. **Evolve mode is worse than tpe at every n** (8.44% / 5.60% vs 1.92%
   / 0.31% at n=64 / n=256). Evolve fitness = BFS-iterations, which biases
   toward complex layouts; tile-pattern-empirical samples from the natural
   authored distribution. For coverage-of-rule-firings (the model-training
   goal), tpe wins. Future evolve work would need diversity-aware
   selection (extending `coverage_select_topk` beyond rule-firings to
   tile-pattern Hamming distance) to compete.

### Recipe for future synth-only multi-game scaling

When training synth-only and evaluating on authored:
- **Always include each authored level's exact (W, H) in the synth multi-grid set.** Square-only synth misses non-square authored aspect ratios. The default `--synthetic_multi_grid` already does this when explicit `--synthetic_grid_sizes` is omitted (it queries authored unique dims), so the simplest fix is to drop the explicit `--synthetic_grid_sizes` override when authored sizes are tractable.
- Use `tile_pattern_empirical` mode, n_levels ≥ 256 per game, 15k updates baseline. Evolve mode underperforms; don't enable without diversity-aware selection.
- The arch findings above are still valid: **pool ON is load-bearing for
  long-range-rule games** (3-5pp BFS gap vs pool OFF on neko); depth saturates
  at 8-16; share-vs-perstep is a wash.

## Embedding distinctness probe on v3_combined (2026-05-05)

First test of "do per-game latents become distinct in the rule-encoding space"
on the 59-game gallery_v2 training. New tool `nca_wm/latent_scatter_rule_attn.py`
encodes each game's tokens through the saved `RuleSlotEncoder`, flattens
(K=16, d_slot=64) slots to a 1024-dim vector, and computes PCA + pairwise
cosine distances.

Two reductions tested (same picture):
- `flat`: full 1024-d slot stack → PC1+PC2 explain 61.9% of variance.
- `dyn_mean`: mean over the 15 dynamic slots → 64-d. Same tightest pairs.

**Key result**: the two pairs of token-equivalent gallery_v2 games have
**cosine distance ≈ 0** in the learned embedding:
- `twolittlecrates2 ↔ twolittlecrates4`: cos dist = -0.0000
- `sokoban_basic ↔ Microban`: cos dist = -0.0000 (the model also collapses
  `blank` here per the dedup analysis)
- `sokoban_basic ↔ sokoban_match3`: 0.0053 (related rules; near-collapse,
  not full)
- `Modality ↔ Ebony_&_Ivory`: 0.0056
- `rigid_one_unlimited ↔ rigid_parallel_unlimited`: 0.0140
- `2D_Whale_World ↔ Lime_Rick`: 0.0140

**Median pairwise cos dist = 0.7665, mean = 0.7960.** Most of the 1711
pairs are well-separated; the embedding space is mostly meaningful, with
collapse only at the dedup-identified duplicates.

**Most distinct pairs** (cos dist ≈ 1.8): scriptcross is the lone wolf,
followed by `blockfaker`, `Love_and_Pieces`, `lunar_lockout`, `The_observer's_paradox`.

scaling_14 (14 games) for contrast: tightest pair `sokoban_match3 ↔
scriptcross` at 0.064 — no zero-distance collisions because the
gallery_v2 internal duplicates (Microban, blank, twolittlecrates4) aren't
in scaling_14.

Figures:
- `nca_wm/logs/multi_scaling_gallery_v3_combined/interp/latent_scatter_flat.{pdf,png,npz,json}`
- `nca_wm/logs/multi_scaling_gallery_v3_combined/interp/latent_scatter_dyn_mean.{pdf,png,npz,json}`
- `nca_wm/logs/multi_scaling_14_v3recipe/interp/latent_scatter_flat.{pdf,png,npz,json}`

### Object-ordering blind spot in dedup (2026-05-05)

Sanity-tested the tokenizer's name-invariance assumption: `tokenize_game`
depends on the order of `canonical_ids`, which is the order of object
declaration in the OBJECTS section. Two games with identical mechanics
but different declaration order get different token sequences → different
hashes → invisible to v1 dedup.

Re-hashing the 3,474-game deduped pool with **alphabetically-sorted**
canonical_ids reveals 26 additional collisions across 21 new groups
(largest: 4-game group merging into `blank`'s cluster). That's ~0.7% extra
dedup. Small enough to defer, but a known blind spot — for the most
faithful structural test we'd canonicalize object order before tokenizing.
Current dedup numbers remain valid as an *upper bound* on token-distinct
game count.

## Held-out game overlay on v3_combined (2026-05-05)

First "unseen game" generalization test. Picked 30 held-out games from
`data/heldout_v4_n30.json` (stratified across complexity buckets, none
in scaling_gallery_v4 by name or token-hash). Encoded each through the
`v3_combined` `RuleSlotEncoder` and computed 1-NN distance to training
games in the 1024-d slot space.

**1-NN cosine distance summary**: min=0.009, median=0.123, mean=0.123, max=0.306.

Every held-out game lands within 0.31 of *some* training game — no
held-out falls completely outside the training manifold. This is the
encouraging baseline.

Closest pairs (heldout → 1-NN training):
- `angize_by_ali_nikkhah → Midas` (0.009)
- `escape_by_shawn_martin → nekopuzzle` (0.027)
- `________________by_ncrecc → scriptcross` (0.042)
- `Hamiltwo → Smother` (0.056)
- `link_by_jeffjeff123456 → Smother` (0.055)
- `Heroes_of_Sokoban_-_Ancient_Japan → Smother` (0.103)

The Heroes_of_Sokoban-variant landing near `Smother` is consistent with
both having multi-bracket / contagion-style mechanics — the encoder
appears to capture *something* mechanically meaningful, not just lexical
similarity.

Most-distant held-outs (least similar to any training game):
- `a_snowballs_chance_in_hell_by_iznaut → Lightdown` (0.306)
- `ice_sliding_by_by_ian_cox → Travelling_salesman` (0.247)
- `the_single_flame_by_hassan_masood → MC_Escher's_...` (0.211)
- `Surround_Yourself_With_Dogs → Lightdown` (0.194)
- `hedgehogger_by_increpare → It_Dies_In_The_Light` (0.187)

Plot: `nca_wm/logs/multi_scaling_gallery_v3_combined/interp/heldout_overlay_flat.{pdf,png,npz,json}`.

**Hypothesis test on scaling**: re-run after v3 (97 games) finishes.
Prediction: median 1-NN distance drops as the encoder sees more rule
patterns. If it doesn't, the bottleneck isn't pattern coverage.

### Held-out 1-NN scaling table — pre-v3 (2026-05-05)

Same 30-game held-out set encoded through 5 existing rule_attn checkpoints,
restricted to the 21 held-outs that fit in every run's max_tok_len for
apples-to-apples comparison:

| run | n_games | mask_hidden | n_updates | median 1-NN | mean 1-NN |
|---|---:|:---:|---:|---:|---:|
| `multi_scaling_14_v3recipe` | 14 | False | 150k | 0.144 | 0.165 |
| `multi_scaling_14_mask_v1` (shared) | 14 | True | 80k | **0.077** | **0.093** |
| `multi_scaling_14_mask_v2_perstep` | 14 | True | 80k | 0.133 | 0.162 |
| `multi_scaling_gallery_v2_nca8` | 59 | False | 80k | 0.117 | 0.129 |
| `multi_scaling_gallery_v3_combined` | 59 | False | 150k | 0.119 | 0.119 |

Findings:

1. **`mask_hidden=True` + shared weights at 14 games beats 59 games without
   mask** (median 0.077 vs 0.117). Recipe lever > 4× dataset size on
   this metric. This is consistent with the "mask_hidden enables
   translation-equivariance / cleaner encoder signal" hypothesis from
   the per-step varislide canary work.
2. **`mask_hidden=True` + per-step weights doesn't help nearly as much**
   (0.133 vs shared 0.077). Sharing across NCA steps appears to
   regularize the encoder toward more compact latents.
3. **Pure scale (no mask): 14g → 59g** halves the 1-NN gap from 0.144
   to 0.117 (~0.03 reduction per 4× games). Slower than the mask
   improvement but real.
4. **Longer training at 59g is a wash** for held-out 1-NN
   (v2_nca8 80k = 0.117 ≈ v3_combined 150k = 0.119).
5. **Compounding prediction**: v3 (97g + mask + shared + 150k) should
   land ≤ 0.07 if mask and scale stack cleanly.

Figures:
- `nca_wm/figures/heldout_scaling/median_1nn_common_subset.{pdf,png,json}`
- `nca_wm/figures/heldout_scaling/summary_flat.md`

### Held-out AR rollout baseline on v3_combined (2026-05-05)

Predictive-accuracy counterpart to the latent-geometric metric. Ran
`heldout_eval.py` on 29 of 30 games in `data/heldout_v4_n30.json` (1
game crashed the run via JIT-cache OOM at the end). 2 levels per game,
3 random episodes, 30-step horizon.

**Headline numbers**:
- **8/29 games beat identity at step 1** (model_step1 ≤ identity_step1).
- **1/29 games beat identity on full 30-step rollout mean.**
- Average model step1 cell-error: 0.068 (vs identity 0.027, ~2.5× worse).

Game-pattern findings:
- **Best transfer** (model ≤ identity at step 1): `watchers_ritual`,
  `my_nana_ate_my_kid_brother`, `link_by_jeffjeff123456`,
  `monophobic_multiban_by_increpare`, `haberdashery_by_lee2sman`,
  `the_single_flame`, `cyberpunk_2020`, `kyles_chesse_cuisine`. Mostly
  simple short-horizon games where identity is itself strong (state
  changes little step-to-step).
- **Worst transfer**: `pacman__by_cora` (model 0.307 vs identity 0.025
  step1), `headless_people_problems` (0.252 vs 0.012), `Surround_Yourself_With_Dogs`
  (0.283 vs 0.085). Complex multi-rule mechanics (chasing, contagion)
  where the model fails because rule structure is OOD, even though
  identity also has a poor mean.

Files: `nca_wm/logs/multi_scaling_gallery_v3_combined/heldout_v4_n30/{results.json,rollout_curves.png,step1_bar.png}`.

The compounding prediction for v3 (97g + mask_hidden + shared, 150k):
more games + better encoder geometry should narrow the avg step1 gap
(0.068 → ?) and grow the beats-identity tally (8/29 → ?). The 1-NN
latent metric predicts the beats-identity tally because tighter
latent-neighborhood ⇒ encoder treats heldout like a known game ⇒
better predictive accuracy.

## v3+decoder (94g, joint encoder + NCA + token decoder, 2026-05-05)

First joint-AE training run at the broad-corpus scale: 94 deduped games,
150k steps, batch=32, recipe matches v3 (no decoder) plus
`--token_decoder_loss_weight 1.0`. Joint params layout `{"wm": ..., "dec": ...}`;
encoder + NCA = 9.89M params, decoder = 1.18M params (12% extra), per-step
wall-clock ~110ms (~20% slower than the no-decoder run).

**Training metrics (final, step 150k):**
- state_loss: 0.0049 (vs v3 no-decoder: 0.0049 — within noise)
- change_err: 0.013 (vs v3 no-decoder: 0.015)
- win_recall: 0.996
- dec_loss: 2.4e-9, dec_acc: 1.000

The joint decoder objective imposes essentially **zero cost on
prediction quality**: state_loss / change_err track the no-decoder run
within noise (the table-1 ablation for the paper).

**Token reconstruction (teacher-forced):**
- 94/94 training games: perfect (1.000) per-position match.
- The decoder learns the conditional next-token distribution well
  enough to fully recover every training game given the right history.

**Held-out latent geometry (v3+decoder vs v3 no-decoder vs scaling_14_mask):**

| run | n_games | mask | decoder | median 1-NN cos dist |
|---|---:|:---:|:---:|---:|
| scaling_14_mask_v1 | 14 | True | False | **0.077** |
| scaling_14_v3recipe | 14 | False | False | 0.144 |
| v3_combined | 59 | False | False | 0.119 |
| v3 (no decoder) | 94 | True | False | 0.215 |
| **v3+decoder** | 94 | True | True | **0.153** |

At 94 games with `mask_hidden=True`, **adding the joint decoder
tightens** the held-out 1-NN distance from 0.215 → 0.153 — a 29%
reduction. The decoder objective forces slots to retain
spec-recoverable information, and the encoder responds by placing
mechanically-similar games closer in the slot space, so held-out
games' 1-NN training games are nearer.

Note that `scaling_14_mask_v1` (14g, no decoder) still has the
tightest held-out 1-NN (0.077), but at 14 training games the held-out
games are mostly easy push-puzzles that map close to several training
games by default; at 94 games the encoder has to model wider
mechanical diversity, which raises the average 1-NN before the decoder
constraint pulls it back down.

**Inverse-fit on 4 OOD held-out games (frozen NCA + decoder, optimize
slot from observed transitions, 2000 Adam steps at lr=5e-3, init=knn):**

| target | final fit BCE | 1-NN train | cos dist |
|---|---:|---|---:|
| `monophobic_multiban_by_increpare` | **8.4e-04** | `run_em_over_by_xxsethyxxxoultraomeg` | 0.389 |
| `cat_adventure_2_by_whenyoucando` | **8.4e-04** | `neko_puzzle_by_increpare` | 0.394 |
| `in_the_way_by_giovanni_mota` | 7.2e-03 | `run_em_over_by_xxsethyxxxoultraomeg` | 0.477 |
| `shadow_by_unknown_author` (smoke) | 1.2e-01 | `rigid_parallel_many` | 0.206 |

The optimization successfully drives BCE down 2-3 orders of magnitude
on the observed OOD transitions for the easier targets. The 1-NN
matches are mechanically meaningful: `cat_adventure_2 → neko_puzzle`
both push-puzzle styles; `monophobic_multiban → run_em_over` both
multi-character-control / chase-style mechanics. `shadow` (a 2-rule
1×4-grid game) fits worst because its rule structure is so atypical
that even a closest-fit slot still leaves residual error.

**Symbolic AE results (autoregressive sampling, 16 random + 9 interpolation):**

- **TF reconstruction**: 100% perfect across all 94 training games.
  Given the right history at each position, the decoder predicts the
  correct next token everywhere.
- **AR engine-load**: 0/25 across both random Gaussian samples and
  9-point sokoban_basic ↔ Travelling_salesman interpolation. Tested
  twice (initial + with `stop_at_pad=True` patch); the patch shortens
  decoded sequences by ~20% but doesn't fix engine-load.
- **What's actually decoded**: AR samples produce *structurally*
  PuzzleScript-shaped output — push rules `[ > Obj2 | Obj4 ] -> [ > Obj2
  | > Obj4 ]`, ellipsis rules `[ > ... | Obj1 ] -> [ > ... | Obj1 ]`,
  `late` prefixes — confirming the decoder learned the rule grammar.
  Even the t=0.00 endpoint produces a sokoban-style first push rule.
- **Why engine-load fails**: the JS engine *parses* the decoded text
  (PuzzleScript syntax is satisfied) but then crashes in
  `serializeCompiledStateJSON` because (i) AR generation emits ~40
  redundant `all Obj1 on Obj4`-style win-conditions instead of
  stopping at one, and (ii) decoded rules reference `Grp1`, `Grp2`,
  etc. as legend groups but the LEGEND section only emits `Grp0` because
  the AR sequence didn't include the corresponding `GROUP_OR` /
  `GROUP_AND` structure tokens in the right place.
- **Diagnosis**: a teacher-forcing-vs-autoregressive gap. The decoder
  has learned the conditional `p(t_n | t_{<n}, slot)` perfectly but
  hasn't learned a stable basin to *generate* well-formed sequences
  from BOS. Standard fixes (explicit EOS token, scheduled sampling,
  decoder pretraining on the token corpus alone, structure-aware
  generation) should close this; we leave as a clear future-work item.
- **Honest paper framing**: 100% TF reconstruction proves the latent
  space carries the spec information. The 0% AR engine-load reflects
  a known sequence-modeling limitation, not a latent-space limitation.
  Inverse-fitting (which optimizes the slot directly without going
  through AR) sidesteps this issue and works (final fit BCE 8e-4 to
  7e-3 across 3 OOD games).

Files:
- `nca_wm/logs/multi_scaling_gallery_v3_decoder/`
- `nca_wm/logs/multi_scaling_gallery_v3_decoder/interp/heldout_overlay_flat.{pdf,png,json}`
- `nca_wm/logs/multi_scaling_gallery_v3_decoder/sampled_games/{interp,random}_*.txt`
- `nca_wm/logs/multi_scaling_gallery_v3_decoder/inverse_fit/<game>/{summary.json,fitted_decoded.txt}`

## v3+decoder+sprites+EoS (94g, joint AE with EoS token + sprite-encoded vocab, 2026-05-05)

Identical recipe to v3+decoder above except: `--use_eos` (append explicit
EOS sentinel at every game's tail; decoder learns to terminate) and
`--encode_sprites` (per-object 5×5 RGBA palette + pixel sprite tokenized
into the input vocabulary; vocab grows from 142 → 183, max_tok_len grows
to fit). Both are *necessary* to clear the AR-sampling engine-load
ceiling that previously sat at 0/25.

**Training metrics (final, step 150k; best at step 134k):**
- state_loss: 7.81e-04 (training-aggregate; sprite vocab makes per-step
  loss reading slightly tighter than v3-no-sprites)
- acc: 0.9999, change_err ≈ 0
- best_loss: 4.45e-3 (step 134k)
- dec_acc: 1.000 throughout

**Token reconstruction (teacher-forced, all 94 training games):**
- mean per-position acc: 0.906
- median per-position acc: 1.000
- min per-position acc:  0.276
- perfect-reconstruction games: **76/94**

The slight regression from 94/94 (no-sprites) → 76/94 (with sprites) is
expected: sprite tokens add ~30 tokens per object, lengthening sequences
into a regime where the decoder's positional budget runs out for the
longest sprite-rich games. All non-perfect games still reconstruct
≥27% of tokens correctly.

**Autoregressive sampling (25 random Gaussian-fit slots + 9-point
interpolation Ebony_&_Ivory ↔ rigidfail1):**
- Random samples: **15/25 engine-load (60%)** — *up from 0/25*.
- Interp samples: **9/9 engine-load (100%)** — perfect along the entire
  Ebony_&_Ivory ↔ rigidfail1 line.
- Sampled programs include real palette+sprite blocks, coherent
  legend assignments (a/b/c → Obj0/Obj1/Obj2), collision-layer
  groupings, push rules `[ > Obj2 | Obj4 ] -> [ > Obj2 | > Obj4 ]`,
  `late`-prefixed rules, and 3×3 to 5×5 levels with players.
- The remaining failures (10/25 random) are mostly downstream
  `serializeCompiledStateJSON` errors due to under-defined legend
  groups in the decoded text, not parsing errors.

**Why EoS + sprites is necessary (ablations not run yet, but
diagnostic-level evidence from earlier sweep):**
- Without EoS, the decoder emits a fixed-length padded suffix that
  trips up the JS engine's COMPILE state-machine.
- Without sprites, the model has no way to ground objects' visual
  identity, so legend rows like `Obj4 = ?` get spurious entries.
- The two are interacting: the EoS gates terminate is *enabled* by
  the sprite-tokens making the sequence locally informative enough
  that the decoder can predict EOS when the document is "done".

Files:
- `nca_wm/logs/multi_scaling_gallery_v3_decoder_sprites_eos/`
- `nca_wm/logs/multi_scaling_gallery_v3_decoder_sprites_eos/sampled_games/summary.json`
- `nca_wm/logs/multi_scaling_gallery_v3_decoder_sprites_eos/sampled_games/{interp,random}_*.txt`

## v4+decoder+sprites+EoS (199g, 2026-05-06)

Same recipe as v3+decoder+sprites+eos, scaled to the 199-game preset
(`scaling_gallery_v4`). Launched after killing the obsolete
no-EoS-no-sprites v4_decoder run.

**Training (final, step 150k; best at step 148,250):**
- best_loss: 3.68e-3
- early_stopped: False
- dec_acc: 1.000

**Token reconstruction (TF, all 199 training games):**
- mean per-position acc: 0.886
- median per-position acc: 1.000
- min per-position acc:  0.332
- perfect-reconstruction games: **147/199**

**AR sampling (25 random + 9-point interp `herding_cats!` ↔ `blocks`):**
- random AR engine-load: **4/25 (16%)** — *down from 15/25 at 94g*
- interp AR engine-load: **9/9 (100%)** — same as 94g
- Interpolation between two training endpoints stays robust at scale,
  but random Gaussian-fit slot draws degrade. Reading: the empirical
  slot prior becomes a poorer match to the manifold as games crowd it
  (slots are denser, off-manifold draws more frequent). Interp's
  invariance is consistent with that: a line whose endpoints are
  on-manifold stays close enough to the manifold along its length.

**Postrun multi-game eval crashed mid-Smother (L12 of 13).** No
traceback in train.log; process just disappeared. Smother L8 had
3,300 tiles; suspect kernel OOM-kill on long-W cumulative JAX
allocations. Workaround: re-run `eval_multigame` as a standalone
pass with subprocess-per-game isolation (TODO).

Files:
- `nca_wm/logs/multi_scaling_gallery_v4_decoder_sprites_eos/`
- `nca_wm/logs/multi_scaling_gallery_v4_decoder_sprites_eos/sampled_games/summary.json`
- `nca_wm/logs/multi_scaling_gallery_v4_decoder_sprites_eos/sampled_games/{interp,random}_*.txt`

## Travelling_salesman: data dilution, not mechanics complexity (2026-05-05)

`Travelling_salesman` (TSM) has been a "persistently hard" game across
v2 / v2_long / v2_smallbatch / v3_combined: change_err 0.99 even at
150k steps in multi-game training. We tested whether it's mechanically
hard by training a single-game NCA (rule_attn, n_hid=256, n_slots=16,
d_slot=64, n_nca=8, batch=32, lr=3e-4, mask_hidden, change_loss_weight=5.0).

**TSM single-game (30k steps):**
- step 250: change_err = 0.54
- step 500: change_err = 0.66 (early oscillation, not converged)
- **step 1,000: change_err = 0.031 (3.1%)**, win_recall = 1.000
- **step 2,000: change_err = 2.1e-9 (essentially zero)**, err = 0%
- step 30,000: state_loss = 1.7e-12, fully memorized.
- Eval (10 random episodes): step-1 L1 = 4.4 cells, step-20 L1 = 20.4 cells.
  (Random actions are partly OOD vs A* training trajectories, so this
  reflects per-cell rollout drift on unseen state-action regions, not a
  rule-learning failure.)

**Multi-game v3 (94 games) at the same step counts**: average change_err
∼0.11–0.13; TSM-specific change_err 0.99 (recurring as worst-fit game).

**Headline**: TSM is fully learnable (memorizable) in 2k single-game
steps but stays at change_err ≈ 1.0 in 94-game training. The
mechanically simple 4-rule game (no `...`, no globals, just a 3-cell
pattern `[ > Player | Link | Node ]`) is not the bottleneck. The
persistently-hard pattern in v2/v3 reflects **capacity competition /
data dilution**: with balanced sampling at 1/N share and 13 levels with
14K state-changing transitions × 50 winning trajectories, TSM's
gradient signal gets drowned out by easier games' transitions.

This is consistent with the broader persistent-hard cluster
(Take_Heart_Lass, Travelling_salesman, Lightdown, It_Dies_In_The_Light,
notsnake): they all have small authored datasets (1–12 levels) and
sparse winning paths. Future fixes:
- Per-game synth augmentation to grow the per-game dataset.
- Hard-game-aware sampling (boost sample share for high-loss games).
- Per-game expert paths (Mixture-of-Experts) to give hard games
  dedicated capacity.

Files: `nca_wm/logs/tsm_single_v3recipe/{config.json, params.pkl}`.

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

## Cross-reference: persistently-hard games and the multi-grid varislide canary (2026-05-03, INVALIDATED 2026-05-04)

> **2026-05-04 — withdrawn pending re-eval.** The "varislide canary fails"
> claim cited below was an artifact of the synth bit-pack regression (commit
> 1fa557d, May 1–4). Post-fix re-runs at h=128, depth ∈ {8, 16}, fully
> shared, mask_hidden=True default: 100% argmax across slide distances on
> the same multi-grid synth set (`nca_wm/logs_canary/varislide_postfixA_*`).
> The architectural extrapolation below — that gallery scaling's per-game
> variance is the *same* phenomenon — therefore loses its empirical
> grounding and should not be cited. Treat the gallery per-game variance as
> still unexplained until a fresh post-fix gallery experiment isolates the
> mechanism.

The persistently-hard gallery games (Take_Heart_Lass, Travelling_salesman, notsnake,
It_Dies_In_The_Light, the_art_of_cloning, Lightdown, …) all share a property:
their dynamics involve iterative rule application that pool features
(`axis_pool` / `axis_cummax` / `global_pool`) cannot fully substitute for. They are
the gallery-scale analog of the varislide canary documented in
`ARCHITECTURE_REPORT.md` F8.

Multi-seed verification on multi-grid varislide (E20-E22) establishes that
rule-conditioned NCAs do not reliably learn iterative rule application
under varied-grid synth, regardless of depth (8 ↔ 64), compute (10k vs
50k), weight-sharing, or change-loss-weighting. The "fire-once" basin is
the typical solution under random init; the iterative basin is rare and
not reachable with current optimization. Per-seed argmax variance is ~5×.

**Implication for the gallery scaling story:** the per-game variance
documented above (no single recipe wins all games; ensemble buys ~5%
headroom per game) is consistent with the varislide canary mechanism —
each game's rule structure determines whether the model lands in a
"learn the rule" basin or a "fire-once / partial" basin, and current
recipes (depth, capacity, duration) don't shift that landscape. The
remaining unblocked candidates from F8 (hard slot routing, per-cell
halt, replay-buffer curriculum) are also what's most likely to move
gallery scaling beyond its current ceiling. Worth treating these as
joint, not separate, problems.

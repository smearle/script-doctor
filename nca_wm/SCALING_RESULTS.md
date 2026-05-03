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

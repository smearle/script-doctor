# Expanded engine comparison, 2026-10-07

This round adds 16 games to the existing 16-game suite. Selection deliberately
emphasizes larger rule sets, many object channels, large boards, and multi-row
patterns. It is a stress sample, not a representative sample of all PuzzleScript
games. Level zero is used throughout.

The [32-game comparison](../../../../paper/figures/complex_throughput_20261007/engine_crossover.pdf),
[new-game curves](../../../../paper/figures/complex_throughput_20261007/complex_game_throughput.pdf), and
[rule-count/board-size comparison](../../../../paper/figures/complex_throughput_20261007/complexity_factors.pdf)
include the retained optimizations. The original sixteen games have their own
curve sheet in the same directory.

All jobs have ended. The figures contain 712 measured batch configurations
and 3,560 timed calls, including the three-point experimental pruning pilot.
C++ has 31 saturated curves and one resource-limited curve. The selected JAX
configuration has timings for 30 games: 23 saturated curves and seven partial
curves. Crate Assembler has no ordinary JAX timing, and Memories of Castlemouse
is excluded for the pre-existing correctness discrepancy described below.
The final raw-data audit recomputes every median/IQR and verifies all plot input
hashes. Sixteen benchmark/reporting tests also pass.

## Where C++ overtakes JAX

These examples reached the empirical plateau/regression stopping criterion in
both engines. Rates are thousands of environment steps per second, selecting
each engine's best measured batch separately. C++ uses the i9-9980XE with up to
32 threads; JAX uses one H200.

| Game | Source rules | C++ ksteps/s | JAX ksteps/s | C++ / JAX |
| --- | ---: | ---: | ---: | ---: |
| Atlas Shrank | 44 | 176.2 | 124.7* | 1.41× |
| Beam Islands | 84 | 91.9 | 43.0* | 2.14× |
| Boxes & Balloons | 50 | 385.7 | 284.0 | 1.36× |
| Caramelban | 71 | 67.3 | 21.0* | 3.21× |
| Heroes of Sokoban III | 79 | 171.1 | 45.4* | 3.77× |
| IceCrates | 50 | 353.9 | 163.4* | 2.17× |
| Unconventional Guns | 112 | 238.8 | 89.2 | 2.68× |
| Castlecloset | 44 | 198.3 | 49.7 | 3.99× |

`*` Prepared reset parameters. This is a comparison of the documented harness
workloads, not a controlled estimate of language or hardware speed. Several
other complex games favor C++ at their shared measured batch sizes but have
censored GPU tails; their incomplete peaks are not included in this table.

Batch size is a major controllable factor. Simple games amortize JAX's cost
over large numbers of environments, reaching roughly 9.6–12.3 million steps/s
in several cases. At a fixed batch, rule-processing complexity is the stronger
cross-game signal than board area. The archived comparison computes Spearman
correlations separately at each shared batch; the plot shows batch 4,096.
At that batch, the 25 available comparisons give correlations of 0.835 for
source rule count, 0.842 for generated rule-group count, and 0.222 for board area
against C++/JAX throughput. Estimated temporary memory gives 0.601.
Source rule count, generated rule functions/groups, and native group-pass
counts are closely related predictors. Board area alone is much weaker here.
The sample is deliberately selected, features are correlated, and the hardest
censored games are missing at larger batches. These are associations, not an
estimate of independent causal contributions.

The implementations suggest why: C++ uses packed masks to reject impossible
rules, rows and columns before detailed matching. JAX expresses many ordered
rule groups and their convergence loops over dense object/force arrays.
Vectorizing state-dependent control flow does not necessarily avoid work for
each inactive environment. More sequential rule work therefore raises the
cost of each GPU batch; a small board need not make a game cheap. Crate
Assembler is an extreme example: 36 cells, 7,964 generated JAX functions, and
about 99.9% of native rule attempts rejected by board masks in the sampled trace.
Object channels, movement and multi-row matching also matter; rule count is
not a sufficient cost model on its own.

Repeated level initialization is a separate, directly tested factor. Preparing
deterministic reset state outside the rollout removes repeated startup-rule
work and reduces several games' runtime substantially. This is stronger causal
evidence than the cross-game correlations because the controlled A/B tests
hold initial states, actions and complete outputs fixed.

## Retained optimizations

C++ rejects a rule with an impossible global object mask before constructing a
match-list return value. Deterministic rules then borrow single-row matches or
stream the Cartesian product of multi-row matches. Row zero still varies
fastest, and every tuple after the first is checked again against the updated
board. Random rule groups retain their materialized sampling population.

The controlled comparison covers 138 paired configurations and 600,600
transitions, with exactly matching full RL outputs. The main 128-configuration,
32-game sample has a geometric-mean speedup of about 7.5%, with individual
medians ranging from 0.990× to 1.709×. Small negative differences are retained in
the results; this is not a claim that every configuration improves. Two longer
Blocks checks (10,000 steps, 21 trials) give median ratios of 1.004× and 1.006×
after an initial 1% slowdown. Two longer Notsnake checks with the same limits
give 1.014× and 1.012× after initial ratios near 0.995×. Both short and long
workloads are retained.

The 58 passing focused C++ tests cover overlapping rows, tuple order,
ellipses, random rules, scores and flags. An independent original-JavaScript
comparison passes all 32 games (3,666 states, through the first win or the
128-action limit). This samples correctness; it is not an exhaustive proof.

JAX gains an opt-in `env.prepare_params(params)` API for deterministic,
fixed-level reset state. A changed level or different environment falls back
to ordinary reset; random and multi-level environments keep the ordinary path.
Close over the prepared parameters when compiling a rollout so XLA can remove
the unused reset branch:

```python
params = env.prepare_params(PJParams(level=env.get_level(0), level_i=0))
step = jax.jit(lambda key, state, action: env.step(key, state, action, params))
```

Passing level arrays as dynamic JIT arguments retains the validity check and
fallback branch, so it need not achieve these specialization gains. Parameter
preparation is outside throughput timing and recorded separately. It does not
change the default API behavior or skip resets after terminal states.

H200 controlled full-output tests use batch 256, 64 steps, two action seeds,
seven alternating trials, and a 17-step episode limit to exercise automatic
resets. All output leaves match exactly. Atlas Shrank improves 2.13–2.21×,
Beam Islands 3.44–3.45×, and Caramelban 3.59–3.60×. Sokoban Basic is unchanged
(1.001×). There are 29 passing reset-cache, environment-switching and wide-object
validation tests. These are separate workloads from the throughput curves;
the ablation ratios must not be multiplied into the paper curves.

The safe API also passes exact full-output comparisons at batch 4,096 on an
RTX 4090 (JAX 0.9.1): Atlas 1.677–1.690×, Beam 3.452–3.454×, and Caramelban
3.597–3.606× across two seeds. These runs exercise over 12,000 automatic resets
per configuration. The earlier monkey-patched prototype remains a separate
artifact and is not the basis for the safe API's claims.

The measured cache engine has SHA-256
`d06ceb53b0c4c615bc5a541c96ef3631b5640c53767a5942e0ba2d33b513ffeb`.
The final production engine in commit `869cc784` has SHA-256
`c8ef56324231e7cb2d68096b635c39c002dbbe0a31d888e300c6c83f1ad8308d`.
Their only difference is a defensive copy of the level during preparation,
outside the timed rollout, so mutating a caller-owned NumPy array cannot also
mutate the cache's validity snapshot. Reset and step code are identical.
The measured benchmarks use immutable JAX arrays; their execution path is
unchanged. Both patches and the exact measured-to-final diff are archived.
The 29-test run covers the final version, including mutable-input invalidation.

## Protocol and provenance

C++ runs on the same Core i9-9980XE as the preceding round, with affinity 0–31,
`min(batch, 32)` OpenMP threads, full Python RL outputs, random action generation,
and win counting inside timing. Each trial starts from a reset board. H200
uses JAX 0.9.0.1, random actions inside the compiled scan, final-carry output,
and continuing rollouts. Five warmed trials give median and IQR at each batch.
These workload differences mean the comparison is useful for these harnesses,
not an isolated estimate of language or hardware speed.
The recorded `compile_s` measures rollout compilation; job wall time also
includes validation, reset compilation, parameter preparation, and warmup.

New H200 sweeps start at batches 16, 256, 1,024 and 4,096, then double.
The stopping rule requires a drop exceeding 10% or two consecutive gains of
at most 3%, considering batches at least 4,096. C++ doubles from one, with a
stopping floor of 1,024. Batch caps and timeouts are recorded separately from
plateaus. Partial measurements are preserved; no missing point is extrapolated.
Equal-batch comparisons and best-saturated comparisons are separate columns.

The frozen baseline is commit `1f3ecd84dc94621e6ae525d8fb47dd0fe9cb0506`,
with JAX engine SHA-256
`ab92c0a647c6efe9809c98cfecc7110a89b23b6dfdea2e7205603429041654fe`.
Original PuzzleScript is pinned to
`4176d350a9a4c7480fe3be0a58a072a7c8fef429` for the throughput curves.
Initial local A/B tests used a concurrently updated original compiler;
`compiler-changed-games.json` identifies the three affected games, and
`guarded-original.json` repeats them with the pinned compiler. `metadata-original`
and `cpp-rule-work-original.json` are the metadata used for analysis. Instrumented
C++ rule-work counts are separate from speed measurements.

`game-inputs/manifest.json` freezes original and simplified source inputs.
`provenance/` contains exact executed driver versions, build recipes, source
patches, library hashes and original-compiler hashes. Active benchmark snapshots
were isolated from subsequent workspace edits. `torch/`, `torch-wide/`,
`torch-terminal/` and `torch-prepared/` preserve separate attempts. `validation/`
selects the applicable ordinary-engine validation, with an origin manifest.
Slurm accounting and `job-status.json` identify resource-limited cases.

The first local CPU driver was interrupted by signal 15 while processing
Travelling Salesman. Its checkpoint and the remaining five games were resumed
with the same driver, library and protocol. Castlecloset, Vacuum and Kettle
received bounded extensions so their high-batch tails could reach the stopping
criterion. Crate Assembler's CPU sweep remains censored while trying batch
2,048; its measured points through 1,024 remain in the figure.

## Validation limits and exclusions

The old JS-state converter packed each cell into one uint64, which failed for
more than 64 object channels. It now decodes all signed 32-bit words directly;
12 tests cover signed words, word boundaries, layout, aliases and legacy input.
The affected games were rerun with the fixed converter before timing.
A missing headless color callback in the Node wrapper is also supplied.

Indigestion matches all 91 states through its first win. Its initial 128-action
comparison diverged only after the terminal state, when the reference retains
a winning flag while further actions are sent. The retained validation therefore
stops at the first win, matching the episode boundary used by the autoresetting
throughput harness. The original failed post-terminal trace is preserved.

Memories of Castlemouse fails on a marker at action 42 in the frozen baseline,
before a win, and its JAX throughput is excluded. The one-action reproducer in
`provenance/reproducer/` reduces the problem to overlapping pattern rows:
`late [ Mouse ] [ MMarker ] -> [ Mouse Marker ] []`. The first row creates a
Marker on the second row's cell; original PuzzleScript clears that entire
collision layer when applying the empty second replacement, while JAX only
clears the originally detected MMarker. This issue predates the reset cache.

Crate Assembler expands to 7,964 JAX rule functions. Its two-hour H200 job times
out before producing validation or throughput results. C++ profiling rejects
about 99.9% of its rule attempts with board masks. This is evidence about
compilation practicality and work avoidance, not a measured C++/JAX speed ratio.

## Experimental fixed-level rule pruning

An additional prototype computes a conservative object-reachability closure
from one initial level and the original compiler's positive object requirements
and possible RHS births. It ignores negative constraints, geometry and movement
when computing reachability, which overestimates what can appear. Only rules
that require an unreachable concrete object are dropped from JAX generation;
unknown property names are retained. Rule order and original group construction
are preserved. The benchmark rejects random or multi-level environments and
forbids external level/state edits. It is not enabled in the production engine.

For Crate Assembler, the closure contains 12 of the game's object types and
reduces 7,964 generated JAX rule functions to 399. All 129 sampled states match
the original JS engine. The H200 pilot then measures:

| Batch | Median steps/s | Rollout compilation |
| ---: | ---: | ---: |
| 16 | 1,265 | 729 s |
| 256 | 17,148 | 598 s |
| 4,096 | 113,861 | 945 s |

Ordinary JAX did not finish its correctness gate within two hours, so there is
no ordinary/pruned runtime speedup ratio. The pruned pilot exceeds C++'s best
measured 56,324 steps/s, but C++ is censored while attempting batch 2,048 and
the three-point pilot does not establish a JAX plateau. This is evidence that
level specialization can change the practical outcome, not a general backend
ranking. The [separate pilot figure](../../../../paper/figures/complex_throughput_20261007/experimental_reachability.pdf)
keeps the specialization out of the production comparison matrix.

The first prototype conservatively missed property-expanded tokens such as
`> crate12`, leaving 1,793 functions. That attempt was stopped during validation
compilation and superseded by the corrected 399-function version. Its source,
structure counts and logs remain archived. A safe general API would need to
enforce the fixed-level assumptions and preserve behavior under state injection
or changing levels; the one-game pilot is insufficient to promote it.

The final benchmark driver also resolves aliases by object index before pruning,
and rejects unmappable reachable names. This avoids incorrectly treating a
reachable object's alternate name as unreachable on other games. Crate's
measured specialization is unchanged: an audit compares all 7,913 generation
decisions and finds exactly the same 399 functions. Four direct alias, negative
pattern and unknown-property guard checks pass. The exact measured source
(`e7b693…`) and final driver (`9c1078…`) are both recorded in the alias audit.

## Rejected experiments

Streaming C++ tuples without the earlier mask guard regressed Crate Assembler;
borrowing single-row matches alone did not resolve that. Both variants are
archived. A rebuilt `-O2` control checks build variation, and `-O3` was generally
1–4% slower on its sampled cases, so the build flags remain unchanged.

RTX 4090 input-buffer donation is neutral (about 0.999–1.003× on the sampled
Sokoban/Atlas batches). Disabling XLA constant folding produces mixed compile
times and slightly slower rollouts (about 0.96–0.99×). Neither is promoted.
The fixed-level monkey-patched reset prototype is archived separately from the
safe prepared-parameter implementation measured on H200.

Splitting a full-output Sokoboros rollout into sequential chunks of 16 at total
batch 256 reduces throughput to 0.639× the monolithic H200 rollout. All outputs
match. Chunking also increases estimated temporary memory from 515 MB to
859 MB in that experiment. The same controlled test on Vacuum gives 0.739×,
again with exactly matching full outputs; it also slightly increases temporary
memory. Neither case supports promoting this chunking configuration.
Whole-turn native tuple-count profiles have only
moderate tails (mean per-step maximum/mean 2.73× for Sokoboros, 2.31× for Vacuum).
These counts do not measure GPU work or per-rule divergence. They do not support
attributing the entire C++ advantage to a few unusually slow environments.

## Reproducing the figures

From the repository root, with the project environment activated:

```bash
MPLCONFIGDIR=/tmp/puzzlejax-mpl python \
  scripts/benchmarks/results/2026-10-07-complex/provenance/reproduce_figures.py
```

The main figure set uses prepared reset parameters for games with a measured
prepared sweep, and ordinary parameters elsewhere. A star identifies prepared
rows. Entire curves are selected by configuration; individual favorable points
are not spliced together. The line plots retain both ordinary and prepared
curves. `ordinary/` separately reproduces the comparison using ordinary JAX
throughout. Every plotted point is also in `throughput.csv`; `comparison.json`
records input hashes, selected variants, stopping reasons and correlations.
The plotter rejects missing directories, mixed protocols, duplicate sweeps,
unvalidated new games and source/compiler mismatches.

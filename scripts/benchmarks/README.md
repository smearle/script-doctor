# Engine performance experiments

Run benchmarks from the repository root with the project environment activated.

## Refreshed C++ curves and sixteen-game regression check (2026-10-06)

The [updated eight-game paper figure](../../paper/figures/throughput_cpp_20261006/random_rollout_profile_h200_updated.pdf)
replaces the archived C++ curve with fresh measurements of the packed-cell
scoring and target-list optimizations. The existing H200 JAX measurements and
historical NodeJS references are retained. The
[sixteen-game supplement](../../paper/figures/adaptive_throughput_cpp_20261006/adaptive_throughput_h200_cpp.pdf)
adds C++ curves for the wider suite, with a
[combined CSV](../../paper/figures/adaptive_throughput_cpp_20261006/adaptive_throughput_h200_cpp.csv).
Raw timings, frozen compiled games, regression hashes and test logs are under
[2026-10-06-cpp-figure](results/2026-10-06-cpp-figure/).

Fresh C++ measurements run on a Core i9-9980XE, pinned to logical CPUs 0–31,
with `min(batch, 32)` OpenMP threads. Each point reports the median and IQR
of five trials after two warmups. Random action generation, full Python RL
outputs, and win counting are timed; reset, compilation and initialization are
excluded. Trials start from a reset board with seeds 42–46. Rollout length is
5,000 at batch one and `max(5000 // batch, 100)` otherwise, following the
original CPU profiler. The episode limit is one larger than the rollout.
Batch size doubles from one. Stopping considers batches at least 1,024 and
requires either a drop exceeding 10% or two successive gains at most 3%.
All sixteen curves completed: **211 batch configurations and 1,055 timed
calls**, with ten plateaus and six regressions as batch size grows. No curve
stopped solely at the safety cap. Peak medians and stopping reasons are in the
[throughput summary](results/2026-10-06-cpp-figure/throughput-summary.json).

The initial stopping floor of 64 ended some curves during a small-batch dip.
The sweep was extended with a floor of 1,024 while preserving those timings.
Per-point driver hashes and extension history record the change; the measured
worker, wrapper, library and rollout protocol did not change. Resume checks
reject incompatible inputs. A separate 16-thread pilot is excluded from the
figures. Unlike the archived C++ curve, the new curve does not select the
fastest timing sample or search over thread counts. Therefore changes relative
to that archive are not controlled estimates of optimization gains.

The controlled comparison uses the same corrected baseline and optimized
extension builds as the preceding round: **both include the batched flag race
fix**. Sixteen games at batches 1/64 and threads 1/8, respectively, plus level
one of Sokoban Basic, Zen, Sokoban Match 3 and Microban, cover **72 paired
configurations and 777,600 transitions**. All observations, rewards, done and
truncation flags, and info fields match exactly. Runs use two seeds, 300 steps,
and nine alternating timing trials; the inactive worker is paused. No sampled
configuration regresses: median speedups range from **1.076× to 1.928×**.
Across the sixteen level-zero games, geometric-mean gains are **25.6%** at batch
one and **36.3%** at batch 64. The
[per-game summary](results/2026-10-06-cpp-figure/regression-summary.json)
retains both seed-specific bounds. These checks sample games and levels; they
do not establish equivalence for every PuzzleScript program.

All **82 focused C++/JAX regression tests** and **12 benchmark/reporting tests**
pass. No additional engine change was required. JAX uses final-carry output
with continuing rollouts, whereas C++ returns full RL outputs and resets between
trials. Those workload differences remain explicit in the figure and prevent
interpreting its curves as an isolated language/backend comparison.

```bash
python -m scripts.benchmarks.benchmark_cpp_throughput \
  --library puzzlescript_cpp/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so \
  --compiled-dir scripts/benchmarks/results/2026-10-06-cpp-figure/compiled-games \
  --max-threads 32 --output-dir /tmp/cpp-throughput-new
# Defaults to the eight paper games. Use --games to include the wider suite.
# Match the affinity/OpenMP environment in the raw metadata when reproducing timings.

MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_engine_throughput \
  --paper-results scripts/benchmarks/results/2026-10-05-adaptive/adaptive-throughput-h200 \
  --cpp-results scripts/benchmarks/results/2026-10-06-cpp-figure/cpp-throughput \
  --output-dir paper/figures/throughput_cpp_20261006
MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_adaptive_throughput \
  --results scripts/benchmarks/results/2026-10-05-adaptive/adaptive-throughput-h200 \
  --cpp-results scripts/benchmarks/results/2026-10-06-cpp-figure/cpp-throughput \
  --output-dir paper/figures/adaptive_throughput_cpp_20261006
```

## Adaptive batches, broader coverage, and CPU audit (2026-10-05)

The updated benchmark doubles the high-batch tail until either a median drops
more than 10% below the best earlier median, or two successive measurements
improve on the previous best by at most 3%. Only batches at least 4096 count
towards stopping. The safety ceiling is 1,048,576; reaching the ceiling is
reported separately from a plateau. Each point retains five warmed timings,
compilation time, and compiler-estimated memory requirements. Results are
checkpointed after every point. Resume rejects changes to the engine hash,
device, JAX version, seed, or rollout protocol.

The workload and frozen JAX engine are unchanged from the preceding paper
figure. The first eight games resume its six original points exactly. Eight
additional games expand the suite to sixteen: Blocks, Nekopuzzle, Sokoban
Match 3, Travelling Salesman, Multi-word Dictionary Game, Microban, Magnet
Jack, and HyperMaze. These are level-zero measurements, not a claim about
all levels or every PuzzleScript game.

All sixteen sweeps completed: **163 configurations and 815 timing samples**,
including the 48 preserved original points. Fifteen games met the plateau
criterion; Atlas Shrank regressed. No run stopped merely at the safety cap.
Sokoban and Notsnake reach 9.56M and 11.54M steps/s, respectively, 52.8% and
63.0% above their preceding batch-16,384 measurements.

| Game | Best measured batch | Median steps/s | Stop |
| --- | ---: | ---: | --- |
| sokoban basic | 1,048,576 | 9,560,741 | plateau |
| notsnake | 1,048,576 | 11,537,635 | plateau |
| Zen Puzzle Garden | 262,144 | 5,110,070 | plateau |
| Slidings | 1,048,576 | 2,359,643 | plateau |
| limerick | 32,768 | 426,135 | plateau |
| kettle | 262,144 | 483,818 | plateau |
| Take Heart Lass | 1,048,576 | 2,880,508 | plateau |
| atlas shrank | 4,096 | 50,942 | regression |
| blocks | 65,536 | 3,296,783 | plateau |
| nekopuzzle | 524,288 | 3,512,366 | plateau |
| sokoban match3 | 524,288 | 5,029,540 | plateau |
| Travelling salesman | 1,048,576 | 12,278,443 | plateau |
| Multi-word Dictionary Game | 524,288 | 3,188,907 | plateau |
| Microban | 1,048,576 | 9,555,941 | plateau |
| Magnet Jack | 16,384 | 216,492 | plateau |
| HyperMaze | 32,768 | 95,271 | plateau |

[Sixteen-game figure](../../paper/figures/adaptive_throughput_20261005/adaptive_throughput_h200.pdf),
[CSV](../../paper/figures/adaptive_throughput_20261005/adaptive_throughput_h200.csv),
and [provenance](../../paper/figures/adaptive_throughput_20261005/adaptive_throughput_manifest.json).
The stopping threshold is an empirical saturation rule, not proof of a global
optimum. Smaller batches near the plateau may offer a better memory trade-off.
HyperMaze's first compilation took 318 seconds; its best measured throughput
was 95,271 steps/s. Compilation costs remain visible in the raw data and CSV.

[Extended paper figure](../../paper/figures/throughput_adaptive_20261005/random_rollout_profile_h200_updated.pdf)
retains the archived CPU measurements unchanged. The reference JSON files
and their hashes are now included under [historical-cpu](results/historical-cpu/)
so a clean checkout can reproduce the figure. The NodeJS native label now
says **engine-only**, making the workload difference explicit.

```bash
# Submit from a prepared checkout on Torch; account/partition are site-specific.
sbatch --account=YOUR_ACCOUNT --partition=h200_tandon \
  scripts/benchmarks/adaptive_throughput.slurm

MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_engine_throughput \
  --paper-results scripts/benchmarks/results/2026-10-05-adaptive/adaptive-throughput-h200 \
  --output-dir paper/figures/throughput_adaptive_20261005
MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_adaptive_throughput \
  --results scripts/benchmarks/results/2026-10-05-adaptive/adaptive-throughput-h200 \
  --output-dir paper/figures/adaptive_throughput_20261005
```

### Further JAX experiments

Two H100 experiments did not justify another production change. Replacing the
movement-delta lookup with arithmetic produced ratios of 0.996–1.009×;
precomputing a shared loop bound produced 0.989–1.001×. Each covers Sokoban,
Blocks and Zen at batches 1/4096/16384, two seeds, and nine alternating trials.
Ratios aggregate the two seed-specific median ratios geometrically. All
**24,577,200 paired transitions** match exactly, including complete rollout
outputs. The [arithmetic](results/2026-10-05-adaptive/movement-deltas.json) and
[loop-bound](results/2026-10-05-adaptive/movement-bounds.json) raw results are
retained; neither candidate is enabled in the engine.

### Why the NodeJS batched curve is low

The native worker pool runs whole rollouts inside NodeJS. It returns aggregate
performance statistics, with no per-step Python crossing, observation, reward,
or info output. Its outer timer includes per-run compilation. In contrast,
`NodeJSBatchedPuzzleEnv` sends each action through Python → a Node controller →
one child process per environment, then gathers complete RL outputs and waits
for every child before the next step. At larger batches it also oversubscribes
CPU cores. These are different workloads; their throughput gap cannot be
attributed simply to the JavaScript engine or multiprocessing implementation.

There were two concrete inefficiencies worth a small check:

* An ordinary transition calculated the same heuristic score twice. The worker
  now reuses the first result unless it reset the board. In nine configurations
  (three games, batches 1/4/16), the median change ranges from effectively zero
  to **+21.6%**. All complete outputs match. The two focused tests also exercise
  wins, truncation, partial reset, and automatic reset enabled/disabled.
* Default child-process JSON serialization expands observation buffers into
  arrays of numbers. Node's binary serialization is now opt-in for experiments.
  It helps several cases but regresses Sokoban at batch 16 by **11.7%**, so JSON
  remains the default. This was not a consistent throughput improvement.

The final score and serialization audits use 2,000 steps and nine alternating
trials after warmup, with frozen source hashes. The [raw measurements](results/2026-10-05-adaptive/)
include all configurations, not just improvements. Earlier short/mutable-source
pilots are explicitly superseded by `nodejs-score-final.json` and
`nodejs-serialization-final.json`.

The NodeJS profiler's moved-file path resolution is corrected. Its Hydra
launcher requests one CPU and defaults to the single-process mode; users who
select the batched mode must also request sufficient CPUs. Archived metadata
does not record the allocation, so this is not evidence that the historical
curves used a one-CPU SLURM allocation. Further optimization work in this round
focuses on C++.

### C++ correctness and scoring

A threaded comparison uncovered an existing `BatchedEngine` race: `dones_`,
`wins_`, and `prev_winning_` used `std::vector<bool>`. OpenMP writes to distinct
environments could modify the same packed word. A lost done flag could also
skip automatic reset, corrupting the next observation. These buffers now use
one byte per environment. Python still receives NumPy boolean arrays.
The new mixed-win regression fails on the unchanged baseline and exercises
32/65/128 environments against a serial reference after the fix.

Scoring and win checks now read packed cell words directly, avoiding temporary
`BitVec` heap allocations for each examined cell. Distance scoring also collects
matching target cells once per condition instead of rediscovering them for every
source cell. The raw score, normalized score, empty-set defaults, property and
aggregate semantics, and win conditions are preserved. The 48 reference cases
cover thin boards, multiword masks, sign bits, empty masks, and mixed conditions.

C++ measurements use isolated extension builds and complete Python RL outputs.
Both final builds include the flag fix. Compilation, initialization, reset,
action generation, correctness hashing and IPC are outside the timer. Two
persistent worker processes alternate measured calls; the idle process is
paused with `SIGSTOP` so its OpenMP workers cannot consume CPU while the other
build runs. The source hashes, library hashes, CPU affinity, thread settings,
all timing samples and full-output hashes are recorded. The shared workspace
extension was not rebuilt, preserving concurrent work.

The early passive-wait pilot and partial target-list ablation are exploratory.
The latter stopped on the pre-existing flag race; only final comparisons against
the corrected baseline are used to report retained speedups.

The final eight-game comparison passes **1,387,200 paired transitions** with
exact complete-output equality, across 48 configurations and 864 timed calls.
Both corrected builds pass all **51 focused C++ tests**. The adaptive-benchmark
and NodeJS regression tests add six passing tests. The earlier JAX engine and
its 131-pass/9-expected-failure regression result are unchanged.

On the Core i9-9980XE, geometric-mean throughput improves **28.4%** at batch one,
**45.3%** at batch 32 with one thread, and **44.7%** at batch 256 with eight
threads. Each cell below is the geometric mean of two seed-specific median
ratios, with nine alternating trials per seed. No tested configuration regresses.

| Game | Batch 1, 1 thread | Batch 32, 1 thread | Batch 256, 8 threads |
| --- | ---: | ---: | ---: |
| sokoban basic | 1.197× | 1.484× | 1.458× |
| blocks | 1.687× | 2.004× | 1.946× |
| Zen Puzzle Garden | 1.226× | 1.344× | 1.325× |
| kettle | 1.506× | 1.680× | 1.632× |
| notsnake | 1.104× | 1.272× | 1.267× |
| Slidings | 1.076× | 1.176× | 1.157× |
| limerick | 1.396× | 1.334× | 1.465× |
| Microban | 1.195× | 1.480× | 1.459× |

[C++ comparison figure](../../paper/figures/cpp_scoring_20261005/cpp_scoring_speedup.pdf),
[summary CSV](../../paper/figures/cpp_scoring_20261005/cpp_scoring_summary.csv),
[raw serial results](results/2026-10-05-adaptive/cpp-final-single.json),
and [raw threaded results](results/2026-10-05-adaptive/cpp-final-threaded.json).
The figures keep these controlled results separate from the historical CPU
references in the paper overlay. Rebuild the C++ extension to activate the
source changes in a working checkout (`python setup_cpp.py build_ext --inplace`).

```bash
# Point these arguments at separate builds, both using byte-backed batched flags.
OMP_WAIT_POLICY=ACTIVE OMP_PROC_BIND=close OMP_PLACES=cores \
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/puzzlejax-mpl \
python -m scripts.benchmarks.benchmark_cpp_scoring \
  --baseline /path/to/corrected-baseline/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so \
  --candidate /path/to/optimized/_puzzlescript_cpp.cpython-313-x86_64-linux-gnu.so \
  --games sokoban_basic blocks Zen_Puzzle_Garden kettle notsnake Slidings limerick Microban \
  --batches 256 --threads 8 --steps 300 --trials 9 --seeds 42 1042 \
  --output /tmp/cpp-threaded.json

MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_cpp_scoring \
  --results scripts/benchmarks/results/2026-10-05-adaptive/cpp-final-single.json \
            scripts/benchmarks/results/2026-10-05-adaptive/cpp-final-threaded.json \
  --output-dir paper/figures/cpp_scoring_20261005
```

## Shared movement loop and sparse updates (2026-10-04–05)

Movement now has an explicit `custom_vmap` rule. A scalar loop index visits
the same ordered force list in every environment, stopping after the longest
list. Finished environments have sentinel coordinates; their writes are
dropped by selecting an out-of-bounds channel. Objects, forces, collisions,
and padded-cell validity are still read from the current board before every
move. The RNG and command flags are unchanged. Dynamic board, coordinate,
collision, and layer-mask arrays are explicit arguments to the batching rule,
including when callers use nested maps or map collision matrices. The constant
movement-delta table is also explicit: `vmap(lax.switch)` can batch branch
constants, which `custom_vmap` rejects if they were captured by its function.
The switch-backend regression tests caught this in the first implementation.

This removes both whole-board selection after rejected moves and the extra
whole-board masking that automatic `vmap(while_loop)` uses for environments
whose loops have finished. Unbatched and singleton-batch execution retain the
original update operations. The latter matters: the first version regressed
Blocks at batch one by about 12%, despite improving larger batches.

Separate H200 ablations at batch 4096 explain why both changes are needed.
Numbers are geometric means of two seed-specific throughput ratios, each
measured with nine alternating trials after compilation and warmup:

| Game | Sparse writes only | Shared loop only | Both |
| --- | ---: | ---: | ---: |
| Sokoban | 0.998× | 0.982× | 1.027× |
| Blocks | 1.000× | 1.011× | 1.212× |
| Zen | 1.001× | 0.997× | 1.034× |

The [raw pilot and ablation data](results/2026-10-04-movement-updates/) include
every timing sample, compilation time, temporary allocation estimate, engine
hash, and benchmark hash. Every timed configuration first compares the reset
and complete 100-step rollout output against the preceding movement method.
The baseline is the preserved `uncached_movement` implementation in
`benchmark_movement_identities.py`, not the potentially modified engine method.

Fresh profiles of the final build at batch 4096 show fewer GPU kernels:
Sokoban 37,989 → 35,197, Blocks 40,358 → 31,899, and Zen 17,265 → 15,969.
These are the same 100-step
learner-facing transition workloads as the preceding profiles. The Blocks
reduction is 21.0%; the coordinate-prefix and coordinate-scatter kernels still
cost about 13.6 ms and 13.7 ms of aggregate instrumented GPU activity. Those
compaction operations and the remaining sequential rule/movement loops are
still potential optimization targets. Trace timings include instrumentation
overhead and are not direct throughput estimates or SM-utilization measures.
[Profile comparison](results/2026-10-04-movement-updates/profile-comparison.json),
[summaries](results/2026-10-04-movement-updates/profile-summary.json), and
[verified complete traces/HLO](results/2026-10-04-movement-updates/profiles.tar.gz).

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_movement_updates.py

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_movement_updates \
  --candidates production --games sokoban_basic blocks Zen_Puzzle_Garden \
  --batches 1 256 4096 --seeds 42 1042 --steps 100 --trials 7 \
  --output /tmp/movement-updates-h200.json
```

`movement_updates.slurm` runs the twelve-game sweep, including the eight paper
games plus Blocks, Nekopuzzle, Sokoban Match 3, and Travelling Salesman.
The focused tests exercise unequal completion, orphan and ACTION-only forces,
overlapping layer masks, custom collision matrices, padding, thin boards,
repeated movement passes, debug execution, singleton batches, nonleading maps,
nested maps, shared initial boards, and mixed movement/identity switch branches.
The initial three-candidate prototype passed 15 tests; the final movement test
module has 17 tests, including the production implementation and the switch
composition regression.

The final source is SHA-256
`ab92c0a647c6efe9809c98cfecc7110a89b23b6dfdea2e7205603429041654fe`.
All 26 focused movement, switch-backend, and padding tests pass on CPU JAX
0.9.0. The 17 movement tests also pass with the declared minimum JAX 0.7.1
in a separate temporary environment, including mapped switch branches.
[Focused integration log](results/2026-10-04-movement-updates/cpu-dispatch-fixed-tests.log)
and [JAX 0.7.1 log](results/2026-10-04-movement-updates/cpu-jax071-tests.log).
The clean complete regression run on the final source passes **131 tests**,
with the same **9 expected failures**, in 28m07s. This includes the original
JavaScript action replays, focused switch-backend checks, padding, independent
cell rules, coordinate compaction, force cleanup, movement refinements,
chunked rollouts, and the new movement cases.
[Complete regression log](results/2026-10-04-movement-updates/cpu-final-regression.log).

### Twelve-game H200 results

The final sweep covers twelve games, batches 1/256/4096, and two seeds. The
three additional level-one checks, three batch-16384 checks, and five-game
H100 check bring the total to **94 seeded comparisons** and **26,831,200 paired
transitions** with exact complete-output equality. The 1,316 timed calls are
separate from compilation, warmup, and correctness comparisons.

| Game | Speedup, batch 256 | Speedup, batch 4096 | Steps/s, batch 4096 |
| --- | ---: | ---: | ---: |
| Sokoban | 1.000× | 1.024× | 2,994,061 |
| Blocks | 1.138× | 1.207× | 2,320,560 |
| Zen | 1.018× | 1.031× | 3,152,890 |
| Notsnake | 0.968× | 1.004× | 3,114,298 |
| Slidings | 0.982× | 0.980× | 715,077 |
| Lime Rick | 0.995× | 0.998× | 314,945 |
| Kettle | 1.007× | 1.033× | 338,160 |
| Take Heart Lass | 0.994× | 1.001× | 642,768 |
| Atlas Shrank | 1.002× | 1.009× | 52,554 |
| Nekopuzzle | 0.994× | 0.980× | 1,562,371 |
| Sokoban Match 3 | 0.997× | 1.006× | 1,957,672 |
| Travelling Salesman | 0.990× | 1.000× | 2,572,770 |

At batch 256, Notsnake is 3.2% slower. Batch-one results range from a 1.6%
slowdown to a 0.4% improvement across these games. The gains are concentrated
in particular workloads; this is not a blanket improvement for every game
and batch size. Every point is retained in the figures and CSV.
The equally weighted geometric-mean speedups across this twelve-game set are
0.998× at batch one, 1.006× at batch 256, and 1.021× at batch 4096.

[Summary and input hashes](results/2026-10-04-movement-updates/final/summary.json),
[complete raw sweep](results/2026-10-04-movement-updates/final/updates-wide-h200/),
[paired speedup figure](../../paper/figures/movement_updates_20261005/movement_updates_speedup_h200.pdf),
[throughput curves](../../paper/figures/movement_updates_20261005/movement_updates_throughput_h200.pdf),
and [larger-batch comparison](../../paper/figures/movement_updates_20261005/movement_updates_large_batch_h200.pdf).
The figures have PNG counterparts and include a CSV and a hashed provenance
manifest. Whiskers/bands show the range across two seeds, not confidence intervals.

```bash
MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_movement_updates \
  --results scripts/benchmarks/results/2026-10-04-movement-updates/final/updates-wide-h200/game-*.json \
  --large-results scripts/benchmarks/results/2026-10-04-movement-updates/final/large-batch-h200.json \
  --output-dir paper/figures/movement_updates_20261005
```

### H100 comparison and larger H200 batches

The final implementation was compared with the preceding movement method on
an H100 80GB HBM3 as well as H200, using JAX 0.9.0.1. Both use full-output
100-step rollouts, level zero, seeds 42 and 1042, and seven alternating timing
trials. Ratios below are geometric means of the two seed-specific median
throughput ratios. The H100 check confirms the same trade-off seen on H200:
strong gains in Blocks, smaller gains in Sokoban and Zen, and small regressions
in Slidings and Nekopuzzle.

| Game | H200 speedup, batch 4096 | H100 speedup, batch 4096 | H100 steps/s |
| --- | ---: | ---: | ---: |
| Sokoban | 1.024× | 1.016× | 2,879,152 |
| Blocks | 1.207× | 1.215× | 2,300,912 |
| Zen | 1.031× | 1.025× | 3,051,236 |
| Slidings | 0.980× | 0.977× | 669,904 |
| Nekopuzzle | 0.980× | 0.973× | 1,504,125 |

At batch 16,384 on H200, Blocks improves **37.4%**, reaching **3,377,909
steps/s**. Slidings is 1.2% slower (1,484,499 steps/s), and Nekopuzzle is 1.0%
slower (2,666,704 steps/s). These are full-output A/B results, distinct from
the final-carry paper workload. Neither a universal speedup nor an optimal
batch size is implied.

Additional level-one comparisons at batch 4096 show gains of 1.6% for Sokoban,
2.9% for Zen, and 2.5% for Sokoban Match 3, with exact complete-output equality
for both seeds.
[H100 raw data](results/2026-10-04-movement-updates/final/comparison-h100.json),
[large-batch H200 raw data](results/2026-10-04-movement-updates/final/large-batch-h200.json),
and [additional-level raw data](results/2026-10-04-movement-updates/final/additional-levels-h200.json).

The completed profiles and source snapshots are archived under this round's
results directory. Torch hit its account file-count quota during the first
sweep; unused game files from our own snapshot were archived, verified by
per-file SHA-256, and removed from that scratch snapshot. All twelve selected
game sources were verified unchanged afterward. No unrelated account data was
changed. Interrupted results are retained separately from the final runs.
[Execution and archive notes](results/2026-10-04-movement-updates/execution-notes.json).

### Updated eight-game paper figure (2026-10-05)

The final source was remeasured on all eight paper games at batches
1/16/256/1024/4096/16384: **48 configurations and 240 timed calls**. This is the
existing paper workload: random actions inside the timed scan, final carry
only, no practical episode-length cutoff, and a continuing carry between
warmup and five measured calls. Curves show median throughput and IQR.
The IQR describes those continuing rollouts, not a confidence interval across
independent seeds. Compilation, reset setup, and warmup are excluded.

The historical Core i9-9980XE C++ and NodeJS references are retained unchanged
and explicitly labeled archived. All 16 reference input files still match
the preceding figure's recorded hashes. Their original selection/statistics
are preserved, so the overlay is historical context rather than a new
controlled CPU/GPU comparison.

| Game | Best tested batch | Median environment steps/s |
| --- | ---: | ---: |
| Sokoban | 16,384 | 6,259,157 |
| Notsnake | 16,384 | 7,076,286 |
| Zen | 16,384 | 4,557,421 |
| Slidings | 16,384 | 1,508,583 |
| Lime Rick | 16,384 | 423,556 |
| Kettle | 16,384 | 434,520 |
| Take Heart Lass | 16,384 | 1,800,501 |
| Atlas Shrank | 4,096 | 50,942 |

[Updated paper figure](../../paper/figures/throughput_20261005/random_rollout_profile_h200_updated.pdf),
[PNG version](../../paper/figures/throughput_20261005/random_rollout_profile_h200_updated.png),
[raw measurements](results/2026-10-05-paper-h200/), and
[plot provenance](../../paper/figures/throughput_20261005/throughput_plot_manifest.json).
These numbers are a separate workload from the full-output paired comparisons
above and should not be mixed to compute speedup ratios.

```bash
MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_engine_throughput \
  --paper-results scripts/benchmarks/results/2026-10-05-paper-h200 \
  --output-dir paper/figures/throughput_20261005
```

## Broader H200 coverage (2026-10-03)

The retained movement refinements were compared with the preceding engine on
nine additional games, at batches 256 and 4096. These use level zero, seed 42,
100-step full-output rollouts with automatic reset, and seven alternating
timing trials after compilation and warmup. All **3,916,800 compared
transitions** match exactly, including reset and every rollout output leaf.
Torch used JAX 0.9.0.1 and NVIDIA H200 GPUs. The engine remains at SHA-256
`38ab0cee2e8eb6d22ce4e2bb9780d3b9862f1eddaf0d2fb4aadb0900c3cea03c`.

| Game | Speedup, batch 256 | Speedup, batch 4096 | Current steps/s, batch 4096 |
| --- | ---: | ---: | ---: |
| Notsnake | 1.035× | 1.036× | 3,021,641 |
| Slidings | 1.004× | 0.996× | 708,623 |
| Lime Rick | 0.995× | 0.991× | 315,502 |
| Kettle | 1.079× | 1.072× | 324,556 |
| Take Heart Lass | 1.017× | 1.018× | 633,517 |
| Atlas Shrank | 1.029× | 1.023× | 51,432 |
| Nekopuzzle | 1.011× | 1.006× | 1,578,561 |
| Sokoban Match 3 | 1.189× | 1.151× | 1,898,686 |
| Travelling Salesman | 1.034× | 1.024× | 2,517,530 |

The larger gains from the earlier three-game sweep do not generalize
uniformly. Sokoban Match 3 and Kettle benefit meaningfully; several other
games are nearly unchanged. Lime Rick is slightly slower, by less than 1%.
The baseline replaces the three movement hooks with sequential cleanup,
channel gathering, and unbarriered compaction, while retaining the earlier
distance-transform, match-enumeration, and independent-cell changes.

[Raw game pairs](results/2026-10-03-h200-wide/),
[PDF](../../paper/figures/wide_movement_20261003/wide_movement_speedup_h200.pdf),
[PNG](../../paper/figures/wide_movement_20261003/wide_movement_speedup_h200.png),
[CSV](../../paper/figures/wide_movement_20261003/wide_movement_throughput_h200.csv),
and [plot manifest](../../paper/figures/wide_movement_20261003/wide_benchmark_manifest.json).
Whiskers span the ratios formed from timing extrema; they are not confidence
intervals. These full-output comparisons are separate from the paper's
final-carry-only throughput workload. Historical CPU curves are unchanged.

The [game metadata](results/2026-10-03-game-metadata.json) records board sizes,
object/layer counts, level counts, and exact game-source hashes for all twelve
games considered across the pilot and wider sweep. Runtime sources and game
files were checked against the [Torch snapshot hashes](results/2026-10-03-torch-runtime-source.sha256).
The [source archive](results/archives/2026-10-03-movement-experiments-source.tar.gz)
and [its manifest](results/archives/2026-10-03-movement-experiments-source.json)
preserve the experimental implementations and plotting code.
[SLURM history](results/2026-10-03-slurm-history.txt) and
[submitted launchers](results/2026-10-03-launchers/) record execution details.
Initial jobs interrupted by account disk quota were rerun after preserving
and compressing our own old profiling traces. Only completed pairs are plotted.

```bash
sbatch --account=YOUR_ACCOUNT scripts/benchmarks/wide_engine_benchmarks.slurm
python -m scripts.plotting.plot_wide_engine_benchmarks \
  --results scripts/benchmarks/results/2026-10-03-h200-wide/*.json \
  --output-dir paper/figures/wide_movement_20261003
```

## Moving-object identity experiment (2026-10-03)

`benchmark_movement_identities.py` tests computing an object-ID map once per
movement pass instead of searching all object channels at every sequential
move. It preserves the original force scan and destination collision checks.
Caching requires disjoint layer masks and complete within-layer collisions:
while a force remains active, its original object cannot have moved or been
displaced by another object in that layer. Moving clears the source forces.
Custom collision matrices and overlapping masks fall back to live reads.
This implementation is experimental; the production engine is unchanged.

The initial H200 comparison (job 19116379, JAX 0.9.0.1) uses two action seeds,
nine alternating trials, and 100-step full-output rollouts. All 2,611,200
compared transitions match exactly. Ratios below span the two seed results:

| Game | Batch 256 | Batch 4096 |
| --- | ---: | ---: |
| Sokoban | 0.979–0.980× | 0.995–1.000× |
| Blocks | 1.001–1.002× | 0.983–0.984× |
| Zen | 0.995–1.009× | 0.989–0.990× |

This pilot supplies no reason to enable caching by default. Its temporary
buffer estimates also grow at batch 4096 (for Zen, 81.6 to 90.5 MB).
[Raw H200 measurements](results/2026-10-03-h200-movement-identities.json).

The follow-up checks games with more object types, at two levels and batch
4096. Each range again spans seeds 42 and 1042:

| Game | Level 0 | Level 1 |
| --- | ---: | ---: |
| Slidings | 0.983× | 0.980–0.983× |
| Take Heart Lass | 0.997–0.998× | 0.999–1.000× |

An additional 3,276,800 compared transitions match exactly. Across both H200
identity experiments, **5,888,000 transitions** match, with no useful runtime
gain to justify enabling the candidate. The production engine is unchanged.
[Raw extended measurements](results/2026-10-03-h200-identities-wide/).

All 16 focused CPU parity tests passed, covering ordered movement, orphan
forces, multiple objects per cell, padded boards, repeated movement passes,
custom-collision fallbacks, and the debug path's first-success behavior.
The uncached benchmark copy is checked against the production implementation.
[Main test log](results/2026-10-03-movement-identities-tests.log) and
[debug test log](results/2026-10-03-movement-identities-debug-tests.log).

An [RTX 4090 pilot](results/2026-10-03-rtx4090-movement-identities-contended.json)
also passed every exact-output comparison, but other experiments started
using the GPU during that run. Its timing samples are contaminated by
contention and are excluded from performance conclusions and figures.

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_movement_identities.py
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_movement_identities \
  --games sokoban_basic blocks Zen_Puzzle_Garden --batches 256 4096 \
  --seeds 42 1042 --trials 9 --output /tmp/h200-identities.json
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_movement_identities \
  --games Slidings Take_Heart_Lass --levels 0 1 --batches 4096 \
  --seeds 42 1042 --trials 9 --output /tmp/h200-identities-wide.json
```

## Remaining costs in the retained engine

H200 job 19117246 profiles warmed, 100-step rollouts at batch 4096 after all
retained changes. The traced workload retains learner-facing transitions and
the final state. Compilation and tracing are excluded from its separate
unprofiled timings. Trace timings include instrumentation overhead.

| Game | GPU kernels per rollout | Median kernel duration | Kernels below 5 µs |
| --- | ---: | ---: | ---: |
| Sokoban | 37,989 | 1.568 µs | 88.1% |
| Blocks | 40,358 | 1.472 µs | 82.6% |
| Zen | 17,265 | 1.728 µs | 73.5% |

Small GPU operations and repeated state copies remain concrete targets.
In Blocks, `loop_select_fusion` executes 1,990 times and accounts for 29.0 ms
of aggregated traced GPU activity. Its optimized HLO selects between complete
`(4096, 1, 20, 11, 13)` board arrays inside the movement loop. Reducing these
copies through masked updates and a batching-aware movement implementation is
a promising next experiment; its benefit has not yet been measured. Any such
change must retain ordered moves and the behavior of completed environments
within a batch. It is a larger change than caching object identities.

Coordinate compaction also remains measurable: the Blocks prefix-sum fusion
and coordinate-scatter fusion account for 13.6 and 13.7 ms respectively in
this trace. Fusion names describe generated kernels, not isolated source
operations. These profiles do not establish a performance ceiling or directly
measure SM utilization.

[Profile summaries](results/2026-10-03-h200-profiles.json),
[unprofiled timings](results/2026-10-03-h200-profile-timings.json),
[complete traces and optimized HLO](results/archives/2026-10-03-h200-profiles.tar.gz),
and [per-file hashes](results/archives/2026-10-03-h200-profiles.sha256).
All twelve archived profile files were verified after transfer. Trace paths
in the summary identify the original temporary capture directory; archive
members are rooted at each game's directory.

## Further movement refinements and controlled chunking

The follow-up experiments target invalid-force cleanup, the force-channel
layout, and the GPU fusion around coordinate compaction. Cleanup can run in
parallel because it reads only object occupancy and writes force bits. It uses
the original selected coordinates: duplicate coordinates commute, truncation
is preserved, and ACTION-only cells retain their previous behavior. Actual
movement resolution remains sequential.

Force extraction can replace a boolean channel gather with a slice over
`(layer, force, height, width)`, followed by the same spatial scan order.
For compaction, two `optimization_barrier` calls separate mask construction,
prefix accumulation, and scatter. The barriers are compiler tuning supported
by measurements on the recorded JAX/hardware versions, not a guarantee for
other compilers or devices.

The retained combination is parallel cleanup, sliced force extraction, and
both barriers. The completed H200 combined sweep (job 19063941, 15 alternating
trials, exact full-output comparisons) improves every tested configuration:

| Game | Batch 256 | Batch 1024 | Batch 4096 | Batch 16384 |
| --- | ---: | ---: | ---: | ---: |
| Sokoban | 1.234× | 1.226× | 1.207× | 1.202× |
| Blocks | 1.615× | 1.591× | 1.505× | 1.473× |
| Zen | 1.151× | 1.150× | 1.153× | 1.152× |

At batch 4096, updated throughput is 2,799,691 / 1,862,750 / 2,978,922 steps/s
for Sokoban / Blocks / Zen. At batch 16384 it is 5,775,331 / 2,436,949 /
4,250,720. Adding the slice gives no consistent further gain once both
barriers are present. These figures compare against the preceding engine,
not the repository's original unoptimized engine. The sweep compares
13,056,000 transitions across its 24 A/B configurations, all exactly equal.

The performance gain costs some temporary storage. For example, Zen at batch
16384 increases XLA's temporary-buffer estimate from 208.6 to 349.7 MB. This
estimate excludes full trajectory output storage and is not total GPU memory.

Raw H200 combined measurements: [Sokoban](results/2026-10-02-h200-combined-refinements-0.json),
[Blocks](results/2026-10-02-h200-combined-refinements-1.json),
[Zen](results/2026-10-02-h200-combined-refinements-2.json).
The runtime candidates are explicit in `benchmark_movement_refinements.py`;
its source snapshot uses the pre-promotion engine, while the chunking runs
and CPU tests use the promoted engine (SHA-256 `38ab0cee2e8eb6d22ce4e2bb9780d3b9862f1eddaf0d2fb4aadb0900c3cea03c`).

The complete RTX 4090 sweep (JAX 0.9.1) also improves Sokoban and Blocks at
all four batch sizes. At batch 4096, gains are 1.171× / 1.353× / 1.030× for
the three games. **Zen at batch 16384 regresses to 0.878×.** A separate
cleanup-only A/B with identical slice/barrier settings isolates a 0.829×
cleanup speed ratio there. Removing the barriers does not fix this case
(the cleanup-plus-slice combination is 0.844× versus the preceding engine).
This is an explicit exception to the H200 gains, not a universal speedup
claim. Chunking is a useful alternative for this RTX workload, as measured
below; it should be chosen per workload and device.

Raw RTX measurements: [complete combined sweep](results/2026-10-02-rtx4090-combined-sweep.json),
[cleanup isolation](results/2026-10-02-rtx4090-zen-cleanup.json),
[without barriers](results/2026-10-02-rtx4090-zen-without-barriers.json).

Initial H200 ablations (job 19062051, JAX 0.9.0.1, 15 alternating trials) at
batch 4096, 100 steps, full rollout outputs:

| Candidate | Sokoban | Blocks | Zen |
| --- | ---: | ---: | ---: |
| Parallel cleanup | 1.103× | 1.229× | 0.995× |
| Slice instead of force-channel gather | 1.065× | 1.161× | 1.035× |
| Barrier after prefix sum only | 0.959× | 0.937× | 0.916× |
| Barriers before and after prefix sum | 1.090× | 1.231× | 1.164× |

Each row is a separate A/B experiment. Layout and barrier ablations already
include parallel cleanup in both variants, so these numbers must not be
multiplied to estimate combined speedup. The prefix-only barrier is rejected.
Cleanup also improves the three games at batch 256 by 1.146× / 1.440× / 1.109×,
and at batch 1024 by 1.120× / 1.424× / 1.070×.

Raw measurements: [cleanup](results/2026-10-02-h200-cleanup-refinements.json),
[layout](results/2026-10-02-h200-layout-refinements.json), and
[compaction barriers](results/2026-10-02-h200-compaction-refinements.json).

The combined benchmark replaces all three hooks for its baseline, reproducing
the preceding engine (sequential cleanup, channel gather, unbarriered prefix
compaction). Both variants retain the earlier distance-transform, rule-order,
and independent-cell optimizations. Every reset and rollout leaf must match
before timing. `--candidates cleanup_barriers all` compares implementations
with and without the slice in otherwise combined candidates.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_movement_refinements \
  --experiment combined --candidates cleanup_barriers all \
  --batches 256 1024 4096 16384 --steps 100 --trials 15 \
  --output /tmp/h200-combined.json

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_chunked_rollouts \
  --batch 16384 --chunks 4096 --steps 100 --trials 9 \
  --output /tmp/h200-chunked.json
```

Chunking preserves the global environment index in every `fold_in` PRNG call,
uses identical runtime action arrays, and restores the original output order.
It compares one batch of 16,384 against four sequential full-rollout chunks of
4,096, including every intermediate state, observation, reward, done flag,
info field, and the final master key. Large comparisons run on-device to avoid
duplicating trajectories in host RAM. `--mode final` is available for separate
final-carry measurements; it must not be mixed with full-output results.

H200 chunking results (nine alternating trials, batch 16,384, four chunks
of 4,096, 100-step full trajectories):

| Game | Monolithic steps/s | Chunked steps/s | Speedup |
| --- | ---: | ---: | ---: |
| Sokoban | 5,763,430 | 2,816,437 | 0.489× |
| Blocks | 2,435,603 | 1,855,188 | 0.762× |
| Zen | 4,280,507 | 3,027,184 | 0.707× |
| Atlas Shrank | 40,335 | 51,468 | 1.276× |

All 6,553,600 compared transitions match exactly. Chunking stays opt-in:
Atlas benefits on H200, while the other three games prefer the full batch.
The corresponding RTX results below have a different preferred configuration;
choose using measurements on the intended workload and software/hardware stack.

Raw H200 chunking measurements: [Sokoban](results/2026-10-02-h200-chunked-rollouts-0.json),
[Blocks](results/2026-10-02-h200-chunked-rollouts-1.json),
[Zen](results/2026-10-02-h200-chunked-rollouts-2.json),
[Atlas](results/2026-10-02-h200-chunked-rollouts-3.json).

On H200, the controlled Atlas comparison finishes at 40,335 steps/s for
one batch and 51,468 steps/s for four chunks: **1.276× throughput** with all
1,638,400 transitions exactly equal. Temporary buffers increase from 5.48 to
6.14 GB, while full output storage stays 9.44 GB; chunking is not a general
memory-saving guarantee. The two rollout compilations take 279 and 286 seconds
and are excluded from these execution timings. The complete Atlas job took
27m30s including initialization, compilation, validation, and timing.
[Raw Atlas chunking measurements](results/2026-10-02-h200-chunked-rollouts-3.json).

RTX 4090 chunking results (machine 210, JAX 0.9.1, nine alternating trials):

| Game | Monolithic steps/s | Chunked steps/s | Speedup | Temporary buffers, mono → chunks |
| --- | ---: | ---: | ---: | ---: |
| Sokoban | 5,747,298 | 3,264,104 | 0.568× | 96.2 → 75.3 MB |
| Blocks | 1,262,762 | 2,035,538 | 1.612× | 317.4 → 138.8 MB |
| Zen | 2,539,753 | 3,382,185 | 1.332× | 349.7 → 144.1 MB |

These are identical-stream comparisons at batch 16,384 and chunk size 4,096;
all 4,915,200 compared transitions match exactly. Full output storage is
unchanged. Temporary buffers are XLA estimates, not total device memory.
Chunking remains optional: it helps Blocks and Zen on this GPU but slows
Sokoban by 43%. These are collection rollouts with precomputed actions, not
an end-to-end policy-training benchmark. The RTX results use a different JAX
version from Torch and should not be interpreted as a controlled hardware
comparison. [Raw chunking measurements](results/2026-10-02-rtx4090-chunked-rollouts.json).

The new cleanup oracle tests exhaust one-cell object/force combinations and
check multi-layer, sparse/dense, duplicate, padded, truncated, ACTION-only,
and auxiliary-channel cases. Layout tests use distinct channel labels to
expose permutations. Chunk tests include stochastic trajectories across chunk
boundaries and an actual random-rule game with automatic resets.

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_custom_action_sequences.py \
  tests/test_padding_mask_jax.py tests/test_env_switch.py \
  tests/test_independent_cell_rules.py tests/test_nonzero_compaction.py \
  tests/test_invalid_force_cleanup.py tests/test_movement_refinements.py \
  tests/test_chunked_rollouts.py
```

The complete command above finished with **98 passed and 9 existing expected
failures** (25m51s, CPU JAX 0.9.0). This includes the full standard JavaScript
replay suite, focused switch replays, padding regressions, and the new tests.
There were no new replay mismatches. The large switch-backend replay excluded
from the previous focused matrix remains outside this validation run.
[Validation record](results/2026-10-02-refinements-validation.json).
The [source and environment record](results/2026-10-02-refinements-environment.json)
identifies the isolated snapshots and successful SLURM jobs.

`movement_refinements.slurm` provides a seven-task launcher: three combined
movement sweeps and four chunking games. Submit from the repository root with
your account; override `--gres`, `--partition`, or the array concurrency as
needed. `MOVEMENT_BENCHMARK_OUTPUT_DIR` selects the destination.

Updated paired plots are saved as vector PDF and 300-DPI PNG in
[`paper/figures/movement_refinements_20261002`](../../paper/figures/movement_refinements_20261002/).
They include H200 ablations, combined gains across batch sizes on each GPU,
and the controlled chunking comparisons. The plot manifest records input
hashes, engine hashes, JAX versions, hardware, and the plotter hash.

```bash
results_dir=scripts/benchmarks/results
MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_movement_refinements \
  --ablation-results "$results_dir"/2026-10-02-h200-{cleanup,layout,compaction}-refinements.json \
  --combined-results "$results_dir"/2026-10-02-h200-combined-refinements-{0,1,2}.json \
    "$results_dir"/2026-10-02-rtx4090-combined-sweep.json \
  --chunk-results "$results_dir"/2026-10-02-h200-chunked-rollouts-{0,1,2,3}.json \
    "$results_dir"/2026-10-02-rtx4090-chunked-rollouts.json \
  --output-dir paper/figures/movement_refinements_20261002
```

The earlier eight-game paper curves below retain their original source hash
and snapshot. They precede this follow-up; the new paired experiments use
full trajectory outputs and fixed action streams, so their absolute FPS
must not be substituted into those final-carry curves.

## Movement coordinate compaction and updated paper figures

Movement now uses `_compact_nonzero_coords` instead of `jnp.argwhere` to build
its force-coordinate list. A prefix sum assigns each matching cell its output
position, then a scatter writes its flat index. Nonmatches and excess matches
are dropped. The resulting coordinates preserve the original scan order,
fixed capacity, truncation, and `-1` padding. The movement updates themselves
remain sequential. This avoids the contended histogram scatter used by the
old `argwhere` lowering.

`test_nonzero_compaction.py` exhausts small 1D/2D/3D masks and capacities,
including zero capacity, empty input, truncation, and padded output. It also
checks sparse/dense masks with the real movement-array dimensions under
JIT/vmap against an independent NumPy oracle (8 tests).

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_nonzero_compaction.py
JAX_PLATFORMS=cpu python -m pytest -q tests/test_custom_action_sequences.py \
  tests/test_padding_mask_jax.py tests/test_env_switch.py tests/test_independent_cell_rules.py
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_movement_coordinates \
  --batches 256 1024 4096 --steps 100 --trials 15 \
  --output /tmp/h200-movement-coordinates.json
```

H200 job 19017202, JAX 0.9.0.1, 15 alternating A/B trials after compilation
and warmup. Both variants retain the previous independent-cell optimization.
Every reset and rollout leaf matches over 1,612,800 transitions. At batch 4096:

| Game | Argwhere steps/s | Compaction steps/s | Speedup |
| --- | ---: | ---: | ---: |
| Sokoban | 2,293,604 | 2,329,523 | 1.016× |
| Blocks | 1,154,930 | 1,229,473 | 1.065× |
| Zen Puzzle Garden | 2,381,287 | 2,587,002 | 1.086× |

The coordinate microbenchmarks at batch 4096 improve 1.13× / 1.26× / 1.27×
on 7×7 / 12×12 / 16×16 boards with three movement layers. For 12×12, estimated
temporary storage falls from 14.9 MB to 8.9 MB; whole-rollout temporary buffers
are largely unchanged. RTX 4090 whole-rollout gains at batch 4096 are
1.036× / 1.130× / 1.168× for the same three games.

Raw [H200 results](results/2026-10-02-h200-movement-coordinates.json) and
[RTX 4090 results](results/2026-10-02-rtx4090-movement-coordinates.json).
The [H200 environment record](results/2026-10-02-h200-environment.json) includes
Python, JAX/JAXlib/CUDA-plugin versions, driver, engine hash, and job IDs.
The two CPU test commands above completed with 72 passes and 9 existing
expected failures in total, including the complete standard JS action replay
suite and focused switch regressions. There were no new replay mismatches.

Follow-up H200 job 19019964 profiled the updated engine at batch 4096 over
100 steps. The transitions traces contain 40,680 / 55,076 / 17,655 GPU kernels
for Sokoban / Blocks / Zen. The movement-coordinate scatter fusion still
accounts for 20.0 / 79.6 / 38.0 ms of aggregated traced kernel time. Inspection
of the Blocks HLO confirms that this fusion includes the force-channel gather
and part of the prefix sum as well as the final scatter; its duration should
not be attributed to the scatter alone. Optimizing that fused computation,
sequential movement, and short-kernel/control overhead remain concrete targets.
These traces do not establish a performance ceiling. Profiled timings include
instrumentation overhead and are separate from the unprofiled A/B speedups.

Raw [updated trace summaries](results/2026-10-02-h200-optimized-profiles.json)
and [accompanying unprofiled timings](results/2026-10-02-h200-optimized-profile-timings.json).
Full Perfetto traces and optimized HLO are preserved in
[`archives/2026-10-03-torch-profiles.tar.gz`](results/archives/2026-10-03-torch-profiles.tar.gz),
with a [per-file SHA-256 manifest](results/archives/2026-10-03-torch-profiles.sha256).
The archive contains `profiles-current-h200/`, `profiles-cells/`, and
`profiles-scaling/`. All 28 files were verified before replacing the raw Torch
directories with `/scratch/se2161/puzzlejax-perf-20261002/profiles-archive-20261003.tar.gz`
to recover account disk quota for further benchmarks.

`benchmark_paper_throughput.py` reproduces the workload behind the selected
eight-game figure in `plots/profiling/random_rollout_profile_5000-step_rollout_select.png`:
level zero, random actions generated inside the compiled scan, final carry
only, automatic reset on terminal states, and no practical episode time limit.
It uses `max(5000 // batch, 100)` steps per call (5000 at batch one), continues
the carry across calls, excludes compilation and warmup, and records five
synchronized trials. The updated statistic is median throughput with IQR;
all measured batch sizes are shown, including any throughput declines.
The IQR describes five continuing rollouts, not a confidence interval across
independent seeds. Game-state evolution affects throughput: for example, Lime
Rick at batch 16384 rises from 364k to 490k steps/s over the five samples, with
a 430k median. These measurements warm the executable, but do not assume a
stationary distribution of game states.
These final-carry timings must not be mixed with the full-output A/B timings.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_paper_throughput \
  --game-index 0 --output-dir /tmp/paper-throughput-h200

MPLCONFIGDIR=/tmp/puzzlejax-mpl python -m scripts.plotting.plot_engine_throughput \
  --paper-results scripts/benchmarks/results/2026-10-02-paper-h200 \
  --movement-results scripts/benchmarks/results/2026-10-02-h200-movement-coordinates.json \
  --output-dir paper/figures/throughput_20261002
```

Run indices 0–7 for the eight games; Torch array 19017419 started with a limit
of two H200s, raised to four for the final independent games. The reusable launcher is
`sbatch --account=torch_pr_84_general scripts/benchmarks/paper_throughput.slurm`
(submit from the repository root; change the account for other clusters).
Raw rows include engine SHA-256, hardware, JAX version,
rollout compilation times, every execution sample, and the rollout configuration.
Initialization and reset setup are also excluded from timed rollouts; their
cost is not included in the recorded `compile_s` field.
The plotter preserves the historical Core i9-9980XE C++ and NodeJS reference
curves and explicitly labels them as archived CPU results. All 32 CPU series
were checked for exact equality with the existing plotter's values. Their existing
selection/statistics differ from the new JAX medians, so the overlay is context,
not a new controlled CPU/GPU comparison. Figures are saved as vector PDF and
300-DPI PNG alongside a manifest with input hashes.

The completed sweep contains 48 configurations and 240 timed samples. Peak
median throughput among the six tested batch sizes is:

| Game | Best tested batch | Environment steps/s |
| --- | ---: | ---: |
| Sokoban Basic | 16,384 | 4,810,560 |
| Notsnake | 16,384 | 6,521,537 |
| Zen Puzzle Garden | 16,384 | 3,716,457 |
| Slidings | 16,384 | 1,513,346 |
| Lime Rick | 16,384 | 429,729 |
| Kettle | 16,384 | 400,737 |
| Take Heart Lass | 16,384 | 1,749,144 |
| Atlas Shrank | 4,096 | 49,518 |

Atlas falls to 39,342 steps/s at batch 16,384 (20.6% below batch 4096).
The largest tested batch is therefore not a universal choice, and this sweep
does not establish optimal batch sizes beyond its six measured points.

The [paper-style figure](../../paper/figures/throughput_20261002/random_rollout_profile_h200_updated.pdf)
and [paired movement comparison](../../paper/figures/throughput_20261002/movement_coordinate_speedup_h200.pdf)
have PNG counterparts in the same directory. The complete per-game measurements
are in [the H200 paper results directory](results/2026-10-02-paper-h200/).

## Batch scaling, GPU profiling, and independent cell rules

`benchmark_rollout_scaling.py` compares full validation trajectories against
observations, actions, rewards, and done flags plus the final state. Both use
`env.step`, including automatic resets. The smaller output records the returned
next observation, including reset observations. It is a collection benchmark,
without a policy network or optimizer, rather than an end-to-end training loop.
Batch prefixes receive identical actions and PRNG streams. Every retained
output is checked against the full trajectory before timing.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_rollout_scaling \
  --batches 256 1024 4096 --steps 100 --trials 9 \
  --profile-dir /tmp/puzzlejax-profiles --profile-batch 4096 \
  --output /tmp/puzzlejax-scaling.json
```

Profiling captures one warmed `transitions` call after ordinary timing, with
offline Perfetto traces and optimized HLO saved under each game/batch directory.
Compilation and correctness comparisons are outside both timing and tracing.
Use `summarize_gpu_trace.py path/to/perfetto_trace.json.gz --output summary.json`
to count kernels and merge their activity intervals. Its active fraction means
time covered by recorded GPU activity, **not SM utilization**; trace durations
also include profiler overhead. Use the unprofiled A/B medians for speedups.
Add `--sequential-cell-rules` to reproduce the pre-optimization scaling sweeps
below; the default measures the current production implementation.

The rule compiler now parallelizes deterministic rules containing exactly one
kernel with one cell on each side. Each point detector/projector is reused on
an isolated cell, including metadata binding, movement bits, and validity-mask
checks. Such replacements cannot invalidate other cells' matches, and consume
no random numbers. All other rules retain sequential application; random-group
selection retains its existing `apply_one` path. This removes coordinate
compaction and match loops for eligible rules without changing rule/group order.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_independent_cells \
  --games sokoban_basic blocks Zen_Puzzle_Garden --batches 256 1024 4096 \
  --steps 100 --trials 15 --output /tmp/puzzlejax-cells.json

JAX_PLATFORMS=cpu python -m pytest -q tests/test_independent_cell_rules.py \
  tests/test_padding_mask_jax.py tests/test_custom_action_sequences.py \
  tests/test_env_switch.py
```

The A/B benchmark disables only the independence proof for its sequential
baseline. It checks every reset and rollout output leaf exactly. Tests also
exercise dense/sparse matches, empty input/output cells, metadata properties,
movement modifiers, commands, padding, and rejection of random/multi-cell rules.

### Preliminary RTX 4090 results (2026-10-02)

Machine 210, JAX 0.9.1. These are separate from the H200 measurements below;
different hardware and JAX versions prevent a controlled hardware comparison.
The single-cell change improves Zen Puzzle Garden by 1.561× / 1.414× / 1.317×
at batches 256 / 1024 / 4096. Sokoban and Blocks have no eligible rules and stay
within 0.2%. All 1,612,800 A/B transitions match exactly. At batch 4096, the
Zen trace drops from 30,012 to 18,952 GPU kernels per 100-step rollout.

The baseline batch sweep shows that larger batches help substantially, but the
best size depends on the game. At 4096, full-rollout throughput is 2.69M / 1.35M /
2.13M steps/s for Sokoban / Blocks / Zen. At 16384, these become 5.08M / 0.95M /
1.66M. Removing diagnostic trajectory state has little effect on execution time
despite roughly halving output storage. These scaling runs disable the new
single-cell path, isolating batching from the compiler optimization.

Raw measurements: [scaling](results/2026-10-02-rtx4090-scaling.json),
[batch 16384](results/2026-10-02-rtx4090-scaling-16384.json),
[single-cell A/B](results/2026-10-02-rtx4090-independent-cells.json),
[trace summaries](results/2026-10-02-rtx4090-profiles.json).

### H200 scaling and profile results (2026-10-02)

Torch H200, driver 610.43.02, JAX 0.9.0.1. The baseline sweep and traces are
from job 19014716 (4m40s), with the single-cell optimization disabled.
Nine alternating full/transitions trials follow compilation and warmup.

| Game | Batch 256 | Batch 1024 | Batch 4096 | Batch 16384 |
| --- | ---: | ---: | ---: | ---: |
| Sokoban | 199,529 | 757,535 | 2,287,155 | 4,637,000 |
| Blocks | 155,222 | 512,510 | 1,158,526 | 1,525,082 |
| Zen Puzzle Garden | 265,103 | 863,735 | 1,877,654 | 2,609,572 |

Values are full-output environment steps/s. Moving from 256 to 4096 improves
throughput by 11.5× / 7.5× / 7.1× respectively. Smaller rollout outputs roughly
halve output buffers but improve throughput by at most 4.7% in this sweep.
Batch 16384 was measured in the follow-up job, also with single-cell
parallelization disabled. It improves all three games on H200, unlike the
RTX 4090 results. This is the largest tested batch, not a demonstrated optimum.
Larger batches trade longer individual rollout latency and memory for throughput.

Warmed transitions traces at batch 4096 contain 41,876 / 57,071 / 30,507 GPU
kernels per 100-step rollout for Sokoban / Blocks / Zen. Median kernel durations
are 1.47 / 1.41 / 1.66 µs; activity intervals cover 47% / 59% / 62% of each trace.
The movement-coordinate `argwhere` lowers to a conspicuous `scatter-add`: its
aggregated kernel duration is 22 / 91 / 43 ms in these traces. The output sizes
(127 / 430 / 433 indices) match the movement force-list capacities in
`apply_movement`, rather than the small rule-coordinate lists. These traces
motivated the stable movement-coordinate compaction measured above.
The remaining sequential movement updates and small-kernel launch/control
overhead also merit attention. Profiling perturbs execution, so activity fractions
do not quantify unprofiled idle time. The traces do not establish a performance ceiling.

Raw [H200 scaling measurements](results/2026-10-02-h200-scaling.json) and
[batch-16384 measurements](results/2026-10-02-h200-scaling-16384.json), plus
[trace summaries](results/2026-10-02-h200-profiles.json).

### H200 independent-cell optimization

Job 19015783 (13m42s including the extended sweep), the same H200 node as the
baseline sweep. Fifteen alternating
A/B trials per workload compare the sequential and parallel rule paths, with
the direct coordinate enumeration and distance-transform changes enabled in both.

| Zen Puzzle Garden batch | Sequential steps/s | Parallel steps/s | Speedup |
| --- | ---: | ---: | ---: |
| 256 | 263,118 | 400,245 | 1.521× |
| 1024 | 862,339 | 1,205,271 | 1.398× |
| 4096 | 1,865,287 | 2,383,399 | 1.278× |

The optimized program has 14 `while` operations instead of 22. Estimated
temporary buffers at batch 4096 shrink from 56.9 MB to 52.2 MB. Sokoban and
Blocks have no eligible rules and remain within 1.3% of their controls, with
overlapping timing ranges. All
1,612,800 transitions match exactly, including every observation, state, reward,
done flag, info field, and final PRNG key; reset outputs also match.

[Raw H200 single-cell A/B measurements](results/2026-10-02-h200-independent-cells.json).

The corresponding batch-4096 transitions trace drops from 30,507 to 18,655
GPU kernels per 100-step rollout. The post-change trace is included in the
H200 trace summaries above.

CPU validation after the single-cell change: 19 new guard/point-parity tests
passed; the complete standard JavaScript replay suite, 12 focused switch
replays, padding regression, and switch reset/turn/group tests produced 45
passes and the same 9 existing expected failures. Total: **64 passed, 9 xfailed**
across the distinct tests. No new mismatches were observed. The previously
unvalidated large switch replay remains outside this focused switch matrix.

## Nearest-target Manhattan heuristics

`benchmark_manhattan_distance.py` compares the original all-pairs implementation
with the production distance transform. Both preserve the `H + W` fallback for
missing targets/sources; the sum heuristic counts only real sources.

```bash
source .venv/bin/activate
mkdir -p data/game_trees
python -m scripts.benchmarks.benchmark_manhattan_distance \
  --sizes 8 16 32 --batches 1 64 256 \
  --games sokoban_basic blocks Zen_Puzzle_Garden \
  --steps 100 --trials 7 --output /tmp/manhattan-results.json
```

The microbenchmark times both distance reductions together. Each call takes
runtime inputs and synchronizes its output, without a compiled loop repeating
unchanged inputs. Compilation, warmup, and timed execution are separate.
Timed executions alternate A/B order after both versions compile and warm up.
Compile times are single samples in baseline-first order and can be affected
by compiler caches; use the repeated execution timings to assess throughput.
`temporary_bytes` is XLA's compiled temporary-buffer estimate, not total device
memory usage.

The game benchmark uses `env.step`, including automatic reset, in a compiled
scan of evolving states and a batch of independent random action sequences.
Both implementations receive identical initial states, actions, and PRNG keys.
It compares every returned observation, state, reward, termination flag, and
info field exactly. These timings include trajectory output storage; they are
not training throughput measurements. `Zen_Puzzle_Garden` has a `no` win
condition and serves as a control that does not use distance heuristics.

The standalone correctness tests use a NumPy oracle, exhaust all source/target
masks on several small grids, and cover sparse, dense, empty, and rectangular
inputs under JIT and vmap:

```bash
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_manhattan_distance_transform.py tests/test_jax_heuristic_fixes.py
```

For gameplay compatibility, run the existing JavaScript replay harness too:

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_custom_action_sequences.py
```

Use the same checkout and device for each A/B comparison. Report full rollout
results separately from heuristic microbenchmarks: serial rule application and
movement can dominate step time even when the heuristic becomes much cheaper.

`benchmark_env_subsystems.py` now repeats synchronized calls on the host and
returns the complete subsystem output. Its old compiled repetition loop let
XLA hoist the computation and only repeat an addition; old per-repetition
numbers are invalid. `--inner-reps` now means host calls per trial (default 10).
Timings include host dispatch overhead; batched costs are amortized per
environment. The tick benchmark supplies the required `do_again=False` argument.

## Validation of the distance-transform change (2026-10-02)

- Existing CPU tests (`test_custom_action_sequences`, `test_jax_heuristic_fixes`,
  `test_padding_mask_jax`, `test_apply_player_force`): 39 passed and 9 expected
  failures before and after the change. The expected failures are existing
  JavaScript/JAX replay mismatches, not newly introduced failures.
- New independent NumPy-oracle tests: 8 passed. The H200 also passed all 5,012
  source/target pairs from the exhaustive and randomized checks.
- H200 gameplay A/B validation: all observation, state, reward, termination,
  and info fields matched for three games at batches 1, 64, and 256, with 100
  steps each (96,300 environment transitions in total).

The performance experiments use the standard `PuzzleJaxEnv`.

## Switch backend repairs

`PuzzleJaxEnvSwitch` now shares the standard reset, rule compilation, and turn
logic. This fixes the invalid `PJState(level_i=...)` construction, stale tick
signature and level shape, startup scoring, and multi-level padding behavior.
It retains dynamic switch dispatch but applies each rule once per group pass;
the group owns convergence and remembers effects from every rule. Late rules
retain their original block and loop grouping.

`tests/test_env_switch.py` compares complete batched reset and rollout outputs
(including automatic reset) with the standard backend, checks that compilation
does not leak dispatch-table tracers, and directly exercises rule/group ordering.
The JavaScript replay tests retain the full standard-backend suite and also
exercise the switch backend on the small custom regression games and Sokoban.
Full switch-backend replay is available explicitly:

```bash
JAX_PLATFORMS=cpu python -m puzzlejax.validate_actions --backend switch \
  --output-dir /tmp/puzzlejax-switch-validation
```

The switch backend remains experimental. On this CPU, its three-move
`Doors_and_Boxes` replay did not finish within 20 minutes and was interrupted;
the standard backend passed that case. That large switch case is unvalidated,
and the repairs should not be interpreted as a switch-backend speedup claim.

```bash
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_env_switch.py tests/test_benchmark_env_subsystems.py \
  tests/test_custom_action_sequences.py
```

## Match coordinate enumeration

The sequential replacement path previously collected row-major coordinates,
constructed ordering keys, sorted every padded list, and gathered the result.
`_ordered_match_coords` now enumerates row-major masks directly and transposes
column-major masks before enumeration, then swaps coordinate components back.
The rule's direction is static at compilation. Padding, match order, mixed-radix
tuple order, and sequential rechecks are preserved. No parallel application of
overlapping replacements is introduced.

`test_match_coordinate_order.py` exhausts all masks on five small shapes and
checks sparse/dense rectangular masks, all eight scan-direction combinations
for three independent kernels, and empty/full masks under JIT and vmap.
`benchmark_coordinate_order.py` retains the original sort path as its baseline,
compares every output leaf, and reports optimized HLO sort counts as well as
timings. Both variants use the same current heuristics and turn logic.

CPU validation after these changes, completed across separate invocations:

- 41 unit/parity checks passed (heuristics, coordinates, padding, player force,
  subsystem timing, and switch reset/turn/group behavior).
- All 33 existing standard-backend JavaScript replay cases completed:
  24 passed and the same 9 expected failures remained.
- All 12 focused switch-backend JavaScript replays passed.
- Total: 77 passed, 9 existing expected failures. The interrupted large switch
  replay described above is excluded from that completed test matrix.

The corrected subsystem benchmark also completed all four cases on CPU.

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/test_match_coordinate_order.py
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_coordinate_order \
  --sizes 8 16 32 64 --batches 1 64 256 \
  --games sokoban_basic blocks Zen_Puzzle_Garden \
  --steps 100 --trials 15 --output /tmp/h200-coordinates.json
```

## H200 coordinate results (2026-10-02)

Torch, NVIDIA H200 (143,771 MiB), driver 610.43.02, JAX 0.9.0.1.
SLURM job 19009988 completed in 4m13s. Measurements use 15 alternating A/B
trials after both compilations and warmup. All 96,300 rollout transitions
matched exactly (three games, batches 1/64/256, 100 steps). These compare the
sort change alone, with the distance transform enabled in both variants.

[Raw coordinate measurements](results/2026-10-02-h200-coordinates.json).

| Workload (batch 256) | Sort baseline | Direct enumeration | Speedup |
| --- | ---: | ---: | ---: |
| 8×8, two coordinate lists | 96.3 µs | 58.5 µs | 1.648× |
| 16×16, two coordinate lists | 105.9 µs | 93.0 µs | 1.139× |
| 32×32, two coordinate lists | 145.2 µs | 116.2 µs | 1.250× |
| 64×64, two coordinate lists | 303.4 µs | 158.8 µs | 1.911× |
| sokoban_basic full rollout | 192,082 steps/s | 196,472 steps/s | 1.023× |
| blocks full rollout | 158,808 steps/s | 158,729 steps/s | 1.000× |
| Zen_Puzzle_Garden full rollout | 246,436 steps/s | 262,815 steps/s | 1.066× |

Sokoban and Zen each go from eight optimized HLO sort operations to zero.
`blocks` already has zero sorts in the baseline and is effectively unchanged.
At batch 64, the respective full-game improvements are 3.1% and 8.9%.
The 64×64 coordinate microbenchmark's estimated temporary buffers shrink from
33.6 MB to 17.0 MB; whole-game memory savings on these small boards are minor.
Microbenchmarks include host dispatch overhead, and rollouts include trajectory
storage. Compilation samples are baseline-first and cache-sensitive; they are
not evidence of a compilation-speed guarantee. Small batch-one rollout timing
ranges overlap, so prefer the batched results when assessing this change.

## H200 distance-transform results (2026-10-02)

Torch, NVIDIA H200 (143,771 MiB), driver 610.43.02, JAX 0.9.0.1.
SLURM job 19008109 completed successfully. These results use 15 alternating
A/B trials, batch size 256, level 0, and seed 42. The source was local
commit `602d419e` plus the distance-transform change.

[Raw measurements](results/2026-10-02-h200-manhattan.json).

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 8×8 heuristic batch | 115.5 µs | 69.1 µs | 1.672× |
| 16×16 heuristic batch | 525.9 µs | 73.9 µs | 7.114× |
| 32×32 heuristic batch | 399.6 µs | 74.4 µs | 5.369× |
| sokoban_basic full rollout | 192,783 steps/s | 195,211 steps/s | 1.013× |
| blocks full rollout | 155,094 steps/s | 160,374 steps/s | 1.034× |
| Zen_Puzzle_Garden full rollout | 249,780 steps/s | 249,907 steps/s | 1.001× |

The whole-engine improvement is modest for these small boards: approximately
1.3% for Sokoban and 3.4% for blocks. The control is effectively unchanged
(0.05%). The Sokoban timing ranges overlap; avoid treating its small median
difference as a general speedup guarantee. The larger heuristic gains do not
translate directly into whole-engine throughput because rule application and
movement still dominate these workloads. Other games, board sizes, levels,
and batch sizes need their own measurements.

Reproduce the final run on a CUDA device:

```bash
source .venv/bin/activate
mkdir -p data/game_trees
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m scripts.benchmarks.benchmark_manhattan_distance \
  --sizes 8 16 32 --batches 256 \
  --games sokoban_basic blocks Zen_Puzzle_Garden \
  --steps 100 --trials 15 --output /tmp/h200-manhattan.json
```

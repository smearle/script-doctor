# Spec: Sokoban-variant world family + scaled model

A hidden-mechanic ("which sokoban am I in?") family for the information-gain
active learner. Same objects and observations across variants; only the **push
rule** differs, so the variant `theta` is hidden and must be discovered by
experiment — the sokoban analog of "press the button / walk to the TV".

## 1. Object family (shared across all variants)

Five canonical objects (fits the existing 32-cell-token bitmask vocab unchanged):

| name | role | sprite color |
|---|---|---|
| `background` | floor | white |
| `target` | goal marker (floor-level) | light gray |
| `wall` | blocker | dark gray |
| `player` | agent | yellow |
| `box` | pushable | brown |

**Collision layers** (this is the one change vs the current toy assembler —
`target` must sit *below* the movement layer so a box can rest on it):

```
Background
Target
Wall, Player, Box
```

A box-on-target cell is simply `box`+`target` bits both set (multihot handles
it); no separate object. Player movement is engine-native (moves in the input
direction unless blocked); variants differ only in what a box does when shoved.

## 2. Variants (theta = which rule-set)

Chosen so that **a single push into open space disambiguates them**, and so the
set spans the cases the IG estimator must handle: unknown-deterministic (3 ways)
and genuinely-stochastic (1 way, to confirm "not mesmerized by randomness").

| variant | push rule(s) | one-push signature | case |
|---|---|---|---|
| **classic** | `[ > Player \| Box ] -> [ > Player \| > Box ]` | box moves exactly 1 | det, informative |
| **inert** | *(no push rule)* | box doesn't move (player blocked) | det, the "noop" |
| **slide** | push + `[ > Box \| no Wall no Box ] -> [ \| > Box ] again` | box slides to first obstacle | det, momentum |
| **chaos** | `[ > Player \| Box ] -> [ Player \| randomDir Box ]` | box flies a random direction | **stochastic** |

Optional 5th once the above work: **multipush**
(`[ > Box \| Box ] -> [ > Box \| > Box ]`, pushes a line) — needs a 2-box
setup to distinguish from `classic`, so it makes identification require richer
configurations.

Distinguishability with one push into >=2 open cells: `inert`=0 cells,
`classic`=1, `slide`=many, `chaos`=1 in a *random* direction (and the haoo'
resample `o'` diverges from `o` only for `chaos` — exactly the reducible-vs-
irreducible signal). `classic` vs `chaos` over a single sample look similar; the
resample (`o != o'`) and repeated probes separate them.

**Risks to compile-test first** (per the test-before-build rule): the `slide`
`again` loop must terminate and move the box cell-by-cell to the first obstacle;
`randomDir` must actually yield a uniformly random direction through the JS->C++
path (validate exactly like `check_double_step.py`). Build/validate the rule
behavior for all 4 variants *before* any training.

## 3. Level distribution

- Small bordered grids, **7x7** (5x5 interior), wall border.
- Place `player`, 1-2 `box`, 1-2 `target`. At least one box must have **>=2
  clear cells** in some push direction (so `slide` is distinguishable from
  `classic`) and **not be adjacent to the player** (so identification requires
  navigation -> informative transitions are sparse under a random policy, the
  regime where active collection wins — cf. the navigation family result).
- Same K layouts baked into all variants (so `obs0` is identical across
  variants -> theta hidden at the start).
- `(no win)` for a pure dynamics/active-learning testbed; optionally
  `all Box on Target` if we later want goal-conditioned planning too.

## 4. Identification probe (mirrors `eval_probes.py`)

- `fresh_unknown`: at `obs0`, IG of the action that pushes toward a box should
  be HIGH (resolves which of the 4 variants).
- `classic_known` / `inert_known` / `slide_known`: after the agent has pushed a
  box once (variant identified), IG of further pushing ~0.
- `chaos_known`: after identifying `chaos`, IG of pushing ~0 **despite** large
  visible change — the "not mesmerized by randomness" check.
- Held-out predictive metric: after k exploration steps, the model's NLL on a
  canonical "push a box into open space" diagnostic transition (a variant-
  identified model predicts the box displacement; an uninformed one does not).
- Collection comparison (online vs offline): random vs IG-planner vs a
  navigate-to-box-then-push oracle, at matched budget — same harness as
  `compare_collection*.py`.

## 5. Scaled model config

Sokoban needs more capacity and context than the 1x6-corridor probe, but the
**vocab is unchanged** (5 objects -> 32 cell tokens; total ~44 tokens incl. a
5th action). Sequence length: 7x7 obs block = 49 cells + 2 markers = 51 tokens;
a ~10-step episode + resample ~= 550-650 tokens.

| field | probe (current) | sokoban (scaled) |
|---|---:|---:|
| `d_model` | 128 | **256** |
| `n_layer` | 4 | **8** |
| `n_head` | 4 | **8** |
| `d_ff` | 384 | **1024** |
| `max_seq_len` | 320 | **1024** |
| params | ~0.66M | **~6.5M** |

Additions:
- **2-D cell position embedding**: add a learned `(row, col)` embedding to each
  cell token within an OBS block (the 1-D RoPE handles the temporal/sequence
  axis; this restores grid geometry the flattening destroys). Cheap, high value.
- Keep next-token CE; keep the `haoo'` RESAMPLE_OBS format unchanged.
- Train ~10-20k updates, batch 64; data gen unchanged (engine rollouts).

This stays a 1-D token transformer — deliberately incremental. It is the
**ceiling** of the token approach: beyond ~8x8 grids or multi-object/multi-game
data, replace the per-cell bitmask + 1-D flatten with a 2-D frame encoder
(small conv/NCA) carrying a recurrent belief state across frames (the NCA-belief
model), which respects geometry and scales to many objects. That is the bridge
to a large multi-game dataset.

## 6. Implementation deltas (onto existing `active_learning/`)

- `worlds.py`: add a sokoban assembler with the 3-layer collision stack (the
  current `rule_game.assemble_game` puts everything on one layer); add a
  `form="sokoban"` with the 4 variant rule-sets; add a `box_pushable` layout
  style (box reachable, >=2 clear cells, not player-adjacent).
- `vocab.py`: rename canonical objects to `[background, target, wall, player,
  box]` (or add a family registry); add 2-D position helper. Vocab size ~unchanged.
- `model.py`: add optional `(row,col)` cell-position embedding; bump config.
- `data.py` / `collect.py` / `eval_probes.py` / `compare_collection*.py`:
  reusable as-is once the family + geometry are set (policies are observation-
  based; navigate-to-box is a 1-line change from navigate-to-seed).
- New `check_sokoban_variants.py` (like `check_double_step.py`): compile all 4
  variants, scripted-push a box, assert the 0/1/many/random signatures.

## Results (steps 2-3, `sokoban_train.py`)

Family validated (`check_sokoban_variants.py`): all 5 variants give the specified
one-push signature through the engine (classic +1, inert 0/blocked, slide to wall
via terminating `again`-loop, swap trades places, chaos randomDir 4/10 distinct).
Family build (`worlds.py form="sokoban"`, 7x8, `box_pushable`): `obs0` identical
across variants (theta hidden); box moves in only ~4-6/60 random 8-step rollouts
(inert 0) — sparse under random.

Scaled model: **6.32M params** (d256/8L/8H/dff1024/ctx1024), trained 8000 updates
on mixed-policy (0.6 navigate) haoo' data; train loss plateaus ~0.027 by step 1k.

**Per-variant held-out predictive NLL** (navigate-collected, resample block):

| variant | NLL |
|---|---:|
| classic | 0.0000 |
| slide | 0.0000 |
| inert | 0.0002 |
| swap | 0.0043 |
| chaos | **0.0160** |

The four deterministic mechanics are predicted exactly; `chaos` floors at 0.016 —
the irreducible-noise signature (a random push direction cannot be predicted to 0).
So the scaled token model learns all five push dynamics in-context.

**Push-IG identification** (mean IG of pushing a box; fresh = variant unknown,
known = after one observed push):

| variant | fresh | known |
|---|---:|---:|
| classic | +1.185 | +0.000 |
| inert | +1.410 | +0.106 |
| slide | +1.334 | -0.000 |
| swap | +0.915 | -0.000 |
| chaos | +1.437 | +0.014 |

Pushing an unknown box is high-IG for every variant and collapses to ~0 once one
push reveals the mechanic. `chaos` +1.44 -> +0.01 is the not-mesmerized check:
after identifying randomness, the agent values poking the box at ~0 despite its
large visible change. (Token IG estimate is slow at 56-cell grids — autoregressive
obs-block decode; batch the `n_samples` decodes for ~8x before step 4 / scaling.)

## 7. Build order

1. `check_sokoban_variants.py` — author + validate the 4 variant rules behave
   as specified (esp. `slide` again-loop and `chaos` randomDir). **Gate.**
2. Sokoban assembler + family + layouts in `worlds.py`; smoke-test obs0 identical
   across variants, sparse-under-random triggers.
3. Scaled model + 2-D positions; train; reproduce the identification probe table.
4. Online-vs-offline + IG-planner collection on the sokoban family.
5. Exploration GIFs (push-to-investigate).

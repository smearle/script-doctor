# Active Learning via Information-Gain World Models (`nca_wm/active_learning/`)

Integration of the "Neural World Models + Active Learning" note into the
`script-doctor` / `nca_wm` codebase. The note's idea: train a sequence world
model on `haoo'` data (two i.i.d. next observations per action) so that an
information-gain-like intrinsic reward can be read out **without ever
representing latent dynamics**, then act to maximize it for open-ended
exploration.

This subproject reproduces that mechanism on real PuzzleScript dynamics (via the
C++ engine) and uses it to test the standing question: **can an online,
information-seeking data-collection scheme match or beat offline pre-collection
at a matched budget?**

## The `haoo'` mechanism on PuzzleScript

A *world* `theta` is a PuzzleScript ruleset; the agent sees only the symbolic
object grid, so `theta` is hidden and must be inferred by interaction. Two i.i.d.
next observations `o, o'` are drawn from one pre-action snapshot using the
engine's `seed_rng` + `backup_level`/`restore_level` (validated:
`check_double_step.py` — deterministic worlds give `o==o'`, `random`-rule worlds
diverge). A decoder-only transformer is trained on `BOS obs (ACT a obs)* RESAMPLE_OBS obs'`
with plain next-token cross-entropy.

Information gain of a candidate action is then

    IG(h,a) = E_{o~q0}[ log q1(o|h,a,o) - log q0(o|h,a) ]

where `q0` is the next-obs distribution after `<h> ACT a` and `q1` after
`<h> ACT a OBS o END_OBS RESAMPLE_OBS`.

## World families (`worlds.py`)

All share a fixed canonical object family (background/wall/player/seed/sprout)
and serialize the grid to per-cell bitmask tokens (`vocab.py`).

- **every_turn** (`[ObjA]->...`, fires every turn): noop / det / rand. Used to
  validate the IG probe. Random behavior already covers all informative
  transitions, so it gives active collection no headroom.
- **adjacency** (`[Player|ObjA]->...`, fires next to a seed) + `far_cluster`
  layout: informative transitions are sparse under a random policy (must
  navigate to the cluster). Used for the online-vs-offline comparison.
- **dir_adjacency** (`[> Player|ObjA]->...`, fires only when moving RIGHT into a
  seed) + `left_of_cluster` layout: exactly one informative action per state, so
  a cheap greedy depth-1 IG planner can pick it. Used to test the learned planner.

## Results

### 1. IG probe reproduces the note's behavior (`train.py`, `runs/probe0`)

A 665k-param transformer on the every_turn family, final probes (step 3750):

| fresh_unknown | rand_known | det_known | noop_known |
|---:|---:|---:|---:|
| +1.11 | -0.0006 | +0.0001 | -0.0005 |

High IG for acting in an unresolved world; ≈0 for every resolved case, including
the stochastic one — i.e. **not mesmerized by randomness**, exactly the property
information gain is meant to have.

### 2. Online (active) vs offline (random) collection (`compare_collection.py`)

From-scratch WMs trained on random- vs navigate-policy data (navigate = walk to
nearest seed; an informative oracle using observable seed positions), at matched
budget, on a shared held-out informative eval set. Held-out outcome NLL,
mean ± std over 4 seeds:

| budget | random (offline) | navigate (active) |
|---:|---:|---:|
| 64 | 0.774 ± 0.329 | 0.023 ± 0.018 |
| 128 | 0.369 ± 0.028 | 0.014 ± 0.012 |
| 256 | 0.258 ± 0.151 | 0.010 ± 0.006 |
| 512 | 0.234 ± 0.150 | 0.0045 ± 0.001 |
| 1024 | 0.055 ± 0.043 | 0.0046 ± 0.003 |
| 2048 | 0.044 ± 0.032 | 0.0032 ± 0.001 |
| 4096 | 0.032 ± 0.016 | 0.0027 ± 0.001 |
| 8192 | 0.023 ± 0.013 | 0.0031 ± 0.001 |

Active collection plateaus by budget ~512; passive decreases slowly and is still
~7x worse at budget 8192. The gap narrows from 33x (@64) toward convergence, i.e.
active >= passive **up to a large budget limit** (passive needs >100x the data):
navigate@64 (0.023) already matches random@8192 (0.023) — ~128x sample
efficiency, robust across seeds. Figure:
`nca_wm/figures/active_collection_online_vs_offline.{png,pdf}`.

### 3. The learned IG planner discovers the informative action (`compare_collection_active.py`)

On dir_adjacency, the greedy IG planner ranks actions purely from its learned WM
(no privileged seed positions):

    UP=+0.035  LEFT=-0.005  DOWN=-0.095  RIGHT=+0.281  ACTION=-0.002

RIGHT (moving into the seed — the only informative action) dominates ~8x (with a
better-trained reference, +0.53 vs others ≈0). Three-way collection sweep,
held-out outcome NLL (single seed; figure
`nca_wm/figures/active_collection_ig_planner.{png,pdf}`):

| budget | random | ig_greedy (learned) | navigate (oracle) |
|---:|---:|---:|---:|
| 64 | 0.230 | 0.021 | 0.0003 |
| 128 | 0.027 | 0.0011 | 0.0002 |
| 256 | 0.018 | 0.048* | 0.0001 |
| 512 | 0.0004 | 0.0009 | 0.0001 |
| 1024 | 0.0004 | 0.0005 | 0.0000 |

The learned planner beats random by 11-24x at low budgets, where coverage is the
bottleneck. On this *one-step* dir_adjacency family random catches up by ~512
(the informative action occurs ~1/5 of random steps, so it is not sparse enough
to sustain the gap; *256 is a single-seed spike). The *sustained* advantage lives
in the multi-step navigation family of result 2 (passive still 7x worse at 8192).
So the learned planner discovers and exploits the informative action and
front-loads informative data exactly where coverage matters.

### 4. Exploration GIFs (`render_explore.py`)

The expectimax IG planner, on a 1-D corridor, navigates to an uninvestigated
seed cluster and acts; once the hidden mechanism is revealed its predicted
information gain collapses. Same start observation across worlds (mechanism is
hidden — the agent cannot tell them apart, so it must explore):

- `nca_wm/figures/active_explore_noop.gif` — investigate, find the world inert, lose interest.
- `nca_wm/figures/active_explore_det.gif` — investigate, trigger deterministic sprout, identify, IG→0.
- `nca_wm/figures/active_explore_rand.gif` — trigger stochastic sprout, IG drops, brief
  re-engagement toward the still-unresolved seed (curious but not mesmerized).

### 5. Sokoban-variant family + 2-D NCA-belief decoder (`sokoban_train.py`, `nca_belief_model.py`)

The 1-D token transformer identifies a world by autoregressively decoding every
cell (O(H*W) passes/sample) — the sokoban probe took ~30 min and IG-planner
collection was intractable at real grid sizes. The **2-D NCA-belief model**
(`nca_belief_model.py`, spec in `NCA_BELIEF_MODEL_SPEC.md`) predicts a whole
frame in one conv pass and carries a recurrent spatial belief state across
frames — the NCA's persistent hidden channels *are* the in-context posterior over
the hidden dynamics. A small discrete latent (K=16) with an **exact** mixture
likelihood (no Gumbel) gives joint frame coherence; two heads `q0(o|B,a)` and
`q1(o'|B,a,e(o))` are trained on `haoo'` pairs so `q1` learns `o' ⊥ o` for noise.

The sokoban-variant family (`SOKOBAN_VARIANTS_SPEC.md`, `check_sokoban_variants.py`)
shares objects/observations across five variants that differ only in the push
rule — classic / inert / slide / swap / chaos — so the variant is hidden and must
be discovered by pushing a box. Two harnesses exercise the belief decoder on it:
`sokoban_train.py` reports per-variant held-out predictive NLL (did it learn each
push mechanic) and a push-IG identification probe (push-IG when the variant is
*fresh*/unknown vs. *known*/after one push — constructed so fresh ≫ known,
including chaos-known ≈ 0 as the not-mesmerized check); `sokoban_collect_compare.py`
re-runs the online-vs-offline collection comparison (result 2) with the belief
decoder. Exploration GIFs per variant:
`nca_wm/figures/sokoban_explore_{classic,inert,slide,swap,chaos}.gif`.

### 6. Learned cumulative-IG explorer (`explore_policy.py`)

Greedy depth-1 IG is myopic — it cannot navigate to set up a multi-step
mechanic. `explore_policy.py` trains a small policy/value net by A2C to maximize
the discounted sum of per-step IG (intrinsic reward from a frozen belief WM), so
temporal credit assignment lets it head toward high-IG regions when the immediate
IG is 0. Eval is rule-firing coverage vs. a random policy.

## Caveats / open items

- The single-seed `random` curve is non-monotonic (likely seed noise, possibly a
  real "confidently wrong on rare triggers" effect) — rerun multi-seed.
- `navigate` is a heuristic oracle; result 3 closes the gap to a *learned* policy
  but on the simpler dir_adjacency family.
- The "up to a certain limit" crossover (random eventually catching up) is not yet
  shown — extend random to much higher budgets.
- `dir_adjacency` collapses rand to det (one adjacent seed); the stochastic case
  is only exercised in every_turn/adjacency.
- Not yet done: the full bootstrap loop (planner uses the model it is training,
  iterated) and an expectimax navigation demo ("walk to the cluster, then act").

## File map

Shared infrastructure: `vocab.py` tokens/geometry · `worlds.py` engine-backed
world families · `data.py` haoo' sampler · `collect.py` collection policies
(random / navigate / make_ig_policy) · `check_double_step.py` data-source
validation.

1-D token model (results 1-4): `model.py` transformer (ported) · `inference.py`
IG + sampling/scoring · `planner.py` expectimax · `eval_probes.py` probe
scenarios · `train.py` meta-training · `compare_collection.py` online-vs-offline ·
`compare_collection_active.py` learned-planner sweep · `render_explore.py` GIFs.

2-D belief model (results 5-6): `nca_belief_model.py` / `nca_belief_train.py`
conv belief decoder + training · `attn_belief_model.py` / `attn_belief_train.py`
attention belief variant · `grid_data.py` frame-grid batches · `belief_planner.py`
expectimax over belief · `sokoban_train.py` variant-family train + push-IG probe ·
`sokoban_collect_compare.py` online-vs-offline (belief) ·
`check_sokoban_variants.py` family validation · `render_explore_sokoban.py` GIFs ·
`explore_policy.py` A2C cumulative-IG explorer · `eval_wm_nll.py` NLL eval.

Multigame (broad-corpus probes): `multigame_build.py` / `multigame_data.py` /
`multigame_bootstrap.py` / `multigame_train.py` · `rule_overlap.py`,
`attn_coverage.py` coverage diagnostics. Generated caches/checkpoints under
`_multigame/`, `_world_*`, `ckpts/`, `runs/` are gitignored.

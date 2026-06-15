# In-Distribution Single-Game Modeling — Investigation Log

Goal (2026-05-20): step back from OOD/cross-game transfer and ask the simpler
question first — **can the NCA world model perfectly learn a game's rules when
trained on that game alone?** Paradigm: train on as much data as possible
(all levels, A* search that does NOT stop on solution so it saves off-optimal
transitions too), hold out a fraction of transitions for validation, and seek
**perfect validation accuracy on held-out transitions**. Val is for measurement
only — never for early stopping. Holding out whole *levels* is unfair (later
levels can introduce mechanics absent from earlier ones), so we hold out a
uniform % of transitions across all levels (`--val_frac`).

Plan: go broad eventually, but start with the simplest games and increase
complexity gradually.

---

## Prior evidence: `per_game_arch` grid (2026-05-05, box 210, single-LEVEL)

6 games trained one-at-a-time on `--level 0` only, 4 arch buckets. Eval split by
trained-level (L0) vs held-out levels. Buckets: A=pool ON/shared,
B=pool ON/per-step, C=pool OFF+skip/shared, D=pool OFF+skip/per-step.

BFS rollout cell-error on the **trained level (L0)**:

| game | A pool-ON | C pool-OFF+skip |
| --- | --- | --- |
| sokoban_basic | 0.0% | 0.0% |
| nekopuzzle | 0.0% | 0.0% |
| Microban | 0.0% | 0.0% |
| Heroes_of_Sokoban | 0.0% | 0.0% |
| Bouncers | 0.9% | 0.2% |
| **Travelling_salesman** | **45.1%** | **0.0%** |

Takeaways:
- Pool-ON fits the trained level fine on **5/6 games**. TSM is the only game
  global pooling breaks. So pooling is not broadly harmful — it's a
  TSM-specific pathology.
- On held-out *levels* of the same game, every game degrades (Heroes 26%,
  TSM 40%, Microban 16%, even sokoban_basic 7%) — but that grid trained on a
  single level, so cross-level transfer is confounded with "memorized one
  level." The all-levels + held-out-transition setup here removes that.

Caveat: under-trained (loss-ratio flag on every cell; TSM at 1.85), single
level, run on a different box. Re-verifying here.

---

## TSM pooling pathology — is pool-ON failure real or under-training?

TSM has only LOCAL rules (no `...`, no `[X][Y]`):
```
[ > Player | HLink | Node ]   -> [ NodeSeen | HLinkSeen | Player NodeSeen ] Sfx1
[ > Player | VLink | Node ]   -> [ NodeSeen | VLinkSeen | Player NodeSeen ] Sfx1
[ > Player | No AnyLink ]     -> [Player | ]
[ > Player | AnyLink | NodeSeen ] -> Sfx0
```
So pooling shouldn't be *needed* — pool-ON breaking it would be a real concern
(spurious global signal destabilizing a purely-local computation).

Experiment (`scripts/run_tsm_pool_diag.sh`, GPU 1, launched 2026-05-20 16:09):
2x2 over (pooling x input_skip), all 12 levels, `--val_frac 0.1`,
`--val_eval_interval 500`, 30k steps, `--patience 0` (no early stop),
rule_attn h=256, n_nca_steps=8 shared (n_repeats=8), batch=16, lr=3e-4.
save_dirs: `logs/tsm_pool_diag/{pool_off_skip_on, pool_on_skip_on,
pool_on_skip_off, pool_off_skip_off}`.

### Results (2026-05-20, all 4 cells, 30k steps)

**1-step held-out-transition validation — ALL FOUR cells reach perfect:**
train change_err = 0.0 and **val change_err = 0.0** for pool ON/OFF × skip ON/OFF.
Convergence speed (val change_err first hits 0): pool-OFF+skip ~3.5k, pool-ON+skip
~5–7k, the no-skip cells a bit later. So pooling does NOT impair learning TSM's
1-step transition map.

**Autoregressive rollout (end-of-run eval, mean wrong tiles/level over 12 levels):**

| config | bfs | astar | random_tf (1-step) | random (AR, 50 steps) |
|---|---|---|---|---|
| pool OFF + skip | 0.00 | 0.00 | 0.00 (max 2) | 12.0 (1.93% cells) |
| pool ON + skip | 0.00 | 0.00 | 0.00 (max 2) | 20.0 (3.33%) |
| pool OFF, no skip | 0.00 | 0.00 | 0.00 (max 2) | 9.7 (1.62%) |
| pool ON, no skip | 0.00 | 0.00 | 0.00 (max 1) | 1.7 (0.20%) |

**Verdict — the per_game_arch "pool-ON breaks TSM (45%)" does NOT reproduce
under all-levels training.** With all 12 levels, pool-ON gets perfect BFS/A*/val,
identical to pool-OFF. The 45% was an artifact of single-LEVEL (L0-only)
training: with one level's data, the global-max pool feature overfits spurious
level-specific global statistics that break during AR generation; with 12 levels
it can't. So pooling is **not** dangerous for local-only games *given enough
data* — good news, we can keep it on for `[...]`/`[X][Y]` games. This is the
data-scarcity story again ([[project_tsm_data_dilution]]), not an architecture
flaw.

**Residual:** the only nonzero error is **random-action AR** rollout
(off-policy, 50 steps), 0.2–3.3% cells. It's generic AR error-compounding
(`random_tf` 1-step is perfect, so the map is right; rare 1–2 tile slips
compound over 50 AR steps under random actions reaching odd states). It's not
cleanly pool-ordered (pool-ON+no-skip is *best* at 0.20%, pool-ON+skip *worst* at
3.33%) → noise, not a pooling effect.

Figure: `figures/tsm_pool_diag/curves.{pdf,png}`; table:
`figures/tsm_pool_diag/summary.md`. Launcher: `scripts/run_tsm_pool_diag.sh`;
plotter: `scripts/plot_tsm_pool_diag.py`.

**Implication for the broad sweep:** "perfect in-distribution" should be measured
on **all-levels** training with held-out *transitions* (not levels). Single-level
training manufactures false hardness. On-policy (BFS/A*) rollout + 1-step
held-out val are the clean perfection metrics; random-AR is a separate
(compounding) axis to track but not the primary target.

---

## Shared-rollout eval: random-AR error ⟺ random 1-step error (2026-05-20)

Why does random-AR show error while BFS/A*-AR is exactly 0? If the model's
1-step prediction were truly 0 everywhere, AR would compound nothing and also be
0. The resolution: it is **not** 0 everywhere — it is 0 only on the
A*-search-covered state distribution (which BFS/A* rollouts stay within and which
the held-out val transitions are drawn from). Random play escapes that coverage
onto states with small residual 1-step error, and AR amplifies it.

To make this airtight, eval now reports teacher-forced (TF, 1-step) and
autoregressive (AR) error **on the same random rollout**:
`_run_eval_rollouts_jax(..., return_both=True)` pre-rolls the C++ env once and
runs both the AR and TF `lax.scan`s over the *identical* action sequence and real
trajectory (`evaluate_multigame` now makes one such call per level instead of two
independent ones). On a shared action sequence the per-episode invariant is
exact: if TF is correct at every step, AR re-derives the very states it would be
fed under TF, so AR is correct too — **any AR error must coincide with a nonzero
1-step (TF) error.**

Verified empirically on all 4 TSM configs (`scripts/tsm_eval_and_ar_gifs.py`,
10 eps × 50 steps): the invariant "TF-perfect episode ⟹ AR-perfect episode"
holds with **0 violations** in every config. Per-level (`tf_max` / `ar_max` =
max wrong cells over steps×eps; `ar_final` = mean wrong cells at step 50;
`first_div` = mean first divergent step, 50 = never):

`pool_on_skip_off` (cleanest — shows the contrast within one model):

| lvl | tf_max | ar_max | ar_final | first_div |
|--|--|--|--|--|
| 2  | 0 | 0  | 0.00  | 50.0 |
| 5  | 0 | 0  | 0.00  | 50.0 |
| 10 | 0 | 0  | 0.00  | 50.0 |
| 6  | 1 | 1  | 0.00  | 49.5 |
| 3  | 1 | 2  | 0.20  | 48.9 |
| 0  | 2 | 3  | 0.33  | 36.4 |
| 11 (19×20) | 2 | 52 | 11.20 | 20.7 |

Where the 1-step error is genuinely 0 (L2/L5/L10), AR never diverges
(`first_div=50`, `ar_final=0`). Where a small 1-step error exists (L0/L11), AR
compounds it; the largest level L11 (19×20) is the worst. All BFS/A*/held-out-val
metrics remain perfect; this is purely the off-policy axis. **The root cause of
that residual is identified below — it is mostly an eval artifact, not a state
coverage gap.**

### Root cause: eval samples the disabled "action" key (`noaction`) — FIXED FRAMING 2026-05-20

The random-AR residual on TSM is **overwhelmingly an eval-side action-space
mismatch**, not state novelty.

`Travelling_salesman.txt` declares **`noaction`** in its prelude — the X/"action"
key is disabled. The C++ search collector respects this correctly:
`actionCount()` (`puzzlescript_cpp/src/solver.cpp:77`) returns 4 for `noaction`
games, so action 4 is never collected — *correct*, and not a collection bug
(non-`noaction` games collect all 5). In the L11 training data, actions 0–3 each
have ~25k edges and **action 4 has 0**.

But the eval random-rollout policy hardcodes `N_ACTIONS = 5` (train.py:46) at
every sampling site (e.g. `_run_eval_rollouts_jax`, `_run_eval_rollout`,
`_render_training_gif`), so it feeds the *disabled* action 4 to `noaction` games.
The real env treats action 4 as a no-op; the model never saw it (correctly), so
it injects spurious 1-step errors that AR compounds. Restricting the random
policy to the game's real action count nearly eliminates the drift:

| level | random ar_mean (0–4) | random ar_mean (0–3, movement) |
|--|--|--|
| 0  | 0.20  | **0.00** |
| 1  | 7.70  | **0.00** |
| 7  | 1.20  | **0.00** |
| 11 | 11.20 | **0.50** |

So the membership check's earlier "coverage gap / action-4 under-explored"
framing was wrong: A* expands *all* enabled actions of every node it pops; action
4 is simply disabled in this game. The genuine state-coverage residual is tiny
(L11: ~0.5 cell with a fair 4-action policy); the bulk (11.2→0.5) was the eval
feeding a disabled action.

**Fix (implemented + validated 2026-05-20).** Added `_enabled_action_count(json_str)`
(train.py) mirroring the C++ `actionCount()` (4 if `noaction` else 5), and routed
EVERY random-action sampling site through it — `_run_eval_rollouts_jax`,
`_run_eval_rollouts_batched`, `_run_eval_rollout`, `evaluate_world_model`,
`_benchmark_eval_impls`, `_render_training_gif`, `_render_rollout_frames`,
`render_rollout_comparison`, plus `heldout_eval.py`, `interpolate_rollout.py`,
`rule_inference.py`. The action one-hot width stays N_ACTIONS=5; only the sampled
range shrinks. (The C++ collector/solvers already respected `noaction`; this only
fixes the Python-side random policy.) Post-fix per-level random-AR on
pool_on_skip_off: **L0–L10 all 0.00 (never diverge); L11 ar_mean 0.50, tf_max=1**
— i.e. TSM is modeled near-perfectly on random off-policy rollouts too, leaving a
genuine state-coverage residual of ≤1 cell on the largest level. BFS/A* and
held-out-val (which only use enabled actions) are unchanged.

(Membership-check method/controls live in `scripts/tsm_check_divergent_in_dataset.py`;
an earlier control failed at 0/74 only because it replayed a cached *solution*
whose action IDs were in a different convention — see [[reference_action_mappings]].)

AR-rollout GIFs (real | NCA
prediction | diff, mispredicted cells tinted red; declared sprites; the
worst-diverging eval episode per level) are in
`figures/tsm_pool_diag/ar_gifs/pool_on_skip_off/L*.gif` — L02/L05/L06/L10 stay
d0 end-to-end, L11 (19×20) climbs to d49 — but note that drift is mostly the
disabled-action-4 eval artifact described above, not state novelty. `_render_training_gif` now works for
any architecture (rule_attn included) — it uses the model's own apply_fn rather
than rebuilding a viz model, and gates learned-sprite rendering on raw (not
sigmoid) sprite logits so decoder-free models render declared sprites instead of
grey mush.

---

## Per-game transition budget: water-fill across levels (2026-05-21)

`--max_transitions_per_game` previously capped a game's training transitions by
truncating greedily, which let a few large levels eat the whole budget and starve
small ones — the Take_Heart_Lass starvation failure mode
([[project_thl_subsampling_starvation]]). Replaced the greedy cap with a
**water-fill** allocation (commit `69cb775`): the per-game budget is distributed
across that game's levels by repeatedly raising a uniform per-level ceiling until
the budget is spent, so small levels keep all their transitions and only the
largest levels get subsampled. This is the allocation used by every in-dist run
below (THL is run at `--max_transitions_per_game 300000`, TSM at `100000`).

---

## LR schedule: cosine vs constant (2026-05-21)

Question (commits `2b882c3`, `1b2a51e`): the default cosine-to-`1e-7` schedule
ties LR decay to a fixed `n_updates`, which makes resuming/extending a run
awkward. Does a **constant** LR reach *and hold* the same perfect val
`change_err`? If so the schedule isn't load-bearing and we can drop it for
resume-friendliness.

Result (`figures/lr_sched_ablation/`, plotter
`scripts/plot_lr_sched_ablation.py`):

| game | cond | final val cerr | first val=0 | max cerr (last 20%) |
|---|---|---|---|---|
| Travelling_salesman | cosine | 0.0 | 26500 | 5.6e-10 |
| Travelling_salesman | constant | 3.6e-4 | 3000 | 1.4e-2 |
| nekopuzzle | cosine | 0.0 | 1000 | 0.0 |
| nekopuzzle | constant | 0.0 | 1000 | 0.0 |
| sokoban_basic | cosine | 0.0 | 16500 | 0.0 |
| sokoban_basic | constant | 0.0 | 25000 | 1.2e-3 |

**Verdict — cosine ≥ constant; keep cosine.** Constant LR *reaches* zero val
`change_err` (often faster — TSM at 3k vs 26.5k) but does not reliably *hold* it:
it leaves a residual on TSM (3.6e-4 final, 1.4e-2 jitter in the last 20%) and
jitters on sokoban_basic (1.2e-3). nekopuzzle is perfect either way. The cosine
decay is doing real work — annealing into a stable zero-error fixed point that
constant LR keeps bouncing out of. For resume/extension we should warm-restart
the cosine schedule against the new horizon rather than switch to constant.

---

## LayerNorm ablation: keep it OFF (2026-05-21)

`--use_layernorm` adds a shared pre-norm LayerNorm on `h` at the start of every
NCA step (the attn/slot/win LayerNorms are always present). It defaults OFF. The
only prior evidence (ARCHITECTURE F4, Bouncers+Collapse, multi-game) called LN
"at best neutral, at worst harmful," but that predates `input_skip` becoming the
default and the in-dist pivot. This sweep regenerates it under the canonical
single-game recipe (launcher `scripts/run_ln_ablation.sh`, analyzer
`scripts/analyze_ln_ablation.py`; figures `figures/ln_ablation/`). Grid: TSM and
Take_Heart_Lass × depth {8, 32} (`n_nca_steps == n_nca_repeats`, max-shared) ×
LN {off, on}, plus a TSM pool-OFF stress regime (skip on/off) where bare deep
training historically diverged and LN gave partial recovery.

| group | depth | LN | best val cerr | conv@ | bfs | astar | rand-AR |
|---|---|---|---|---|---|---|---|
| TSM pool-ON+skip | 8 | off | 0.000% | 3000 | 0.000% | 0.000% | 0.001% |
| TSM pool-ON+skip | 8 | **on** | 0.000% | 8500 | 0.034% | 0.000% | 0.040% |
| TSM pool-ON+skip | 32 | off | 0.000% | 3500 | 0.000% | 0.000% | 0.000% |
| TSM pool-ON+skip | 32 | **on** | 0.000% | 4000 | 0.011% | 0.000% | 0.065% |
| THL pool-ON+skip | 8 | off | 0.000% | 19500 | 0.435% | 0.552% | 0.134% |
| THL pool-ON+skip | 8 | **on** | 0.000% | 14500 | **4.882%** | **5.534%** | **2.970%** |
| THL pool-ON+skip | 32 | off | 0.000% | 17000 | 2.995% | 3.002% | 0.920% |
| THL pool-ON+skip | 32 | **on** | 0.000% | 22000 | 2.908% | 4.024% | **5.269%** |
| TSM pool-OFF+skip | 32 | off | 0.000% | 3500 | 1.066% | 0.000% | 1.397% |
| TSM pool-OFF+skip | 32 | **on** | 0.000% | 2000 | 0.000% | 0.000% | 0.000% |
| TSM pool-OFF,no-skip | 32 | off | 0.000% | 3500 | 0.000% | 0.000% | 1.647% |
| TSM pool-OFF,no-skip | 32 | **on** | 0.000% | 3500 | 0.000% | 0.000% | 0.727% |

**Verdict — keep LN OFF (the current default is correct).** Three reads:
1. **1-step held-out val is perfect (0%) in every cell**, LN on or off. LN never
   affects the quantity we actually optimize; the differences are entirely on
   autoregressive rollouts.
2. **In the shipped regime (pool-ON + skip), LN is neutral-to-harmful.** On TSM
   it's a wash (sub-0.1% rollout noise, and it slows convergence: 3k→8.5k at
   d8). On the harder Take_Heart_Lass it is clearly harmful: at d8 it inflates
   BFS rollout error 0.4%→4.9%, A\* 0.6%→5.5%, random-AR 0.13%→2.97% — a ~10×
   regression — and at d32 it triples random-AR (0.92%→5.27%). The 1-step map is
   identical; LN just makes the deep unroll less robust off the val distribution.
3. **The historical "LN partially recovers divergent training" only applied to
   the bare pool-OFF/no-skip regime that is no longer the default.** There LN
   does help (pool-OFF+skip rollouts 1.4%→0%), but `input_skip` already supplies
   that stabilization, so LN is redundant rather than additive under the current
   defaults.

So `--use_layernorm` stays off. Note THL is *not yet perfectly modeled even with
LN off* (BFS/A\* rollout 0.4–3.0%) — that residual is the separate
data-starvation axis ([[project_thl_subsampling_starvation]]), now mitigated but
not eliminated by the water-fill budget; it is the next thing to push on, not an
LN effect.

---

## Next mechanic-ladder set: Microban / Heroes / Bouncers (2026-06-15)

Climbing past the already-perfect games (sokoban_basic, nekopuzzle, TSM) along
three axes. Canonical recipe (rule_attn h256, depth-8 `input_skip`, LN off,
cosine LR, water-fill 300k cap, 30k steps, `val_frac 0.1`). Launcher:
`scripts/run_in_dist_next.sh`. Run across local GPU 1, ssh 210, and the torch
SLURM cluster (`scripts/torch_in_dist.sbatch`).

| game | axis | val change_err | AR rollout | verdict |
|---|---|---|---|---|
| Microban | data-scale (~150 levels) | **0.0** | — | **perfect** ✓ |
| Bouncers (d8) | iterative `again` | **0.0** | 3/5 levels perfect | val perfect; small rollout residual |
| Bouncers (d16) | iterative `again`, deeper | **0.0** | 3/5 levels perfect | **= d8; depth doesn't help** |
| Heroes_of_Sokoban | local-rule richness | 2.8e-3 | 5/22 perfect | residual (data + mechanics) |
| Heroes_2xdata (600k, 60k) | Heroes + 2× data/steps | 7.4e-4 | 8/22 perfect | improved ~4×, still not perfect |

Findings:
- **Microban is perfect** — the water-fill budget handles a large authored level
  bank (~150 small levels) with zero held-out error. Data-scale axis is clean.
- **Bouncers: depth-8 is enough.** Both depth-8 and depth-16 reach val
  change_err 0 and an identical 3/5-levels-perfect AR rollout (the 2 residual
  levels have ≤1-cell errors at both depths). The iteration-extrapolation
  hypothesis ([[project_nca_iter_extrap]]) — that `again`-heavy games need
  deeper shared unrolls — is **not supported here**: deeper did not move the
  residual. Bouncers' again-chains resolve within 8 steps at these level sizes.
- **Heroes is the live frontier.** Doubling data (300k→600k cap) and steps
  (30k→60k) improved val change_err 2.8e-3→7.4e-4 and perfect levels 5→8/22, so
  data scarcity is a real factor — but 14/22 levels still carry residual, with
  the worst AR drift concentrated on specific levels (16, 17, 3) that likely
  exercise rarer hero-unit mechanics. So Heroes is part data-coverage, part
  mechanics-coverage; more raw data alone won't close it. The May synth-level
  experiments (`logs_heroes_synth`: novelty/rule-coverage/solvability seeding)
  are the relevant prior art for closing the mechanics-coverage gap.

### Interactive WM server + two padding-masking fixes (2026-06-15)

While serving the Heroes checkpoint for interactive play (`serve_wm.py`), two
padding-masking issues surfaced:
1. **`serve.py` AR-feedback leak** (commit `dd352f4`): the interactive rollout
   fed the full padded prediction back without cropping to the real `(n_objs,
   H, W)`. Because the architectural mask is **input-derived** (`x.sum>0`),
   nonzero junk the model paints in off-grid padding (loss never penalizes it)
   was read as "real" next step — the player could "leave" through a wall into
   padding and be carried back later (off-screen memory). Fixed by routing both
   `/api/step` and `/api/step_dream` through `_feedback_pred()` (crop + re-pad
   with zeros), mirroring the train/eval AR path. See [[feedback_ar_padding_leak]].
2. **`--mask_padded_loss` made mandatory** (commit `d7e4c3b`): the flag had crept
   back as a CLI option; padding-loss masking should never be optional. Removed
   the flag and every unmasked fallback branch in the jitted loss/metrics
   (`_heads_loss`/`_ponder_loss`/`_metrics`) and the mask builders; a real
   spatial_mask is now always built and passed. Hidden-state masking was already
   unconditional in all model classes. Behavior is identical to the prior default
   (it defaulted True). See [[feedback_padding_always_masked]].

### Heroes: longer training closes most of the gap (2026-06-15)

The Heroes residual is **rare-transition underfitting, not a coverage gap**. Train
change_err reads exactly 0 on most batches (the model fits the common transitions);
the nonzero val is a small set of *rare* states — notably the 3-crate **chain
push** in L1 — that appear infrequently in batches and stay underfit, flipping
right/wrong across checkpoints (the val "descent" was really oscillation around a
floor, [[feedback_train_change_err_misleading]]).

Tested by extending training (`Heroes_long`, torch job 10882807): 150k steps,
600k data cap, depth-8, cosine-to-150k — isolating step count vs the 60k
`Heroes_2xdata` run (same data/code).

| run | steps / data cap | val change_err | levels perfect | L1 chain-push (bfs / astar) |
|---|---|---|---|---|
| Heroes (baseline) | 30k / 300k | 2.8e-3 | 5/22 | RESIDUAL (3 wrong cells) |
| Heroes_2xdata | 60k / 600k | 7.4e-4 | 8/22 | RESIDUAL (1) |
| **Heroes_long** | **150k / 600k** | **≈0** | **15/22** | **0 / 0 — on-policy perfect** |

**Verdict — training longer substantially helps.** 8→15/22 perfect levels at fixed
data, val change_err → 0, and L1's chain push is now on-policy-perfect (bfs/astar
both 0; only a 0.12-cell random-AR off-policy residual). 20/22 levels reach
bfs_wc=0; the genuine on-policy holdouts shrink to L12/L13. The mechanism matches
the LR-schedule ablation: the val oscillation is set by rare transitions at
moderate LR, and **stretching the cosine anneal to 150k gives a long low-LR
refinement phase that settles them to zero** — it is "more steps *with the
schedule stretched to match*," not raw step count. More data also helped
(30k/300k→60k/600k took 5→8/22), but the dominant lever for Heroes was annealed
training length. Practical implication: for rare-mechanic games, prefer a longer
run with a full-length cosine schedule over short runs; the prior 30k default
under-trained them.

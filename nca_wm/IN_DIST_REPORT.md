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

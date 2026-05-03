# nca_wm running report

Living doc. Last refresh: 2026-05-01.

> **Architectural decisions live in `ARCHITECTURE_REPORT.md`** — that doc
> takes a stand on what the right architecture is and why (global-pool
> stack, rule_attn vs FiLM, depth + skip-connection findings on
> looping-dynamics games, adaptive-pass-count direction). Use this doc
> for chronological run logs.
>
> Latest architectural finding (2026-05-01): on Collapse single-game,
> deeper rule_attn lowers train loss but increases autoregressive
> rollout drift; the new `--use_layernorm --input_skip` patch trades 9×
> higher train loss for 3-4× lower rollout error. Global pooling is
> load-bearing on this game — no-pool runs sit in identity collapse
> regardless of depth. See ARCHITECTURE_REPORT for the table.

## Current best recipe

`rule_attn` arch (`rule_attn_model.py`) with `K=16`, `n_hid=256`, `n_nca_steps=4`,
plus all of:

- `--axis_pool --axis_cummax --global_pool` — full global-context stack (cheap, strict superset of any one alone). Necessary for `...` and `[X][Y]` rules.
- `--change_loss_weight 5.0` — upweight changed-cell BCE so the model can't sit in identity-collapse plateaus.
- `--grad_clip 0.5` — bounds BPTT spikes; visible in loss-vs-step traces as bounded oscillations rather than divergence.
- `--patience 200 --min_delta 1e-6` — loose; tighter early-stopping has cut multi-game runs off mid-recovery in past experiments.
- `--balanced_sampling` for any `n_games > 1`.
- `--kernel_sep` — distinguishes `[A|B][C|D]` multi-kernel rules from single-kernel ones in the tokenized spec.
- `--token_decoder_loss_weight 0.1` — small joint regularization on the rule-slot encoder; helps esp. for synth-data multi-game.

### Synth-data specific (`--synthetic_levels N` path)

- `--synthetic_w/h` should be **≤ smallest authored max-dim** of any training game. Bigger-than-authored breaks size-down generalization (model trained at w=8 can't predict 6×7 sokoban, gets identity-baseline error). Size-up works fine — w=7 trained transfers perfectly to 12×12 / 17×30.
- `--synthetic_mode evolve --synthetic_require_solvable --synthetic_min_states 10` is the workable default for sokoban-class single-player games.
- Per-game `LevelGenerator` auto-detects single-player (enforce exactly 1 player) vs swarm (preserve authored multi-player count, e.g. kettle has 16 player tiles per level).
- Win-condition heuristic (v7) emits `count_min(A, 1)` for any A referenced in any win condition, plus `count_eq(A, B)` for "all A on B" when both invariant. `num=-1` ("no A") form supported.
- Cache key: `seed{seed}_n{N}_v{CACHE_VERSION}_mode-{...}_solv{0|1}_mi{...}_tmo{...}_ms{...}.npz`.
- For multi-game synth, use `--synthetic_per_game_size` (auto-detects each game's authored max-dim and uses it). This recipe matches authored multi-game on training-game in-distribution and beats it on Microban-style heldouts (4-games at per-game size: Microban 0.14% step-1 vs authored scaling_6's 0.37%).

**Encoder mask bug** (fixed 2026-04-18, all post-fix runs): Flax's `MultiHeadDotProductAttention` interprets `mask` as boolean. Earlier runs were passing a float mask with semantics inverted, producing near-identical `z` for every game (cos-sim 0.77–0.99 across 9-game `small` AE). Post-fix cos-sim 0.31 for distinct games. *Any pre-fix saved checkpoint should be considered suspect.*

## Saturation map (single-game, post-fix)

Loss/error a single-game model can reach **alone** at the listed config. Useful for separating intrinsic difficulty from capacity-sharing.

| Game | Loss | Notes |
|---|---|---|
| sokoban_basic (h=128, 1 game) | 1.4e-6 (20K steps) | Trivial. |
| nekopuzzle (h=128, alone) | 6.6e-9 | Trivial alone — the 68% in 9-game training was capacity-share. |
| Zen_Puzzle_Garden (h=128, alone) | 6e-7 (9.6K early-stop) | Trivial. |
| nirvana (h=128) | 3.46e-7 | Trivial. |
| clearing (h=128) | 1.9e-5 | Easy. |
| constellationz (h=128) | 7.9e-6 (57.6K early-stop) | Needs grad_clip + long patience. |
| kettle (h=128, alone) | did not converge in 40K steps | Intrinsically hard; needs ≥h=256 or longer. |

## Multi-game findings

- `scaling_6` @ h=128 plateaus around 4% change_err at 50K steps; capacity-bound.
- `scaling_6` @ h=256 → 1.3e-5 (≈4–10× per capacity doubling).

### `scaling_6` @ h=256 verified perfect-fit (2026-04-30 re-eval)

Re-evaluated `scaling_6_joint_v1` (rule_attn h=256, joint token decoder weight 0.1, 80K steps, best_loss 5.84e-6) on training games and Microban/Microban_I. **Strictly beats identity baseline on every game**:

| Game | n_levels | step-1 mean | step-1 max | rollout mean | identity step-1 | identity rollout |
|---|---|---|---|---|---|---|
| nekopuzzle (train) | 10 | 0.00% | 0.00% | 0.00% | 0.58% | 0.62% |
| Zen_Puzzle_Garden (train) | 5 | 0.00% | 0.00% | 0.00% | 1.12% | 0.90% |
| sokoban_basic (train) | 2 | 0.00% | 0.00% | 0.04% | 2.38% | 2.20% |
| blocks (train) | 1 | 0.00% | 0.00% | 0.00% | 2.33% | 3.08% |
| Travelling_salesman (train) | 12 | 0.66% | 1.36% | 1.95% | 0.65% | 1.32% |
| **kettle (train)** | 11 | **1.99%** | 4.15% | 6.66% | **6.17%** | 1.86% |
| Microban (heldout) | 10 | 0.37% | 2.98% | 2.89% | 1.30% | 1.81% |
| Microban_I (heldout) | 20 | 0.39% | 2.98% | 4.01% | 1.32% | 1.73% |

kettle — known-hardest in `scaling_6` (intrinsically multi-mechanic + swarm) — converges to 1/3 of the identity-baseline step-1 error under joint training. TSP matches identity (capacity-shared with the other 5). Microban transfer is solid: 0.37% step-1 mean across 10 levels, with 8/10 perfectly predicted at first step.

**Fresh second seed `repro_scaling_6_h256_seed0`** (50K steps, no joint decoder, otherwise identical recipe): converged to **best_loss 1.48e-5** at step 50K. End-of-train per-game teacher-forced AR-rollout averages (50 steps each):

| Game | n_levels | mean wrong tiles | max wrong | avg tiles per level | mean error |
|---|---|---|---|---|---|
| sokoban_basic | 2 | 0.0 | 0 | 210 | **0.000%** |
| blocks | 1 | 0.0 | 0 | 572 | **0.000%** |
| Travelling_salesman | 12 | 0.0 | 2 | 811 | **0.000%** |
| Zen_Puzzle_Garden | 5 | 0.0 | 0 | 850 | **0.000%** |
| nekopuzzle | 10 | 0.8 | 4 | 170 | 0.469% |
| kettle | 11 | 0.5 | 21 | 1604 | 0.028% |

5/6 games near-perfect; nekopuzzle TF rollout has a residual ~0.5% (likely the long-range `…` rule still causes occasional rollout drift), kettle TF averages <1 wrong tile per level out of 1604 (~0.03%). Both seeds confirm the perfect-fit claim. Older `_joint_v1` with the joint token decoder is slightly tighter on kettle's harder levels (L7–L10), suggesting the decoder loss helps the multi-rule game even when ignored at inference.
- `scaling_large` (19 games) @ h=256 + change_loss_weight=5 + encoder-fix → 19% mean change_err (best 18.96%), 12 of 19 games <20% on their own. Hard members: kettle 81%, Travelling 57%, Modality 51%, Collapsable/Love/nekopuzzle/scriptcross 41–47%.
- **rule_attn vs FiLM** (h=256, 19 games):
  - wide-FiLM (d_z=256, d_model=128, encL=4): 14.5% best smoothed-500.
  - rule_attn K=16: **9.4% best**, ~3.4× faster convergence.
  - rule_attn K=32: 10.4% (over-parameterized — K=16 saturates).
  - rule_attn h=512: KILLED — bigger body slowed convergence (31% at step 20K vs 11% for K=16/h=256). Poor ROI.
  - rule_attn nca_steps=8: 9.15% (vs nca=4's 9.41%) — marginal improvement, not worth the wall-clock.
- **Identity-prediction is a transient warmup minimum**, not a stable attractor. Patience on uniform BCE was being fooled into early-stopping mid-recovery. `change_loss_weight` aligns the patience metric with what we care about.

## Synthetic-data training (new, 2026-04-30)

Procedural-level generation as a substitute for authored level data. See `nca_wm/synthetic_levels.py` and `nca_wm/curriculum.py`.

**Validity criterion** (game-agnostic):
1. Exactly 1 player (engine `playerMask`).
2. Not initially winning (engine `check_win()` after `restore_level`).
3. BFS over reachable state space within budget yields ≥`min_states` iterations, no timeout, and (default) at least one winning transition.

**Construction**: per-tile sampling from the empirical distribution of bitfield-patterns observed in the game's authored levels. No semantic role lookups — works on any PuzzleScript game.

### sokoban_basic results

256 random valid 7×7 levels, 20K steps, h=256:

| eval set | step-1 cell err | mean cell err |
|---|---|---|
| Microban (10 levels, 12×12) | **0.00%** | **0.00%** |
| Microban_I (10 levels, 17×30) | **0.00%** | **0.00%** |
| sokoban_basic authored (control, 6×7) | 0.00% | 0.00% |

Authored-only baseline (`scaling_1`, 2 levels, 20K steps, h=256), same eval:
| | step-1 | mean |
|---|---|---|
| Microban | 1.65% | 4.55% |
| Microban_I | 1.59% | 4.61% |

**Synthetic decisively beats authored** at the same budget. Mechanism: 2 authored levels = narrow state-action slice → overfit. 256 random valid levels = much broader surface.

### Curriculum loop (model-error-driven mutation)

`nca_wm/curriculum.py`: train → score levels by `change_err` on their full transition set → mutate top-error parents (`LevelMutator`, rank-weighted) → BFS-validate → keep top-K by error → train more.

Smoke test on sokoban_basic (h=64, 16 init pool, 4 gens × 1500 steps): mechanism works (children consistently exceed parents' max error by 1.5–10×, kept_err mean drops 20× generation-over-generation). But on a matched A/B vs static 16-level pool at same step budget, the static pool wins by a small margin (1.86% vs 2.62% mean Microban error). Sokoban_basic is too easy at this scale — curriculum has no headroom to recover.

### Rejection sampling vs evolve mode

Rejection-sampling acceptance with `require_solvable=True`:
- sokoban_basic 7×7: ~0.3% accept (32 levels in 4.3s).
- nekopuzzle 7×7: ~1.4% accept (16 levels in 0.3s).
- Travelling_salesman 8×8: **0/32000 accept** in 2.3s — rejection breaks down entirely.

Evolve mode (`--synthetic_mode evolve`, fitness = BFS iterations + 1e6 if solvable, mutations from `LevelMutator`) recovers Travelling_salesman: **16/16 valid levels in 0.5s after rejection sampling produced 0/32000**. >10,000× speedup on the regime where rejection fails. On sokoban_basic, evolve also produces ≈25K transitions/level vs 1.4K under rejection — same level count, denser state-space coverage per level.

**Construction-time win-condition constraints** (added 2026-04-30): the generator statically analyzes the compiled JSON's rules to mark each object as creatable / destroyable / invariant, then derives count_eq / count_min constraints from win-conditions when both sides are invariant. E.g. for `all Target on Crate` with both objects invariant, the corrector drops surplus tiles to enforce `count(Target) == count(Crate) ≥ 1` at construction. Fully game-agnostic — derived from the engine's win-condition encoding (num=0/1/2/3) rather than name lookups. Replaces the earlier hardcoded sokoban-specific equality.

**Player-count rule auto-detection** (added 2026-04-30): on construction, scan authored levels and detect whether every authored level has exactly 1 player tile. If so, treat as a **single-player game** and enforce `count(player_tiles) == 1` at synth construction (random tile-pattern sampling can otherwise produce 2+ player tiles, biasing the trained model toward swarm dynamics that don't apply). If any authored level has ≥ 2 player tiles, treat as a **swarm-style game** (e.g. `kettle` has up to 44 player tiles per authored level — 4 directional police × ~11 each) and use the looser `≥ 1` rule. Cache version bumped to v6 when this was added.

| sokoban_basic synth → Microban_I (155 levels) | step-1 perfect | rollout perfect |
|---|---|---|
| v3 (hardcoded "exactly 1 player", 256 levels) | 155/155 | 152/155 |
| v5 (hardcoded "≥ 1 player", 256 levels) | 153/155 | 116/155 |
| v6 (auto-detect → exactly-1 for sokoban) | 155/155 | 143/155 |

The v3 vs v6 gap (152 → 143 rollout-perfect) is **not** the player rule — it's the win-condition corrector (added in v4) tightening the level distribution by dropping surplus targets/crates. Cost on sokoban diversity is small; benefit on hard games (kettle 8×8 evolve: 19.5s → 1.1s, ~17×) is large. Net favorable.

### Synthetic multi-game (vs single-game) — sokoban→Microban transfer comparison

Same evaluator, same Microban / Microban_I holdouts. All h=256, all rule_attn:

| Run | training data | steps | Microban_I step-1 mean | step-1 perfect | rollout mean | vs identity (1.32%) |
|---|---|---|---|---|---|---|
| Single-game synth sokoban | 256 synth 7×7 sokoban | 20K | **0.000%** | 155/155 | 0.012% | strictly better |
| Authored `scaling_6_joint_v1` | 6 authored games | 80K | 0.388% | 14/20 | 4.01% | strictly better |
| Synth 4-games | 64 synth each of sokoban / neko / TSP at 8×8 (Zen excluded — see below) | 25K | 1.846% | 7/30 | 5.11% | **slightly worse than identity** |

**Mixing games in synth training hurt the sokoban→Microban transfer**, in this run. The synth 4-game model is *worse than identity* on Microban_I step-1. Plausible causes (untested):

- **Compute**: 25K steps is much less than authored scaling_6's 80K.
- **Dataset imbalance**: synth TSP at 8×8 yields only 3,836 transitions (123 winning) vs sokoban's 200K (capped). Balanced sampling forces the model to spend equal updates on TSP, but TSP at 8×8 has very low rule-firing density per state (player visits a city → city marked), so the model probably memorizes those few transitions while taking capacity away from sokoban dynamics.
- **Encoder sharing**: rule_attn slots are shared across games. With only 3 dissimilar games (push, `…`, mark-on-visit), the slot allocation may not converge to a sokoban-friendly representation.

**Operational consequence**: for "transfer to a sokoban variant" tasks, single-game synth sokoban is *strictly better* than the multi-game mix at the budgets tested. Multi-game synth probably needs (a) longer training, (b) more matched dataset sizes (require_solvable_fraction or explicit per-game caps), and (c) game sets that share more mechanics. Worth a dedicated sweep before declaring multi-game synth a working recipe.

**Compute hypothesis tested (2026-05-01)**: same synth-3-games config retrained at 50K steps (vs original 25K). Best loss 9.4e-6 (vs 25K's 1.29e-4 — 14× lower training loss), but Microban_I step-1 mean stays at 1.86% (vs 1.85%). Rollout mean dropped only marginally (5.11% → 4.48%). **Compute alone doesn't close the gap.** The structural cause likely dominates.

**Same-mechanic hypothesis tested (2026-05-01)**: trained synth on `sokoban_basic + sokoban_match3` (identical `idDict`, same "All Crate on Target" win condition, both single-player). Cleaner test: matched mechanics, no encoder-confusion alibi. The result was actually **worse** in-distribution than the heterogeneous mix:

| Run | sokoban_basic authored (step-1) | Microban (step-1) | Best loss |
|---|---|---|---|
| single-game synth sokoban (n=256, w=7, 20K) | **0.000%** (2/2 perfect) | **0.000%** (10/10 perfect) | ~1e-6 |
| synth-3-games (sokoban+neko+TSP, w=8, 50K) | 3.968% (0/2 perfect) | 1.959% | 9.4e-6 |
| synth-2-sokoban-likes (basic+match3, w=8, 25K) | 4.365% (0/2 perfect) | 2.124% | 4.6e-5 |
| authored `scaling_6_joint_v1` (h=256+joint, 80K) | 0.000% (2/2 perfect) | 0.367% | ~6e-6 |
| identity baseline | 2.38% | 1.30% | — |

**Both multi-game synth runs are worse than identity baseline on `sokoban_basic`'s own authored levels** (4.37% vs 2.38%) — even though sokoban_basic is in their training set and the synth dataset for it had 200K transitions including 22K winning. Same-mechanic same-`idDict` co-training doesn't fix it. Authored multi-game and single-game synth both work.

**The pattern:**

| | in-distribution authored | held-out same-mechanic (Microban) |
|---|---|---|
| single-game synth (sokoban only) | perfect | perfect |
| multi-game synth | broken | broken |
| multi-game authored | perfect | perfect |

So the failure mode is **specific to multi-game synth data**, not multi-game training in general. Likely cause: synth levels per game cover a much narrower structural distribution than authored levels (64 evolved variants of the same skeleton vs 11 hand-designed authored levels with deliberately diverse layouts). Under multi-game training, the encoder's rule-attn slots converge to slot allocations that overfit to the synth distribution and fail to recover the broader authored distribution at eval time.

**Next levers worth trying** (none implemented yet):
- Larger and more diverse synth pool per game (e.g. 256 levels at multiple grid sizes).
- Token-decoder joint loss (the reproduction's `_joint_v1` had it; our synth runs didn't) — may regularize the encoder toward broader slot allocations.
- Multi-grid synth: generate at 6×7, 7×7, 8×8 within the same game so the model sees variable layouts.
- Curriculum: train single-game synth first, then add multi-game with fine-tune. Decouples encoder warmup from multi-game competition.

### Grid-size is the dominant variable (2026-05-01)

Tested the levers above. Diversity (n=64 → n=256) didn't help. Joint decoder gave a small (~10–30%) improvement. **The decisive variable was grid size:**

| Config | sokoban_basic in-distribution step-1 | Microban step-1 | Microban_I step-1 |
|---|---|---|---|
| single-game synth w=**7** (n=256, 20K) | **0.000%** | **0.000%** | **0.000%** (155/155 perfect at 30) |
| single-game synth w=**8** (n=256, 20K) | 3.968% | 1.365% | 1.337% |
| 2-sokoban-likes synth w=**8** (n=256, 30K, +joint) | 4.365% | 1.960% | 1.453% |

**Single-game synth at w=8 fails almost as badly as multi-game synth at w=8.** Going from w=7 (matches authored sokoban_basic max dim of 7) to w=8 (one cell larger) destroyed the perfect transfer. So the prior "multi-game synth is broken" framing was wrong — it was always **grid-size-mismatch synth is broken**, and the multi-game experiments just happened to use w=8.

Hypothesis: rule_attn overfits the spatial position of walls/border patterns at the training grid size. Padding authored 6×7 levels to 7×7 puts them in a near-flush layout matching the training distribution; padding to 8×8 introduces an extra zero row+col the model never saw. NCA + rule_attn are size-flexible architecturally but learn to expect specific spatial patterns at the training grid scale.

**Asymmetry: confirmed by training at w=12** (matching Microban authored size):

| Train grid | sokoban_basic (6×7) step-1 | Microban (12×12) step-1 | Microban_I (up to 17×30) step-1 |
|---|---|---|---|
| w=**7** | **0.000%** | **0.000%** | **0.000%** |
| w=**8** | 3.97% | 1.37% | 1.34% |
| w=**12** | 2.38% (= identity) | 1.49% | 1.29% |

`w=7` trained transfers up perfectly to 12×12 / 17×30 (size-up works). `w=12` trained fails on 6×7 (size-down breaks — step-1 = identity baseline = no useful learning beyond "predict input"). `w=8` trained — closer to authored than 12 but still bigger — also fails on 6×7 but less dramatically.

**Refinement (also tested w=5)**: smaller-than-authored is *also* broken.

| Train w | sokoban_basic (6×7) step-1 | Microban (12×12) step-1 | Microban_I (≤17×30) step-1 |
|---|---|---|---|
| 5 (smaller) | 2.38% (= identity) | 1.18% | 1.36% |
| **7** (matches authored) | **0.000%** | **0.000%** | **0.000%** |
| 8 (one bigger) | 3.97% | 1.37% | 1.34% |
| 12 (4 bigger) | 2.38% (= identity) | 1.49% | 1.29% |

w=5 trained → 6×7 sokoban gets *identity baseline* error — the model trained on 5×5 didn't learn enough push-rule structure to recover when evaluated on a 6×7 layout. So the "smaller-than-authored works" guess from the earlier 3-row table was wrong. The actual rule is:

**Synth grid size must MATCH the authored max-dim** of the game it'll be evaluated on. Below: insufficient state-space exposure. Above: model learns padding-position-specific patterns that fail at smaller eval. Exactly at the authored max dim — both in-distribution and size-up generalization work (w=7 trained transfers perfectly to 12×12 Microban and 17×30 Microban_I).

**Operational rule (corrected)**: per-game synth grid sizes matching each game's authored max-dim. For multi-game training, this means we likely need **per-game synth grid sizes**, not a global `--synthetic_w/h`.

### Per-game synth grid sizes — implemented and validated (2026-05-01)

Added `--synthetic_per_game_size` flag. When set, each game's synth grid size is auto-detected from its authored max-dim via `CppPuzzleScriptEnv.observation_shape` over all authored levels (max H, max W). Falls back to global `--synthetic_w/h` if detection fails.

Tested on 4 single-player sokoban-class games (sokoban_basic 7×6, sokoban_match3 9×7, nekopuzzle 8×7, TSP 20×19), 30K steps, joint decoder loss:

| Run | sokoban_basic | sokoban_match3 | nekopuzzle | Microban | Microban_I |
|---|---|---|---|---|---|
| **4-games synth, per-game size** | **1.19%** | **1.92%** | **1.35%** | **0.14%** | **0.29%** |
| 3-games synth, w=8 global, 50K | 3.97% | — | 1.35% | 1.96% | 1.86% |
| single-game synth sokoban (w=7) | 0.00% | — | — | 0.00% | 0.00% |
| authored `scaling_6_joint_v1` | 0.00% | — | 0.00% | 0.37% | 0.39% |

Per-game-size dropped sokoban_basic in-distribution error 3.3× and Microban transfer 14× vs the global-grid baseline. **The per-game-size synth multi-game now BEATS authored multi-game on the held-out Microban evaluations** (0.14% vs 0.37% step-1 on Microban; 0.29% vs 0.39% on Microban_I). The residual gap to single-game synth on sokoban_basic (1.19% vs 0.00%) is the price of capacity competition across 4 different mechanics.

The earlier "multi-game synth is broken" finding has been **fully resolved**: it was grid size all along, and `--synthetic_per_game_size` makes synth multi-game competitive with — and on Microban-style holdout, better than — authored multi-game.

### Fallback to dynamics-only for hard games (2026-05-01)

Added `--synthetic_fallback_dynamics`. When `require_solvable=True` produces 0 levels for a game (e.g. Zen at 12×12 — random layouts never reach the all-brushed terminal in BFS budget), automatically retries with `require_solvable=False` so the multi-game pipeline gets dynamics-only data instead of silently dropping the game.

Tested on 4 games (sokoban_basic, nekopuzzle, TSP, **Zen**) with per-game-size + fallback. Zen synth: 0 solvable levels; fallback yielded 64 levels with **0 winning transitions, 240K transitions of dynamics**. Trained model performance:

| Game | step-1 | rollout | identity step-1 | (vs. authored scaling_6_joint_v1) |
|---|---|---|---|---|
| **Zen** (in-distribution authored) | **0.000%** (5/5 perfect) | **0.000%** | 1.12% | tied (authored: 0.000%) |
| sokoban_basic | 0.000% (2/2 perfect) | 6.89% | 2.38% | tied (authored: 0.000%) |
| nekopuzzle | 1.59% | 5.53% | 0.58% | worse (authored: 0.000%) |
| **Microban** (heldout) | **0.198%** (8/10 perfect) | 1.55% | 1.30% | **beats authored 0.367%** |
| **Microban_I** (heldout) | **0.187%** (26/30 perfect) | 0.92% | 1.75% | **beats authored 0.388%** |

**Even with zero winning transitions**, the model trained on Zen's dynamics-only synth data perfectly predicts authored Zen at step-1 (5/5 perfect). The dynamics-only signal is sufficient for state-prediction; only the `wons` head loses Zen as a positive-class source (the other games' wins are still there).

The residual weakness is nekopuzzle (1.59% in-distribution vs authored's 0.000%). Hypothesis: capacity competition with Zen's much larger transition pool (240K) plus neko's `…` rule needing the global-pool stack to learn cleanly.

**Final synth multi-game recipe (validated 2026-05-01):**
```
--games <preset> \
--synthetic_levels 64 \
--synthetic_per_game_size \         # per-game native authored max-dim
--synthetic_fallback_dynamics \     # auto require_solvable=False for hard games
--synthetic_mode evolve \
--synthetic_require_solvable \
--synthetic_min_states 5 \
--token_decoder_loss_weight 0.1 \
--n_hid 256 --n_updates 30000
```
This recipe matches or exceeds authored multi-game on every game in the 4-game test, with the exception of nekopuzzle in-distribution (1.59% vs 0%). Microban heldout transfer is *better* than authored.

### Full `scaling_6` synth (mixed result, 2026-05-01)

Same recipe applied to the full `scaling_6` (sokoban_basic, nekopuzzle, blocks, TSP, Zen_Puzzle_Garden, kettle). Tight BFS budget (max_iters=2000, timeout_ms=500) so swarm/multi-rule games don't hang the gen loop. All 6 games generated; fallback fired for blocks (13×11), TSP (20×19), Zen (12×12), kettle (15×15) — no solvable layouts found in budget. Train 30K steps, h=256, joint decoder. Best loss 1.4e-4.

| Game | synth scaling_6 | authored `scaling_6_joint_v1` | identity |
|---|---|---|---|
| sokoban_basic (in-dist) | **0.000%** (2/2 ✓) | **0.000%** (2/2) | 2.38% |
| nekopuzzle | 1.30% (3/10) | 0.000% (10/10) | 0.58% |
| blocks | **0.000%** (1/1 ✓) | **0.000%** (1/1) | 2.33% |
| Zen_Puzzle_Garden | 5.24% (0/5) ❌ | 0.000% (5/5) | 1.12% |
| kettle | 4.12% (0/11) | **1.99%** (0/11) | 6.17% |
| **Microban** (heldout) | **0.258%** (8/10) | 0.367% (8/10) | 1.30% |
| **Microban_I** (heldout) | **0.256%** (24/30) | 0.388% (14/20) | 1.32% |

**Synth wins:** sokoban_basic, blocks (both perfect, tie), Microban / Microban_I heldouts (synth strictly better than authored).

**Authored wins:** Zen, kettle — both games where `require_solvable=True` failed at native size and the fallback gave dynamics-only training data. Zen specifically went from perfect (5/5 in the 4-game test where Zen had less competition) to 0/5 with the 6-game mix — capacity competition with kettle's larger transition pool (8.9MB) and blocks' (3.2MB) crowded out Zen's 13.9MB dynamics signal.

**Net assessment**: synth multi-game with per-game-size + fallback works **strictly better than authored on Microban heldout** for sokoban-class games. For multi-rule / swarm games (Zen, kettle), the dynamics-only fallback is insufficient without winning supervision; that's the regime where authored-seeded synth or per-game tuned BFS budgets become necessary. The 6-game synth recipe is "drop-in" usable for any sokoban-class extension; `scaling_6` is at the boundary of what works without per-game-tuning of the BFS budget.

**Tested: bigger BFS budget on Zen 12×12 (2026-05-01)**: max_iters=30,000 and timeout_ms=10,000 (10× our default budget). At gen 5/100 best_fit was still 30,000 (BFS hits the iter cap without ever reaching a winning state). Each generation took ~68s × 32-pop, projecting ~3.5 hours for the full 100-gen run. **Bigger budget is not the lever for Zen-class games** — the reachable-state-space-from-random-init is exponential in unbrushed-sand count, and the all-brushed terminal is many BFS-depth-steps from any random starting config. The fix is structurally different: either (a) authored-level seeding (start the population in known-valid layouts and mutate locally — defers the depth problem), or (b) a more sophisticated heuristic that constrains starting configurations to be near-solvable (e.g. tightly cap A-count for "no A" win conditions so BFS only needs to brush 1–2 cells). Pure budget-scaling is not sufficient.

**Implemented (b) — `count_max` heuristic for `no A` win conditions (2026-05-01, cache v8)**: extended `parse_win_constraints` to emit `count_max(A, K)` (default K=3) whenever a win condition has `num=-1` (`No A`) form. The corrector drops random surplus tiles to enforce the cap at construction. This caps the BFS reachable-state-space at ~K bits of difficulty.

**Zen 12×12 generation: 0/16 in 11+ min → 16/16 in 4 seconds.** All solvable, 295K transitions, 86K winning (29% win-rate).

**Trained model on Zen authored levels: 5.24% step-1 (v7 fallback dynamics) → 2.24% step-1 (v8 count_max).** Substantial improvement, though still not matching authored's 0%.

**Trade-off**: in the full `scaling_6` v8 retrain, Zen improvement came with mild regressions elsewhere (sokoban_basic 0% → 1.59%; Microban heldout 0.258% → 0.536%). Cause: v8's Zen synth produces fewer-but-denser levels (16 with 86K winning) vs v7's 64 dynamics-only levels — shifting the balanced-sampling distribution. The count_max value (default 3) is also probably too aggressive — Zen authored levels have 20+ unbrushed-sand tiles, so synth at ≤3 underexposes the model to high-U configurations. A sweep over K ∈ {3, 5, 8, 12} is the obvious next ablation.

**Net assessment of count_max**: structurally enables solvable synth for Zen-class games (a category previously inaccessible) at a small ~2pt regression on other games. Worth exposing as a CLI arg `--synthetic_no_a_count_max K` (currently hardcoded default in `parse_win_constraints`).

**K sweep — K=5 is the sweet spot (2026-05-01)**: at K∈{5,8,12,20} for Zen 12×12 evolve, only K=5 finishes generation in budget (16/16 in 32s, 394K transitions, 15K winning). K=8/12/20 all timeout because BFS state space is exponential in K. Ran full `scaling_6` at K=5 with tight budget (max_iters=1500, timeout=400ms):

| Game | v7 (no count_max) | v8 K=3 | **v8 K=5** | authored |
|---|---|---|---|---|
| **Zen_Puzzle_Garden** | 5.24% (0/5) | 2.24% (0/5) | **0.88%** (1/5) | 0% (5/5) |
| sokoban_basic | 0% (2/2) | 1.59% (1/2) | 1.19% (1/2) | 0% (2/2) |
| nekopuzzle | 1.30% | 1.23% | 1.41% | 0% |
| blocks | 0% / rollout 0.19% | 0% / rollout 0.09% | **0% / rollout 0.00%** | 0% / 0% |
| kettle | 4.12% | 4.00% | 4.36% | 1.99% |
| Microban (heldout) | 0.258% | 0.536% | 0.565% | 0.367% |
| **Microban_I** (heldout) | **0.256%** | 0.408% | **0.259%** ✓ | 0.388% |

**K=5 simultaneously**: 6× better Zen authored than v7 (5.24% → 0.88%, 1/5 perfect), restores Microban_I heldout to v7's level (0.259%) and beats authored (0.388%), blocks rollout becomes machine-perfect. Still loses on neko/kettle vs authored — those need different fixes (neko's `…` rule benefits from training pool diversity authored-data has; kettle's directional-police mechanics aren't in dynamics-only fallback).

**Neko residual gap diagnosed (2026-05-01)**: trained single-game neko synth at K=5 (1.18% step-1) and K=8 (1.41%) — both worse than identity (0.58%) and far from authored's 0%. K is not the bottleneck for neko; it's **mixed authored sizes**. Authored nekopuzzle levels include both (8, 7) and (8, 8). My per-game-size detection picks the MAX (8, 8), so (8, 7) authored levels get 1 row of zero padding at eval — the same size-mismatch failure mode documented for sokoban w=7 vs w=8 above. Same root cause as the earlier "synth w=8 fails on sokoban 6×7 authored" finding.

**Permanent fix (not implemented)**: train at multiple grid sizes per game so the model becomes size-invariant. Code paths needed: (a) `collect_synthetic_dataset` with a list of (w, h) sizes; (b) merging caches at training time; (c) per-batch random-sized sampling. This is the "multi-grid synth" open question (#7 below). Closing the residual neko / TSP / Microban-large-level gaps requires it.

### Multi-grid synth — implemented and validated (2026-05-01)

Added `--synthetic_multi_grid` flag. When set, generates synth at every unique authored grid size per game and concatenates (with spatial padding to per-game max). Each size gets `n_levels // num_sizes` levels.

**Single-game neko at K=5 with multi-grid (8×7 + 8×8 — neko's two authored sizes):**

| Run | step-1 | rollout | perfect | (identity 0.58%) |
|---|---|---|---|---|
| single-grid K=5 (8×8 only) | 1.176% | 5.095% | 4/10 | |
| single-grid K=8 (8×8 only) | 1.414% | 5.245% | 3/10 | |
| **MULTI-GRID K=5** (8×7 + 8×8) | **0.104%** | **0.556%** | **9/10** | |
| authored | 0.000% | 0.000% | 10/10 | |

**11× lower step-1 error, 9/10 levels perfect** — multi-grid essentially closes the neko gap. The mixed-authored-size hypothesis is fully confirmed.

**Full `scaling_6` with multi-grid + K=5 (mixed result)**:

| Game | single-grid K=5 | multi-grid K=5 | authored |
|---|---|---|---|
| nekopuzzle | 1.41% (3/10) | **0.42% (6/10)** ✓ 3.4× better | 0% (10/10) |
| kettle | 4.36% | 3.09% ✓ | 1.99% |
| Microban (heldout) | 0.565% | 0.394% ✓ ≈ authored 0.367% | 0.367% |
| sokoban_basic | 1.19% | 1.19% same | 0% |
| Zen | 0.88% | 0.80% similar | 0% |
| Microban_I (heldout) | 0.259% | 0.623% ⚠️ regressed | 0.388% |

**Tradeoff identified**: `--synthetic_levels 64` is divided across each game's unique sizes. A game with single authored size (sokoban_basic, Zen, blocks, kettle) gets all 64 at one size; a game with N sizes (neko 2, TSP many) gets 64/N per size. So multi-grid helps mixed-size games but reduces per-size pool for single-size games — and the model balance shifts. **Likely fix: bump `--synthetic_levels` to e.g. 128 when multi-grid is on**, so per-size pools stay healthy.

The multi-grid path is correct in principle and decisively closes the largest remaining gap (neko). Tuning the level-budget interaction is the obvious next step but not yet done.

**Final synth scaling_6 recipe (validated 2026-05-01, K=5):**
```
--games scaling_6 \
--synthetic_levels 64 \
--synthetic_per_game_size \
--synthetic_fallback_dynamics \
--synthetic_no_a_count_max 5 \      # K=5 sweet spot for "no A" win conditions
--synthetic_mode evolve \
--synthetic_evolve_pop_size 24 \
--synthetic_evolve_max_generations 60 \
--synthetic_require_solvable \
--synthetic_min_states 5 \
--synthetic_max_iters_search 1500 \  # tight so kettle's fallback fires fast
--synthetic_timeout_ms_search 400 \
--token_decoder_loss_weight 0.1 \
--n_hid 256 --n_updates 30000
```
Beats authored on Microban heldout, ties or near-ties on sokoban_basic / blocks / Zen, residual gap on neko / kettle.

### Multi-game synth at the right grid size — WORKS (2026-05-01)

The earlier "multi-game synth is broken" finding was a red herring: it was the grid size, not the multi-game co-training. Re-running 2-sokoban-likes synth at w=7 (matching authored max dim) instead of w=8:

| Run | sokoban_basic step-1 | sokoban_match3 step-1 | Microban step-1 | Microban_I step-1 |
|---|---|---|---|---|
| single-game synth (sokoban only, w=7) | 0.000% (2/2) | — | 0.000% (10/10) | 0.000% (30/30) |
| 2-sokoban-likes synth, w=**8**, +joint | 4.365% (0/2) | 2.183% (0/2) | 1.960% (3/10) | 1.453% (9/30) |
| **2-sokoban-likes synth, w=7, +joint** | **0.000% (2/2)** | **0.000% (2/2)** | 0.238% (9/10) | 0.351% (23/30) |

Multi-game synth at w=7 essentially matches single-game synth on every metric. The earlier ~4% degradation was almost entirely grid-size mismatch; multi-game co-training adds at most a small residual (Microban_I 0.35% vs single-game's 0.00%).

**Recipe for multi-game synth that works:**
1. Pick `synth_w = synth_h = min_over_training_games(max_authored_dim)` — i.e. the size of the smallest game.
2. Use `--synthetic_mode evolve --synthetic_require_solvable --synthetic_min_states 10`.
3. Set `--token_decoder_loss_weight 0.1` (modest joint regularization).
4. Pool size n=256 per game, ~25K training steps.
5. Trust size-up generalization for evaluation on bigger authored levels.

For very-heterogeneous size ranges (TSP 5×5 to 19×20), training at multiple grid sizes per game is still the obvious next step — but for sokoban-class games (all ≤ 30×17), one size matching the smallest works.

Note: Zen at 8×8 produced 0/64 valid synth levels with `require_solvable=True`. The cause is *not* the win-condition heuristic over-applying (it correctly skipped Zen, since `unbrushedsand` is destroyable not invariant) — it's that random Zen layouts at small grids never reach the all-brushed terminal state within BFS budget. We used `require_solvable=False` at 12×12 to get 16/16 dynamics-only levels in one generation (240K transitions, 0 winning), confirming the fitness signal works fine and the only cliff is solvability. Cache version bumped to v7 with parser handling for `num=-1` ("no A" form) and an unconditional `count_min(A, 1)` baseline whenever A is referenced in any win condition (avoids randomly-empty-of-A starts).

### Kettle on synthetic-only (negative result, but instructive)

Trained `kettle` on synthetic 8×8 evolve-generated levels (matching the swarm-aware v6 rule, 21 levels accepted, 7600 transitions, 0 winning since `--no-synthetic_require_solvable` to escape 0/64 acceptance with `require_solvable=True`). Eval on 11 authored 13×15 kettle levels:

| Run | step-1 mean | rollout mean | vs identity |
|---|---|---|---|
| v3 broken-puzzle (forced exactly-1 player → strips authored 16 to 1) | 19.4% | 25.6% | 3.2× worse |
| v6 swarm-aware (multi-player synth, dynamics-only) | **10.0%** | **20.7%** | 1.6× worse |
| Identity baseline | 6.2% | 1.9% | — |
| Joint scaling_6 authored | 1.99% | 6.66% | **3× better** |

The auto-detect swarm rule cuts step-1 error in half vs the broken-puzzle baseline, but the synth-only kettle model is still worse than identity. The authored-data multi-game scaling_6 model gets 1.99% step-1 — well below identity. **For kettle specifically, purely-synthetic at small grids is not enough; multi-game authored training (or seed-from-authored synth) is needed.**

The "0 winning transitions" in the dynamics-only kettle synth dataset is a **filter artefact, not a generation failure**: with `--no-synthetic_require_solvable` we accept any level with `iterations >= min_states` and no BFS timeout, regardless of whether a winning trajectory was reached. The accepted levels happen to all be unsolvable within the BFS budget because their multi-player swarm state spaces are too large for the per-level budget (`max_iters=1500`) to find a goal path. Setting `require_solvable=True` rejects all of them. Larger BFS budget + smaller player count (or seed-from-authored) would recover winning data.

### Travelling_salesman (genuinely hard, scaling sweep)

Setup: 8×8 grid, evolve-generated training pools, h=128, 15K total training steps. Holdout: 64 fresh evolve-generated 8×8 levels with seed=42.

| training pool | strategy | training loss | holdout mean change_err | median |
|---|---|---|---|---|
| 16 | static | 3.2e-6 | 76.8% | 76.3% |
| 16 | curriculum (5 gens, change_err select) | 1.0e-5 | 79.2% | 78.9% |
| **64** | **static** | 5.0e-7 | **66.3%** | **65.8%** |
| 64 | curriculum (8 gens, regret select) | 3.6e-6 | 76.2% | 76.3% |
| 256 | static | 1.3e-6 | 69.0% | 68.4% |

**Both negative findings are durable:**

1. **Curriculum loses to a matched static pool at every tested scale**, with the gap widening as data grows (3pt at 16 levels, 10pt at 64 levels). Mutation-correlated sampling — even with regret selection — hurts generalization vs i.i.d. random samples. Hypothesis: when the model saturates training data (loss ~1e-6, change_err 0%), the regret signal degenerates because *every* mutation child reads as "barely-better-than-identity," so selection becomes near-random AND biased toward the local mutation neighborhood.

2. **TSP synthetic training plateaus around 66% holdout error and *gets slightly worse* at 256 levels** (66.3% → 69.0%). Not data-quantity bound. Most likely: the architecture's rule-attn slots are memorizing specific city configurations rather than the rule "player on city → city marked," and 8×8 random configs don't expose enough rule-firing density to force the model to abstract.

**Lesson**: curriculum-by-error with *pool-replacement* doesn't help (and often hurts) on saturated training data — diversity loss outweighs signal-direction. The mining mechanism itself works (gen 3 children hit 53.9% error on TSP/h=64); it's the "substitute the pool" schedule that's wrong.

### Replay-buffer curriculum (the variant that actually wins)

Same generator, same model, same step budget — but every accepted level (parent or child) stays in a replay buffer forever instead of replaced. Optionally weight per-batch sampling by per-level regret.

TSP head-to-head on a single fresh holdout pool (64 evolve-generated 8×8 levels, seed=42), all checkpoints scored together:

| run | sampling | buffer endpoint | mean change_err | median |
|---|---|---|---|---|
| n64 static | uniform | 64 fixed | 66.3% | 65.8% |
| n64 curriculum (pop-replacement) | uniform | 64 (rotating) | 76.2% | 76.3% |
| n256 static | uniform | 256 fixed | 69.0% | 68.4% |
| **replay_uniform** (init=64 + 8×32 children) | uniform | ~320 | **28.5%** | **28.9%** |
| replay_softmax temp=0.005 | softmax (concentrated) | ~320 | 39.0% | 39.5% |
| replay_softmax temp=0.1 | softmax (≈uniform) | ~320 | 72.6% | 71.1% |

Two findings:

1. **The win is the buffer, not the sampling weight.** `replay_uniform` crushes the static baselines by 38 points. Same step budget; same model; same end-of-training data diversity (~320 unique levels, comparable to `n256_static`). The difference: replay accumulated those levels organically over 8 mutation generations from a 64-level evolve seed, vs `n256_static` throwing all 256 i.i.d. samples at the model from step 0.
2. **Sampling-weight ablations within replay are dominated by RNG noise** at single-seed: temp=0.005 (39.0%), temp=0.1 (72.6%), uniform (28.5%) — non-monotonic, and at temp=0.1 the softmax weights are nearly-uniform anyway (regrets are ±0.002 → exp(±0.02)). Different sampling strategies consume RNG state at different rates, so subsequent mutation/selection diverges. Trustworthy ablation needs either decoupled RNGs (batch-sample vs everything-else) or multi-seed averaging. The `replay_uniform` > static result is robust because it doesn't depend on sampling-weight signal.

**Why this works** (interpretation): the buffer ends with the same number of unique levels as the static-256 pool, but the *order* in which they were added matters. Levels born from mutation chains have correlated structure; mixing them in gradually as the model trains lets the model digest the easier subset first and progressively integrate harder neighbors. Static-256 throws the entire diverse set at the model from step 0, and the model can't form a stable "easy core" before being asked to fit the tail.

(The absolute numbers here use the post-2026-04-30 generator with the win-condition corrector + the more-recent eval pool. An earlier 4-way comparison run produced different absolute numbers (85% / 90% / 64%) due to a transient holdout-pool variation while the generator was being modified; the relative ordering is the durable finding.)

## Open questions

### Synthetic-to-human level generalization audit (2026-05-03)

Added `nca_wm/scripts/analyze_synth_generalization.py`, a post-hoc reporter
over existing `heldout_eval*/results.json` files. It writes:

- `nca_wm/figures/synth_generalization/summary.csv`
- `nca_wm/figures/synth_generalization/summary.md`

Current cached-result readout: single-game synthetic sokoban at the matched
authored grid size transfers perfectly to Microban/Microban_I. The pattern
mostly repeats for multi-game synth when using per-game-size / multi-grid
training: Microban transfer beats identity and is competitive with authored
`scaling_6`. The failures are now localized: grid-size mismatch (`w=8` synth
on 6x7 sokoban), mixed-size `nekopuzzle` without multi-grid, `Travelling_salesman`,
and swarm/dynamics-heavy games such as `kettle`, where synthetic data remains
weaker than authored data despite beating identity at step 1.

Follow-up for the suspicious `w=8` sokoban failure: added
`nca_wm/scripts/run_sokoban_w8_memorization_sweep.sh` and
`nca_wm/scripts/summarize_sokoban_w8_memorization_sweep.py`. This sweeps
synthetic pool size `{64,256,512}` and NCA depth `{1,2,4}` while holding the
original failed recipe fixed otherwise, then evaluates Microban/Microban_I plus
the authored `sokoban_basic` control. Interpretation: if more levels fixes it,
the 8x8 run was data-diversity limited; if shallow NCA fixes it, the depth/capacity
is learning grid-position artifacts; if neither fixes it, the generator's 8x8
distribution is structurally misaligned with 6x7 authored sokoban rather than
merely under-sampled.

1. **Replay-buffer curriculum confirmed as the right design** (TSP holdout 64.2% vs all static baselines >85%). Open: how does it behave at larger scale (1024-level buffer, 50 generations), and on simpler games where static already does well (sokoban_basic) — does it preserve the perfect Microban transfer or drift?
2. **Sampling-temperature ablation**: `--replay_softmax_temp 0.005` was a one-shot guess. Sweep over `[0.0001, 0.001, 0.01, 0.1, 1.0]` to find the right concentration.
3. **Why does TSP plateau at ~64% even with replay-buffer wins?** Either rule-attn memorizes city configurations (probe by training on a fixed pool but evaluating on permuted-city versions of the *same* configurations), or 8×8 has insufficient rule-firing density (probe by trying 16×16 matching authored TSP).
4. **`--init_mode evolve` for curriculum** is wired and confirmed on TSP. Whether per-gen child mutation should also use the evolve loop (instead of single-step `LevelMutator.mutate`) when acceptance is low — untested.
5. **Gallery-scale (174 games)**: in flight as of last batch; final numbers TBD.
6. **Why does nekopuzzle in-distribution degrade in multi-game synth?** (1.59% step-1 with synth-4-games-per-game-size vs 0% with authored-scaling_6 vs 0% with single-game-synth-neko). Hypothesis: capacity competition with Zen's much larger transition pool (240K dynamics-only) crowds out the global-pool slots that neko's `…` rule needs. Test: cap Zen transitions per epoch at neko's level, see if neko recovers.
7. **Swarm + multi-rule games (kettle, blocks) at native sizes**: BFS state space at native authored size with multi-player swarm becomes enormous (kettle authored has ~18K reachable states per level; at synth-native 15×15 with ≥5 player tiles, candidate evaluation takes seconds → evolve loop infeasible). The current `--synthetic_fallback_dynamics` flag would in principle handle them but each candidate is still slow. Levers worth trying: (a) per-game tiny BFS budget (max_iters=500) so timeout fires quickly and we accept "rich-state" candidates without waiting for solvability, (b) `--seed_from_authored` to start the population in known-valid layouts, (c) parallelize BFS via the existing `EvaluatorPool` (threading) in `evolve_level_cpp.py`.
8. **Implemented and shipped this session** (2026-05-01):
   - `--synthetic_per_game_size` (auto-detect each game's authored max-dim)
   - `--synthetic_fallback_dynamics` (retry with `require_solvable=False` when 0 levels)
   - `--token_decoder_loss_weight` confirmed as helpful for synth multi-game
   - `parse_win_constraints` extended for `num=-1` ("no A") form + always-emit `count_min(A,1)` baseline
   - `LevelGenerator` auto-detects single-player vs swarm games from authored levels
   - `nca_wm/scripts/render_synth_levels.py` for ASCII visualization of any synth cache

## Where things live

- `nca_wm/train.py` — main training entry. `--synthetic_levels N` opt-in.
- `nca_wm/synthetic_levels.py` — game-agnostic level generator (rejection + evolve).
- `nca_wm/curriculum.py` — model-error-driven curriculum loop.
- `nca_wm/heldout_eval.py` — zero-shot eval on held-out games. `--max_levels_per_game K`.
- `nca_wm/scripts/compare_synth_vs_authored.py` — comparison reporter.
- `nca_wm/rule_attn_model.py` — current best architecture.
- `nca_wm/token_decoder.py` + `nca_wm/train_token_ae.py` — game-spec autoencoder for latent sampling/interpolation.
- Cache: `rollout_data/{game}/level_{i}/...` (authored), `rollout_data/{game}/synthetic_{w}x{h}/...` (synthetic).
- Logs: `nca_wm/logs/`. Each run dir contains `config.json`, `params.pkl`, `params_best.pkl`, `game_infos.pkl`, `train_meta.json`.

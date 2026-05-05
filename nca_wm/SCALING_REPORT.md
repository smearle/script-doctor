# NCA World Model Scaling Report

This is the running log of multi-game scaling experiments. Future agents: keep this file
up to date as you launch / finish runs. Newer rows go on top.

For empirical results (headline metrics, paired-Δ tables, per-game comparisons,
findings), see `SCALING_RESULTS.md`. **Update both files: this one with run state, the
results file with numbers and findings.**

## Index of runs

| run | preset | n_games | recipe | n_updates | batch | save_dir | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **scaling_v3_n94_decoder** | scaling_gallery_v3 | 94 | v3 recipe + mask_hidden + token_decoder_loss_weight=1.0 | 150k | 32 | `multi_scaling_gallery_v3_decoder` | training (launched 2026-05-05, GPU 0) |
| **scaling_v3_n94** | scaling_gallery_v3 | 94 | v3 recipe + mask_hidden=True | 150k | 32 | `multi_scaling_gallery_v3` | training (launched 2026-05-05, GPU 1) |
| tsm_single_v3recipe | Travelling_salesman | 1 | v3 recipe + mask_hidden | 30k | 32 | `tsm_single_v3recipe` | DONE — train change_err 1e-9 by step 2k |
| scaling_v3_aborted_n96 | scaling_gallery_v3 (w/ master_zombie + felix) | 96 | v3 recipe + mask_hidden=True | 150k | 32 | discarded | ABORTED — JAX OOM on master_zombie's 27×51 grid |
| scaling_v3_aborted_n97 | scaling_gallery_v3 (w/ broken) | 97 | v3 recipe + mask_hidden=True | 150k | 32 | `multi_scaling_gallery_v3_aborted_n97` | ABORTED — A* hung on broken_by_beekie18 |
| scaling_14_v3recipe | scaling_14 | 14 | v3 recipe (batch=32, n_nca=8, 150k) | 150k | 32 | `multi_scaling_14_v3recipe` | DONE |
| v3_combined | scaling_gallery_v2 | 59 | batch=32 + n_nca=8 + 150k (combined recipe) | 150k | 32 | `multi_scaling_gallery_v3_combined` | DONE |
| v2_smallbatch | scaling_gallery_v2 | 59 | baseline + batch=32 only | 80k | 32 | `multi_scaling_gallery_v2_smallbatch` | DONE |
| v2_long | scaling_gallery_v2 | 59 | baseline + 200k | 200k | 64 | `multi_scaling_gallery_v2_long200k` | DONE |
| v2_big | scaling_gallery_v2 | 59 | n_hid=512, n_slots=24, n_nca=6, batch=32 | 80k | 32 | `multi_scaling_gallery_v2_big_nca6` | DONE |
| v2_nca8 | scaling_gallery_v2 | 59 | n_nca=8, batch=32, baseline n_hid/n_slots | 80k | 32 | `multi_scaling_gallery_v2_nca8` | DONE |
| v2 | scaling_gallery_v2 | 59 | baseline (n_hid=256, n_slots=16, n_nca=4, n_app_slots=1) | 80k | 64 | `multi_scaling_gallery_v2_level-None_nca-4_hid-256_lr-0.0003_pat-2500_s-0` | DONE |
| v1 | scaling_gallery_v1 | 39 | baseline | 50k | 64 | `multi_scaling_gallery_v1_level-None_nca-4_hid-256_lr-0.0003_pat-1500_s-0` | DONE |

Shared training-time defaults (unless overridden in the recipe column):
`lr=3e-4`, cosine schedule to `lr_min=1e-7`, `grad_clip=0.5`, `change_loss_weight=5.0`,
`max_transitions_per_game=200_000`, `search_timeout_ms=60_000`, `n_search_steps=100_000`,
`balanced_sampling=True`, `seed=0`, `--patience` set per run.

## Findings to fold into design

- **gallery_v2 effective game count is 57, not 60** (2026-05-05): under
  `tokenize_game(encode_sprites=False)`, three preset entries are token-
  identical and the encoder cannot distinguish them — `sokoban_basic ≡
  blank ≡ Microban` (one shared cluster, presumed the default PuzzleScript
  template) and `twolittlecrates2 ≡ twolittlecrates4`. The model is
  effectively training on 57 distinct games, with three latents that are
  forced to collapse. Implication for embedding-distinctness experiments:
  expect at most 57 distinct points from gallery_v2 training.
- **Available game pool, post-dedup**: applying token-hash dedup to
  metadata-passing games (n_rules∈[1,20], n_objects≤20, n_levels∈[1,30],
  max_level_area≤30) across {custom, gallery, scraped, increpare} yields
  **3,474 token-distinct games** (cut 1,098 of 4,572 by dedup; biggest
  collision group is 445 copies of the default template). 56 from
  `gallery/`, 286 from `scraped/`, 3,132 from `increpare/`. Tool:
  `nca_wm/scripts/dedup_games.py`. Output: `data/dedup_candidates_v2.json`.
- **scaling_gallery_v3 proposal (n=97 = 57 deduped gv2 + 40 stratified
  additions)** at `data/scaling_gallery_v3_proposal.json`. Stratified
  buckets: 12 low, 16 mid, 8 high, 4 xhigh rule-counts. **Launched
  2026-05-05 on GPU 1**, save_dir `multi_scaling_gallery_v3`. Once it
  finishes, run `latent_scatter_rule_attn` and compare embedding
  distinctness vs v3_combined (59g). Recipe matches v3_combined except
  mask_hidden=True (current default; v3_combined predates that flip).
- **scaling_gallery_v4 (n=200) preset is staged** in `train.py`
  MULTI_GAME_PRESETS. A* cache warming for the 138 new-vs-v3 games
  launched 2026-05-05 in parallel with v3 training (CPU-only via
  `nca_wm/scripts/warm_caches.py`, log `/tmp/warm_v4_minus_v3.log`).
  When v3 training finishes, v4 launch will skip A* search since the
  per-level caches are already populated.
- **Held-out tooling**: `nca_wm/scripts/pick_heldout.py` picks N games
  disjoint from a training preset (by name AND token-hash); output
  `data/heldout_v4_n30.json`. `nca_wm/latent_overlay_heldout.py` encodes
  the held-outs through a trained checkpoint and overlays them on the
  training-set scatter, reporting k-NN by cosine distance. Baseline on
  v3_combined: 1-NN median = 0.123, max = 0.306 — every held-out lands
  within 0.31 of a training game. See SCALING_RESULTS.md "Held-out game
  overlay on v3_combined".
- **Held-out AR rollout eval (`heldout_eval.py`) hits OOM** on
  v3_combined (2026-05-05) — JAX tries to allocate 32 GiB for a single
  op that survives rematerialization, exceeding GPU 0's 24 GiB. Likely
  cause: per-game JIT compile cache accumulates across the 30
  shape-diverse held-outs, or one of the larger-grid games (Heroes-of-
  Sokoban-Ancient-Japan, headless_people_problems with 12 levels and
  283-token spec) blows up the model's intermediate buffers. Latent
  overlay (geometric metric) is unaffected and remains the primary
  scaling-curve metric. Predictive-accuracy eval deferred until
  `heldout_eval.py` is patched (e.g., per-game JIT clear, or a
  max_level_area filter on the heldout list).
- **Open: per-game data scarcity via synth/evolved levels.** The
  persistently-hard games in v2 (Take_Heart_Lass, Travelling_salesman,
  Lightdown, etc.) have 1-12 authored levels — likely data-starved.
  Nekopuzzle precedent (SCALING_RESULTS "Nekopuzzle synth distribution
  sweep") shows synth multi-grid that includes the authored aspect
  ratio closes a 3–8% gap to 0.3%. For multi-game scaling, the natural
  follow-up is `--synthetic_levels K --synthetic_multi_grid` per game
  in the preset so each game gets +K synth levels. Static synth first
  (cheaper than evolve); failure-driven evolve as a follow-up if static
  doesn't move hard games. Wire-up cost: add per-game synth pool to
  collect_unique_transitions; current synth path is single-game-only.

## Active runs

- **Varislide iteration-extrapolation** (2026-05-04 → 2026-05-05): DONE,
   hypothesis validated. Two-part experiment:
   (a) Fixed-depth (D_train ∈ {2,4,8}, 3 seeds, pool OFF + ON control):
   model learns the per-step shift rule (D_eval=k → slides up to d=k
   work; peak at D_eval=D_train), extrapolates ~1 doubling beyond
   D_train, then collapses at D_eval ≥ 4× D_train. Rest state is not a
   fixed point of the learned dynamics.
   (b) **Uniform-halt followup (2026-05-05, T_max=32, 3 seeds, pool OFF):**
   `--adaptive_halt --halt_mode uniform --halt_kl_weight 0` supervises
   every per-step readout equally, forcing rest-state idempotence.
   Result: **100% argmax-correct at every D_eval ∈ {4, 8, 16, 32, 64}
   on every slide distance**, including D_eval=64 (2× T_max). D_eval=1
   still only solves d=1 and D_eval=2 only d≤2 — model genuinely uses
   iteration, doesn't shortcut.
   (c) **Width-OOD bonus (2026-05-05):** the uniform-halt checkpoints
   also generalize to grids wider than seen at training. New synth
   caches at W ∈ {20, 24, 32} (max train W=16); accuracy at D_eval up
   to 128 (= 4× T_max) holds at 0.99–1.00 across the entire matrix —
   the body is a stable local rule that's grid-size agnostic on the
   training distribution.
   (d) **Handcrafted long-slide stress test (2026-05-05) — claim
   tempered.** Per-seed evaluation on authored-layout (`#P...##`)
   wide grids with controlled slide_d ∈ {3,6,12,18,24,30} shows the
   "depth-arbitrary iteration" claim is layout-specific. Seed 0 at
   D_eval=8 nails d=6–30 (impossible via pure iteration with 3×3 conv;
   the model uses a local stop-position pattern detector, not
   iteration). Seeds 1/2 fail across the board. D_eval ≥ 16 collapses
   for everyone. The earlier 100%-at-D_eval=128 result was a synth-
   distribution artifact (synth has d≤5 dominating); on authored-
   layout long slides, the body is seed- and depth-fragile.
   See SCALING_RESULTS.md "Handcrafted long-slide stress test" for
   the corrected interpretation.
   Figures: `nca_wm/figures/varislide_depth_extrap/{fig_argmax_vs_Deval,
   fig_argmax_by_distance, fig_argmax_by_distance_pool}.{pdf,png}` —
   the heatmap's D_train=32 panel is fully yellow vs the diagonal-only
   stripes of D_train ∈ {2,4,8}. Logs `logs_depth_extrap/`. Scripts:
   `run_varislide_depth_extrap_sweep.sh`, `run_varislide_uniform_halt_sweep.sh`,
   `eval_varislide_depth_extrap.py`, `plot_varislide_depth_extrap.py`.
   See SCALING_RESULTS.md "Varislide iteration extrapolation".
- **Nekopuzzle synth-arch sweep** (2026-05-04): DONE. 12-config arch sweep
   plus follow-up synth-distribution sweep + v5 data/compute extension.
   Headline: **the user's "perfect generalization to authored" goal is
   achieved when the synth multi-grid set includes the authored grid
   size**. With 5-sizes (incl 8x7) at n=256: 0.31% BFS / 1.64% RAR /
   0.58% holdout / 0.39% OOD. v5 n=512/5sizes is a co-winner with better
   RAR (1.00%) but slightly worse BFS (0.67%); v5 n=256/30k/5sizes
   regressed everywhere — once 8x7 is in the multi-grid set, data×compute
   is saturated. Original v1 3-8% gap was almost entirely a
   grid-size-mismatch artifact ({5x5,6x6,7x7,8x8} vs authored 8x7). Arch
   findings still valid: pool ON dominant (3-5pp), depth saturates at
   d=8-16, per-step ≈ shared. GIFs of v4 + v5 at
   `logs_neko_arch/{neko_d16_pool_perstep_tpe_n256_5sizes_s0,
   neko_d16_pool_perstep_tpe_n512_5sizes_s0}/gifs/rollout_nekopuzzle_bfs.gif`.
   Full followup table in `SCALING_RESULTS.md`. Logs in
   `nca_wm/logs_neko_arch/`. Scripts: `run_neko_synth_sweep.sh`,
   `run_neko_synth_v2_evolve.sh`, `run_neko_synth_v3_more_data.sh`,
   `run_neko_synth_v4_authored_size.sh`, `run_neko_synth_v5_push.sh`,
   `reeval_neko_arch_sweep.py`, `aggregate_neko_arch_summary.py`,
   `plot_neko_arch_sweep.py`, `render_rule_attn_gifs.py`.
- **Bouncers L×R sweep**: DONE (2026-05-03). 6 configs, results in
   `nca_wm/figures/bouncers_lr_sweep/` and SCALING_RESULTS.md.
- **v3_combined**: still in eval (per-step n=8, 59 games).
- **Varislide post-bitpack-fix sweep** (2026-05-04): DONE. 34/34 runs at
   100% argmax — every architectural variant solves multi-grid varislide.
   Findings written up in SCALING_RESULTS.md "Varislide post-bitpack-fix"
   and ARCHITECTURE_REPORT F8 (annotated as withdrawn). Figures:
   `nca_wm/figures/varislide_postfix/{argmax_by_depth, argmax_by_LR,
   argmax_pool_onoff}.{pdf,png}` + `summary.{csv,md}`. The pre-fix
   "varislide depth/compute/sharing flat" claims (F8) were entirely a
   bitpack regression artifact.
- **Heroes_of_Sokoban L0–L7 → L8–L21 transfer** (2026-05-04): DONE.
   Recipe C (pool OFF + input_skip + shared) at d=4 and d=8, 20k updates.
   Headline: training on 8 authored levels cuts BFS heldout error ~60%
   (d=4: 21.31% → 8.24%; d=8: 26.75% → 11.38%). TF heldout falls 76–79%.
   d=4 still beats d=8 on transfer. See SCALING_RESULTS.md
   "Heroes_of_Sokoban L0–L7 → L8–L21 transfer" for full table.
- **Heroes_of_Sokoban L0 depth × sharing × pool sweep** (2026-05-04, RE-EVAL):
   DONE + corrected. The first pass used buggy centered-vs-top-left
   slicing in `train.py` `_run_eval_rollout` — corrected via re-eval
   (`scripts/reeval_via_render_only.sh`, npz files saved as
   `eval_multigame_tlfix.npz`). Corrected headline:
   - **L0 fit:** C and D (pool OFF + input_skip) achieve **0.00% at every
     depth**; A degrades 0→0.38→1.54% as d grows; **B collapses to 91.35%
     at d=32**.
   - **Held-out (L1–L21, mean):** all variants best at d=4 (21–24%);
     C wins at d=32 (25.39%); B blows up to 88.89% at d=32.
   - **Recipe:** C (pool OFF + input_skip + shared) most robust overall.
     The d=4 pool-ON penalty in the original analysis was a slicing
     artifact; the d=32 per-step+pool collapse is real.
   See SCALING_RESULTS.md "Heroes_of_Sokoban …RE-EVAL" section for full
   per-config table. Figures: `nca_wm/figures/heroes_sweep/heroes_bfs_by_depth.{pdf,png}`
   (two panels: L0 vs heldout) + `summary.{csv,md}`.
- **Multi-grid varislide canary** (pre-fix runs, INVALIDATED 2026-05-04 by
   bitpack regression 1fa557d). All `varislide_depth*_s*`,
   `varislide_long50k_d16_s*`, and `varislide_perstep_d16_s*` runs in
   `logs_canary/` saw all-zero states/next_states during training; the
   "fire-once" / "depth flat" / "5× per-seed argmax variance" findings
   are artifacts. Re-running via the postfix sweep above. Old figures in
   `nca_wm/figures/varislide_depth_sweep/` should not be cited.

## Open questions / next experiments

- **Per-step intermediate-state supervision on varislide.** The 2026-05-05
   handcrafted long-slide stress test exposed that uniform-halt's apparent
   "depth-arbitrary iteration" was partly a synth-distribution artifact and
   partly a non-iterative equilibrium shortcut. Existing infra
   (`scripts/train_varislide_perstep.py` + `eval_varislide_intermediates.py`,
   originally invalidated by the May 4 bitpack bug) supervises every NCA step
   against the engine's intermediate state after k `again` iterations —
   forcing genuine per-step iteration, blocking the equilibrium shortcut.
   Suggested sweep: 3 seeds × {pool OFF + input_skip}, T=16 with per-step
   loss weight ∈ {0 (final-only control), 0.5, 1.0}. Hypothesis: with
   per-step supervision, all 3 seeds nail the handcrafted long-slide
   benchmark, and D_eval=k strictly solves d=k. *Discuss before launching*
   per `feedback_discuss_before_fix.md`.
- **L×R sweep on a harder game.** Bouncers turned out too easy to discriminate
   the configs on capability — only on parameter efficiency. Re-run the same
   sweep on Mirror Isles or Heroes of Sokoban (looping is non-axis-aligned and
   long, so global pooling can't substitute for actual iteration). Expect
   bigger gaps between `(L=8, R=1)` and `(L=1, R=8)` if the architecture has
   a real iterative-reasoning bottleneck.
- **v3_combined eval (in progress)** is the cleanest dataset-size comparison vs
   scaling_14 — same recipe, same 150k steps, only the game set differs. Once it
   finishes, update SCALING_RESULTS.md with per-game v3_combined vs scaling_14 deltas
   and confirm/refute "scale hurts (or is neutral) at this capacity."
- **Why does longer training hurt It_Dies_In_The_Light / Modality?** They're not in
   v1 (where v2 baseline beat its successors here). Possible: late training overfits
   to abundant easy-game transitions and forgets a rare rule firing in these games.
   Could be tested with per-game loss weighting or by holding them out.
- **Does adding more games make existing games easier (rule transfer)?** Partial
   answer in `SCALING_RESULTS.md`: at current capacity, recipe matters far more than
   game-set size. The ~80-game v3 preset experiment is still worth running once we
   verify v3_combined eval, to test whether transfer flips at larger scales.
- **Per-game capacity vs per-game data.** mazetest's train change_acc never got >91%
   in v2_big — yet its rollout is OK. notsnake's train accuracy is fine, but rollout
   is 80% wrong. The notsnake gap suggests teacher-forcing leak; could test by
   evaluating fixed-state-noise rollout or by widening the bottleneck rule_attn slot
   count.
- **Drop the (32×64)-bucket games?** mazetest, MC_Escher, constellationz are the only
   contributors. If we exclude them, batch=64 stays clear and biggest model fits.
- **scaling_gallery_v3 preset.** Most candidate gallery games beyond v2 are over the
   v2 complexity caps (≥25 rules, ≥25 objects). v3 expansion will need either a
   loosened cap (~30 rules) or scraped-games sourced additions.

## Operational notes

- GPU 0 / GPU 1 are both ~24 GiB. Two single-GPU experiments run in parallel; multi-GPU
   for one run is *not* supported by the codebase (no `pmap` / sharding wiring).
- `gdrtodd` sometimes occupies GPU 1 — check `nvidia-smi --query-compute-apps` first
   (project memory `feedback_gpu_sharing_gdrtodd.md`).
- A* search now bounded by both `max_transitions_per_game / n_levels` (write-time cap)
   and a 60s per-level wallclock timeout. constellationz / Take_Heart_Lass / Midas would
   loop forever otherwise.
- Per-level cache filename is `astar_transitions_v5_{max_iters}_{timeout_ms}_cap{N}.npz`.
   Cache load falls back to any matching `cap{N}` file regardless of `timeout_ms`, so
   tightening timeout doesn't invalidate prior caches.
- Datasets cached at `rollout_data/_merged/dataset_{hash}.npz` (format_version 14, packed).

## Logs (all in /tmp/, not under git)

- v1: `/tmp/gallery_v2.log` (yes, named v2 because it superseded v1's log mid-experiment)
- v2: same file as v1 above (file overwrite)
- v2_nca8: `/tmp/gallery_v2_nca8.log`
- v2_big: `/tmp/gallery_v2_big.log`
- v2_long: `/tmp/gallery_v2_long.log`
- v2_smallbatch: `/tmp/gallery_v2_smallbatch.log`
- v3_combined: `/tmp/gallery_v3_combined.log`
- scaling_14_v3recipe: `/tmp/scaling_14_v3recipe.log`

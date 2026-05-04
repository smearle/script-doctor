# NCA World Model Scaling Report

This is the running log of multi-game scaling experiments. Future agents: keep this file
up to date as you launch / finish runs. Newer rows go on top.

For empirical results (headline metrics, paired-Δ tables, per-game comparisons,
findings), see `SCALING_RESULTS.md`. **Update both files: this one with run state, the
results file with numbers and findings.**

## Index of runs

| run | preset | n_games | recipe | n_updates | batch | save_dir | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scaling_14_v3recipe | scaling_14 | 14 | v3 recipe (batch=32, n_nca=8, 150k) | 150k | 32 | `multi_scaling_14_v3recipe` | DONE |
| v3_combined | scaling_gallery_v2 | 59 | batch=32 + n_nca=8 + 150k (combined recipe) | 150k | 32 | `multi_scaling_gallery_v3_combined` | training (launched 2026-05-02) |
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

## Active runs

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
- **Heroes_of_Sokoban L0 depth × sharing × pool sweep** (2026-05-04): DONE.
   16 configs. Headline: pool OFF + input_skip + shared (bucket C) is the
   most robust recipe across depth ∈ {4, 8, 16, 32}; pool ON + per-step
   collapses at d=32 (1.94% → 15.35%). Pool ON harms at d=4 (~6.5% vs ~1.4%
   no-pool). See SCALING_RESULTS.md "Heroes_of_Sokoban" section for
   per-config table and recommendation. Figures:
   `nca_wm/figures/heroes_sweep/{heroes_bfs_by_depth, summary}.*`.
- **Multi-grid varislide canary** (pre-fix runs, INVALIDATED 2026-05-04 by
   bitpack regression 1fa557d). All `varislide_depth*_s*`,
   `varislide_long50k_d16_s*`, and `varislide_perstep_d16_s*` runs in
   `logs_canary/` saw all-zero states/next_states during training; the
   "fire-once" / "depth flat" / "5× per-seed argmax variance" findings
   are artifacts. Re-running via the postfix sweep above. Old figures in
   `nca_wm/figures/varislide_depth_sweep/` should not be cited.

## Open questions / next experiments

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

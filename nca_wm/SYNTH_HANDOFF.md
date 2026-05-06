# Synth-only within-game generalization — handoff

**Goal.** Establish, on five mechanic-class games × four architecture buckets, that the rule-conditioned NCA can be trained purely on synthetic levels and still hit ~authored fidelity on every authored level. Backs the headline claim of `nca_wm/paper/sections/results.tex` §`sec:results-within-game`.

**Status (2026-05-06).** Anchor result validated: `Microban`, bucket D, 7×7 synth, K=128, 30k updates → **0.60% BFS / 0.02% TF** on all 10 authored levels (1-step *better* than authored baseline 0.11%). Full 5×4 sweep partially launched.

## Validated recipe

- **Smallest-by-area authored size** per game (an *actual* authored size, tiebreak by sum then `(w,h)`). NOT bounding-box max (`--synthetic_per_game_size`) and NOT multi-grid — both tried, both ~8% BFS on Microban.
- **K = 128** levels at that size.
- **30k updates**, rule_attn body, `n_hid=256`, T=8.
- **Coverage GA**: `--synthetic_track_rules_fired --synthetic_rule_coverage_weight 100 --synthetic_coverage_select_topk`.
- Full flag set in the launcher header comment.

## Files

- `nca_wm/scripts/run_per_game_arch_synth_grid.sh` — sweep launcher; `min_size_for()` shell function picks the per-game size at launch time. Defaults: 5 games × 4 buckets, K=128, 30k updates, single GPU.
- `nca_wm/scripts/prewarm_synth_caches.py` — pre-warms synth caches in parallel (CPU only). `--size_mode min_area` (default, matches launcher) or `--size_mode all_sizes` (multi-grid for the appendix comparison). `--workers N`.
- `nca_wm/scripts/make_within_game_synth_artifacts.py` — reads run logs, writes `paper/figures/per_game_arch_synth/{anchor_microban_d.tex, within_game_synth_table.tex, heatmap_bfs.{pdf,png}, summary.csv}`. Re-run after any sweep cells land; paper auto-picks-up via `\input`.
- `nca_wm/scripts/summarize_per_game_arch_grid.py` — older summarizer; `--all_authored_as_heldout` mode for synth-only runs.
- `nca_wm/logs_per_game_arch_synth/` — run dirs (`<game>__<bucket>_d8`) and prewarm logs.
- `nca_wm/paper/sections/results.tex` § `sec:results-within-game` — anchor + sweep tables `\input` from the figures dir.
- `nca_wm/paper/sections/appendix.tex` § `app:synth-only` — recipe documentation + design rationale.

## Background processes when handoff written

Check with `ps -eo pid,etime,pcpu,cmd | grep -E "train.py|spawn_main" | grep -v grep`.

- `bash nca_wm/scripts/run_per_game_arch_synth_grid.sh` — GAMES="Microban Heroes_of_Sokoban" BUCKETS="A B C D" sweep on GPU. Logs: `nca_wm/logs_per_game_arch_synth/sweep_microban_heroes.run.log`.
- `prewarm_synth_caches.py --size_mode min_area --workers 1` — single-process CPU prewarm for Bouncers / neko / TSP min-area caches. Logs: `prewarm_minarea_sp.log`.
- `prewarm_synth_caches.py --size_mode all_sizes --workers 1` — single-process CPU prewarm for the multi-grid appendix comparison. Logs: `prewarm_multigrid_sp.log`.

When the prewarms finish, kick the second-batch sweep:
```sh
GAMES="Bouncers nekopuzzle Travelling_salesman" N_UPDATES=30000 \
  bash nca_wm/scripts/run_per_game_arch_synth_grid.sh
```

## After every sweep batch

```sh
.venv/bin/python3 nca_wm/scripts/make_within_game_synth_artifacts.py
```

regenerates the paper artifacts. Paper picks up the change on next compile.

## Gotchas

- **`node` must be on PATH.** `puzzlescript_cpp` imports `backends.nodejs` which spawns `node`. After a box restart prepend `/home/jupyter-earle/.nvm/versions/node/v24.15.0/bin` to PATH for any subprocess that touches train.py / synth-gen / parser. Symptom: `FileNotFoundError: [Errno 2] No such file or directory: 'node'` during import.
- **Single-process for prewarm right now.** The 8-worker pool exhausted memory on this box once and required restart. Stick to `--workers 1` until otherwise instructed.
- **Cache key includes** `_rc{weight}_cstop` for coverage runs; baseline (no coverage) caches lack the suffix. Side-by-side coexistence is intentional for the appendix comparison.
- **TSP at 5×5 is a known-likely-fail case.** All-pairs combinatorial structure of the 20×19 authored levels isn't present in 5×5 synth. Pre-registered prediction in the paper says it'll need authored-seeded fallback; not a recipe bug.
- **Bouncers GA is slow.** Looping-projectile mechanic + 12×9 grid → 1+ hour per cache at single-process. If it stalls, `fallback_dynamics=True` will at least produce a `solv0` cache.
- **Anchor run lives at non-standard path** `Microban__D_d8_w7_k128_30k`. The artifact script slots it into `(Microban, D)` if no standard `Microban__D_d8` cell exists; once the launcher writes a standard `Microban__D_d8` (at the launcher's smallest-by-area = 6×7), that takes priority. Both can coexist.

## Memory pointers

`/home/jupyter-earle/.claude/projects/-home-jupyter-earle-script-doctor/memory/`

- `project_synth_recipe.md` — validated recipe details (May 6 update at top).
- `feedback_per_game_size_pitfall.md` — why bounding-box auto fails for varied-size games.
- `reference_runtime.md` — venv path, node-on-PATH, C++ engine API.
- `project_box_scope.md` — division of labor (this box does single/few-game; main paper claims on the other box).
- `project_synth_unpack_bug.md` — bit-pack regression May 1–4, fixed 1fa557d. Numbers from that window are invalid.

## Open questions for the next agent

1. Once the sweep finishes: do the per-game numbers replicate the Microban anchor across A/B/C/D? Pre-registered prediction in `results.tex`.
2. Does TSP fail under pure synth (as predicted)? If so, document and try authored-seeded fallback (`--seed_from_authored` in synth-gen — needs CLI plumbing through train.py if not already there).
3. Multi-grid vs single-size comparison row in the appendix once both prewarm batches finish (cache files are already side-by-side; just need to run the multi_grid sweep cells too).

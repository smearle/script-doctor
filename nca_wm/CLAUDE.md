# NCA World Model (`nca_wm/`)

Subproject of the larger `script-doctor` repo. Trains a Neural Cellular
Automaton (NCA) world model that predicts PuzzleScript transitions
`f(state_t, action_t) → state_{t+1}`, plus win probability and (optionally)
sprite visuals. Other subprojects in the repo (RL training, LLM agents,
search, data pipelines, etc.) are unrelated and can be ignored when working
here.

## Dependencies on sibling code
- `puzzlescript_cpp` — C++ engine + gym wrapper used to roll out trajectories
  and supply observation shapes / canonical object IDs.
- `puzzlescript_jax.utils.init_ps_lark_parser` — used by `tokenize_game.py`.
- `train.py` inserts the repo root onto `sys.path` so it can import these
  while living in the subdir; follow the same pattern for new scripts.

## Files

### Models & training
- **`train.py`** (~2.9k lines) — main entry point (`main()` + `build_parser()`
  CLI, the `train()` loop, `make_train_step`, and `MULTI_GAME_PRESETS`). Much
  of what used to live here has been split into the cohesive modules below;
  train.py re-imports the names it still uses so existing
  `from nca_wm.train import …` call sites keep resolving.
  - Models defined in `models.py`: `NCAWorldModel` (unconditional),
    `ConditionalNCAWorldModel` (FiLM-modulated by a `GameSpecEncoder` over game
    tokens), with optional `axis_pool` / `axis_cummax` / `global_pool`
    global-context flags for multi-bracket / axis-aligned rules.
  - Training loop: per-game partitioned datasets, balanced sampling, gradient
    clipping, periodic per-game eval, early stopping, W&B logging,
    checkpointing. Optional `--history`, `--cond_mask`, and VQ flags.
- **`state_ops.py`** — dependency-free pure-numpy leaf helpers: bitpack /
  unpack (`_pack_states` / `_unpack_states`), padding (`_pad_obs` /
  `_pad_packed`), and multihot conversion (`_dats_to_multihot_batch` /
  `_multihot_to_objects`). Imported everywhere; imports nothing from the package.
- **`data_collection.py`** — solver-based transition collection
  (`collect_unique_transitions`, BFS / A* via the C++ backend), per-level and
  merged-dataset caches, `collect_multigame_dataset[_synthetic]`, the
  predecessor-graph helpers for `--history` sampling, and `_build_sprite_tensor`.
- **`inference.py`** — training-loop-independent `model.apply` wrappers shared
  by training / eval / rendering (`make_apply_fn`, `make_eval_forward`, `_wm_p`,
  `_pad_state_for_model`, `_unpad_pred`).
- **`eval.py`** — autoregressive multi-step rollout evaluation
  (`evaluate_multigame`, the numpy / jitted-jax rollout impls, scorecard).
- **`gif_rendering.py`** — training-time and post-training visualization
  (rollout GIFs, activation grids, sprite overlays, interactive playback).
- **`scaling_gallery_presets.py`** — curated `scaling_gallery_v{1..5}` game
  lists (superseded by `--n_per_rule_games`); merged into `MULTI_GAME_PRESETS`.
- **`rule_attn_model.py`** — Perceiver-style alternative. `RuleSlotEncoder`
  turns game tokens into K rule slots; `RuleAttnNCAWorldModel` does
  per-cell cross-attention to those slots at every NCA step. Selected via
  CLI flag in `train.py`.

### Tokenization & token autoencoder
- **`tokenize_game.py`** — domain-general PS-AST → integer token sequence.
  Two vocab tiers: `VOCAB_SIZE_BASE` (141, mechanics only) and
  `VOCAB_SIZE_EXT` (~183, includes sprite palette + 5x5 grids). Objects
  are encoded as channel indices (`ch0`, `ch1`, …) so naming is irrelevant.
  Entry point: `get_game_tree_from_js` → `tokenize_game`.
- **`token_decoder.py`** — autoregressive Transformer decoder
  `latent z → token sequence` with helpers `decoder_loss`, `shift_right`,
  `sample_tokens` (greedy / temperature).
- **`train_token_ae.py`** — standalone game-token autoencoder. Reuses
  `GameSpecEncoder` (optionally initialized from a trained world-model
  checkpoint for behavior-grounded latents) + `TokenDecoder`. Reads
  `game_infos.pkl` produced during multi-game collection.
- **`sample_latent_games.py`** — loads a trained token AE, decodes training
  latents, interpolates between two games, and samples from an empirical
  Gaussian over training latents. Token-level output only (no engine
  detokenizer yet).

### Sweeps & ops
- **`sweep.py`** (~1k lines) — named experiment presets (`CondVsUncond`,
  `HiddenSize`, `NCASteps`, `NGamesScaling`, `GlobalArchSingleGame`, …).
  Generates Cartesian arg sweeps, optionally submits to SLURM, then
  rediscovers completed runs by `sweep_name` in `config.json` and renders
  curves / heatmaps / per-game error breakdowns.
- **`scripts/parallel_collect.py`** — multiprocess wrapper around
  `collect_unique_transitions`. Defaults: `astar`, 100k iters, 60s timeout
  per level, skip-if-cached.
- **`scripts/recompress_caches.py`** — atomic in-place recompression of
  `rollout_data/**/*.npz` (multihot states compress ~100–800x). Uses
  tmp-file + `os.replace` so concurrent readers are safe.
- **`scripts/status.py`** — scans `nca_wm/logs/`, groups by `sweep_name`,
  reports latest step / loss / change-err / live status.
- **`scripts/summarize_run.py`** — for a single run dir, prints final train
  metrics + per-game change-error trajectories + per-rollout-type eval
  errors (random / BFS / A*).

## Key concepts

### Rollout cache layout
```
rollout_data/{game}/level_{i}/random.npz
rollout_data/{game}/level_{i}/search_{algo}_{budget}_{timeout}.npz
```
Random caches store contiguous episodes plus an episode-boundary index,
and are appended to when more episodes are requested. Search caches are
keyed by `(algo, budget, timeout)` and reused as-is when present.

### State representation
Bitpacked engine state → `(C, H, W)` `uint8` multihot via
`_dats_to_multihot_batch` (in `state_ops.py`). Channels are canonical object
indices (deduped against `raw_to_canonical`) so they match the model's input
shape across games.

### Conditional model
`GameSpecEncoder` consumes `(tokens, mask)` → latent `z`. A `FiLMAdapter`
turns `z` into per-step scale/shift vectors applied to NCA hidden state,
letting one model serve many games. Replaceable by the rule-attention
variant for spatially-distributed conditioning.

### Global context flags
- `axis_pool` — per-cell row/column max (axis-aligned rules).
- `axis_cummax` — directional prefix max along an axis.
- `global_pool` — grid-wide max (for multi-bracket rules like `[X] [Y]`).

## Typical CLI usage

```bash
# Single-game, unconditional
.venv/bin/python nca_wm/train.py --game pipe_bend

# Multi-game conditional
.venv/bin/python nca_wm/train.py --games small --conditional --n_hid 128

# Rule-attention variant
.venv/bin/python nca_wm/train.py --games small --conditional --architecture rule_attn --n_slots 16

# Data collection (parallel A* over the gallery preset)
.venv/bin/python -m nca_wm.scripts.parallel_collect --games gallery --workers 16 --skip-existing

# Recompress caches
.venv/bin/python -m nca_wm.scripts.recompress_caches --root rollout_data --workers 8

# Sweep: launch then plot
.venv/bin/python nca_wm/sweep.py cond_vs_uncond --mode train
.venv/bin/python nca_wm/sweep.py cond_vs_uncond

# Run dashboards
.venv/bin/python nca_wm/scripts/status.py
.venv/bin/python nca_wm/scripts/summarize_run.py nca_wm/logs/<run_dir>

# Token AE + latent sampling
.venv/bin/python nca_wm/train_token_ae.py --init_from <wm_ckpt_dir> --save_dir <ae_out>
.venv/bin/python nca_wm/sample_latent_games.py --ae_dir <ae_out> --n_random_samples 5
```

`parallel_collect.py --games` only expands the literal `gallery` preset. For
other multi-game sets, pass a comma-separated game list rather than a preset
name like `small` or `scaling_14`.

## Conventions when editing
- Keep imports of `puzzlescript_cpp` / `puzzlescript_jax` working from the
  subdir by inserting repo root onto `sys.path` (see top of `train.py`).
- New models should accept the same `(state, action[, tokens, mask])` API
  so they can be swapped via the `--architecture` flag in `train.py`.
- Cache files are append-friendly for `random` and immutable for `search_*`;
  do not break that contract — downstream scripts assume it.
- Run metadata lives in `<run_dir>/config.json` keyed by `sweep_name`;
  the sweep / status / summarize scripts all rely on this field.

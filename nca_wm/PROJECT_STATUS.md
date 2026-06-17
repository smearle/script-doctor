# NCA World Model Project Status

Last updated: 2026-06-17

This file is the current entry point for the `nca_wm/` subproject. It
summarizes the project goal, what has been implemented, what the experiments
currently show, and which generated files are safe to treat as cache or
scratch. For deeper detail, follow the source documents listed below rather
than the older tracker tables.

## Goal

The project is training a Neural Cellular Automaton (NCA) world model for
PuzzleScript games. The model takes:

- the current symbolic grid state,
- an action input,
- and, for multi-game models, a tokenized game rule specification,

and predicts the next symbolic grid state.

The motivation is that PuzzleScript itself is a sequence of local rewrite
rules applied in engine order, including repeated `again` execution until the
state stabilizes. An NCA has a compatible local-update bias, but it is not a
literal engine clone: it learns a fixed number of neural update passes and must
approximate ordered rule execution, long-range rule patterns, and loop
dynamics from data.

## Current Implementation

The main implementation lives in `nca_wm/`.

- `train.py` is the main training and data-collection entry point.
- `rule_attn_model.py` is the current default conditional architecture.
- `tokenize_game.py` converts PuzzleScript game ASTs into token sequences.
- `token_decoder.py` and related scripts support latent-to-game decoding.
- `synthetic_levels.py` and `curriculum.py` generate synthetic training levels.
- `heldout_eval.py` evaluates checkpoints on unseen games.
- `scripts/` contains launchers, plotting, cache, and post-processing tools.

The current best world-model architecture is a rule-attention NCA:

- a `RuleSlotEncoder` maps game tokens into K rule slots,
- each grid cell cross-attends to these slots during the NCA update,
- global context features are usually enabled for long-range PuzzleScript
  patterns such as ellipsis rules and multi-bracket rules,
- changed-cell loss weighting is used to avoid identity-prediction collapse.

## What Works

Single-game fitting is often easy after the encoder-mask and cache bugs were
fixed. Several games can be fit nearly perfectly or exactly under the right
recipe, which supports the basic claim that the NCA body can learn many
PuzzleScript transition systems.

Synthetic-level training can work well when the generated level distribution
matches authored level geometry. The strongest recurring finding is that grid
size and aspect-ratio coverage matter more than simply adding more synthetic
levels. For some games, synthetic training reaches authored-level fidelity.

Multi-game training works in-distribution for moderate sets, but capacity,
data balance, and rule coverage matter. Rule-attention conditioning outperformed
the earlier FiLM-style conditioning in the documented scaling runs.

The encoder latent space is meaningful enough to cluster exact token
duplicates and many related mechanics. Adding a token decoder can regularize
the latent space and preserve more reconstructable game-spec information.

## What Does Not Work Yet

The learned model is not a drop-in replacement for the PuzzleScript engine on
arbitrary unseen games.

Held-out predictive transfer is still weak. In the documented `v3_combined`
held-out rollout baseline, only 8 of 29 held-out games beat the identity
baseline at step 1, and only 1 of 29 beat identity over the full rollout mean.

Autoregressive rollout error can diverge from teacher-forced one-step loss.
Some deeper or higher-capacity models get lower train loss while producing
worse multi-step rollouts.

The latent game decoder improved substantially after explicit EOS and sprite
tokens were added, but random latent samples are still unreliable at larger
scale. Interpolations between known training endpoints are much more stable
than Gaussian samples from the fitted latent prior.

Perfect single-game overfit is not guaranteed for every game at every recipe.
Failures usually point to one of these issues:

- insufficient NCA update depth for loop-like dynamics,
- missing or harmful global context features for the game class,
- identity-collapse under sparse state changes,
- train/eval grid-size mismatch,
- data dilution in multi-game training,
- or teacher-forced loss not matching autoregressive behavior.

## Current Best Reading of the Research Question

The evidence supports a careful version of the claim:

An NCA can learn many PuzzleScript transition rules and can often fit a game or
a family of related games well, especially with rule-attention conditioning and
well-matched synthetic data.

The evidence does not yet support the stronger claim:

One shared rule-conditioned NCA can zero-shot simulate arbitrary PuzzleScript
games as reliably as the real engine.

The next useful work should focus on the gap between these claims: held-out
predictive accuracy, autoregressive stability, and whether the learned latent
rule space can adapt to new games from a small number of observed transitions.

## Active Track: VQ Codebook Regularization

The most concrete latent-space follow-up is to reduce VQ codebook collapse.
Prior token-ablation results showed that `vq1024+decoder` was the only
conditioning setup where zeroing or permuting tokens materially hurt both
teacher-forced and autoregressive performance. However, that run still used a
small number of active codebook entries, so the latent space was informative
but highly collapsed.

This track is scoped as the latent/regularization path: the goal is to test
whether the game latent can become an adaptation interface for novel games,
separate from the parallel work on more data, history-conditioned dynamics, and
code-conditioning dropout. The main evidence should come from token-ablation
sensitivity and few-shot inverse-fit behavior, not from training loss alone.

New code adds a differentiable usage-entropy regularizer to the VQ slots:

- `--vq_usage_loss_weight` adds `log(K) - H(mean soft assignment)` to the loss.
- `--vq_entropy_temp` controls the soft assignment temperature used only for
  this regularizer and its diagnostics.
- Training curves now log `vq_usage_losses` and `vq_soft_perplexities` in
  addition to hard code utilization.

Early `scaling_14` results are mixed but useful. Weights `0.001` and `0.01`
raise soft codebook perplexity to about 1024 without hurting decoder accuracy
or transition error, but hard nearest-code utilization remains collapsed at
about 3 active entries. The `0.05` run is still in progress and was stable
past 120k/200k steps, with hard utilization around 4. This means the current
regularizer improves the soft latent geometry and adaptation diagnostics, but
it is not yet a complete hard-code collapse fix.

Short token-ablation and inverse-fit pilots after this change are the main
reason to keep the branch:

- token zeroing hurts the regularized checkpoints more than the `w0` control,
  so rule-token conditioning still matters;
- few-shot inverse fitting improves more for `0.001` and `0.01` than for
  `w0`, after correcting the inverse-fit change-loss weighting.

The intended first sweep keeps the successful VQ+token-decoder recipe fixed
and varies only the usage regularizer:

```bash
GPU=0 \
GAMES=scaling_14 \
WEIGHTS="0 0.001 0.01 0.05" \
N_UPDATES=200000 \
nca_wm/scripts/run_vq_usage_ablation.sh
```

Readout criteria:

- hard `vq_util` and soft perplexity should increase versus the `weight=0`
  baseline,
- teacher-forced and autoregressive transition errors should not regress,
- token ablation should still hurt the model, otherwise the model may be using
  a superficially diverse but non-semantic codebook,
- held-out/few-shot inverse-fit should be rerun on the best checkpoint because
  the main question is whether a less-collapsed latent space helps map new rule
  sets into useful slots.

## Documentation Map

Use these files as the authoritative docs:

- `nca_wm/ARCHITECTURE_REPORT.md` - architectural decisions and major findings.
- `nca_wm/SCALING_RESULTS.md` - empirical results and tables.
- `nca_wm/SCALING_REPORT.md` - run index and operational notes.
- `nca_wm/RUNNING_REPORT.md` - older chronological notes and recipes.
- `nca_wm/SYNTH_HANDOFF.md` - synthetic-level handoff and open questions.
- `nca_wm/SUPPLEMENT_README.md` - paper/supplement reproduction notes.
- `nca_wm/VQ_REGULARIZATION_PLAN.md` - current VQ usage-regularization
  experiment plan and runbook.
- `nca_wm/CLAUDE.md` - compact subproject guide, currently untracked.

Be careful with these:

- `nca_wm/refine-logs/EXPERIMENT_TRACKER.md` is stale. It still lists many
  early milestones as TODO even though later experiments moved past them.
- `nca_wm/SCALING_REPORT.md` has some rows that were later superseded by
  `SCALING_RESULTS.md` and paper artifacts.
- Any result from the May 1-4 bitpack-regression window should be treated as
  suspect unless it was explicitly re-evaluated.

## Generated Files and Cleanup Policy

Do not delete generated artifacts blindly. Many are expensive to regenerate or
are referenced by paper tables. Use this policy first.

| Path | Current role | Cleanup guidance |
| --- | --- | --- |
| `rollout_data/` | Per-game rollout/search caches | Regenerable but expensive. Do not remove without confirming the affected experiments. |
| `rollout_data/_merged/` | Merged dataset caches, currently the largest disk user | Regenerable cache. Prefer `nca_wm/scripts/cleanup_merged_cache.sh` or an explicit manifest-based cleanup, not manual deletion. |
| `nca_wm/logs/` | Checkpoint/run outputs | Gitignored. Some old logs are the only local artifacts for early runs. Archive before deleting. |
| `nca_wm/logs_canary/` | Canary/invalidated experiment logs | Mostly historical. Keep until invalidated findings have been fully documented elsewhere. |
| `nca_wm/figures/` | Result summaries and generated plots | Some files are paper inputs. Do not bulk-delete. Regenerate only through the matching scripts. |
| `nca_wm/paper/` | Paper source and paper-local figures | Source-of-truth for manuscript text. Keep under version control discipline. |
| `.venv/` | Local Python environment | Rebuildable from setup instructions. Safe to recreate if disk pressure requires it. |
| `build/`, `*.so`, `__pycache__/` | Build artifacts and caches | Rebuildable. Safe cleanup candidates. |
| `.claude/`, `.aris/`, `.codex/` | Agent/tool state | Not part of research code. Leave alone unless intentionally cleaning tool state. |

The biggest current disk issue is not many small scripts; it is merged rollout
datasets. A disk cleanup should start by listing `rollout_data/_merged/*.npz`
with sizes, mapping them to active runs if possible, and deleting only stale
datasets through a documented cache-cleaning step.

## Recommended Next Steps

1. Make this file the first-read status document for `nca_wm/`.
2. Decide whether to track `nca_wm/CLAUDE.md` or fold its useful content into
   this file.
3. Run the inverse-fit few-shot diagnostic before launching larger training
   sweeps. This tests whether a frozen NCA body can adapt to a held-out game
   when the slot latent is optimized from a small number of observed
   transitions.
4. Re-run or repair held-out evaluation with subprocess-per-game isolation so
   held-out predictive accuracy is measured without JAX cache OOM failures.
5. Run a non-destructive cache audit for `rollout_data/_merged/` and produce a
   deletion manifest before removing anything.
6. Continue the synthetic-level line only with explicit authored-size coverage;
   avoid broad synthetic sweeps that do not match the target level geometry.
7. Treat token-decoder and inverse-fit experiments as latent-space evidence,
   not as proof of fully generative game synthesis.
8. Run the VQ usage-regularization ablation and compare code utilization,
   soft perplexity, token-ablation sensitivity, and rollout quality.

## Current Experiment: Inverse-Fit Few-Shot Diagnostic

New harness:

```bash
.venv/bin/python -m nca_wm.scripts.run_inverse_fit_grid \
    --load nca_wm/logs/20260430-165528_heldout5_15g_vq1024_dec_200k \
    --games wrappingrecipe \
    --shot_counts 10 \
    --n_steps 2 \
    --init knn \
    --out_subdir inverse_fit_grid_smoke
```

Smoke result: the harness can load a checkpoint, initialize from nearest
training slots, run slot optimization, and write per-run summaries. The first
full diagnostic should use more optimization steps and several held-out games:

```bash
.venv/bin/python -m nca_wm.scripts.run_inverse_fit_grid \
    --load nca_wm/logs/20260430-165528_heldout5_15g_vq1024_dec_200k \
    --games wrappingrecipe,kettle,notsnake,scriptcross \
    --shot_counts 10,50,100,500 \
    --n_steps 1000 \
    --init knn \
    --out_subdir inverse_fit_grid_v1 \
    --skip_done
```

If the post-fit loss and cell error improve strongly with more shots, the
frozen NCA body has useful held-out dynamics capacity and the encoder/latent
mapping is the bottleneck. If post-fit remains near the zero-shot/identity
baseline, the next work should target the NCA body or training distribution
rather than just the encoder.

The inverse-fit implementation now records pre-fit and post-fit metrics and
uses the same changed-cell weighting as training:
`1.0 + change_loss_weight * changed`. Results before that fix should not be
compared directly.

Pilot result after the loss-weighting fix, using 200 optimization steps on the
VQ usage ablation checkpoints:

```bash
.venv/bin/python -m nca_wm.scripts.run_inverse_fit_grid \
    --load nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p001 \
    --games wrappingrecipe,notsnake \
    --shot_counts 10,50 \
    --n_steps 200 \
    --init knn \
    --out_subdir inverse_fit_grid_vq_usage_pilot_lossfix \
    --skip_done
```

Summary:

| run | game | shots | pre cell err | post cell err | identity cell err | post loss |
|---|---|---:|---:|---:|---:|---:|
| `w0` | `notsnake` | 10 | 0.0100 | 0.0025 | 0.0200 | 0.0005616 |
| `w0` | `notsnake` | 50 | 0.0095 | 0.0080 | 0.0290 | 0.003429 |
| `w0` | `wrappingrecipe` | 10 | 0.0327 | 0.0347 | 0.0367 | 0.1454 |
| `w0` | `wrappingrecipe` | 50 | 0.0339 | 0.0282 | 0.0327 | 0.1207 |
| `w0p001` | `notsnake` | 10 | 0.0050 | 0.0000 | 0.0200 | 0.00005917 |
| `w0p001` | `notsnake` | 50 | 0.0110 | 0.0010 | 0.0290 | 0.0004532 |
| `w0p001` | `wrappingrecipe` | 10 | 0.0327 | 0.0245 | 0.0367 | 0.08382 |
| `w0p001` | `wrappingrecipe` | 50 | 0.0343 | 0.0269 | 0.0327 | 0.05054 |
| `w0p01` | `notsnake` | 10 | 0.0025 | 0.0000 | 0.0200 | 0.00002857 |
| `w0p01` | `notsnake` | 50 | 0.0095 | 0.0015 | 0.0290 | 0.0004232 |
| `w0p01` | `wrappingrecipe` | 10 | 0.0857 | 0.0306 | 0.0367 | 0.1013 |
| `w0p01` | `wrappingrecipe` | 50 | 0.0604 | 0.0188 | 0.0327 | 0.05766 |

Interpretation: the regularized checkpoints are better adaptation candidates
than `w0` in this pilot. Both `0.001` and `0.01` reach near-zero error on
`notsnake`; `0.001` is stronger on `wrappingrecipe` 10-shot, while `0.01` is
stronger on `wrappingrecipe` 50-shot. This is still a small pilot and should be
followed by more games and autoregressive evaluation before making a broad
claim.

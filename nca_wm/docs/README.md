# `nca_wm/` documentation index

Findings, reports, and design notes for the NCA world-model subproject, grouped
by topic. (Project instructions live in `nca_wm/CLAUDE.md`; per-experiment figure
summaries live next to their figures under `nca_wm/figures/*/summary.md`.)

## scaling/ — multi-game scaling & curriculum
- **SCALING_REPORT.md** — running state of the scaling effort (read before launching).
- **SCALING_RESULTS.md** — numbers and findings from scaling runs.
- **GAME_CURRICULUM_REPORT.md** — game-curriculum experiments.
- **TRUNCATION_ANALYSIS.md** — Heldout-30: impact of rule-token truncation on the OOD aggregates.

## architecture/ — model architecture
- **ARCHITECTURE_REPORT.md** — NCA world-model architecture report (the "F#" findings, e.g. F4 = input_skip default).
- **token_ablation_summary.md** — does the game-token encoder actually steer the NCA?

## vq/ — vector-quantized latent codebook
- **VQ_COLLAPSE_DIAGNOSTICS_AND_NEXT_STEPS.md** — VQ codebook-collapse diagnostics.
- **VQ_REGULARIZATION_PLAN.md** — VQ codebook regularization plan.

## synth/ — synthetic game generation
- **RULE_GP_DESIGN.md** — rule-grammar GP curriculum design sketch (referenced by `nca_wm/rule_gp.py`).
- **SYNTH_HANDOFF.md** — synth-only within-game generalization handoff.

## status/ — running logs & status
- **PROJECT_STATUS.md** — project status overview.
- **RUNNING_REPORT.md** — running report (referenced by `nca_wm/train.py`).
- **SUPERVISOR_LOG.md** — supervisor log.
- **IN_DIST_REPORT.md** — in-distribution single-game modeling investigation log.

## paper/
- **SUPPLEMENT_README.md** — "Modeling Many Worlds: Rule-Conditioned NCA" code supplement. (Paper sources proper live in `nca_wm/paper/`.)

## See also (docs that live with their feature)
- `nca_wm/active_learning/REPORT.md` — information-gain / belief world models + active data collection.
- `game_synth/FINDINGS.md` — GP↔WM environment co-evolution.
- `nca_wm/autumn/` — Autumn-engine port docs (`FINDINGS.md`, `DIVERGENCES.md`, `WORKLOG.md`).

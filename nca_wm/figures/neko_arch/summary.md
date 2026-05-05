# Nekopuzzle synth-level architecture sweep — eval summary

Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4, 15k updates, mask_hidden=True default. Synth: 64 levels at multi-grid {5x5, 6x6, 7x7, 8x8}. Authored levels are 8x7 — held-out from training.

Columns: TF/AR rollout cell-error on authored 10 levels (step-1 + 30-step mean), holdout = held-out synth at trained sizes (different seed), ood = synth at 9x9 (size-OOD).

| depth | share | pool | TF step1 | TF mean | AR step1 | AR mean | holdout | ood |
|---|---|---|---|---|---|---|---|---|
| 8 | per-step | OFF | 1.31% | 0.89% | 1.31% | 5.43% | 0.76% | 0.53% |
| 8 | per-step | ON | 0.81% | 0.53% | 0.81% | 4.44% | 0.58% | 0.42% |
| 8 | shared | OFF | 1.54% | 0.90% | 1.54% | 5.59% | 0.79% | 0.48% |
| 8 | shared | ON | 0.82% | 0.54% | 0.82% | 4.76% | 0.63% | 0.42% |
| 16 | per-step | OFF | 1.48% | 1.14% | 1.48% | 6.04% | 0.65% | 0.49% |
| 16 | per-step | ON | 0.28% | 0.28% | 0.28% | 3.05% | 0.56% | 0.41% |
| 16 | per-step | ON | 0.65% | 0.40% | 0.65% | 5.00% | 0.66% | 0.55% |
| 16 | per-step | ON | 1.00% | 0.59% | 1.00% | 4.37% | 0.58% | 0.43% |
| 16 | per-step | ON | 0.48% | 0.25% | 0.48% | 2.51% | 0.57% | 0.37% |
| 16 | per-step | ON | 0.17% | 0.19% | 0.17% | 1.12% | 0.56% | 0.40% |
| 16 | per-step | ON | 0.10% | 0.17% | 0.10% | 0.86% | 0.58% | 0.39% |
| 16 | per-step | ON | 0.06% | 0.02% | 0.06% | 0.65% | 2.87% | 15.20% |
| 16 | per-step | ON | 0.11% | 0.18% | 0.11% | 1.33% | 0.56% | 0.36% |
| 16 | per-step | ON | 0.06% | 0.12% | 0.06% | 0.65% | 0.58% | 0.38% |
| 16 | per-step | ON | 0.24% | 0.25% | 0.24% | 2.30% | 0.57% | 0.38% |
| 16 | shared | OFF | 1.60% | 1.20% | 1.60% | 6.03% | 0.93% | 0.47% |
| 16 | shared | ON | 0.94% | 0.68% | 0.94% | 5.02% | 0.62% | 0.45% |
| 32 | per-step | OFF | 1.41% | 1.07% | 1.41% | 6.59% | 0.72% | 0.52% |
| 32 | per-step | ON | 0.65% | 0.48% | 0.65% | 4.30% | 0.61% | 0.45% |
| 32 | shared | OFF | 1.90% | 1.25% | 1.90% | 6.76% | 0.92% | 0.54% |
| 32 | shared | ON | 1.29% | 0.76% | 1.29% | 5.62% | 0.61% | 0.41% |

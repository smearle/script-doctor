# Nekopuzzle synth-arch sweep — combined summary

Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4, 15k updates, mask_hidden=True default, change_loss_weight=5.0, balanced_sampling. Synth training data: 64 levels (16 per size) at multi-grid {5x5, 6x6, 7x7, 8x8} = 2,130 transitions. Authored neko levels are 8x7, held-out from training.

Eval columns:
- **BFS authored** = final-step cell-error rate on BFS-solution rollouts on the 10 authored levels (in-train eval, gold standard)
- **RAR authored** = final-step cell-error on random-action AR rollouts on authored levels
- **TF authored** = final-step cell-error with teacher-forcing on authored levels
- **Holdout synth** = single-step cell-error on a fresh synth pool (different seed) at trained sizes (5x5–8x8)
- **OOD synth** = single-step cell-error on synth pool at 9x9 (size-OOD, larger than max trained size)

| depth | share | pool | BFS authored | RAR authored | TF authored | Holdout synth | OOD synth |
|---|---|---|---|---|---|---|---|
| 8 | perstep | OFF | 7.25% | 5.87% | 0.51% | 0.76% | 0.53% |
| 8 | perstep | ON | 3.66% | 4.93% | 0.40% | 0.58% | 0.42% |
| 8 | shared | OFF | 7.95% | 6.46% | 0.57% | 0.79% | 0.48% |
| 8 | shared | ON | 4.40% | 4.77% | 0.26% | 0.63% | 0.42% |
| 16 | perstep | OFF | 5.98% | 6.71% | 1.11% | 0.65% | 0.49% |
| 16 | perstep | ON | 5.60% | 4.36% | 0.44% | 0.56% | 0.41% |
| 16 | perstep | ON | 8.44% | 7.65% | 0.25% | 0.66% | 0.55% |
| 16 | perstep | ON | 3.15% | 5.41% | 0.65% | 0.58% | 0.43% |
| 16 | perstep | ON | 2.10% | 3.54% | 0.00% | 0.57% | 0.37% |
| 16 | perstep | ON | 1.18% | 2.30% | 0.35% | 0.56% | 0.40% |
| 16 | perstep | ON | 0.31% | 1.64% | 0.00% | 0.58% | 0.39% |
| 16 | perstep | ON | 0.00% | 0.88% | 0.00% | 2.87% | 15.20% |
| 16 | perstep | ON | 1.92% | 2.62% | 0.07% | 0.56% | 0.36% |
| 16 | perstep | ON | 0.67% | 1.00% | 0.00% | 0.58% | 0.38% |
| 16 | perstep | ON | 3.50% | 3.78% | 0.42% | 0.57% | 0.38% |
| 16 | shared | OFF | 5.83% | 6.65% | 0.45% | 0.93% | 0.47% |
| 16 | shared | ON | 5.31% | 5.51% | 0.33% | 0.62% | 0.45% |
| 32 | perstep | OFF | 5.83% | 7.20% | 0.94% | 0.72% | 0.52% |
| 32 | perstep | ON | 4.75% | 5.69% | 0.18% | 0.61% | 0.45% |
| 32 | shared | OFF | 6.85% | 7.12% | 1.03% | 0.92% | 0.54% |
| 32 | shared | ON | 3.17% | 5.56% | 0.73% | 0.61% | 0.41% |

## Best per metric

- **BFS authored**: d=16 / perstep / pool=ON → 0.00%
- **Random-AR authored**: d=16 / perstep / pool=ON → 0.88%
- **TF authored**: d=16 / perstep / pool=ON → 0.00%
- **Holdout synth**: d=16 / perstep / pool=ON → 0.56%
- **OOD synth**: d=16 / perstep / pool=ON → 0.36%

# Per-game × per-architecture grid

Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4, n_nca_steps=8, --balanced_sampling, change_loss_weight=5.0. Trained on level 0 of each game.

Buckets:

- **A** — A: pool ON, shared
- **B** — B: pool ON, per-step
- **C** — C: no-pool + skip, shared
- **D** — D: no-pool + skip, per-step


## BFS rollout cell-error — training level (L0)

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Bouncers | 0.93% | 0.20% | 0.20% | 0.20% |
| Heroes_of_Sokoban | 0.00% | 0.00% | 0.00% | 0.00% |
| Microban | 0.00% | 0.00% | 0.00% | 0.00% |
| Travelling_salesman | 45.14% | 26.86% | 0.00% | 0.00% |
| nekopuzzle | 0.00% | 0.00% | 0.00% | 0.00% |
| sokoban_basic | 0.00% | 0.00% | 0.00% | 0.00% |

## BFS rollout cell-error — held-out levels (L1+; mean)

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Bouncers | 10.07% | 8.38% | 7.24% | 7.35% |
| Heroes_of_Sokoban | 25.56% | 27.71% | 27.64% | 26.92% |
| Microban | 22.91% | 21.87% | 25.86% | 15.55% |
| Travelling_salesman | 56.84% | 52.05% | 43.33% | 40.48% |
| nekopuzzle | 10.02% | 8.30% | 7.83% | 7.32% |
| sokoban_basic | 9.97% | 9.23% | 12.95% | 6.70% |

## Teacher-forced 1-step cell-error (random actions)

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Bouncers | 6.81% | 5.91% | 5.07% | 5.00% |
| Heroes_of_Sokoban | 24.09% | 24.28% | 23.26% | 23.35% |
| Microban | 10.15% | 6.06% | 6.44% | 5.19% |
| Travelling_salesman | 16.24% | 21.05% | 12.88% | 10.44% |
| nekopuzzle | 3.47% | 2.40% | 3.95% | 2.93% |
| sokoban_basic | 3.31% | 3.59% | 3.83% | 2.62% |

## Best train loss (cross-entropy)

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Bouncers | 6.02e-02 | 6.08e-02 | 6.06e-02 | 6.09e-02 |
| Heroes_of_Sokoban | 7.39e-02 | 7.53e-02 | 7.49e-02 | 7.59e-02 |
| Microban | 8.52e-02 | 8.68e-02 | 8.62e-02 | 8.72e-02 |
| Travelling_salesman | 1.70e-03 | 1.72e-03 | 1.70e-03 | 1.72e-03 |
| nekopuzzle | 1.45e-02 | 1.48e-02 | 1.45e-02 | 1.46e-02 |
| sokoban_basic | 1.26e-07 | 7.91e-08 | 8.68e-08 | 1.63e-07 |

## Convergence diagnostic

Ratio of mean loss in training window 50–75% to mean loss in window 75–100%. ~1.0 means flat; **⚠ flagged** when ratio>1.05 *and* absolute final loss > 1e-4 (numerical-floor cells, like fully-fit sokoban_basic, get noisy ratios but no flag). Compare same-row cells: if all four are still descending, extend the budget; if one bucket asymptotes much higher than its neighbors, that's a real architectural fit ceiling.

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Bouncers | 1.18 ⚠ | 1.18 ⚠ | 1.18 ⚠ | 1.18 ⚠ |
| Heroes_of_Sokoban | 1.18 ⚠ | 1.18 ⚠ | 1.18 ⚠ | 1.18 ⚠ |
| Microban | 1.18 ⚠ | 1.18 ⚠ | 1.18 ⚠ | 1.18 ⚠ |
| Travelling_salesman | 1.85 ⚠ | 1.85 ⚠ | 1.85 ⚠ | 1.85 ⚠ |
| nekopuzzle | 1.18 ⚠ | 1.18 ⚠ | 1.17 ⚠ | 1.17 ⚠ |
| sokoban_basic | 2.89 | 2.23 | 2.16 | 2.05 |

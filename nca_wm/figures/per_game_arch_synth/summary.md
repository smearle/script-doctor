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
| Microban | — | — | — | — |

## BFS rollout cell-error — held-out levels (L1+; mean)

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Microban | 9.16% | — | — | 8.38% |

## Teacher-forced 1-step cell-error (random actions)

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Microban | 1.19% | — | — | 0.96% |

## Best train loss (cross-entropy)

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Microban | 5.48e-02 | — | — | 5.83e-02 |

## Convergence diagnostic

Ratio of mean loss in training window 50–75% to mean loss in window 75–100%. ~1.0 means flat; **⚠ flagged** when ratio>1.05 *and* absolute final loss > 1e-4 (numerical-floor cells, like fully-fit sokoban_basic, get noisy ratios but no flag). Compare same-row cells: if all four are still descending, extend the budget; if one bucket asymptotes much higher than its neighbors, that's a real architectural fit ceiling.

| game | A: pool ON, shared | B: pool ON, per-step | C: no-pool + skip, shared | D: no-pool + skip, per-step |
|---|---|---|---|---|
| Microban | 1.21 ⚠ | — | — | 1.21 ⚠ |

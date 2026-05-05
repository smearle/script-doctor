# Heroes_of_Sokoban L0 depth × sharing × pool sweep

Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4, 15k updates, --balanced_sampling, change_loss_weight=5.0. Single-level (level 0).

`bfs_err`/`astar_err` = mean per-step cell error over BFS/A* oracle rollouts. `random_err`/`random_tf_err` = same for random-action rollouts (autoregressive / teacher-forced).

Re-evaluated 2026-05-04 with the top-left slicing fix. `bfs L0` is the training level (cell-error should be ~0 if the model fit it). `bfs heldout` is the mean over authored levels L1–L21, which the model never saw during training — these test in-game generalization across map sizes/topology.


## A: pool ON, shared

| depth | best loss | final change_acc | bfs L0 | bfs heldout | astar L0 | astar heldout | random_tf err | random err |
|---|---|---|---|---|---|---|---|---|
| 4 | 7.53e-02 | 100.0% | 0.00% | 23.45% | 0.00% | 23.46% | 21.27% | 23.76% |
| 8 | 7.36e-02 | 100.0% | 0.00% | 26.14% | 0.00% | 26.17% | 25.78% | 26.11% |
| 16 | 7.26e-02 | 100.0% | 0.38% | 25.86% | 0.38% | 25.82% | 25.53% | 25.89% |
| 32 | 7.20e-02 | 100.0% | 1.54% | 28.34% | 1.54% | 28.38% | 27.61% | 28.19% |

## B: pool ON, per-step

| depth | best loss | final change_acc | bfs L0 | bfs heldout | astar L0 | astar heldout | random_tf err | random err |
|---|---|---|---|---|---|---|---|---|
| 4 | 7.61e-02 | 100.0% | 0.00% | 23.50% | 0.00% | 23.52% | 22.85% | 23.42% |
| 8 | 7.54e-02 | 100.0% | 0.00% | 29.73% | 0.00% | 28.34% | 23.60% | 29.34% |
| 16 | 7.34e-02 | 100.0% | 0.58% | 29.03% | 0.58% | 29.33% | 27.52% | 28.71% |
| 32 | 7.07e-02 | 100.0% | 91.35% | 88.89% | 91.35% | 88.82% | 88.37% | 88.29% |

## C: pool OFF + skip, shared

| depth | best loss | final change_acc | bfs L0 | bfs heldout | astar L0 | astar heldout | random_tf err | random err |
|---|---|---|---|---|---|---|---|---|
| 4 | 7.87e-02 | 100.0% | 0.00% | 21.31% | 0.00% | 20.61% | 15.21% | 21.70% |
| 8 | 7.44e-02 | 100.0% | 0.00% | 26.75% | 0.00% | 26.50% | 23.81% | 27.29% |
| 16 | 7.28e-02 | 100.0% | 0.00% | 25.54% | 0.00% | 25.60% | 24.69% | 26.04% |
| 32 | 7.33e-02 | 100.0% | 0.00% | 25.39% | 0.00% | 25.44% | 24.76% | 25.41% |

## D: pool OFF + skip, per-step

| depth | best loss | final change_acc | bfs L0 | bfs heldout | astar L0 | astar heldout | random_tf err | random err |
|---|---|---|---|---|---|---|---|---|
| 4 | 7.67e-02 | 100.0% | 0.00% | 21.39% | 0.00% | 21.57% | 17.07% | 22.98% |
| 8 | 7.62e-02 | 100.0% | 0.00% | 25.11% | 0.00% | 25.12% | 22.34% | 25.07% |
| 16 | 7.45e-02 | 100.0% | 0.00% | 27.59% | 0.00% | 27.15% | 24.13% | 27.69% |
| 32 | 7.27e-02 | 100.0% | 0.00% | 26.13% | 0.00% | 26.16% | 24.80% | 26.33% |

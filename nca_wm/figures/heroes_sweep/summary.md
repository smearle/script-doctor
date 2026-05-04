# Heroes_of_Sokoban L0 depth × sharing × pool sweep

Recipe: rule_attn, h=256, n_slots=16, n_app_slots=1, batch=16, lr=3e-4, 15k updates, --balanced_sampling, change_loss_weight=5.0. Single-level (level 0).

`bfs_err`/`astar_err` = mean per-step cell error over BFS/A* oracle rollouts. `random_err`/`random_tf_err` = same for random-action rollouts (autoregressive / teacher-forced).


## A: pool ON, shared

| depth | best loss | final loss | final change_acc | bfs err | astar err | random_tf err | random err |
|---|---|---|---|---|---|---|---|
| 4 | 7.53e-02 | 7.54e-02 | 100.0% | 6.48% | 6.49% | 6.49% | 6.47% |
| 8 | 7.36e-02 | 7.37e-02 | 100.0% | 1.61% | 1.61% | 1.59% | 1.63% |
| 16 | 7.26e-02 | 7.27e-02 | 100.0% | 1.57% | 1.57% | 1.57% | 1.59% |
| 32 | 7.20e-02 | 7.21e-02 | 100.0% | 1.78% | 1.78% | 1.73% | 1.78% |

## B: pool ON, per-step

| depth | best loss | final loss | final change_acc | bfs err | astar err | random_tf err | random err |
|---|---|---|---|---|---|---|---|
| 4 | 7.61e-02 | 7.62e-02 | 100.0% | 6.45% | 6.46% | 6.45% | 6.45% |
| 8 | 7.54e-02 | 7.55e-02 | 100.0% | 1.89% | 1.77% | 1.40% | 1.96% |
| 16 | 7.34e-02 | 7.35e-02 | 100.0% | 1.94% | 1.97% | 1.87% | 1.94% |
| 32 | 7.07e-02 | 7.08e-02 | 100.0% | 15.35% | 15.12% | 15.51% | 15.84% |

## C: pool OFF + skip, shared

| depth | best loss | final loss | final change_acc | bfs err | astar err | random_tf err | random err |
|---|---|---|---|---|---|---|---|
| 4 | 7.87e-02 | 7.88e-02 | 100.0% | 1.50% | 1.46% | 1.04% | 1.59% |
| 8 | 7.44e-02 | 7.45e-02 | 100.0% | 1.71% | 1.66% | 1.45% | 1.77% |
| 16 | 7.28e-02 | 7.29e-02 | 100.0% | 1.58% | 1.59% | 1.52% | 1.62% |
| 32 | 7.33e-02 | 7.34e-02 | 100.0% | 1.50% | 1.51% | 1.46% | 1.53% |

## D: pool OFF + skip, per-step

| depth | best loss | final loss | final change_acc | bfs err | astar err | random_tf err | random err |
|---|---|---|---|---|---|---|---|
| 4 | 7.67e-02 | 7.68e-02 | 100.0% | 1.41% | 1.42% | 1.12% | 1.53% |
| 8 | 7.62e-02 | 7.63e-02 | 100.0% | 1.57% | 1.56% | 1.45% | 1.58% |
| 16 | 7.45e-02 | 7.46e-02 | 100.0% | 1.74% | 1.70% | 1.51% | 1.78% |
| 32 | 7.27e-02 | 7.28e-02 | 100.0% | 1.63% | 1.63% | 1.50% | 1.69% |

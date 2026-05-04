# Heldout transfer comparison

Per held-out game: AR step-1 cell-error rate (model / identity), with `*` = beats identity at single-step prediction.

## Checkpoints

| run | preset | n_steps | n_repeats | mask_hidden | n_updates |
|---|---|---|---|---|---|
| multi_scaling_14_v3recipe | scaling_14 | 8 | None | None | 150000 |
| multi_scaling_14_mask_v1 | scaling_14 | 8 | 8 | True | 80000 |
| multi_scaling_14_mask_v2_perstep | scaling_14 | 8 | 1 | True | 80000 |

## AR step-1 cell error (model / identity)

| game | multi_scaling_14_v3recipe | multi_scaling_14_mask_v1 | multi_scaling_14_mask_v2_perstep |
|---|---|---|---|
| sumo | 8.21% / 5.71%  | 12.14% / 5.71%  | 5.71% / 5.71%  |
| the_undertaking | 0.49% / 0.91%* | 0.50% / 0.91%* | 0.37% / 0.91%* |
| wrappingrecipe | 4.49% / 3.27%  | 6.94% / 3.27%  | 2.04% / 3.27%* |
| rigidfail1 | 2.59% / 5.93%* | 3.70% / 5.93%* | 2.59% / 5.93%* |
| constellationz | 4.53% / 1.01%  | 0.12% / 1.01%* | 2.66% / 1.01%  |

## AR rollout mean (over 30 steps)

| game | multi_scaling_14_v3recipe | multi_scaling_14_mask_v1 | multi_scaling_14_mask_v2_perstep |
|---|---|---|---|
| sumo | 34.43% / 4.70% | 50.99% / 4.70% | 39.49% / 4.70% |
| the_undertaking | 2.02% / 0.81% | 5.03% / 0.81% | 9.41% / 0.81% |
| wrappingrecipe | 25.77% / 3.35% | 28.94% / 3.35% | 29.47% / 3.35% |
| rigidfail1 | 9.72% / 2.79% | 13.74% / 2.79% | 33.17% / 2.79% |
| constellationz | 15.63% / 0.81% | 1.33% / 0.81% | 10.82% / 0.81% |

## Beats-identity tally (step-1)

| game | multi_scaling_14_v3recipe | multi_scaling_14_mask_v1 | multi_scaling_14_mask_v2_perstep |
|---|---|---|---|
| count | 2/5 | 3/5 | 3/5 |

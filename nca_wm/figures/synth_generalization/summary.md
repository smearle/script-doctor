# Synthetic Level Generalization Summary

Cell error rates are means across evaluated levels. Step-1 uses the
autoregressive evaluator's first predicted transition; identity is the
baseline that predicts the next state equals the current state.

## Held-Out Human Levels

| run | game | levels | step-1 | rollout | identity step-1 | perfect | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| authored_scaling6 | Microban | 10 | 0.367% | 2.894% | 1.302% | 8/10 | beats identity |
| single_synth_sokoban_w7 | Microban | 10 | 0.000% | 0.000% | 1.302% | 10/10 | beats identity |
| single_synth_sokoban_w8 | Microban | 10 | 1.365% | 5.228% | 1.302% | 3/10 | fails identity |
| synth_4games_per_game_size | Microban | 10 | 0.139% | 0.745% | 1.302% | 8/10 | beats identity |
| synth_scaling6_k5 | Microban | 10 | 0.565% | 2.903% | 1.302% | 7/10 | beats identity |
| synth_scaling6_multigrid_k5 | Microban | 10 | 0.394% | 3.515% | 1.302% | 5/10 | beats identity |
| synth_scaling6_v7_fallback | Microban | 10 | 0.258% | 2.223% | 1.302% | 8/10 | beats identity |
| authored_scaling6 | Microban_I | 20 | 0.388% | 4.007% | 1.322% | 14/20 | beats identity |
| single_synth_sokoban_w7 | Microban_I | 30 | 0.000% | 0.021% | 1.747% | 30/30 | beats identity |
| single_synth_sokoban_w8 | Microban_I | 30 | 1.337% | 4.163% | 1.747% | 10/30 | beats identity |
| synth_4games_per_game_size | Microban_I | 30 | 0.288% | 2.047% | 1.747% | 22/30 | beats identity |
| synth_scaling6_k5 | Microban_I | 30 | 0.259% | 1.757% | 1.747% | 25/30 | beats identity |
| synth_scaling6_multigrid_k5 | Microban_I | 30 | 0.623% | 3.478% | 1.747% | 14/30 | beats identity |
| synth_scaling6_v7_fallback | Microban_I | 30 | 0.256% | 1.761% | 1.747% | 24/30 | beats identity |

## Authored Training-Game Controls

| run | game | levels | step-1 | rollout | identity step-1 | perfect | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| authored_scaling6 | Travelling_salesman | 12 | 0.657% | 1.945% | 0.647% | 4/12 | fails identity |
| synth_scaling6_multigrid_k5 | Travelling_salesman | 12 | 2.801% | 28.726% | 0.647% | 0/12 | fails identity |
| authored_scaling6 | Zen_Puzzle_Garden | 5 | 0.000% | 0.000% | 1.120% | 5/5 | beats identity |
| synth_scaling6_k5 | Zen_Puzzle_Garden | 5 | 0.880% | 6.241% | 1.120% | 1/5 | beats identity |
| synth_scaling6_multigrid_k5 | Zen_Puzzle_Garden | 5 | 0.795% | 5.300% | 1.120% | 0/5 | beats identity |
| synth_scaling6_v7_fallback | Zen_Puzzle_Garden | 5 | 5.236% | 16.694% | 1.120% | 0/5 | fails identity |
| authored_scaling6 | blocks | 1 | 0.000% | 0.000% | 2.331% | 1/1 | beats identity |
| synth_scaling6_k5 | blocks | 1 | 0.000% | 0.000% | 2.331% | 1/1 | beats identity |
| synth_scaling6_v7_fallback | blocks | 1 | 0.000% | 0.194% | 2.331% | 1/1 | beats identity |
| authored_scaling6 | kettle | 11 | 1.988% | 6.656% | 6.168% | 0/11 | beats identity |
| synth_scaling6_k5 | kettle | 11 | 4.357% | 21.536% | 6.168% | 0/11 | beats identity |
| synth_scaling6_multigrid_k5 | kettle | 11 | 3.086% | 17.775% | 6.168% | 0/11 | beats identity |
| synth_scaling6_v7_fallback | kettle | 11 | 4.121% | 20.891% | 6.168% | 0/11 | beats identity |
| authored_scaling6 | nekopuzzle | 10 | 0.000% | 0.000% | 0.580% | 10/10 | beats identity |
| synth_4games_per_game_size | nekopuzzle | 10 | 1.354% | 5.637% | 0.580% | 3/10 | fails identity |
| synth_scaling6_k5 | nekopuzzle | 10 | 1.406% | 5.620% | 0.580% | 3/10 | fails identity |
| synth_scaling6_multigrid_k5 | nekopuzzle | 10 | 0.417% | 5.148% | 0.580% | 6/10 | beats identity |
| synth_scaling6_v7_fallback | nekopuzzle | 10 | 1.295% | 5.907% | 0.580% | 3/10 | fails identity |
| authored_scaling6 | sokoban_basic | 2 | 0.000% | 0.040% | 2.381% | 2/2 | beats identity |
| single_synth_sokoban_w7 | sokoban_basic | 2 | 0.000% | 0.000% | 2.381% | 2/2 | beats identity |
| single_synth_sokoban_w8 | sokoban_basic | 2 | 3.968% | 5.860% | 2.381% | 0/2 | fails identity |
| synth_4games_per_game_size | sokoban_basic | 2 | 1.190% | 6.429% | 2.381% | 1/2 | beats identity |
| synth_scaling6_k5 | sokoban_basic | 2 | 1.190% | 2.593% | 2.381% | 1/2 | beats identity |
| synth_scaling6_multigrid_k5 | sokoban_basic | 2 | 1.190% | 5.608% | 2.381% | 0/2 | beats identity |
| synth_scaling6_v7_fallback | sokoban_basic | 2 | 0.000% | 2.566% | 2.381% | 2/2 | beats identity |
| synth_4games_per_game_size | sokoban_match3 | 2 | 1.918% | 5.089% | 2.712% | 0/2 | beats identity |

## Readout

- Microban transfer is real: synthetic sokoban at the matched authored grid
  size is perfect on the cached Microban/Microban_I evals.
- The main repeatable failure is grid-size mismatch: the same synthetic
  sokoban recipe at 8x8 stops beating identity on the 6x7 authored
  sokoban control and becomes much worse on Microban.
- Per-game-size and multi-grid synthetic training recover most of the
  multi-game Microban transfer, roughly matching authored scaling_6.
- Residual failures concentrate in games where synthetic data is sparse,
  mixed-size, or dynamics-only: especially nekopuzzle, Zen, and kettle.

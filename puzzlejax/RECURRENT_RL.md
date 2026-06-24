# Recurrent RL agents on human-authored PuzzleScript games

A recurrent (GRU) PPO agent trained per-game on a variety of human-authored
PuzzleScript games, with GIFs of the best (greedy) agents.

## What was built

- `puzzlejax/models.py` — `ScannedRNN` (PureJaxRL-style GRU scan with done-masked
  carry resets) + `ConvEncoder` + `ActorCriticRNN`.
- `puzzlejax/train_jax_rnn.py` — recurrent PPO: hidden state + last-done threaded
  through the rollout, GAE, and a BPTT loss that re-scans the GRU over each
  trajectory; minibatching over the environment axis (time order preserved).
- `puzzlejax/enjoy_rnn.py` — renders greedy + stochastic GIFs from the latest
  checkpoint.
- `scripts/training/run_rnn_showcase.sh` — per-game train + render loop.
- `scripts/training/make_showcase_grid.py` — tiles winning GIFs into
  `showcase_grid.gif` / `showcase_contact_sheet.png`.

Train command (per game):
```
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.72 \
.venv/bin/python3 -m puzzlejax.train_jax_rnn game=<game> level=-1 \
    n_envs=256 num_steps=64 hidden_dims=[128,128] total_timesteps=1500000 \
    render_freq=0 ENT_COEF=0.02 wandb_mode=disabled
```

## Results (greedy win rate over 6 sampled levels)

| game | mechanic | max return | greedy wins | notes |
|---|---|---|---|---|
| kettle | cooking / push | 1.77 | **6/6** | near-perfect |
| Slidings | sliding tiles | 1.10 | **6/6** | near-perfect |
| sokodig | sokoban-dig | 1.95 | 4/6 | strong |
| Travelling_salesman | routing | 1.40 | 1/6 (sample 4/6) | wins solvable levels |
| Multi-word_Dictionary_Game | word | 1.34 (mid-train) | 0/6 | PPO late-collapse; won mid-training |
| sokoban_match3 | match-push | 0.94 | 0/6 | weak / didn't converge |

`showcase_grid.gif` and `showcase_contact_sheet.png` show the four winning agents
(Slidings, Travelling_salesman, kettle, sokodig) side by side.

## On the "multiple resets / undos" question

This was the central difficulty, confirmed empirically: pure recurrent PPO with
the heuristic-shaped reward **collapses on dead-end-prone single levels**. On
`sokoban_basic level=0` the policy converged to the step-penalty floor (return
−1.85, mean == max → zero exploration) and **never won** — once a box is pushed
into a corner the level is unsolvable but the agent has no way to recover within
an episode.

Two mitigations are visible in these results:
- **`level=-1` (sample all levels)** gives the agent a curriculum: easy/small
  levels are winnable, providing gradient signal that single hard levels do not.
  This is why every game above is trained with `level=-1`.
- The env auto-resets on episode end, so the agent gets *many attempts across*
  episodes, but cannot reset/undo *within* an episode.

A direct next step is to add a **restart action** (a 6th action that reloads the
level) so the agent can abandon a doomed attempt rather than burn 200 steps —
the literal "multiple resets" idea. Undo (step-back) is more expensive in JAX
(needs per-env state history) and is a later option.

## On the "expert-iteration" question

The repo's ExIt code (`scripts/training/exit_train_jax.py`) trains a neural
*heuristic for A\* search*, not a standalone playable policy — it's a
search-based solver, complementary to (not the same as) an RL agent. A genuine
ExIt *policy* loop (search finds wins on dead-end-prone games → imitation +
PPO) is the most promising route to crack the sokoban-class games that pure PPO
cannot, and would reuse the A\*/JAXtar search already present.

## Generalist (one net across games) vs per-game specialists

`puzzlejax/train_jax_rnn_multi.py` trains ONE shared GRU over all games at once
(parallel envs partitioned into per-game blocks; obs padded to a common shape;
conditioned on a learned game-ID embedding).

**win4 (4 games, 1 net, 4M steps)** converged win rates: kettle 0.99,
Slidings 1.00, sokodig 0.57, TSP 0.75 — matches the 4 separate specialists
(and beats the TSP specialist). One net does the job of four.

**Scaling (gen6, 6 games, 5M steps):** kettle 0.99, Slidings 1.00, match3 1.00,
sokodig 0.46, TSP 0.45, mwd 0.00. Easy games stay solved as game count grows;
harder games degrade gracefully (per-game capacity/data dilution); mwd is
RL-hard for everyone.

Compute caveat: the fully-jitted multi-game graph compiles slowly (XLA flags it)
— porting envs to C++/PufferLib (run outside the autodiff graph) is the right
fix for many-game scale.

## Resets / meta-RL (does "multiple resets" help?)

Three experiments, all honest about the outcome:

1. **Restart action** (`RestartActionWrapper`, single game): on the dead-end
   `sokoban_basic level=0`, baseline never won (return −1.85) and the
   restart-action agent did *worse* (−2.0). Resets don't fix dead-end *search*.

2. **Multi-trial meta-RL on visually-distinct games** (`trials_per_meta=K`,
   `use_game_id=False`): per-game few-shot curves are FLAT (e.g. kettle
   0.41/0.37/0.37/0.37 across trials), and K=1-no-ID ≈ game-ID oracle and beats
   K=4. The games self-identify from a single frame (distinct sprites/sizes), so
   there is nothing to disambiguate and persistent memory only adds optimization
   difficulty.

3. **Control-permutation meta-RL** (`train_jax_rnn_meta.py`: one game under N
   action-permutations — visually identical, mechanically different, so the
   agent must probe): the genuinely-ambiguous regime. The probing skill did not
   emerge in budget (kettle N=4/K=4 stuck at 0.04, flat few-shot; even the
   no-ambiguity sanity is only 0.20 at the short trial budget).

Takeaway: few-shot-via-resets is well-motivated but only matters under genuine
observational ambiguity, and learning to probe is a hard meta-RL problem needing
a clean base task and far more env steps — i.e. the compute regime that a
C++/PufferLib port unlocks. Rule-conditioning (rule_attn/FiLM) remains the path
to *zero-shot* generalization to unseen games.

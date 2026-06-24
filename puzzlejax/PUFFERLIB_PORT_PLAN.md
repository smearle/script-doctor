# PufferLib port of PuzzleScript RL — scoping plan

## Why

The JAX recurrent-PPO stack works (per-game agents win; one generalist matches
specialists; see `RECURRENT_RL.md`) but is **compute-bound by XLA compilation**:
the whole training loop is one fused `jax.jit`, so each per-game `env.step`
subgraph is inlined. A 4-game generalist compiles for ~10 min (XLA prints "very
slow compile"); 9 games is impractical. Meta-RL / few-shot probing additionally
needs *far* more env steps than the JAX path can afford.

Porting the env to **PufferLib** (envs run in C/C++ outside the autodiff graph,
policy in PyTorch) removes both problems: no mega-compile, and very high env
throughput so we can run the many-game and meta-RL regimes that JAX can't reach.

## Feasibility — already most of the way there (measured)

The repo's C++ engine already exposes a natively-batched gym env. Measured on
this box (`scripts`/scratch smoke test):

- `sokoban_basic` compiles (JS → JSON → C++) in **0.13 s** (vs ~10 min XLA).
- `CppBatchedPuzzleScriptEnv` throughput: **~93k steps/s at B=1024, ~284k at
  B=4096** (CPU, OpenMP, GIL released). Scales ~linearly with batch.
- Obs: `uint8` multihot `(B, n_objs, H, W)`; actions: `Discrete(5)`; reward
  computed in C++ (`score_delta + win_bonus - 0.01`); auto-reset supported.

So the heavy lifting (fast vectorized env, rendering, search) exists. The port
is mostly: a thin PufferLib env adapter, a PyTorch recurrent policy, and a
multi-game wrapper.

## Key building blocks (existing)

- `puzzlescript_cpp.CppBatchedPuzzleScriptEnv(json_str, batch_size,
  level_indices, max_episode_steps, auto_reset, num_threads)` — `reset()`,
  `step(int32[B]) -> (obs, rewards, dones, truncated, infos)`,
  `observation_shape`, `num_actions`, `set_levels`, `set_num_threads`.
- `puzzlescript_cpp.CppPuzzleScriptBackend.compile_and_serialize(parser, game)`
  → JSON (needs Node, which is on PATH; `init_ps_lark_parser()` for `parser`).
- `puzzlescript_cpp.Renderer.render_batched_env(be, env_idx)` → `(H,W,3)` uint8
  for GIFs.
- One `BatchedEngine` = **one game** across B level-instances. Multi-game ⇒ hold
  N batched envs (one per game) and concatenate (pad obs to a common shape, as
  the JAX generalist already does).

## Missing pieces

- `pufferlib` and (optionally) `cleanrl` are NOT in the `.venv`. `gymnasium 1.2.3`
  and `torch 2.10+cu128` are present.
- A PyTorch recurrent actor-critic (port of `ActorCriticRNN`: conv encoder → GRU
  → actor/critic). PufferLib ships a recurrent PPO (`clean_pufferl`) we can drive.

## Architecture

```
CppBatchedPuzzleScriptEnv (C++, OpenMP)         # 1 game, B envs
        │  obs (B, n_objs, H, W) uint8
        ▼
PufferPSEnv  (PufferLib PufferEnv adapter)      # single-game
        │
MultiGamePSEnv  (holds N PufferPSEnv, pads obs   # generalist
        │        to (Cmax,Hmax,Wmax), adds game-id)
        ▼
clean_pufferl PPO + RecurrentPolicy (PyTorch)   # GRU actor-critic on GPU
```

- **Env adapter** (`puffer_ps_env.py`): wrap `CppBatchedPuzzleScriptEnv` to the
  PufferLib `PufferEnv` API (it expects a contiguous obs buffer; our obs is
  already a single `(B, …)` uint8 array — minimal copy). Map reward/done/truncate
  through. Expose `single_observation_space` / `single_action_space`.
- **Multi-game**: a wrapper holding one `CppBatchedPuzzleScriptEnv` per game with
  per-game batch slices; pad each game's `(n_objs,H,W)` to the max; concatenate to
  `(ΣB, Cmax,Hmax,Wmax)`; carry a per-env `game_id`. (Mirrors
  `train_jax_rnn_multi.py` exactly, but with no fused graph.)
- **Policy** (`puffer_models.py`): conv encoder (CHW→feat) → `nn.GRU` → actor
  (Discrete 5) + critic. Optional `nn.Embedding(game_id)` and, later, a
  rule-encoding input (rule_attn/FiLM) for zero-shot generalization.
- **Training**: PufferLib's recurrent PPO. Reuse its vectorization, LSTM/GRU
  handling, and logging; we only supply env + policy.
- **Meta-RL**: the multi-trial / control-permutation logic moves into the env
  adapter (reset level each trial, keep policy hidden state across trials, reset
  it at meta boundaries; apply per-env action permutation). This is where the
  throughput pays off — few-shot probing needs 10–100× more steps.

## Milestones

- **M0 — deps + env smoke** (½ day): `pip install pufferlib` (pin a version that
  supports recurrent policies; verify it builds against torch 2.10/cu128). Wrap
  `CppBatchedPuzzleScriptEnv` as a PufferEnv; random-rollout smoke + throughput.
- **M1 — single-game recurrent PPO** (1 day): GRU policy; reproduce the JAX
  per-game wins (kettle, Slidings ~1.0) — *parity check*. Confirm no compile wall
  and measure wall-clock vs JAX.
- **M2 — multi-game generalist** (1–2 days): multi-game wrapper + game-id;
  reproduce win4/gen6 numbers, then push to **many games (gen9, 20, 40…)** —
  the regime JAX couldn't compile. Plot win-rate vs game-count.
- **M3 — meta-RL at scale** (2–3 days): multi-trial + control-permutation in the
  env adapter; run the few-shot probing experiment with a clean base task (solved
  ~1.0) and 10–100× the steps. This is the proper test of the resets/few-shot
  hypothesis that the JAX budget couldn't reach.
- **M4 — eval/GIF + rule-conditioning** (1 day): render via the C++ `Renderer`;
  add a rule-encoding policy input for zero-shot transfer to unseen games.

## Risks / open questions

- **PufferLib version churn**: the API (PufferEnv, `clean_pufferl`) changes across
  releases; pin a version and follow its env contract. Mitigation: start from a
  Puffer example env and swap in our backend.
- **Reward parity**: C++ reward is `score_delta + win - 0.01` using the JS
  heuristic *score*; the JAX env used a distance-to-win heuristic. Confirm they're
  close enough, or align them, before reading parity into M1/M2 numbers.
- **CPU↔GPU transfer**: at huge B the `(B, n_objs, H, W)` uint8 obs transfer to GPU
  each step can bottleneck; keep obs uint8, pin memory, possibly async copy.
  (uint8 multihot is small, so likely fine.)
- **Multi-game obs padding**: different `n_objs/H/W` per game → pad to a common
  shape (as in JAX). Channel semantics still differ per game → game-id (or rule
  encoding) conditioning needed; not zero-shot without rules.
- **Determinism / seeding**: C++ RNG seeding via `Engine.seed_rng` / `load_level`
  seed — wire through for reproducibility.
- **Node dependency** for compilation only (one-time per game → JSON); not on the
  hot path. Cache compiled JSON to disk.

## Dependencies to add

```
.venv/bin/pip install pufferlib   # pin a recurrent-capable version; verify cu128
# (cleanrl optional if we prefer ppo_lstm over clean_pufferl)
```
`gymnasium`, `torch`, `numpy`, the C++ `.so` (built, importable) are already present.

## First concrete step

M0: add `puzzlejax/puffer_ps_env.py` wrapping `CppBatchedPuzzleScriptEnv` as a
PufferEnv, plus a 30-line random-rollout + throughput script, gated behind a
`pufferlib` install. Everything downstream reuses the existing C++ engine.

## STATUS / findings (2026-06-24)

**Dependency conflict (important).** `pufferlib==3.0.0` hard-pins `numpy<2.0` and
`gymnasium<=0.29.1`, which conflict with this repo's JAX stack (`numpy>=2`,
`gymnasium 1.2.3`). Installing it into the main `.venv` downgraded numpy and
broke jax/nca_wm; the main venv has been **restored** (numpy 2.2.6, gymnasium
1.2.3, gym 0.26.2; pufferlib removed). Worse, importing the `puzzlescript_cpp`
*wrapper* pulls in `backends.nodejs` → `from javascript import require` and
`puzzlescript_jax.utils` (→ jax → numpy>=2), so the wrapper can't even import in a
numpy<2 venv. ⇒ A true in-pufferlib run needs an **isolated venv** that uses the
raw `_puzzlescript_cpp` .so (Engine/BatchedEngine/Renderer) directly, bypassing
the Python wrapper and `backends`.

**Built and validated in the main venv (conflict-free, no pufferlib import):**
- `puzzlejax/puffer_ps_env.py` — `PuzzleScriptVecEnv`: multi-game batched env over
  the C++ engine (one `CppBatchedPuzzleScriptEnv` per game, obs padded to a common
  (C,H,W), per-env `game_ids`), gym-vector / PufferEnv-shaped API
  (`single_observation_space`, `single_action_space`, `reset`, `step`). Smoke: 4
  games padded to (12,19,20), 4096 envs, ~33k env-steps/s (the per-game engines
  step sequentially in Python — parallelizing them, or a single multi-game C++
  engine, is the obvious throughput win).
- `puzzlejax/puffer_models.py` — `RecurrentPolicy` (torch): conv→GRU→actor/critic
  with done-masked carry reset + optional game-id embedding. End-to-end
  env(CPU)↔policy(GPU) smoke passes (obs→encode→GRU→sample→step), 3.3M params, no
  XLA compile. Precompiled game JSON cached under `puffer_assets/games_json/`.

**Recommended path (decision point):**
- **(A) No-pufferlib, main venv:** write a cleanrl-style torch recurrent PPO loop
  directly on `PuzzleScriptVecEnv` (the C++ env already provides the
  vectorization). Zero dependency conflict, uses existing torch 2.10+cu128, ready
  now. Gets the full compute win (fast C++ envs, no compile wall).
- **(B) True PufferLib, isolated venv:** create `.venv_puffer` (numpy<2 + pufferlib
  + torch), import the raw `_puzzlescript_cpp` .so (no wrapper/backends), wrap as a
  PufferEnv, drive `clean_pufferl`. More setup + a duplicate torch, but gets the
  puffer ecosystem (tuned PPO, sweeps, dashboards).

(A) is the faster route to M1–M3 results; (B) buys the Puffer tooling. The env
adapter + policy above are shared by both.

**Decision: went with (A).** Also fixed the jax-drag: `backends/__init__.py` now
lazy-imports the nodejs backend (PEP 562), so `import puzzlescript_cpp` loads
neither jax nor javascript.

**M1/M2 RESULT (path A, `puffer_train.py`):** single-game Slidings learns in
seconds (no compile). The **win4 generalist (4 games, game-id embed, 8M steps)
converged to ~1.00 on ALL four games** (kettle/Slidings/sokodig/TSP), *beating*
the JAX generalist (which had sokodig 0.57, TSP 0.75) — at ~10k SPS, 8M steps in
~13 min, starting instantly (no XLA wall). Caveat: the C++ env reward
(`score_delta + win - 0.01`) differs from the JAX heuristic reward, so this is not
a perfectly controlled comparison, but it removes the compute bottleneck and
matches-or-exceeds quality. Next: M2 scale to many games (gen9/20/40 — the regime
JAX couldn't compile), M3 meta-RL at scale, and throughput (parallelize the
per-game engines / single multi-game C++ engine; currently they step
sequentially in Python).

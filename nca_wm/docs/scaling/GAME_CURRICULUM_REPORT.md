# Game Curriculum Report

This is the running log of the **LLM-driven game-curriculum loop**
(`nca_wm.game_curriculum`) — the game-level analogue of `nca_wm.curriculum`.
Instead of mutating *levels* for one fixed game, this loop evolves fixed-size
*sets of complete PuzzleScript games* by querying a local OpenAI-compatible
vLLM server. Each surviving set ships with a ready-to-run `train_command.sh`
that calls `nca_wm.train --games ...`.

For per-game world-model scaling runs, see `SCALING_REPORT.md` /
`SCALING_RESULTS.md`. For the broader chronological narrative, see
`RUNNING_REPORT.md`.

## Index of runs

| run | model | seeds | pop × set × gens × children | best score | save_dir | status |
| --- | --- | --- | --- | --- | --- | --- |
| smoketest_2026-05-03 | `vllm-qwen3-8b` (Qwen/Qwen3-8B on GPU 1) | `scaling_4` | 4 × 4 × 3 × 4 | **8.331** | `nca_wm/logs/game_curriculum_vllm_qwen3_8b_1777847610_0` | DONE |

## Smoketest run (2026-05-03)

First end-to-end exercise of the loop. Goal was just to validate that every
phase fires (seed scoring → LLM gen → C++ compile → BFS search → tokenize →
set-level scoring → artifact emission → elitist replacement) and to surface
operational quirks.

### Results

| gen | top score | median score | top set solvable_frac |
| --- | --- | --- | --- |
| 0 (seeds) | 5.889 | 5.889 | 0.50 |
| 1         | 5.928 | 5.889 | 0.50 |
| 2         | 5.928 | 5.889 | 0.50 |
| 3         | **8.331** | **7.831** | **0.75** |

The gen-3 jump came from a `mutate` of seed `blocks` that produced an
11-level `Travelling Salesman: Eulerian Path` variant with all levels
solvable (max search iters 4376, 184 tokens). The set's other slots were
inherited seeds — single-slot replacement is the only mutation per child.

Best set composition (`set_g003_003_6fd3de`):

```
blocks                                    (seed,  unsolved, 27000 iters, score 2.80)
nca_gc_..._g003_mutate_cb                 (mutate, solvable, 4376 iters, score 10.21)
sokoban_basic                             (seed,  solvable,   900 iters, score 8.06)
nekopuzzle                                (seed,  solvable,    26 iters, score 4.52)
```

Train command emitted at `<best>/train_command.sh`:

```
.venv/bin/python -m nca_wm.train \
    --games "blocks,nca_gc_vllm_qwen3_8b_1777847610_0_g003_mutate_cb,sokoban_basic,nekopuzzle" \
    --save_dir ".../wm_run" \
    --conditional --architecture rule_attn --n_updates 5000 \
    --max_transitions_per_game 50000
```

Has not been launched yet — separate workstream, paired with `scaling_4`
runs in `SCALING_REPORT.md`.

### Compile / solvability by mutation mode

12 LLM-generated child slots across 12 children (3 gens × 4 children),
counted before any retry/repair upgrades the slot's `mode` tag. (Repair is
booked as a `mode+repair` re-eval, hence the high count of `+repair` rows.)

| mode | compile_ok / N |
| --- | --- |
| `targeted+repair` | 5/5 |
| `crossover+repair` | 2/2 |
| `mutate` | 1/1 |
| `targeted` | 1/1 |
| `crossover` | 0/1 |
| `init` | 0/2 |

Open-ended `init` was 0/2 in this run; the bare `init`/`crossover` paths are
where the LLM most often omits a required section (typically `WINCONDITIONS`)
and `preprocess_ps` raises the unpack error described below. The
solver-feedback `repair` pass cleans these up reliably and is currently the
loop's main safety net. **15 unique LLM games** ended up materialized to
`custom_games/nca_gc_vllm_qwen3_8b_1777847610_0_*` (more than 12 because
repair reissues a fresh name).

solvability across all valid LLM children: 2/12 fully solvable. Search
hardness is the main score driver, but `solvable_frac` directly multiplies
the per-game `score`, so the loop is heavily pressured to fix unsolvable
slots — visible in the gen-3 winner.

## Operational quirks discovered

These all bit during the smoketest. Worth fixing before the next run.

- **`--model` rejects bare HF ids.** `Qwen/Qwen3-8B` raises
  `Unknown vLLM model alias`. Use the alias (`vllm-qwen3-8b`) — the
  resolver is in `puzzlescript_jax/utils.py:resolve_vllm_model`. The bare
  `vllm` alias defers to `VLLM_MODEL` env var if you need a custom model.
- **`srun_vllm_server.sh` assumes SLURM**, which isn't installed on this
  host. The launch sequence used here was:
  ```
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m vllm.entrypoints.openai.api_server \
      --model 'Qwen/Qwen3-8B' --port 8000 --max-model-len 32768 \
      > /tmp/vllm_server.log 2>&1 &
  ```
  Server cold-start ~30 s; ~22.5 GB on a 4090.
- **Python output is line-buffered when piped to a file.** Running the
  curriculum with `> log 2>&1` makes the per-child `print()` lines invisible
  for minutes at a time. The most reliable progress signal is
  `ls <save_dir>/gen_*/`. Pass `-u` to Python (or set `PYTHONUNBUFFERED=1`)
  to fix.
- **Noisy preprocessing tracebacks.** When the LLM emits a game missing the
  `SOUNDS` block (or any of the 8 standard sections), `preprocess_ps` raises
  `ValueError: not enough values to unpack (expected 8, got 7)` and prints
  the full traceback. `_eval_candidate` catches it and assigns `score=-1.0`,
  so the run continues — but the log looks like it crashed. Either swallow
  these in `preprocess_ps` (return `None`) or in `_eval_candidate` add a
  short-form `f"preprocess failed: {e}"` log line.
- **`init` mode is brittle.** 0/2 compiled in this run; it's also the
  hardest mode to repair because there's no parent to fall back to. Worth
  either lowering its weight (currently 0.10 in `_breed_set`) or
  always running it through repair regardless of compile status.

## Launching

Server (any machine, GPU-only):

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model 'Qwen/Qwen3-8B' --port 8000 --max-model-len 32768
```

Curriculum (CPU + remote LLM):

```bash
.venv/bin/python -u -m nca_wm.game_curriculum \
    --model 'vllm-qwen3-8b' \
    --vllm_base_url 'http://localhost:8000/v1' \
    --seed_games scaling_4 \
    --set_size 4 --pop_size 4 \
    --n_generations 3 --n_children_per_gen 4
```

Defaults take ~25 min wall on a 4090 + Qwen3-8B (~12 LLM rounds + per-game
BFS at 10 s timeout). Larger model (e.g. `vllm-qwen3-32b`) trades wall time
for a better compile rate on `init`.

## Output layout

```
nca_wm/logs/game_curriculum_<model_tag>_<unix_ts>_<seed>/
  config.json             # full argparse dump
  history.json            # per-gen population snapshots (sorted by score)
  best/                   # symlink-style copy of the best set's gen-N dir
  gen_000/                # seed sets
    set_<uid>/
      games.txt           # comma-separated game names (newline-terminated)
      summary.json        # set score + metrics + per-game candidate dicts
      train_command.sh    # ready-to-run nca_wm.train invocation (chmod +x)
  gen_001/, gen_002/, gen_003/
    set_g00N_<i>_<hash>/  # one dir per child set
      ...

custom_games/
  nca_gc_<run_tag>_g<gen>_<mode>_<hash>.txt   # raw .txt for every materialized
                                              # LLM game (15 in this run)
```

Repaired games keep the parent's `mode` prefix and add `+repair` (e.g.
`mode = targeted+repair`); the filename uses the repair mode tag itself.

## What we did *not* validate

- **No actual world-model training was run.** The curriculum's score is the
  cheap proxy described in the docstring (search hardness + token diversity
  + shape sanity, gated by solvability). Whether sets that score well in the
  proxy actually train better world models is the next experiment — would
  require running `train_command.sh` for top-K sets and feeding that back.
- **Single seed**, single model, no replicates. A 5.889 → 8.331 jump in 3
  generations could just as easily be variance over LLM samples. Need a
  multi-seed sweep before drawing any conclusions about whether evolution is
  doing real work over single-shot sampling.

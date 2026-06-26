# Autumn ELM — evolving novel variants of AutumnBench environments

LLM-driven (ELM) evolution of Autumn `.sexp` programs. Start from an authored
seed (e.g. `mario`), mutate one rule at a time with an LLM, keep variants that
are **valid** and produce **distinct rollouts under random play** (the user's
acceptance criterion).

## Pipeline

| File | Role |
|------|------|
| `sexp.py` | S-expr span parser; exposes *mutable targets* = whole `(on ...)` handlers + `initnext` NEXT-clauses. Object defs, `GRID_SIZE`, and all INIT-clauses are frozen, so **the initial level layout never changes**. |
| `rollout_worker.py` | Child process: load a program, run a fixed-seed random probe, emit a behavioral signature (covered `on`-clause indices via `get_covered_on_clause_indices` + trajectory hash). |
| `rollout.py` | Parent wrapper. Runs the worker in a **subprocess** (the C++ interpreter *segfaults* on bad programs — must isolate). `evaluate_multi` runs K seeds; novelty = any trajectory differs. |
| `mutate.py` | LLM mutation operator. Feeds the program + the marked target + all 15 authored-env docstrings (mutation palette); asks for a named mutation + a single replacement S-expression. Backends: `vllm` (local, free, default) or `anthropic` (paid, opt-in). |
| `evolve.py` | ELM loop: sample parent → pick target → mutate → splice → validate → fingerprint → accept iff valid, non-dead, novel. Writes `variants/*.sexp`, `manifest.jsonl`, `summary.json`. |

## Run

```bash
# vLLM server (free, local) — cap mem to fit a shared 4090:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --gpu-memory-utilization 0.48 --max-model-len 8192 \
  --enforce-eager --port 8011 --served-model-name qwen3-4b

PYTHONPATH=game_synth/autumn .venv/bin/python3 game_synth/autumn/evolve.py \
  --seed-game mario --iters 20 --backend vllm --probe-seeds 3
```

## Findings (2026-06-24)

- **End-to-end works.** Qwen3-4B, mario, 20 iters: `parse_fail 0`, `valid_rate 0.80`.
- **Novelty probe is the lever.** Single random seed → accept 5% (most mutations
  add a branch random play never triggers → trajectory identical to seed →
  "dup"). 3-seed union → accept 30% (more handlers exercised, differences
  surface). This is the AutumnBench analog of the PuzzleScript "activated
  mechanics" lesson: source-distinct ≠ behavior-distinct under a weak policy.
- Accepted variants are coherent (enemy-direction flips, click-to-shoot changes,
  bullet-retreat) and re-validate independently.

## Notes / gotchas

- `restore_environment` exists but its serialized state is **not** cleanly
  round-trippable, so a coverage-walk probe (try-each-action-from-backup) is not
  available; multi-seed random probing is the restore-free substitute.
- vLLM is NOT broken on this box — the default `gpu_memory_utilization=0.9`
  collided with other users' memory. Cap it.

# Modeling Many Worlds: Rule-Conditioned NCA — Code Supplement

This is the anonymized code supplement for the NeurIPS submission
*Modeling Many Worlds: Rule-Conditioned Neural Cellular Automata as
Universal Puzzle Game Engines*.

It contains:

- `nca_wm/` — the world-model code (training, evaluation, ablations,
  paper LaTeX sources, and result aggregators).
- `puzzlescript_jax/` — the PuzzleScript-to-JAX preprocessing pipeline
  (parser, tokenizer, environment wrapper).
- `puzzlescript_cpp/` — the C++ PuzzleScript engine used as the ground
  truth for caching transitions; ships precompiled `.so` for Python
  3.12/3.13 plus source under `puzzlescript_cpp/src/` for rebuilds.
- `backends/`, `conf/` — engine-backend abstractions and
  Hydra training configs.
- `gallery_games/`, `custom_games/`, `data/scraped_games/` — the
  human-authored PuzzleScript corpus the dataset is built from.
- `evolve_level_cpp.py`, `evolve_games_agentic.py` — search /
  level-generation utilities used by the synth-data pipeline.

The bundled rollout-transition caches are *not* included (they are
~91 GB on disk). The repro instructions below regenerate the caches
from the bundled C++ engine and game corpus.

## Setup

```bash
export REPO="$(pwd)"          # the repo root inside the unzipped dir
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r nca_wm/requirements.txt   # if shipped; otherwise:
pip install jax[cuda12] flax optax numpy hydra-core wandb \
            imageio pillow matplotlib
```

The C++ engine is shipped as a precompiled extension for Python 3.12
and 3.13 on x86_64-linux. For other platforms, rebuild from source:

```bash
cd puzzlescript_cpp/src && make    # or follow the README in src/
```

## Hardware

Every experiment in the paper was trained on a single 24 GB consumer
GPU (NVIDIA RTX 4090). No multi-GPU or distributed training is needed.

## Reproducing the main tables

| Paper section | Table | Launcher |
|---|---|---|
| §4.1 Single-game architectures | Tab. 1 (Microban) | `bash nca_wm/scripts/run_authored_sokoban_repro.sh` |
| §4.1 Multi-grid synth | Tab. 2 (sokoban_basic) | `bash nca_wm/scripts/run_baselines_synth.sh` |
| §4.2 Per-feature ablation | Tab. 3 (pool / input-skip) | `bash nca_wm/scripts/run_ablation_pool.sh` |
| §4.3 Multi-game scaling | Tab. 4 (Train-14) | `bash nca_wm/scripts/run_scaling_14.sh` |
| §4.4 Cond vs uncond | Tab. 5 | `bash nca_wm/scripts/run_cond_vs_uncond.sh` |
| §4.5 Broad-corpus fidelity | Tab. 6 (Train-94) | `bash nca_wm/scripts/run_train_94.sh` |
| App. C OOD geometry | Tab. 7 | `bash nca_wm/scripts/run_heldout_geometry.sh` |

> **Note.** Specific launcher script names may differ from the table
> above; an authoritative `nca_wm/scripts/REPRO.md` will be added in a
> follow-up update before the deadline. The launchers themselves are
> straightforward: each one calls `python -m nca_wm.train` with the
> hyper-parameters listed alongside its corresponding table in the paper
> (Section 3, §3.3 *Architectures and hyper-parameters*).

Each launcher caches per-game transitions under `${REPO}/rollout_data/`
the first time it runs, then trains and writes outputs under
`${REPO}/nca_wm/logs/<run_name>/`. Re-runs of the same launcher reuse
the cache.

## Regenerating transition caches manually

```bash
python evolve_level_cpp.py game=Microban   # writes rollout_data/Microban/
```

For the broad-corpus presets (Train-14, Train-94, Train-199), the
launchers above already invoke the cache step.

## Evaluating a trained checkpoint

```bash
python -m nca_wm.heldout_eval \
    --run_dir nca_wm/logs/<run_name> \
    --eval_games heldout_v4_n30
```

## Repository layout

- `nca_wm/train.py` — main training entrypoint.
- `nca_wm/rule_attn_model.py` — the rule-conditioned NCA architecture.
- `nca_wm/baselines.py` — CNN / U-Net / ViT baselines (shared encoder,
  matched I/O).
- `nca_wm/synthetic_levels.py` — synth level generation.
- `nca_wm/heldout_eval.py` — held-out evaluation harness.
- `nca_wm/scripts/` — launcher shell scripts and aggregators.
- `nca_wm/paper/` — LaTeX sources and figures.

## Anonymization notes

- All references to absolute paths and author identity have been
  scrubbed. Where launchers refer to `${REPO}`, set
  `export REPO="$(pwd)"` before running.
- The supplemental zip is built by `nca_wm/scripts/build_supplement.sh`
  in the (de-anonymized) source repo.

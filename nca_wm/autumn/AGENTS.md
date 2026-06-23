# AGENTS.md — working on Autumn NCA world models

Onboarding for an agent (or person) continuing this subproject. Read `README.md`
for *what it is* and the pipeline, `FINDINGS.md` for *results + architecture*, and
`WORKLOG.md` for the chronological detail. This file is *how to work here*.

## What this is, in one paragraph

We train single-game Neural Cellular Automaton (NCA) world models that predict
`f(state, action) → next_state` for [AutumnBench](../../../mara/MARA/domains/autumnbench)
programs. The AutumnBench interpreter is always the ground-truth oracle. The whole
point of the study: most environments have **hidden state** (a variable that drives
dynamics but isn't in the visible grid), and **each kind of hidden state needs a
different architectural lever**. See the taxonomy below.

## The mental model (this is the payoff — internalize it)

Diagnose an environment, then pick the lever. The diagnostic is the **noop-imperfect
signature** plus the **per-action breakdown** (`train.py` prints per-action exact /
changed-cell accuracy):

| symptom | cause | lever |
|---|---|---|
| `noop` exact ≈1, just imperfect overall | Markovian but hard / under-data'd | more **data/compute**; more **NCA depth** (`n_steps`) for long-range propagation |
| `noop` exact < ~0.95 on a *deterministic* game, error is motion-readable | hidden variable acting **every step** (e.g. a direction) | **1-frame history** (`--keep_prev` / `--history`) |
| `noop` perfect, error only under *specific actions* | hidden **counter/identity/mode** set intermittently, persists | **recurrent** hidden grid (`train_recurrent.py`, BPTT) |
| a hidden signal is set at one cell but acts globally (a mode button) | **spatial locality** — local conv can't broadcast it | **max pooling** (`--pool meanmax`, now the recurrent default) |
| object overlaps collapse to one color | discrete-grid **representation** | object channels (`objects.py`, multi-hot + BCE + `pos_weight`) |
| residual is an unpredictable draw (seeded PRNG) | **irreducible** randomness | read the **calibrated marginal**; don't argmax |

"Deterministic ≠ Markovian" — several games first labeled Markovian (gravity,
disease) turned out to be hidden-state once evaluated on harder data. Always
re-check on freshly-collected diverse data, not the easy in-distribution val split.

## Environment / running things

- **Always** prefix engine-touching commands with `PYTHONPATH=/home/jupyter-smearle/mara/MARA`
  (so the `mara-autumn-cpp` interpreter import resolves) and use the repo `.venv`:
  `PYTHONPATH=/home/jupyter-smearle/mara/MARA .venv/bin/python3 -m nca_wm.autumn.<tool>`.
  Training (`train.py` / `train_recurrent.py`) doesn't need the interpreter, only the
  `.npz` data, but the PYTHONPATH prefix is harmless.
- **Build the interpreter** if missing: `python3 -m pip install /home/jupyter-smearle/mara/MARA/Autumn.wasm/`.
- Programs live in `mara/MARA/.../Autumn.wasm/tests/*.sexp` — **read the `.sexp` first**;
  it tells you the hidden state directly (look for `(: x String|Int|Bool)`,
  `initnext ... (prev x)`, `% timestep`, `randomPositions`, `on (clicked button) ...`).

## Typical workflow to add / fix an environment

```bash
PP=/home/jupyter-smearle/mara/MARA
# 1. collect transitions (single-frame) or ordered episodes (recurrent)
PYTHONPATH=$PP .venv/bin/python3 -m nca_wm.autumn.collect --game GAME --profile agent \
    --rollouts 800 --rollout_len 100 --out nca_wm/autumn/data/GAME.npz            # single-frame
PYTHONPATH=$PP .venv/bin/python3 -m nca_wm.autumn.collect --game GAME --profile agent \
    --sequences 1200 --rollout_len 60 --out nca_wm/autumn/data/GAME_seq.npz        # recurrent
# add --keep_prev for history models
# 2. train
PYTHONPATH=$PP .venv/bin/python3 -m nca_wm.autumn.train           --data ...GAME.npz     --save_dir nca_wm/autumn/runs/GAME
PYTHONPATH=$PP .venv/bin/python3 -m nca_wm.autumn.train_recurrent --data ...GAME_seq.npz --save_dir nca_wm/autumn/runs/GAME_recurrent  # --pool meanmax default
# 3. visualize (engine vs WM, side by side)
PYTHONPATH=$PP .venv/bin/python3 -m nca_wm.autumn.render_gif --save_dir nca_wm/autumn/runs/GAME --demo generic
# 4. interactive viewer (auto-discovers every runs/*/config.json on startup)
PYTHONPATH=$PP .venv/bin/python3 -m nca_wm.autumn.serve_compare --runs_dir nca_wm/autumn/runs --port 8766
```

Rigorous comparison: when comparing two models, eval **both on the identical val
split** (same data file, same `np.random.default_rng(0)` permutation). Raw val
numbers across different data look misleading because harder datasets have harder
val splits.

## Gotchas that have bitten us (don't relearn these)

- **Engine protocol**: one transition = register an input method (`click(x,y)`,
  `left`, …) **then** `step()`. Only the last-registered input applies per frame.
- **seed=0 is degenerate**: `randomPositions` collapses to (0,0) (ants food always
  top-left). The viewer uses a fresh non-zero seed each reset; do the same.
- **Background color** is per-game (mario is white, not black) — use `get_background()`.
- **GPU sharing**: GPUs on this box are shared; check `nvidia-smi` first, use
  `--device auto` (freest GPU) or a specific free one. When local GPUs are full,
  **offload to 210** (`ssh 210`: free GPU + a `~/script-doctor/.venv` with torch;
  rsync the `*.py` + the `.npz` data — training needs no interpreter). **torch** is a
  **SLURM cluster** (`sbatch`) — use it for real fan-out (many seeds/games), not one-offs.
- **Background jobs**: don't launch a long job with a trailing `&` inside a
  `run_in_background` Bash call — the wrapper exits and the job detaches/orphans
  (you get a premature "completed" and no real completion signal). Run the long
  command *as* the background command, or `ssh host "cmd"` directly.
- **`train_recurrent` metric naming**: it prints `fire_recall`, which is
  Mario-specific; for non-purple games "best `fire_recall`" in the save line is
  actually the changed-cell accuracy (the selector falls back to it).
- **Don't pollute** `custom_games/` or commit `runs/` / `data/` (gitignored — large
  binaries; regenerate from the pipeline).

## Module map

- `collect.py` — `AutumnGame` wrapper + data collection. Profiles: `agent`
  (arrows+clicks), `generic`, `ca` (clustered seeding for GoL), and heuristics
  (`mario_heuristic`, `snake_heuristic`, `waterplug_heuristic`). `--sequences` for
  recurrent, `--keep_prev` for history.
- `model.py` — `AutumnNCA` (stateless `f(state,action)→next`, supports `history`,
  `global_pool`) and `RecurrentAutumnNCA` (hidden grid across env steps, `pool=
  none|mean|max|meanmax`). Representation-agnostic (logits in/out).
- `objects.py` — per-object-channel (multi-hot) representation for overlaps.
- `train.py` / `train_recurrent.py` — single-frame / BPTT training, per-action metrics.
- `infer.py` — load_run + step functions (`wm_step`, `recurrent_step`, object steps).
- `serve_compare.py` — the port-8766 viewer (auto-discovers `runs/`).
- `render_gif.py` — engine-vs-WM side-by-side GIFs.
- `cond_*.py`, `train_cond*.py`, `latent.py`, `diffusion.py`, `correlated_test.py` —
  research threads (conditional/latent/diffusion variants; latent/diffusion did NOT
  beat the calibrated marginal on ants — see FINDINGS).

## Open problems

See `FINDINGS.md` § Open problems. The live ones: mario over-firing (needs a
platformer navigation policy to get multi-coin data — deferred), waterplug liquid
flow (capacity-limited), and coherent joint sampling of irreducibly-random spawns.

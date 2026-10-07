# PuzzleScript game generator

A from-scratch transformer that writes whole PuzzleScript games. It is a prior for program-space UED, in which a
generator proposes games (mechanics and levels) for an exploring agent. It is trained on the public corpus
[`smearle/puzzlescript-gists`](https://huggingface.co/datasets/smearle/puzzlescript-gists) and scored with the
reference engine in `puzzlescript_nodejs/` (headless wrapper of the original JS engine in `PuzzleScript/`).

Experiment records (pre-registered protocol, logs, galleries, archives) live in the infogain-world-models repo under
`experiments/puzzlescript_generator_20261006/` and `results/puzzlescript_generator_20261006/`. `PROTOCOL.md` here is
a copy of the protocol.

## Pipeline

| step | script |
|---|---|
| engine verdict per game (compile, playable; with `--dynamics`, a seeded 200-step random rollout and a 20k-node BFS) | `ps_check.js`, run in parallel by `check_games.py` |
| raw corpus: dedup representatives that compile, split by title and mechanics components, line-pre-split BPE | `prepare_data.py` |
| canonical corpus: the engine's own parse, renamed and stripped of aesthetics, verified equivalent to the original, one document per distinct mechanics | `ps_extract.js`, `canonicalize.py`, `ps_equiv.js`, `prepare_canonical.py` |
| model and training (GPT with RoPE, RMSNorm and SwiGLU; length-bucketed batches; resumable) | `model.py`, `train_lm.py` |
| samples scored by the engine, novelty against train | `sample_eval.py` |
| canonical-space diversity: distinct mechanics, memorized share, dynamic and puzzle subsets | `canonical_eval.py` |
| re-score saved samples with the current checker | `recheck_eval.py` |
| engine-rendered contact sheets | `gallery.py`, `ps_render.js` |
| sampling-path audit (KV cache vs full forward) | `check_generation.py` |
| regression of a new engine version against the current one | `engine_regress.js`, `regress_engines.py` |
| level-first corpus (stage 3): one document per (mechanics, level), the level first and objects numbered by it, so a model can be prompted with a level | `level_first.py`, `prepare_level_first.py`, `test_level_first.py` |
| mechanics sampled for held-out levels, scored by the engine, by canonical mechanics and by behaviour under shared random action sequences | `level_eval.py`, `ps_probe.js` |
| the exact pod pipelines used (stages 1-3) | `run_pod.sh`, `run_canon.sh`, `run_level_first.sh`, `smoke.sh` |

The JS scripts load the engine through `ps_engine.js`. Inside the engine's VM realm, it does three things:
- sets `IDE = false`: the committed wrapper otherwise throws on editor hooks;
- makes message and sound output no-ops: otherwise an in-level message leaves the headless engine in text mode, and it
  never checks for a win again;
- provides `engine.clearLog()`: loop guards log warnings on some turns, and at 100 logged messages the engine throws
  "Too many errors/warnings". The dynamics loops, the BFS solver included, clear the log before every turn.

Pass the repository root as `ENGINE_DIR`.

```bash
cd puzzlescript_generator
python prepare_canonical.py --out data-canon --revision 0bf0f66fb3cb96ac5b2a68f803b0f6e04bf01013 --engine-dir .. --workers 24
python train_lm.py --data data-canon --out runs/m30 --n-layer 8 --n-head 8 --d-model 512 --dropout 0.1 --lr 1e-3 --epochs 26
python sample_eval.py --data data-canon --run runs/m30 --out eval --engine-dir ..
python canonical_eval.py --eval eval --data data-canon --canonical data-canon --engine-dir .. --out canonical_eval.json
```

## Canonical form

A canonical game is still valid PuzzleScript, built from the engine's own parse (`serializeParsedState` plus the
compiled levels).
- **Names:** objects become o1.., other legend names l1.., numbered by first use (rules, win conditions, layers).
  `background` and `player` keep their names.
- **Aesthetics:** one colour per object, no sprites. Sounds, messages, comments and metadata are dropped.
- **Dynamics-relevant structure:** flags that change dynamics or view are kept. Collision layers keep the engine's
  order, so object ids are unchanged.
- **Levels:** the compiler's initial cells, with glyphs assigned by first appearance.
- **Verification:** `ps_equiv.js` checks every game against its original: initial cells of all levels, and 60 seeded
  random actions on two levels with the same engine RNG seed. 22,998 of 23,193 games pass.

## Results (seed 0, 30M parameters)

| | stage 1: raw corpus | stage 2: canonical corpus |
|---|---|---|
| training data | 22,320 games, 45.1M tokens | 16,185 distinct mechanics, 11.3M tokens |
| T=0.8, 256 samples: playable / distinct mechanics | 60 / 12 | 34 / 30 |
| T=0.8: most common mechanics' share / copied from train | 75% / 92% | 15% / 18% |
| T=1.0, 512 samples: playable / distinct mechanics | 13 / 6 | 25 / 25 |
| new levels for 64 held-out games (4 each): playable | 5% | 36%, all levels new |
| distinct mechanics that are dynamic (T=0.8 / new levels) | 9 / 5 | 19 / 38 |
| distinct real puzzles: dynamic, not won by random play, BFS-solved in >= 5 moves (T=0.8 / new levels) | 6 / 1 | 3 / 7 |

- In stage 1, valid samples collapse onto the editor's Sokoban template, `[ > player | crate ] -> [ > player | > crate ]`.
  In the canonical corpus that template is one document (763 games).
- Canonical, mechanics-deduplicated training removes the collapse.
- Validity is not interest: real puzzles stay rare. Choosing samples worth exploring needs a teacher signal (engine
  dynamics, solvability, or an agent's learning progress).

## Stage 3: mechanics for a given level

Level-first documents put one level first, with its objects numbered by their order in the level, so an observation
channel means the same cells in every game written for that level. The corpus has 66,293 documents from 16,221 mechanics.

| 24 rich held-out levels, 64 samples each per temperature | playable | distinct behaviours |
|---|---:|---:|
| whole mechanics, T=0.8 / 1.0 | 0.2% / 0.3% | 3 / 5 |
| rules only (the level's own flags, objects, legend and layers given), T=0.6 / 0.8 | 9.0% / 6.9% | 121 / 96 |

- **Whole mechanics fail** on global consistency at the size of these games: objects in no layer, undefined names,
  malformed rules.
- **Rules only works:** writing the rules and win conditions for a given level and object inventory gives varied,
  mostly dynamic games, none copied from training.

## Engine notes

- The `PuzzleScript/` submodule moved from `4176d350` to upstream `dfdeabcd` (105 commits). Results above use
  `4176d350`.
- Regression on all 25,845 representatives (`regress_engines.py`):
  - dynamics are identical in 99.93% of the games both versions accept;
  - 79 newly accepted (repeated objects on a rule's right-hand side are no longer an error);
  - 23 newly rejected (comments inside rules are now an error).
- The parsed legend now flattens nested properties, so canonical mechanics keys differ between engine versions for
  about 20% of games. Never mix keys across versions.

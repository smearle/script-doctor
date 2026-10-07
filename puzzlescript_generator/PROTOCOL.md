# PuzzleScript game generator: pretraining on human games (2026-10-06)

Owner: Claude session 7cc6da6f. Results: `results/puzzlescript_generator_20261006/`.

## Question

UED for Double Take so far generates maps for fixed dynamics (Craftax terrain). The
target setting is a black-box simulator that turns a string into an MDP; the generator
must write programs and initial conditions in the simulator's own language. PuzzleScript
is the first instance: a game is a string (rules + levels), and the reference engine
either compiles it into a playable MDP or reports errors.

This campaign trains the generator's prior: a from-scratch transformer over whole
PuzzleScript source files, trained on the human corpus (user's stand-in for zero-data
bootstrapping). It answers: how often do its samples compile into playable games with
non-trivial dynamics, how novel are they, and can it write new initial conditions
(levels) for held-out rules? No DT or teacher objective is involved yet.

## Data

- `smearle/puzzlescript-gists` (public HF dataset), pinned revision
  `0bf0f66fb3cb96ac5b2a68f803b0f6e04bf01013` (38,661 gists; 35,842 rows after the PS+
  exclusion; 25,845 dedup representatives).
- Kept: representatives that the reference engine (script-doctor's headless wrapper of
  the original JS engine, `ps_check.js`; file hashes in `attempt-01/engine-manifest.sha256`)
  compiles with no error and that have >= 1 playable level. The dataset's Lark
  `parse_status` disagrees with the engine in both directions and is not used.
- Split 95/2.5/2.5 by connected components that link games sharing a normalized title or
  a mechanics fingerprint; components above 40 games always go to train. Template
  derivatives and an author's successive drafts therefore never straddle splits.
- Byte-level BPE, vocab 8192, pre-split only after newlines (tokens never span two lines),
  trained on train only. One game per sequence (`<|bos|> ... <|eos|>`), truncated at
  8192 tokens.

## Model and training

Decoder-only transformer: RoPE, RMSNorm, SwiGLU, tied embeddings, bf16, context 8192.
Length-bucketed batches of ~131k padded tokens (no packing across games), AdamW
(0.9, 0.95, wd 0.1), warmup 200 steps, cosine to 10%, grad clip 1. Four val evaluations
per epoch; `best.pt` keeps the lowest val loss. Configurations (seed 0 each):

| name | layers x width | dropout | cosine horizon | peak lr |
|---|---|---|---|---|
| m85-e8-d01 | 12 x 768 | 0.1 | 8 epochs | 6e-4 |
| m85-e4-d0 | 12 x 768 | 0.0 | 4 epochs | 6e-4 |
| m30-e8-d01 | 8 x 512 | 0.1 | 8 epochs | 1e-3 |

The lowest best-val-loss configuration is selected for evaluation.

## Readouts (`sample_eval.py`)

- Test loss (nats/token; bits/byte over games that fit the context).
- Arms: 512 unconditional samples at T=1.0, 256 at T=0.8; new LEVELS sections for 64
  held-out test games (4 samples each, T=1.0, prompt = the game up to its LEVELS
  header); the held-out human test games as the reference.
- Per text, with the reference engine: compiles; playable (>= 1 level); a 200-step
  seeded random rollout on the first level (state changes, distinct states, random win);
  a 20k-node BFS (solved, solution length).
- Novelty against train: exact copies; verbatim RULES sections; fraction of token
  32-grams found in train. The human test arm calibrates each.

Expectations (written before results): unconditional T=1.0 playable rate 40-70%, higher
at T=0.8; level-conditional playable rate >= 80%; a substantial share of copied RULES
blocks, because the corpus is template-heavy.

## Operations

- RunPod `se-psgen-20261006-r1` (1x H100 SXM, $3.49/h), compute-board job
  `puzzlescript-generator-20261006-r1`, pipeline `run_pod.sh` (stage markers; training
  resumes from `last.pt`).
- Follow-up: systemd user unit `igwm-psgen-20261006-watch` (`controller.py watch`) keeps
  the lease and writes one event file (complete / failed / stalled / transport); the
  owning session waits on that directory.
- Archive: 209's disk is below the 20 GiB operating floor, so only compact results
  (reports, samples, logs, split lists, tokenizer) are mirrored and hash-verified to
  209. Checkpoints and token arrays stay on the pod's persistent volume until they are
  archived to Torch or another store chosen by the user; only then is the pod deleted.

## Amendments

- 2026-10-06, before any training: the first prep used BPE with no pre-split. On a
  250-game sample that variant trained 14x slower (23.3 s vs 1.7 s) at equal compression
  (2.42 vs 2.38 bytes/token), and the full-corpus fit was still running after 7 min, so
  prep was restarted with `--pretok lines`.

## Stage 2: canonical, deduplicated corpus (written 2026-10-07, before its prep or training)

User request (2026-10-06), after stage 1's playable samples collapsed onto the editor's
Sokoban template: "We need to preprocess the dataset by deduplicating functionally
equivalent mechanics/levels. We could even collapse all the aesthetics/semantics (no
sprites, no object names)."

**Canonical form** (`canonicalize.py`). It works on the reference engine's own parse (`ps_extract.js`): objects,
collision layers, legend, raw rule lines, win conditions, flags, and each level's compiled initial cells.
- Kept: the flags that change dynamics, usable inputs or view.
- Renamed: objects become o1.., other legend names l1.., numbered by first use in rules, then win conditions, then
  layers; `background` and `player` keep their names.
- Simplified: one fixed colour per object and no sprites.
- Dropped: comments, sounds, sfx, message commands, message levels, title, author and other metadata.
- Unchanged: layer order and object-id order. Rule, win-condition and level order are kept.
- Levels: the compiled initial cells, with glyphs assigned by first appearance.

**Equivalence check** (`ps_equiv.js`, per game; a game is kept only if it passes):
- the canonical game compiles;
- every playable level has the same initial cells;
- 60 seeded random actions on each of the first two levels give the same cells and win flag after every step.

Both games use the same engine RNG seed, and message and sound output are no-ops in both.

**Deduplication.**
- One document per distinct canonical mechanics: the text without levels, hashed.
- Its levels are the distinct compiled levels of all member games.
- At most 32 levels and 24k characters per document; a seeded subset keeps the levels' order.
- Every document is compiled again.

**Split.** Documents are linked when member games share a normalized title or a dataset mechanics hash. Fractions and
the 40-document cap are as in stage 1.

**Model.** Tokenizer (BPE 8192, line pre-split) and model recipe from stage 1's selected configuration (8x512, dropout
0.1, lr 1e-3).
- Two cosine horizons: 8 epochs, and the horizon that gives stage 1's 2,808 optimizer steps (skipped if under 10
  epochs).
- The run with the lower best val loss is evaluated.

**Readouts.**
- `sample_eval.py` arms as in stage 1.
- `canonical_eval.py` for stage 2's model and for stage 1's re-checked samples, so both are scored in canonical space:
  - distinct mechanics among playable samples;
  - the most common mechanics' share;
  - memorized share (mechanics equal to a training game's);
  - distinct unseen mechanics;
  - for generated levels, the share copied from the source game.

**Bug fix carried into both stages.** An in-level message left the headless engine in text mode, so it never
checked for a win again. `ps_check.js` now makes message output a no-op. Stage 1's arms are re-checked
(`recheck_eval.py`), and their rollout and BFS columns replace the old ones.

**Expectations** (written before results):
- the Sokoban template becomes one document;
- the unconditional playable rate rises, since objects, sprites and names leave little to get inconsistent;
- the most common mechanics' share among playable samples falls well below stage 1's;
- most playable samples still reuse a training mechanics.

**Operations.** Same pod and board job. Pipeline `run_canon.sh` (outputs in `/workspace/psgen/canon`), follow-up
`controller.py watch canon`. Results go to `results/puzzlescript_generator_20261006/canonical-01/`.

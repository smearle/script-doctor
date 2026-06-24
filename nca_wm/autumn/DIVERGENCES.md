# Divergence audit — where the WMs visibly disagree with the engine

The per-game accuracy numbers are **teacher-forced, on the agent/random action
distribution**. A human poking at the viewer does something different: sustained
inputs, mode-switches, click-spam, and **autoregressive** free-running (the WM
feeds its own predictions). This audit exercises every best model under adversarial
action policies (noop/autonomous, repeated-action, oscillating, click-spam, random)
autoregressively for 40 steps and records the first divergence. Randomness games
(ants, particle_2/particles, colour_lines, tetris, dino) are excluded per request.

Tool: `/tmp/divfind.py` (AR rollout vs engine under policies), `/tmp/divtraj.py`
(per-step diff trajectory), `/tmp/tf_vs_ar.py` (teacher-forced vs AR — isolates
single-step error from drift).

## Headline: the divergences are real and quick — but mostly NOT exposure-bias drift

Teacher-forced and autoregressive first-divergence are at the **same step** for the
fast-diverging games — so these are **genuine single-step errors**, not AR drift.
That matters: scheduled-sampling/exposure-bias fixes alone won't help; the model is
wrong even when fed the engine's real state.

## Robust (no divergence in 40 steps, all policies)
gameOfLife, paint, egg, lights. These are genuinely solid.

## Divergence inventory (first divergence; AR, 40 steps)

| game | policy | step | cells (max) | class |
|---|---|---|---|---|
| carrace | noop | 0 | 8 (const) | **phase lock-in** |
| balls | noop | 0 | 2 (const) | **velocity lock-in** |
| space_invaders | noop | 2 | 1→3 grow | drift/accumulation |
| gravity_2 | spamclick | 2 | 3→**116** | **click-spawn catastrophe** |
| logic_gates | spamclick | 3 | 8 | click/compute |
| mario | random | 5 | 1 (max 30) | click (bullet) + spawn |
| waterplug | spamclick | 6 | 24 | **mode-latch (intermittent)** |
| buoyancy | spamclick | 7 | 1 (max 23) | click-spawn |
| pacman | repeat-L | 11 | 4 (max 5) | **mid-episode error** |
| bbq | random | 12 | 1 (max 2) | counter |
| grow | repeat-L | 13 | 3 (max 23) | sustained-action edge |
| gravity_3 | repeat-L | 1 | 1 | early small |
| arc_slack | random | 13 | 1 | minor |
| gravity | spamclick | 14 | 30 (max 49) | click-spawn |
| charge | spamclick | 16 | 1 (max 4) | click |
| disease | random | 21 | 2 (max 4) | mid-episode |
| wind | noop | 20 | 2 | mid-episode |
| sand | repeat-L | 0 | 1 | sustained-action edge |

## The four divergence classes, with fixes

### 1. Recurrent phase / velocity lock-in (carrace, balls, space_invaders)
The hidden state (frameCount phase, per-ball velocity, march phase) is **unobservable
from a single frame**, so at episode start (h=None) the model *must* guess. It commits
to one estimate and **never corrects it** — carrace stays a constant 8 cells off
(car one frame out of phase), balls a constant 1–2 (one ball's velocity wrong). This
is the most insidious because the metrics (teacher-forced, mid-episode) never see it.
**Fixes:**
- **Warm-up the hidden state**: feed the model 2–3 real engine frames before trusting
  its output (the viewer can hold predictions for N frames). Cheap, immediate.
- **Train phase *correction***, not just phase *tracking*: currently BPTT starts from
  a known episode start; train with **random-offset starts** (begin a sequence
  mid-episode with h=0) so the model learns to *infer and update* the phase from
  ongoing motion rather than lock in at t=0. This is the real fix.
- Information-theoretic floor: the very first transition after a cold start is
  genuinely unpredictable for a hidden-velocity/phase var — accept ≤1-step error.

### 2. Click-spawn catastrophe (gravity_2 0→116, gravity, buoyancy, logic_gates, mario)
Spam-clicking spawns objects (blobs, water, bullets, logic toggles) whose appearance/
dynamics the model mis-predicts, and the error **accumulates** as more spawn. The
agent profile rarely spam-clicks, so these states are **out of the training
distribution** — a pure coverage gap. gravity_2 is worst (blobColor counter + 2×2
multi-cell blobs + spawn).
**Fixes (get fancy with collection):**
- **Adversarial click collection**: a policy that spam-clicks varied positions
  (empty cells, occupied cells, buttons, edges) at high rate — directly covers the
  spawn dynamics the agent profile misses.
- **DAgger (the principled version)**: roll the model out autoregressively under
  spam-click, diff against the engine each step, and add the **(divergent-state,
  engine-correct-next)** transitions to the training set; retrain; repeat. This
  targets the model's *own* failure states exactly — the literal "analyze erroneous
  transitions → generate relevant data → retrain" loop, but with the relevant states
  discovered by the model's rollout rather than guessed.

### 3. Genuine mid-episode errors (pacman step 11, wind step 20, disease step 21, bbq)
Perfect for many steps, then a specific transition is wrong (a ghost interaction at a
particular mod-3 phase; a wind change; a disease-spread event). Real model mistakes on
rare-but-deterministic events.
**Fix:** DAgger again — the model's own rollout surfaces these exact states; add the
engine-correct transitions and retrain. Also worth **rendering these as GIFs** (the
viewer's disagreement panel already localizes them) to confirm the mechanic.

### 4. Sustained-action edge cases (sand+left @0, grow+left @13)
Repeated single arrows drive objects into edges/walls — boundary configurations rare
in mixed-action training.
**Fix:** include sustained/repeated-action episodes in collection (a "hold one key"
policy), plus DAgger.

## Recommended program (in priority order)
1. **DAgger harness** — generic: for a model, roll out under each adversarial policy,
   collect the diverged states + engine-correct next-states, append to train data,
   retrain, repeat until divergence-free under the policies. Directly fixes classes 2–4.
2. **Random-offset / warm-up recurrent training** — fixes class 1 (the lock-in), which
   DAgger alone won't (the first-frame error is information-theoretic; the *persistence*
   is the bug).
3. **Adversarial collection profiles** (`spamclick`, `hold-key`) added to `collect.py`.
4. **Viewer**: a few-frame warm-up before scoring, and a "first-divergence" readout so
   divergences are visible/measurable, not just felt.

## Demonstration: adversarial collection partially fixes gravity (honest result)
Added a `spamclick` collection profile (heavy varied clicking), collected 35k gravity
transitions, augmented gravity's history data (79k→114k), retrained. Re-measured the
spam-click divergence (3 seeds): the **small coverage-gap divergence was fixed**
(seed 11: diverge@7 → none) with **no accuracy regression** (0.964), but the **large
divergences were unchanged** (seeds 3/7: 49–93 cells, still diverge @14/29).
**Why:** the big divergences are NOT a coverage gap — by step 14–29 of spam-clicking,
dozens of blobs have spawned and the model's tiny per-blob errors **compound
autoregressively** into 49–93 wrong cells. So:
- **Coverage-gap divergences** (a state-type the agent profile never visits) → one
  round of adversarial collection fixes them.
- **AR-accumulation catastrophes** (slightly-wrong-per-object × many objects) → need
  **iterated DAgger** (re-roll the improved model, collect its *new* failure states,
  repeat) and/or a near-perfect per-step model so errors don't compound. One round
  isn't enough; this is the genuinely hard residual.

## The honest summary
The aggregate metrics were measuring the wrong thing for
interactive use. Under a human's targeted, autoregressive poking, **most models
diverge within ~5–20 steps**, via four well-defined mechanisms — all addressable with
divergence-targeted (DAgger) collection + a recurrent cold-start fix, none requiring
the open compositional architecture.

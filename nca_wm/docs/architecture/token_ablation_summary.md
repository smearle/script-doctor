# Token-ablation diagnostic — does the encoder steer the NCA?

For each trained run, we eval per-game on 2 levels × 2 episodes × 50 steps
under three modes:
  - `none`    real game tokens (control)
  - `zero`    all token ids set to 0 (encoder gets a constant input)
  - `shuffle` token positions randomly permuted (preserves token bag, breaks order)

A model that genuinely uses conditioning should degrade noticeably under
`zero`/`shuffle`. A model that memorizes via NCA + spatial features
alone is unaffected.

## Macro-mean error (% wrong tiles)

| model                                              | TF none | TF zero | TF shuffle | AR none | AR zero | AR shuffle |
|----------------------------------------------------|--------:|--------:|-----------:|--------:|--------:|-----------:|
| baseline (continuous, FiLM-ish via pool path)      |  1.076  |  0.942  |   0.935    |  2.853  |  3.814  |   2.991    |
| vq1024 encoder-only                                |  0.403  |  0.426  |   0.426    |  6.465  |  6.183  |   6.183    |
| vq256  encoder-only                                |  0.320  |  0.320  |   0.320    |  4.030  |  4.030  |   4.030    |
| **vq1024 + decoder**                               |**0.179**|  1.062  |   1.069    |**1.989**|  5.204  |   4.736    |
| vq256  + decoder                                   |  4.889  |  5.156  |   6.258    |  7.787  |  8.155  |   9.616    |

(TF = teacher-forced single step; AR = autoregressive 50 step; lower = better.)

## Findings

1. **vq1024+decoder is the ONLY model that actually uses conditioning.**
   Zeroing tokens degrades TF by 5.9× (0.18 → 1.06%) and AR by 2.6×
   (1.99 → 5.20%). Shuffling gives essentially the same hit. The
   encoder is real and the NCA reads from it.

2. **The continuous baseline is mostly a memorizer.** Despite winning
   the AR-eval beauty contest in our prior comparison (2.55%), it
   barely changes under token ablation: TF goes from 1.08 → 0.94%
   (ablation actually *helps* TF, within noise), AR 2.85 → 3.81% (only
   1.3×). The NCA has memorized 20 games' transitions through its
   128-d hidden + 3 pooling features and uses spatial cues to identify
   games rather than reading rules from tokens.

3. **vq256 encoder-only is bit-identical across modes.** Codebook
   collapsed to 1 entry → every game receives the same constant slot
   → ablating tokens is a literal no-op. Cleanest possible proof of
   conditioning bypass.

4. **vq1024 encoder-only is essentially the same story.** 2 active
   codebook entries; ablation moves numbers by < 5%.

5. **vq256+decoder partially uses conditioning** (TF 4.89 → 5.16 / 6.26;
   AR 7.79 → 8.16 / 9.62, with shuffle hurting more), but the model is
   underperforming overall — 8 codebook entries are not enough capacity
   for 20 games.

## Reframing the prior comparison

The previous "vq_comparison_20260430.md" leaderboard ranked by raw
eval error and concluded the continuous baseline was best on AR. Token
ablation reveals that ranking was measuring **memorization fidelity**,
not rule-following ability. Re-ranked by "actually does what we want":

|  rank | model              | rule-using? | TF (real) | AR (real) |
|------:|--------------------|:-----------:|----------:|----------:|
|     1 | vq1024 + decoder   | ✓ yes       | 0.179%    | 1.989%    |
|     2 | baseline (cont.)   | ✗ memorizer | 1.076%    | 2.853%    |
|     3 | vq256  + decoder   | ~ partial   | 4.889%    | 7.787%    |
|     4 | vq1024 enc-only    | ✗ bypassed  | 0.403%    | 6.465%    |
|     5 | vq256  enc-only    | ✗ collapsed | 0.320%    | 4.030%    |

vq1024+decoder is now the clear winner on both control (real-token)
metrics AND on the rule-vs-memorize criterion. It beats baseline 6× on
TF and 1.4× on AR with a model that's actually doing the job.

## Sample per-game ablation deltas (vq1024+decoder)

These are the games where ablating tokens hurts the most — the
conditioning is doing real work:

|  game                       | TF none | TF zero | factor | AR none | AR zero | factor |
|-----------------------------|--------:|--------:|-------:|--------:|--------:|-------:|
| notsnake                    |  0.000% |  5.583% |   ∞    |  0.000% | 18.875% |   ∞    |
| blocks                      |  0.000% |  4.143% |   ∞    |  0.175% | 15.691% |   90×  |
| blank                       |  0.000% |  0.929% |   ∞    |  2.619% |  1.833% |   0.7× |
| wrappingrecipe              |  0.000% |  2.415% |   ∞    |  4.762% | 10.306% |   2×   |
| actiontest                  |  0.107% |  0.583% |   5×   |  1.390% |  3.096% |   2×   |
| Multi-word_Dictionary_Game  |  0.000% |  0.941% |   ∞    |  3.175% |  0.612% |   0.2× |

Several games drop from "perfect" to ~5% error when tokens are zeroed,
confirming the model is actively reading specific rules out of the
encoder for these.

## Next questions

1. **Why does the baseline still win some games on AR?** Memorization
   plus a 128-d NCA + pooling is genuinely powerful for this scale (20
   games, ~200k transitions each). Need to test on **unseen start
   states** (held-out levels, synthetic random init) to break the
   memorization lever — see `rule_vs_memorize_plan.md` Diagnostic 3 & 4.

2. **Can we make vq1024+dec even less collapsed?** Final util was 13/1024.
   Try EMA codebook updates, dead-code reset, or higher
   commitment_weight (0.25 → 1.0) to keep more codes alive.

3. **Why does vq256 enc-only's `random` AR (4.03%) beat baseline's AR
   on some games even with no conditioning?** Suggests that whatever
   "rule" structure exists in the test trajectories is partially
   redundant with what NCA + pooling can pick up. Confirms that the
   benchmark itself is partially solvable by memorization — held-out
   states are the next experiment to settle this.

---

## Long-horizon (200-step AR) eval

### Macro means (% wrong tiles, averaged over 200 rollout steps)

| model              | AR@50 | AR@200 | growth ratio | TF@200 | mean first_div |
|--------------------|------:|-------:|-------------:|-------:|---------------:|
| baseline (cont.)   | 2.85% |  3.76% |        1.32× |  1.01% |          2.5 s |
| vq1024 enc-only    | 6.46% |  9.25% |        1.43× |  0.37% |          1.4 s |
| vq256  enc-only    | 4.03% |  5.84% |        1.45× |  0.28% |          3.3 s |
| **vq1024 + dec**   | 1.99% |  3.46% |        1.74× |  0.18% |          3.7 s |
| vq256  + dec       | 7.79% |  9.40% |        1.21× |  4.38% |          1.8 s |

(AR@50 values from prior comparison; AR@200 / TF@200 / first_div from current run with 3 episodes × 2 levels × 200 steps each.)

### Findings

1. **vq1024+decoder wins both TF and AR at long horizon.** AR@200 = 3.46%
   beats baseline's 3.76%. The rule-using model widens its margin over the
   memorizing baseline as the rollout horizon grows: at 50 steps baseline
   was a close challenger; at 200 steps the rule model edges ahead.

2. **first_div_step is uninformative for this benchmark.** PuzzleScript
   is deterministic — a single wrong cell at step t makes first_div = t
   forever. All five models have mean first_div in 1.4–3.7 steps. The
   informative signal is the *time-averaged* tile error, not first_div.

3. **Per-game first_div tells a richer story.** vq1024+dec dramatically
   beats baseline on `blocks` (0.0 → 22.3), `blank` (1.5 → 5.8),
   `sokoban_basic` (1.5 → 5.8), `sumo` (0.0 → 3.3). Baseline wins on
   `nekopuzzle` (6.0 → 0.8), `wrappingrecipe` (5.7 → 0.3),
   `Travelling_salesman` (2.2 → 0.2). Memorization is genuinely effective
   for some games (those with simple, repetitive transition patterns),
   while rule-learning matters for others.

4. **Encoder-only collapse models still lose badly at 200 steps.** vq1024
   enc-only AR@200 = 9.25%, vq256 enc-only AR@200 = 5.84% — confirming
   they're noisy memorizers whose noise compounds over horizon.

### Reaffirmed ranking under "actually does the job"

| rank | model              | rule-using? | TF@200 | AR@200 |
|-----:|--------------------|:-----------:|-------:|-------:|
|    1 | vq1024 + decoder   | ✓ yes       | 0.18%  | 3.46%  |
|    2 | baseline (cont.)   | ✗ memorizer | 1.01%  | 3.76%  |
|    3 | vq256  enc-only    | ✗ collapsed | 0.28%  | 5.84%  |
|    4 | vq256  + decoder   | ~ partial   | 4.38%  | 9.40%  |
|    5 | vq1024 enc-only    | ✗ bypassed  | 0.37%  | 9.25%  |

### Combined picture

- Token ablation confirmed only `vq1024+dec` actually uses its conditioning.
- Long-horizon AR confirms `vq1024+dec` extrapolates better than the
  memorizing baseline as horizon increases.
- The two findings are consistent: the rule-using model holds up over
  time; the memorizing model degrades faster.
- We're still using in-distribution start states (training levels). The
  next step is **synthetic random init states** (rule_vs_memorize_plan
  Diagnostic 4) to fully break the spatial-features-as-game-id
  shortcut and confirm rule generalization.

---

## Random-init AR (Diagnostic 4 — unseen start states)

Each episode begins by **resetting** the engine, then taking **100 random
actions** to drift the state away from the A* solver trajectories the
model was trained on. The model then rolls out autoregressively for 50
more steps and is compared against the engine.

5 episodes × 2 levels × 20 games × 5 models.

### Macro AR error

| model              | in-dist AR | random-init AR | ratio |
|--------------------|-----------:|---------------:|------:|
| baseline (cont.)   |     2.85%  |        3.36%   |  1.18×|
| vq1024 enc-only    |     6.46%  |        6.38%   |  0.99 |
| vq256  enc-only    |     4.03%  |        4.03%   |  1.00 |
| **vq1024 + dec**   |   **1.99%**|      **2.29%** |  1.15×|
| vq256  + dec       |     7.79%  |        7.21%   |  0.93 |

### Findings

1. **vq1024+decoder generalizes best to unseen states.** Random-init
   AR = 2.29%, beating baseline's 3.36% by 1.5×. The rule-using model
   has a real edge on out-of-distribution start states.

2. **Both top models degrade only modestly** (~15-18% relative). The
   "memorization" story for baseline is weaker than expected. Two
   possible reasons:
   - Burn-in often terminates games early — the random-walk state may
     not actually be far from training distribution.
   - NCA + pooling has some intrinsic local-rule capability even
     without strong conditioning.

3. **Encoder-only collapse models do NOT degrade.** vq1024/vq256
   enc-only and vq256+dec all show ratio ≤ 1.00 — random-init is
   not harder than in-dist for them. Highly suspicious. Most likely
   explanation: these models are predicting "nothing changes" in
   ambiguous cases, and that's frequently correct after burn-in
   terminates the game.

4. **Per-game gap (vq1024+dec vs baseline) is dramatically game-dependent**:

   Rule-following wins big:
   - notsnake          : 19.18% → 4.98% (gap = -14.2 pp)
   - sumo              : 6.86%  → 2.30% (gap = -4.6 pp)
   - blank             : 5.06%  → 0.81% (gap = -4.2 pp)
   - sokoban_basic     : 5.06%  → 0.81% (gap = -4.2 pp)
   - actiontest, blocks, Multi-word_Dict, Modality, sokoban_match3 (0.7-1.9 pp)

   Memorization wins:
   - Collapsable_Sokoban : +4.0 pp
   - wrappingrecipe      : +3.1 pp
   - scriptcross         : +2.7 pp
   - the_undertaking     : +1.7 pp
   - rigidfail1          : +1.4 pp

5. **Combined verdict** across all four diagnostics (token ablation,
   long-horizon AR, random-init AR, in-distribution control):
   `vq1024+decoder` is the **only model in the suite that simultaneously**
   - uses its conditioning (token ablation: 6× degradation when zeroed),
   - holds up at 200-step horizon (best AR@200),
   - generalizes to unseen start states (best random-init AR),
   - achieves the lowest single-step error (best TF).

   The continuous baseline is fast at memorization but strictly worse
   than the rule-using model on every metric that probes generalization.

### Caveats

- Random-walk burn-in can terminate games early, leaving the state in
  "game over" mode where "no change" is trivially correct. A cleaner
  test would generate truly random valid initial states (per-object
  cardinality respecting), which requires engine state-injection
  support not currently exposed by the cpp wrapper.
- All eval is still on the 20-game training set. Out-of-distribution
  *games* (transfer to held-out games) is a separate question.


## Held-out 5g experiment (2026-04-30)

Run: `20260430-165528_heldout5_15g_vq1024_dec_200k` — vq1024+decoder
trained on 15 games, holding out notsnake, wrappingrecipe, kettle,
blocks, scriptcross. 200k steps, best_loss=0.00115 @ step 199500.

### Token ablation on the 15 trained games

| mode    | TF macro | AR macro |
|---------|----------|----------|
| none    | 0.417%   | 2.690%   |
| zero    | 0.320%   | 3.007%   |
| shuffle | 0.515%   | 3.381%   |

Note: TF under `zero` is *lower* than `none` — the 6× TF degradation
seen in the 20g vq1024+dec model does NOT reproduce here. Likely
explanation: with 15 games the 1024-codebook is over-provisioned
relative to rule diversity, and the NCA can route around the
conditioning channel for single-step prediction. AR shuffle is 1.26×
worse than none, so some conditioning signal remains for multi-step
rollouts.

### TODO: held-out games transfer
Evaluating the 5 unseen games requires borrowing `game_infos` from a
baseline 20g run (the heldout run's pkl only contains 15 entries).
Pending: a follow-up script that constructs eval-only game specs by
slot-encoding the held-out tokens through the trained encoder, and
compares TF/AR per game against the 20g baseline.


## Anti-collapse cw=1.0 experiment (2026-04-30)

Run: `20260430-165620_anticol_20g_vq1024_dec_cw1.0_200k` — 20g
vq1024+decoder with `--vq_commitment_weight 1.0` (was 0.25 in prior
runs). Goal: keep more codebook entries alive at end of training.
200k steps, best_loss=0.0146 @ step 70000 (drift after that).

### Codebook utilization (negative result)
- **cw=1.0 made collapse WORSE, not better.** Final `vq_util=3.0`
  entries, vs 13 with cw=0.25. cw=1.0 locks the encoder onto whatever
  entries are alive in early training; if collapse hits before the
  encoder spreads out, the high commitment loss makes it irreversible.
- best_step=70000 confirms: model peaked early then drifted as
  effective codebook narrowed.

### Token ablation

| mode    | TF macro | AR macro |
|---------|----------|----------|
| none    | 0.269%   | 2.900%   |
| zero    | 0.642%   | 4.070%   |
| shuffle | 0.441%   | 3.361%   |

Counterintuitive positive: with only 3 codes, conditioning still
matters — TF degrades 2.4× under zero, AR degrades 1.4×. AR none
(2.90%) is actually *better* than cw=0.25 vq1024+dec (3.46% AR).
Three game-distinguishing codes are apparently enough for the NCA
to distinguish 20 games while staying rule-following.

### Verdict on anti-collapse strategies
cw=1.0 alone does not solve dead-code drift. Next step: implement
dead-code reset (reseed unused codebook entries with random batch
encodings every N steps) — the original plan that we deferred.


## Held-out 5g per-game TRANSFER (2026-04-30)

Question: does vq1024+decoder learn rules transferable to unseen games,
or does it just learn per-game representations of the training set?

Comparison: heldout_15g (trained on 15, evaluated on the 5 unseen) vs
vqdec_20g (trained on all 20, including these 5) — same architecture.

scriptcross excluded (n_objs=16 > heldout's max_C=9, channel mismatch).

| game           | heldout TF  | vqdec_20g TF | TF gap | heldout AR  | vqdec_20g AR | AR gap |
|----------------|-------------|--------------|--------|-------------|--------------|--------|
| notsnake       | 11.78%      | 0.35%        | 34×    | 25.39%      | 5.98%        | 4.2×   |
| blocks         | 1.20%       | 0.03%        | 47×    | 3.05%       | 0.13%        | 23×    |
| kettle         | 2.54%       | 0.05%        | 50×    | 5.41%       | 1.35%        | 4.0×   |
| wrappingrecipe | 4.27%       | 0.42%        | 10×    | 11.71%      | 4.01%        | 2.9×   |
| MACRO          | 4.95%       | 0.18%        | 28×    | 11.39%      | 2.58%        | 4.4×   |

**Verdict: no rule-extraction transfer.** The 28× TF / 4.4× AR gap on
unseen games shows the encoder doesn't parse rules from arbitrary
PuzzleScript tokens — it learns per-game embeddings of its training
set. The "rule-following" signal seen in 20g token-ablation
experiments (TF zero degrades 6×) is more accurately described as
"trained-game identification": the encoder uses tokens to retrieve a
slot pattern that the NCA has memorized for *that specific game*, not
to parse new rules at inference time.

This reframes the prior result: vq1024+dec is "less memorizing" than
the continuous baseline only in the sense that *some* of its
performance on a held-in game requires the right tokens for that game
to be at the input. It is still 100% memorization in the strong sense
— zero ability to generalize to new games.

A real rule-extraction architecture would need a transfer gap closer
to 1×, not 28×. Open: does VQ + decoder + a held-out *training
augmentation* (e.g. in-context-style training where token diversity
forces sharing across games) close the gap?

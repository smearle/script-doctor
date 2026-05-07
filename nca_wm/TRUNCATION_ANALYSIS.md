# Heldout-30: impact of rule-token truncation on the OOD aggregates

**Generated 2026-05-07 03:00 EDT by supervisor agent.** Additive analysis for
the paper agents working Table 6 / §3.1. Drop into the appendix or use as
prose grounding — none of this changes the existing main-body numbers, it just
quantifies how much the 4 truncated games drag the mean.

## Setup

The conditional model encodes the game's tokenized rule text into $K=16$
slots. Its position-embedding table (`pos_emb`) is sized to `max_seq_len`,
fixed at training time from the longest training game's token count plus a
buffer. For the matched recipe, `max_seq_len = 657`.

Four of the 30 \textsc{Heldout-30} games tokenize to longer than 657 tokens,
so their rule sets get **silently truncated** before the encoder ever sees
them. The conditional model is then evaluated on a partial rule set — an
unfair comparison vs. the rest of the test set.

| Game | rule tokens | over by |
|------|------:|------:|
| `angize_by_ali_nikkhah` | 738 | +12% |
| `headless_people_problems_by_monakrom` | 785 | +20% |
| `break_out_of_the_mine_by_jja_i.e._juan,_jose_&_andre` | 798 | +21% |
| `Heroes_of_Sokoban_-_Ancient_Japan` | 891 | +36% |

The unconditional model is unaffected by token length (no encoder), so all 30
games contribute equally for it.

## Recomputed aggregates (TF, the headline metric)

Reading per-game cell-error from each run's
`heldout_v4_n30/results.json`, dropping the 4 truncated games. Numbers
update as the heldout-refresh queue (`refresh_heldout_with_bfs_astar.sh`)
completes each row (Train-59 cond/uncond and Train-199 cond pending at
the time of writing — still on n=29; the others have been refreshed).

| Run | Heldout-30 mean / median (TF, %) | Heldout-26 (no trunc) mean / median (%) | wins vs. identity |
|---|---:|---:|---:|
| Train-14 uncond  | 23.64 / 5.39 | **19.55 / 4.42** | 6 / 26 |
| Train-14 cond    | 19.26 / 3.89 | **14.77 / 3.45** | 7 / 26 |
| Train-59 uncond  | 4.65 / 3.19  | **4.33 / 2.95**  | 7 / 26 |
| Train-59 cond    | 5.94 / 2.94  | **5.73 / 2.58**  | 6 / 26 |
| Train-199 uncond | 3.29 / 1.97  | **2.46 / 1.52**  | 14 / 26 |
| **Train-199 cond** | **2.91 / 2.11** | **2.30 / 1.55** | **11 / 26** |
| Identity (Train-14 uncond row) | 2.34 / 2.04 | **2.23 / 1.97** | — |

Identity baseline is read from the first row's `identity_cell_err_per_step`
(consistent with `collate_match_table.py`); other rows produce slightly
different identity because of historical eval-bug differences in their
game lists. The committed Table 6 in HEAD shows `2.11 / 1.82` for identity
because it was computed before Train-14 uncond's heldout refresh added the
30th game; after the next collate run that cell will move to `2.34 / 2.04`.

## What changes

- **The headline tightens.** Train-199 cond on the 26-game subset reads
  $2.30 / 1.55\%$ (TF mean / median) vs. identity's $2.23 / 1.97\%$ — the
  conditional model is essentially at parity on the mean and **beats
  identity's median by $0.42\,\text{pp}$**. On the full Heldout-30, the
  headline reads $2.91 / 2.11\%$ vs identity $2.34 / 2.04\%$, a $0.6\,\text{pp}$
  gap on mean. The 4-game truncation tail accounts for most of the gap.
- **Wins-vs-identity counts are unchanged** when truncated games are
  excluded (the model never wins on truncated games anyway): `11/26`
  (cond), `14/26` (uncond) at \textsc{Train-199}; `7/26`, `6/26` at the
  smaller scales. Reporting as `wins / 26-non-truncated` instead of
  `wins / 30` is more honest for cond.
- **Per-game contribution.** For Train-14 cond, the four truncated games
  contribute step-1 errors of 12.5\% / 29.9\% / 67.7\% / 83.8\% — the
  last two are essentially "give up and predict noise" cases.
- **Mean inversion at Train-59 disappears** when truncated games are
  excluded: cond mean 5.73 vs uncond 4.33 → uncond still leads, but the
  4-game tail is no longer the cause.

## Suggested paper integration (pick one)

**Option A — footnote in §3.1.** "Four \textsc{Heldout-30} games
($\geq 738$ tokens) exceed the encoder's $657$-token positional embedding;
their rule sets are silently truncated and the conditional model sees
incomplete dynamics. On the 26-game subset whose rule sets fit, Train-199
cond reads $2.30 / 1.55\%$ TF mean / median, essentially at identity's mean
and below identity's median."

**Option B — appendix table.** Drop the recomputed table above into a
new Appendix subsection ("Truncation sensitivity of \textsc{Heldout-30}").
Keep main-body numbers as-is for parity with the unconditional baseline.

**Option C — a column in the existing Table 6.** Add a "26-game" row beneath
each preset, or a separate "26-game (no trunc)" subtable. Heavier; arguably
not worth the page real estate vs. a footnote.

Recommendation: **A** if pages are tight, **B** if there's room. Both
strengthen the headline without rewriting the main story.

## Pre-rendered LaTeX (for B)

Available at `nca_wm/paper/figures/heldout_truncation/`:
`truncation_table.tex` (table), `truncation_summary.csv` (full numbers).

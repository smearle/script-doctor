# Supervisor Log (Opus 4.7)

Started 2026-05-07 02:34 EDT. Goal: monitor running paper experiments, squeeze
small side-results, keep an audit trail so the user can rewind if anything is
lost.

## Snapshot at start (02:34)

### Running training queues
- **GPU 1** PID 1776032 — `run_overnight_paper_variance.sh` (started 00:12)
  - currently in **Run 1: Train-14 uncond s1** (`multi_scaling_14_uncond_match_s1`)
  - at step ~34k/150k → ETA finish ~04:06; heldout adds ~10 min
  - then Train-14 uncond s2 → Train-14 cond s2 → Train-59 uncond s1 (last won't fit)
- **GPU 0** PID 1864887 — `run_overnight_paper_variance_gpu0.sh` (started 00:28)
  - currently in **Run 1: Train-199 cond s1** (`multi_scaling_gallery_v4_cond_match_s1`)
  - at step ~30k/150k. Per ~30s/250steps it's ~7.5 step/s → ETA ~05:00; heldout ~15 min
  - then Train-14 cond s2 (will collide with GPU1; first to finish wins via
    `eval_multigame.npz` skip), then Train-14 uncond s3 (slack)

### CPU/auxiliary
- PID 1882985 — `nca_wm.heldout_eval` re-eval of `multi_scaling_14_uncond_match_s0`
  (part of `refresh_heldout_with_bfs_astar.sh` PID 1833210). Adds BFS/A* (AR)
  columns; runs on CPU per `JAX_PLATFORM_NAME=cpu` (small persistent ~390 MiB
  on each GPU is XLA preamble, not compute).
  - Refresh queue: `s0` of cond_match_14, uncond_match_14, gallery_v2 cond,
    gallery_v2 uncond, gallery_v4 cond. Started 01:46; appears to be on the
    second run (uncond_match_14) at 02:25.

### Paper state at start
- Latest commit `64a14f4` (Paper: compact prose, fix bugs).
- `cond_vs_uncond_match/match_table` has full TF cells for Train-{14,59,199}
  cond+uncond, but BFS/A* (AR) columns are all `--`. The CPU refresh job
  fills these in once it completes for each checkpoint.
- Working tree changes (uncommitted):
  - `paper/figures/id_ood_teaser/teaser.{pdf,png}` regenerated
  - `paper/figures/indist_intersection/intersection_heatmap.pdf` regenerated
  - `paper/nca_arch_v1.pdf` regenerated
  - `scripts/build_id_ood_teaser.py` modified
  - untracked: `data_cache/`, `paper/additional_refs.bib`,
    `paper/figures/id_ood_teaser/rollouts/`, `paper/scripts/page_budget.py`,
    `scripts/dump_ood_rollout_frames.py`

### Stale prose flagged
- `sections/results.tex` Fig 2 caption: "The Train-199 unconditional held-out
  eval is in flight." — but the table has the row populated already
  (Train-199 uncond: 3.29 / 1.97 % TF mean/median). Stale text; needs
  rephrasing / removal. Will fix unless another agent claims it.

## Other-agent activity observed

- **02:41 commit `216fb6e`**: another agent already patched
  `collate_match_table.py` (BFS/A* cells), regenerated `match_table.tex`,
  noted the Train-14 cond TF mean shift (16.18→19.26) is driven by
  `break_out_of_the_mine` (798 rule tokens vs `max_seq_len=657`,
  truncated). My patch attempt produced an identical file (no-op).
- **02:39 commit `ede655b`**: teaser figure redesigned (NCA-WM block,
  loss feedback, real Train-199 OOD rollouts).
- **02:35 commit `67323d8`**: results section reordered (ID/OOD scaling
  leads, NCA arch ablation last).
- Inference: paper is being actively rewritten, fast cadence (3 commits
  in 10 min). Side-experiments must avoid touching prose, structure,
  or main figures — work on additive supplements only.

## Open observations / candidates for additive work

- **Page budget 11/9** — main body is over by 2 pages. Discussion needed
  before cutting; paper agents may already be aware.
- **Truncation footnote**: `break_out_of_the_mine` (798 tokens) and
  potentially others exceed `max_seq_len=657`. Counting how many of
  the 30 heldouts truncate, and how much they distort the mean,
  could earn a clean appendix-level paragraph or a footnote.
- **Per-game wins-vs-identity ladder**: which 11 (cond) / 14 (uncond)
  Train-199 heldout games beat identity? Useful as an appendix table or
  supplement scatter — concrete grounding for an aggregate claim.
- **OOD rollout GIFs for supplement**: easy CPU-bound add if the paper
  agents want qualitative examples.
- **Supplemental README** for the released code/data drop: low risk.

## Active monitors

- `b3dux084y` GPU 0 queue (Train-199 cond s1 → ...): persistent
- `bkadptu7m` GPU 1 queue (Train-14 variance): persistent
- `btv9ypsd1` heldout BFS/A* refresh: persistent

## Actions taken (this session)

(prepended; newest first)

- 03:05 — quantified Heldout truncation: 4 of 30 games tokenize past
  `max_seq_len=657`, dragging Train-199 cond's TF mean from 2.30
  (no-trunc) to 2.91 (full set). On Heldout-26, cond beats identity's
  median ($1.55$ vs $1.97$) and matches identity's mean within
  $0.07$pp. Wrote `nca_wm/TRUNCATION_ANALYSIS.md`,
  `nca_wm/scripts/build_truncation_analysis.py`, and pre-rendered
  `nca_wm/paper/figures/heldout_truncation/{truncation_table.tex,
  truncation_summary.csv}` so paper agents can drop into a footnote
  or appendix subsection. Numbers will update automatically as the
  remaining heldout refreshes (Train-59 cond/uncond, Train-199 cond)
  finish on CPU.
- 02:55 — set up persistent monitors on all three log streams (GPU 0/1
  queues + heldout refresh) to catch failure / completion events
  without polling
- 02:50 — patched `collate_match_table.py` to read BFS/A* keys; turned
  out to be a no-op (HEAD already had the fix from concurrent agent
  commit `216fb6e`); kept tasks consistent
- 02:34 — created supervisor log; cataloged running training queues,
  CPU heldout refresh, paper state, page budget overflow, stale prose


# Rerun Notes

- 2026-05-07: Rerun heldout evaluations before using final paper numbers.
  Heldout/raw-token padding was fixed to reserve one encoder position for the
  internally-prepended CLS token, and search caches now require BFS/A*
  provenance before reuse.

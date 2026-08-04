"""Adapter: recurrent_data trajectory batches -> event-decoder batches.

Consumes the same ``load_dataset_from_caches`` + ``build_trajectory_batch``
pipeline as ``train_recurrent`` (RecurrentNCA) and
``active_learning/mario_transformer_baseline`` (AttnBeliefModel), so the
event world model trains on byte-identical trajectory distributions.

Per step t of a length-L trajectory the transition is
``states[b, t] --actions[b, t]--> targets[b, t]`` (both frames come from the
same cache row, so a ``-1`` hole at t+1 never corrupts step t's delta).
Invalid steps (``valid=False``) contribute no decoder tokens and are masked
out of encoder attention.
"""

from __future__ import annotations

import numpy as np

from nca_wm.recurrent_data import (  # noqa: F401  (re-exported for callers)
    GameData,
    build_trajectory_batch,
    load_dataset_from_caches,
)

from .tokenizer import Geom, encode_step


def to_event_batch(g: Geom, states, actions_onehot, targets, valid,
                   max_len: int):
    """(states, actions_onehot, targets, valid) -> event-decoder batch dict.

    Returns numpy arrays:
      obs        (B, L, C, H, W) float32 — encoder input frames (states)
      actions    (B, L) int32
      dec_in     (B, max_len) int32 — [BOS, e1..ek] per valid step, concat
      dec_tgt    (B, max_len) int32 — [e1..ek, EOF] aligned; PAD elsewhere
      dec_step   (B, max_len) int32 — env step of each position (255 = pad)
      step_valid (B, L) bool
    Trajectories whose token stream exceeds ``max_len`` are truncated by
    dropping their *earliest* steps' tokens (the final target step is always
    kept); overflow is counted in the returned ``n_overflow``.
    """
    B, L = valid.shape
    dec_in = np.full((B, max_len), g.pad, dtype=np.int32)
    dec_tgt = np.full((B, max_len), g.pad, dtype=np.int32)
    dec_step = np.full((B, max_len), 255, dtype=np.int32)
    n_overflow = 0
    for b in range(B):
        seqs: list[tuple[int, list[int]]] = []
        for t in range(L):
            if not valid[b, t]:
                continue
            seqs.append((t, encode_step(g, states[b, t], targets[b, t])))
        # total positions = sum(len(ev) + 1) per step (BOS/EOF share slots)
        while sum(len(ev) + 1 for _, ev in seqs) > max_len and seqs:
            seqs.pop(0)
            n_overflow += 1
        pos = 0
        for t, ev in seqs:
            k = len(ev) - 1                       # ev ends with EOF
            dec_in[b, pos] = g.bos
            dec_in[b, pos + 1: pos + 1 + k] = ev[:k]
            dec_tgt[b, pos: pos + k + 1] = ev
            dec_step[b, pos: pos + k + 1] = t
            pos += k + 1
    return {
        "obs": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions_onehot).argmax(-1).astype(np.int32),
        "dec_in": dec_in,
        "dec_tgt": dec_tgt,
        "dec_step": dec_step,
        "step_valid": np.asarray(valid, dtype=bool),
    }, n_overflow


def audit_batch(g: Geom, batch, targets) -> None:
    """Bijection audit: re-apply each step's events, compare to targets."""
    from .tokenizer import apply_events

    obs, dec_in, dec_step = batch["obs"], batch["dec_in"], batch["dec_step"]
    B, L = batch["step_valid"].shape
    for b in range(B):
        for t in range(L):
            pos = np.flatnonzero(dec_step[b] == t)
            if len(pos) == 0:
                continue
            assert dec_in[b, pos[0]] == g.bos
            events = [int(x) for x in dec_in[b, pos[1:]]] + [g.eof]
            out = apply_events(g, obs[b, t], events)
            if not np.array_equal(out.astype(bool),
                                  targets[b, t].astype(bool)):
                raise AssertionError(f"bijection audit failed b={b} t={t}")

"""Frame-delta event tokenization, geometry-generic.

Port of ``infogain-world-models`` ``mario_milestone/transformer/tokenizer.py``
(branch ``se_nca_h2h``) with the grid geometry (C, H, W) as a runtime value
instead of module constants, and without the ``RESET`` token (PuzzleScript
trajectories have no identical-world-restart invariant).

A step's target is the canonically ordered list of cell-change events
transforming ``o_t`` into ``o_{t+1}``, terminated by ``EOF``. The canonical
encoding is a *bijection* between frame deltas and token sequences, so the
decoder's autoregressive product is an exactly normalized likelihood over
next frames.

Vocabulary layout for ``N = C*H*W`` cells (ids are canonical rank):

* ``0 .. N-1``    — REMOVE events, id = (channel*H*W + y*W + x)
* ``N .. 2N-1``   — ADD events, id = N + cell
* ``EOF  = 2N``   — end of this step's events (a quiet step is just EOF)
* ``BOS  = 2N+1`` — decoder start token
* ``PAD  = 2N+2``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geom:
    c: int
    h: int
    w: int

    @property
    def n_cell(self) -> int:
        return self.c * self.h * self.w

    @property
    def add0(self) -> int:
        return self.n_cell

    @property
    def eof(self) -> int:
        return 2 * self.n_cell

    @property
    def bos(self) -> int:
        return 2 * self.n_cell + 1

    @property
    def pad(self) -> int:
        return 2 * self.n_cell + 2

    @property
    def vocab(self) -> int:
        return 2 * self.n_cell + 3


def encode_step(g: Geom, prev: np.ndarray, nxt: np.ndarray) -> list[int]:
    """Canonical token sequence for one transition (no BOS, ends with EOF)."""
    pf = prev.reshape(-1).astype(bool)
    nf = nxt.reshape(-1).astype(bool)
    removes = np.flatnonzero(pf & ~nf)          # ascending == canonical
    adds = np.flatnonzero(~pf & nf)
    return [int(i) for i in removes] + \
           [g.add0 + int(i) for i in adds] + [g.eof]


def apply_events(g: Geom, frame: np.ndarray,
                 tokens: list[int] | np.ndarray) -> np.ndarray:
    """Apply a canonical token sequence to ``frame`` -> next frame.

    Raises on any invalid event (training data must never trigger this;
    asserted by the pretokenization audit)."""
    out = frame.reshape(-1).copy()
    for tok in tokens:
        tok = int(tok)
        if tok == g.eof:
            break
        if tok < g.add0:                    # remove
            if not out[tok]:
                raise ValueError(f"remove of empty cell {tok}")
            out[tok] = 0
        elif tok < g.eof:                   # add
            cell = tok - g.add0
            if out[cell]:
                raise ValueError(f"add of occupied cell {cell}")
            out[cell] = 1
        else:
            raise ValueError(f"unexpected token {tok}")
    return out.reshape(g.c, g.h, g.w)


def frame_legality_mask(g: Geom, frame: np.ndarray) -> np.ndarray:
    """``(vocab,)`` bool: events consistent with ``frame`` (order aside)."""
    flat = frame.reshape(-1).astype(bool)
    mask = np.zeros(g.vocab, dtype=bool)
    mask[:g.n_cell] = flat
    mask[g.add0:g.add0 + g.n_cell] = ~flat
    mask[g.eof] = True
    return mask

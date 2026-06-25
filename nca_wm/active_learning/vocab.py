"""Vocabulary for the engine-backed active-learning probe.

Sequence format (mirrors the prototype, but observations are PuzzleScript grids):

    BOS  OBS <cell_0> ... <cell_{HW-1}> END_OBS
         ( ACT <action> OBS <grid> END_OBS )*
         RESAMPLE_OBS OBS <grid> END_OBS

Each world in the family shares a FIXED canonical object family and a FIXED grid
geometry, so an OBS block is always exactly ``H*W`` cell tokens. A cell token
encodes the *set* of canonical objects present in that cell as a bitmask, so the
hidden world identity (which ruleset) is never revealed by the observation
alone — only by interaction.
"""
from __future__ import annotations

# --- Fixed canonical object family (shared across all worlds) --------------
# Index = canonical bit position. Order is ours, independent of any per-game
# engine id_dict; the world generator maps engine ids -> these by NAME.
CANONICAL_OBJECTS = ["background", "wall", "player", "seed", "sprout"]
NAME_TO_BIT = {n: i for i, n in enumerate(CANONICAL_OBJECTS)}
N_CANON = len(CANONICAL_OBJECTS)

# --- Grid geometry (height, width) ----------------------------------------
# Mutable so different world families can use different grid sizes. The token
# vocab itself is geometry-INDEPENDENT (cell tokens enumerate canonical
# bitmasks, not positions), so only N_CELLS changes — STOI/ITOS are unaffected.
GRID_H = 3
GRID_W = 5
N_CELLS = GRID_H * GRID_W


def set_geometry(h: int, w: int) -> None:
    global GRID_H, GRID_W, N_CELLS
    GRID_H, GRID_W, N_CELLS = h, w, h * w

# --- Special / control tokens ---------------------------------------------
PAD = "<pad>"
BOS = "<bos>"
OBS = "OBS"
END_OBS = "END_OBS"
ACT = "ACT"
RESAMPLE_OBS = "RESAMPLE_OBS"

# --- Action tokens ---------------------------------------------------------
# PuzzleScript / cpp engine input ids: 0=up 1=left 2=down 3=right 4=action.
UP, LEFT, DOWN, RIGHT, ACTION = "UP", "LEFT", "DOWN", "RIGHT", "ACTION"
ACTIONS = [UP, LEFT, DOWN, RIGHT, ACTION]
ACTION_TO_INPUT = {UP: 0, LEFT: 1, DOWN: 2, RIGHT: 3, ACTION: 4}

# --- Cell tokens: one per possible canonical bitmask -----------------------
def _cell_tok(mask: int) -> str:
    return f"CELL_{mask}"

CELL_TOKENS = [_cell_tok(m) for m in range(1 << N_CANON)]

# --- Master vocab ----------------------------------------------------------
_SPECIAL = [PAD, BOS, OBS, END_OBS, ACT, RESAMPLE_OBS]
_ALL = _SPECIAL + ACTIONS + CELL_TOKENS

STOI: dict[str, int] = {t: i for i, t in enumerate(_ALL)}
ITOS: list[str] = list(_ALL)
VOCAB_SIZE = len(ITOS)
PAD_ID = STOI[PAD]


def encode(tokens: list[str]) -> list[int]:
    return [STOI[t] for t in tokens]


def decode(ids: list[int]) -> list[str]:
    return [ITOS[i] for i in ids]


def serialize_obs(cell_masks: list[int]) -> list[str]:
    """Row-major list of canonical cell bitmasks -> OBS ... END_OBS block."""
    assert len(cell_masks) == N_CELLS, (len(cell_masks), N_CELLS)
    return [OBS] + [_cell_tok(m) for m in cell_masks] + [END_OBS]


def serialize_action(action: str) -> list[str]:
    assert action in ACTIONS, action
    return [ACT, action]


if __name__ == "__main__":
    print(f"VOCAB_SIZE = {VOCAB_SIZE}  (grid {GRID_H}x{GRID_W}, {N_CANON} objects)")

"""Game-agnostic synthetic-level generation for PuzzleScript world-model training.

Generates random valid levels and collects their full transition sets via the
C++ solver, producing a dataset shape-compatible with
``nca_wm.train.collect_unique_transitions``.

Design (deliberately game-agnostic):

  Construction
    For each output tile, sample a tile *bit-pattern* (a tuple of
    ``STRIDE_OBJ`` int32 words) from the empirical distribution of tile
    patterns observed across the game's authored levels. This implicitly
    respects collision layers (authored tiles do), needs no semantic
    knowledge of which object is "player"/"wall"/"crate"/etc., and adapts
    to any PuzzleScript game.

  Player invariant
    After per-tile sampling, force exactly one player tile (re-place if
    zero, randomly drop extras if > 1). This is the only role-based
    intervention; ``playerMask`` is exposed by the compiled JSON for any
    game that has a player concept (which is essentially all of them in
    PuzzleScript).

  Validity (search-based, fully generic)
    A level passes iff *all* of:
      1. Restoring the level into the engine doesn't immediately win
         (``engine.check_win()`` returns False).
      2. BFS over the reachable state space, budgeted by
         ``max_iters`` / ``timeout_ms``, observes at least one transition
         with ``won==True`` (level is solvable within budget).
      3. ``iterations >= min_states`` (reachable state space is non-trivial
         — filters out player-walled-in or no-op levels).
      4. BFS did not time out (so we trust that the explored space is
         exhaustive and the level is well-bounded).

Cache layout:
    rollout_data/{game}/synthetic_{w}x{h}/seed{seed}_n{N}_v{version}_*.npz

Notes
  - Authored levels typically include the game's full tile vocabulary
    (player/walls/goals/etc.) but not all combinatorial possibilities, so
    sampling from authored tile patterns gives natural-looking levels that
    match the game's visual idiom without being copies of authored ones.
  - Objects spawned by rules at runtime (rather than being placed in the
    initial level) are handled naturally by BFS solvability — we don't
    require them to appear in the initial state.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp._puzzlescript_cpp import (
    Engine,
    LevelBackup,
    collect_transitions_bfs,
)
from puzzlescript_cpp import CppPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser


# ---------------------------------------------------------------------------
# Bitfield helpers
# ---------------------------------------------------------------------------

def _to_i32(val: int) -> int:
    val &= 0xFFFFFFFF
    if val >= 0x80000000:
        val -= 0x100000000
    return val


def _bit_set(words: list[int], obj_idx: int) -> bool:
    word_i, bit = divmod(obj_idx, 32)
    if word_i >= len(words):
        return False
    return bool(words[word_i] & (1 << bit))


def _set_bit(words: list[int], obj_idx: int) -> None:
    word_i, bit = divmod(obj_idx, 32)
    words[word_i] = _to_i32(words[word_i] | (1 << bit))


def _clear_bit(words: list[int], obj_idx: int) -> None:
    word_i, bit = divmod(obj_idx, 32)
    words[word_i] = _to_i32(words[word_i] & ~(1 << bit))


def _decode_bitmask(mask_words: list[int]) -> list[int]:
    """Return the list of object indices set in a multiword bitmask."""
    out = []
    for word_i, word in enumerate(mask_words):
        for bit in range(32):
            if word & (1 << bit):
                out.append(word_i * 32 + bit)
    return out


# ---------------------------------------------------------------------------
# Tile-pattern extraction (mirrors evolve_level_cpp.extract_tile_patterns,
# inlined to avoid a cross-module import inside this leaf file)
# ---------------------------------------------------------------------------

def extract_tile_patterns(
    json_state: dict,
    level_indices: Optional[list] = None,
) -> list[tuple[int, ...]]:
    """Unique per-tile bitfield patterns across authored levels.

    Each pattern is a length-``STRIDE_OBJ`` tuple of int32 words.
    Sorted (stable hash key for caching).
    """
    stride = json_state["STRIDE_OBJ"]
    levels = json_state["levels"]
    if level_indices is not None:
        levels = [levels[i] for i in level_indices if i < len(levels)]
    seen = set()
    for lev in levels:
        if not isinstance(lev, dict) or lev.get("type") != "level":
            continue
        dat = lev["objects"]
        n_tiles = lev["width"] * lev["height"]
        for tile in range(n_tiles):
            base = tile * stride
            seen.add(tuple(dat[base:base + stride]))
    return sorted(seen)


def extract_authored_dats(
    json_state: dict,
    *,
    level_indices: Optional[list] = None,
) -> list[tuple[list[int], int, int]]:
    """Return ``(dat, width, height)`` tuples for each authored level.

    ``dat`` is a flat list of ``width * height * STRIDE_OBJ`` int32 words —
    the same format LevelBackup expects. Use this when seeding evolve from
    real authored layouts.
    """
    stride = int(json_state["STRIDE_OBJ"])
    levels = json_state["levels"]
    if level_indices is not None:
        levels = [levels[i] for i in level_indices if i < len(levels)]
    out = []
    for lev in levels:
        if not isinstance(lev, dict) or lev.get("type") != "level":
            continue
        w = int(lev["width"])
        h = int(lev["height"])
        objs = list(lev["objects"])
        out.append((objs, w, h))
    return out


def crop_dat(
    dat: list[int],
    src_w: int, src_h: int,
    dst_w: int, dst_h: int,
    stride_obj: int,
    rng: np.random.Generator,
    *,
    bg_idx: int | None = None,
) -> list[int]:
    """Random crop / pad of a level dat to ``(dst_w, dst_h)``.

    - If both src dims fit, pad with ``bg_idx``-only tiles centered.
    - If src is bigger in one axis, take a uniformly random window.

    Column-major layout: ``tile_idx = x * src_h + y`` in source.
    """
    out = [0] * (dst_w * dst_h * stride_obj)
    bg_word = 0
    if bg_idx is not None and stride_obj == 1:
        bg_word = _to_i32(1 << bg_idx)
        for t in range(dst_w * dst_h):
            out[t * stride_obj] = bg_word

    if src_w >= dst_w:
        x0 = int(rng.integers(0, src_w - dst_w + 1))
        sx = 0
        out_x_start = 0
        copy_w = dst_w
    else:
        x0 = 0
        sx = 0
        out_x_start = (dst_w - src_w) // 2
        copy_w = src_w

    if src_h >= dst_h:
        y0 = int(rng.integers(0, src_h - dst_h + 1))
        sy = 0
        out_y_start = 0
        copy_h = dst_h
    else:
        y0 = 0
        sy = 0
        out_y_start = (dst_h - src_h) // 2
        copy_h = src_h

    for dx in range(copy_w):
        for dy in range(copy_h):
            src_tile = (x0 + dx) * src_h + (y0 + dy)
            dst_tile = (out_x_start + dx) * dst_h + (out_y_start + dy)
            for s in range(stride_obj):
                out[dst_tile * stride_obj + s] = int(dat[src_tile * stride_obj + s])
    return out


def extract_tile_pattern_freqs(
    json_state: dict,
    level_indices: Optional[list] = None,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Same as extract_tile_patterns but also returns empirical frequencies."""
    stride = json_state["STRIDE_OBJ"]
    levels = json_state["levels"]
    if level_indices is not None:
        levels = [levels[i] for i in level_indices if i < len(levels)]
    counts: dict[tuple[int, ...], int] = {}
    for lev in levels:
        if not isinstance(lev, dict) or lev.get("type") != "level":
            continue
        dat = lev["objects"]
        n_tiles = lev["width"] * lev["height"]
        for tile in range(n_tiles):
            base = tile * stride
            key = tuple(dat[base:base + stride])
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return [], np.array([], dtype=np.float64)
    keys = sorted(counts.keys())
    freqs = np.array([counts[k] for k in keys], dtype=np.float64)
    freqs /= freqs.sum()
    return keys, freqs


# ---------------------------------------------------------------------------
# Static analysis: rule invariance + win-condition count constraints
# ---------------------------------------------------------------------------
#
# Goal: derive game-agnostic, construction-time hard rules that bias random
# levels toward potentially-solvable layouts, *without* peeking at object
# names. All inputs come from the compiled JSON (rules, winconditions).
#
# Pipeline:
#   1. Scan every cell of every rule to determine which objects can be
#      created (RHS-set without LHS-present) or destroyed (LHS-present
#      without RHS-set), or randomly spawned (randomEntityMask).
#      Conservative: when in doubt, mark non-invariant.
#   2. Parse winconditions:
#        num=1 ("All A on B"): if all bits of A and B are invariant, enforce
#          count(A) == count(B) >= 1 at start.
#        num=1 ("All A", no mask2): if A invariant, enforce count(A) >= 1.
#        num=2/3 ("Some A" / "Any A"): if A invariant, enforce count(A) >= 1.
#        num=0 ("No A on B"): handled by engine.check_win() post-restore
#          (already-winning rejection).
#   3. After random-tile-pattern sampling, run a corrector that drops
#      surplus tiles to satisfy the count_eq / count_min constraints.
#      We never *add* tiles (would require knowing layer-safe placement);
#      if there's nothing to drop and the constraint isn't met, the level
#      is invalid by construction and search_validate will reject it.

def _bitmask_words_to_set(words: list[int], n_objs: int = 0) -> set[int]:
    """Decode a multiword bitmask (list[int32]) to a set of bit indices."""
    out: set[int] = set()
    for word_i, word in enumerate(words):
        if not word:
            continue
        for bit in range(32):
            if word & (1 << bit):
                idx = word_i * 32 + bit
                if n_objs and idx >= n_objs:
                    continue
                out.add(idx)
    return out


def _walk_all_cells(rules):
    """Yield every cell dict from the compiled-rules tree.

    PuzzleScript compiles each authored rule into a *list* of direction-
    substituted variants, and each variant has 'patterns' = list of kernels,
    each kernel = list of cell dicts.
    """
    if not rules:
        return
    for rule_or_group in rules:
        if isinstance(rule_or_group, list):
            yield from _walk_all_cells(rule_or_group)
            continue
        if not isinstance(rule_or_group, dict):
            continue
        for kernel in rule_or_group.get("patterns", []) or []:
            for cell in kernel or []:
                if isinstance(cell, dict):
                    yield cell


def analyze_rule_invariance(json_state: dict) -> dict:
    """Per-object 'creatable' / 'destroyable' flags from compiled rules.

    Returns a dict: {
        'creatable': set[int]    (object indices that some rule may create)
        'destroyable': set[int]  (object indices that some rule may destroy)
        'invariant': set[int]    (n_objs - creatable - destroyable)
    }
    Conservative: prefers false positives in creatable/destroyable
    (which just means fewer count constraints are applied; never wrong).
    """
    creatable: set[int] = set()
    destroyable: set[int] = set()
    n_objs = int(json_state.get("objectCount", 0))
    rules = json_state.get("rules", [])
    for cell in _walk_all_cells(rules):
        obj_present = _bitmask_words_to_set(cell.get("objectsPresent", []) or [], n_objs)
        # OR groups: any of these *might* be present in the matched LHS.
        # Treat them as part of obj_present for invariance reasoning (so an
        # object only gets flagged 'destroyable' if RHS doesn't preserve it
        # in *any* of the OR alternatives).
        for grp in cell.get("anyObjectsPresent", []) or []:
            grp_set = _bitmask_words_to_set(grp if isinstance(grp, list) else [grp], n_objs)
            obj_present.update(grp_set)
        repl = cell.get("replacement", {}) or {}
        obj_set = _bitmask_words_to_set(repl.get("objectsSet", []) or [], n_objs)
        rand_ent = _bitmask_words_to_set(repl.get("randomEntityMask", []) or [], n_objs)
        creatable.update(rand_ent)
        # Created at this cell: bits set in RHS that weren't in LHS-present.
        creatable.update(obj_set - obj_present)
        # Destroyed at this cell: bits in LHS-present that aren't in RHS-set.
        destroyable.update(obj_present - obj_set)
    invariant = set(range(n_objs)) - creatable - destroyable
    return {
        "creatable": creatable,
        "destroyable": destroyable,
        "invariant": invariant,
    }


# Win-condition quantifiers (PuzzleScript engine numeric encoding).
# 0 = "No A on B" (negation, with explicit mask2)
# 1 = "All A on B" / "All A" (universal)
# 2 = "Some A" (existence)
# 3 = "Any A" (existence; same family as Some for our purposes)
# -1 = "No A" without explicit mask2 (mask2=[-1] is the universe sentinel —
#       i.e. "no A anywhere", equivalent to "level reaches a state with zero A")
WIN_NO_ON = 0
WIN_ALL = 1
WIN_SOME = 2
WIN_ANY = 3
WIN_NO_GLOBAL = -1


def parse_win_constraints(json_state: dict, invariance: dict, *,
                            no_a_count_max: int = 3) -> dict:
    """Turn winconditions + invariance flags into actionable count constraints.

    Returns a dict: {
        'count_eq':   list[(mask_a_set, mask_b_set)]  - require equal counts
        'count_min':  list[(mask_set, n)]              - require count >= n
        'parsed':     list[dict]                        - human-readable summary
    }
    Each mask_*_set is a frozenset[int] of object indices participating in
    the bitmask.

    Design principles:
      - **Presence baseline**: whenever a win condition references a mask A
        (mask1), we want ``count(A) ≥ 1`` at start, so the puzzle is
        non-trivial regardless of polarity ("all A", "some A", "no A" all
        get easier-to-vacuously-satisfy with zero A). The previous version
        skipped count_min when A was creatable; we now keep it because for
        almost all reasonable puzzle designs A is "the thing the player
        manipulates" and starting with zero of it makes the level dull or
        already-won.
      - **Equal counts** for "all A on B" only when *both* sides are
        invariant — otherwise we'd over-constrain games that spawn B (or A)
        from rules.
      - **No-A-on-B** ("`No A on B`", num=0): at start, requiring
        count(A) ≥ 1 is enough; engine.check_win() catches the
        already-won degenerate.
      - **No-A-globally** ("`No A`", num=-1, mask2=[-1] sentinel): same as
        above plus, since the puzzle goal is to *eliminate* A, A is
        typically destroyable and we definitely want ≥ 1 at start.
    """
    invariant: set[int] = invariance["invariant"]
    n_objs = int(json_state.get("objectCount", 0))
    count_eq: list[tuple[frozenset, frozenset]] = []
    count_min: list[tuple[frozenset, int]] = []
    count_max: list[tuple[frozenset, int]] = []
    parsed: list[dict] = []
    for wc in json_state.get("winconditions", []) or []:
        num = int(wc.get("num", -2))
        m1 = frozenset(_bitmask_words_to_set(wc.get("mask1", []) or [], n_objs))
        m2_raw = wc.get("mask2", []) or []
        # mask2 = [-1] is the engine's "no positional target" sentinel —
        # used by "no A" / "all A" / "some A" forms (vs the on-B variants).
        m2_is_universe = isinstance(m2_raw, list) and m2_raw == [-1]
        m2_words = m2_raw if (isinstance(m2_raw, list) and not m2_is_universe) else []
        m2 = frozenset(_bitmask_words_to_set(m2_words, n_objs))
        m1_inv = bool(m1) and m1.issubset(invariant)
        m2_inv = bool(m2) and m2.issubset(invariant)
        entry = {
            "num": num,
            "mask1_objs": sorted(m1),
            "mask2_objs": sorted(m2),
            "mask2_is_universe": m2_is_universe,
            "mask1_invariant": m1_inv,
            "mask2_invariant": m2_inv,
            "applied": False,
            "reason": None,
        }
        applied_reasons = []
        # Baseline: any win condition that references A → count(A) ≥ 1 at start.
        if m1:
            count_min.append((m1, 1))
            applied_reasons.append("count_min(A,1) baseline (A involved in win)")
        # "No A" globally (num=-1, mask2=[-1] universe sentinel): the puzzle
        # goal is to eliminate A. Cap A-count to keep BFS reachable-state-space
        # tractable — without this, random Zen 12×12 has ~30 unbrushed-sand
        # tiles and BFS state space is exponential in count(A) so no winning
        # state is reachable in any practical budget.
        if num == WIN_NO_GLOBAL and m2_is_universe and m1:
            count_max.append((m1, no_a_count_max))
            applied_reasons.append(f"count_max(A,{no_a_count_max}) — 'no A' goal cap for BFS tractability")
        # "All A on B": equal counts when both fully invariant.
        if num == WIN_ALL and m2 and m1_inv and m2_inv:
            count_eq.append((m1, m2))
            applied_reasons.append("count_eq(A,B) — both invariant 'all on'")
        # "No A on B": want at least one A to need eliminating, plus a B for
        # the relation. count_min(B,1) only when B invariant (else B might
        # spawn at runtime).
        if num == WIN_NO_ON and m2 and m2_inv:
            count_min.append((m2, 1))
            applied_reasons.append("count_min(B,1) — B invariant 'no on'")
        if applied_reasons:
            entry["applied"] = True
            entry["reason"] = "; ".join(applied_reasons)
        else:
            entry["reason"] = "no constraint emitted (e.g. unknown num, empty mask)"
        parsed.append(entry)
    return {"count_eq": count_eq, "count_min": count_min,
            "count_max": count_max, "parsed": parsed}


# ---------------------------------------------------------------------------
# Generic level generator
# ---------------------------------------------------------------------------

class LevelGenerator:
    """Game-agnostic random-level generator.

    All game-specific information comes from the compiled JSON state; no
    hardcoded object-role names.

    Construction modes:
      - "tile_pattern_empirical" (default): sample each tile from the
        empirical pattern distribution observed in authored levels.
      - "tile_pattern_uniform": sample each tile uniformly from the set of
        unique authored tile patterns.

    Player invariant: after per-tile sampling, force exactly one player
    tile by re-placing or removing as needed.
    """

    def __init__(self, json_state: dict, *, mode: str = "tile_pattern_empirical",
                  apply_win_constraints: bool = True,
                  no_a_count_max: int = 3):
        self.json_state = json_state
        self.stride = int(json_state["STRIDE_OBJ"])
        self.n_objs = int(json_state["objectCount"])
        self.mode = mode
        # Player bits (which object indices count as players)
        player_mask = json_state["playerMask"][1]
        self.player_indices = _decode_bitmask(player_mask)
        if not self.player_indices:
            raise RuntimeError("Game has no player object — refusing to generate.")
        # Layer masks (per-layer multiword bitmasks)
        self.layer_masks = [list(m) for m in json_state["layerMasks"]]
        # Find the collision layer that contains the player(s); used to
        # determine "layer-2-empty" tiles when placing/replacing the player.
        self.player_layer_idx: int | None = None
        for li, lmask in enumerate(self.layer_masks):
            if any(_bit_set(lmask, p) for p in self.player_indices):
                self.player_layer_idx = li
                break
        if self.player_layer_idx is None:
            raise RuntimeError("Player object not assigned to any collision layer.")
        # Authored tile patterns + empirical frequencies
        self.patterns, self.pattern_freqs = extract_tile_pattern_freqs(json_state)
        if not self.patterns:
            raise RuntimeError(
                "Game has no authored levels to draw tile patterns from. "
                "Pure-uniform per-layer construction is not yet supported."
            )
        self._patterns_arr = np.array(self.patterns, dtype=np.int64)  # (P, stride)
        # Pre-classify each pattern: does it contain the player? does it
        # occupy the player layer (and thus block player placement)?
        self._is_player_pattern = np.array(
            [self._pattern_has_player(p) for p in self.patterns],
            dtype=bool,
        )
        self._blocks_player_layer = np.array(
            [self._pattern_blocks_layer(p, self.player_layer_idx)
             for p in self.patterns],
            dtype=bool,
        )
        # Static rule analysis + win-condition count constraints.
        self.invariance = analyze_rule_invariance(json_state)
        self.win_constraints = parse_win_constraints(
            json_state, self.invariance, no_a_count_max=no_a_count_max,
        )
        self.apply_win_constraints = apply_win_constraints
        # Auto-detect single-player vs swarm-style games from authored levels.
        # If every authored level has exactly 1 player tile, the game is
        # designed for single-player and we enforce that on synth levels too
        # (matches authored distribution → better holdout transfer). Otherwise
        # we use the looser "≥ 1 player" rule that preserves swarm structure.
        max_authored_players = 0
        all_have_one = True
        seen_any = False
        for src_dat, sw, sh in extract_authored_dats(json_state):
            seen_any = True
            n_p = self.count_players(list(src_dat), sw, sh)
            if n_p > max_authored_players:
                max_authored_players = n_p
            if n_p != 1:
                all_have_one = False
        # If we can't tell (no authored levels) default to single-player.
        self.single_player_game = (not seen_any) or all_have_one
        self.max_authored_player_count = max_authored_players

    # ----- pattern-level predicates -----

    def _pattern_has_player(self, pattern: tuple[int, ...]) -> bool:
        return any(_bit_set(list(pattern), p) for p in self.player_indices)

    def _pattern_blocks_layer(self, pattern: tuple[int, ...], layer_idx: int) -> bool:
        """True if the pattern has *any* object on collision layer ``layer_idx``."""
        lmask = self.layer_masks[layer_idx]
        words = list(pattern)
        for i in range(min(len(lmask), len(words))):
            if words[i] & lmask[i]:
                return True
        return False

    # ----- tile-pattern empty (per layer) -----

    def _empty_player_layer_pattern(self) -> tuple[int, ...]:
        """A pattern with the player layer cleared (preserves other layers).

        Used when forcing player placement: we pick an existing pattern that
        doesn't already block the player layer, but if all sampled tiles
        block it, we may need to clear the player-layer bits before
        placing the player. This returns a 'cleared' template for that.
        """
        return tuple([0] * self.stride)

    def _strip_player_layer(self, pattern: tuple[int, ...]) -> tuple[int, ...]:
        words = list(pattern)
        lmask = self.layer_masks[self.player_layer_idx]
        for i in range(min(len(lmask), len(words))):
            words[i] = _to_i32(words[i] & ~lmask[i])
        return tuple(words)

    # ----- generation -----

    def random_dat(
        self, rng: np.random.Generator, width: int, height: int,
    ) -> list[int] | None:
        """Sample a random level. Returns dat as a flat list of ``w*h*stride``
        int32 words (column-major: ``tile = x * height + y``), or None if
        construction failed (no tile available for the player).
        """
        n_tiles = width * height
        n_patterns = len(self.patterns)

        if self.mode == "tile_pattern_empirical":
            probs = self.pattern_freqs
        elif self.mode == "tile_pattern_uniform":
            probs = np.full(n_patterns, 1.0 / n_patterns)
        else:
            raise ValueError(f"Unknown synth mode {self.mode!r}")

        # Sample patterns for every tile in one shot (vectorized).
        choices = rng.choice(n_patterns, size=n_tiles, replace=True, p=probs)
        # Materialize dat as flat list of int32 words
        dat = self._patterns_arr[choices].astype(np.int64).reshape(-1).tolist()
        # Wrap back into i32 (np->python may give ints outside i32 range)
        dat = [_to_i32(int(v)) for v in dat]

        # Force exactly one player.
        dat = self._force_one_player(rng, dat, width, height, choices)
        if dat is None:
            return None
        # Apply win-condition count corrections (game-agnostic, derived from
        # static rule analysis).
        if self.apply_win_constraints:
            dat = self._apply_win_constraints(dat, width, height, rng)
        return dat

    def _ensure_at_least_one_player(
        self,
        rng: np.random.Generator,
        dat: list[int],
        width: int,
        height: int,
        choices: np.ndarray | None = None,
    ) -> list[int] | None:
        """Ensure player count matches the game's authored regime.

        For single-player games (every authored level has exactly 1 player
        tile), enforce exactly 1 — random sampling otherwise produces
        multi-player layouts that don't reflect the game's design and bias
        the model toward swarm dynamics. For swarm games (any authored level
        has >1 player tile, e.g. kettle), allow ≥1 (don't strip extras).

        ``choices`` is an array of per-tile pattern indices into
        ``self.patterns`` (only available when ``random_dat`` produced this
        ``dat``). For authored crops or other code paths it can be None;
        we'll inspect the actual tile content instead.
        """
        n_tiles = width * height
        n_players = self.count_players(dat, width, height)
        if self.single_player_game:
            if n_players == 1:
                return dat
            if n_players > 1:
                # Strip extras randomly, keep one.
                player_tiles = []
                for tile in range(n_tiles):
                    base = tile * self.stride
                    words = dat[base:base + self.stride]
                    for p in self.player_indices:
                        if _bit_set(words, p):
                            player_tiles.append((tile, p))
                            break
                keep_idx = int(rng.integers(len(player_tiles)))
                for i, (t, p) in enumerate(player_tiles):
                    if i == keep_idx:
                        continue
                    base = t * self.stride
                    words = dat[base:base + self.stride]
                    _clear_bit(words, p)
                    dat[base:base + self.stride] = words
                return dat
            # n_players == 0: fall through to add-one branch below.
        else:
            # Swarm game: any positive count is fine.
            if n_players >= 1:
                return dat
        # No player anywhere; need to add one. Find a tile whose player-layer
        # is currently unoccupied; if all are occupied, strip a random tile's
        # player-layer bits and stamp the player there.
        if choices is not None:
            non_blocking = [
                t for t in range(n_tiles)
                if not self._blocks_player_layer[choices[t]]
            ]
        else:
            non_blocking = []
            for t in range(n_tiles):
                base = t * self.stride
                words = dat[base:base + self.stride]
                blocks = False
                for o in range(self.n_objs):
                    if _bit_set(words, o) and self.obj_to_layer_idx(o) == self.player_layer_idx:
                        blocks = True
                        break
                if not blocks:
                    non_blocking.append(t)
        if non_blocking:
            t = int(rng.choice(non_blocking))
        else:
            t = int(rng.integers(n_tiles))
            base = t * self.stride
            words = dat[base:base + self.stride]
            stripped = list(self._strip_player_layer(tuple(words)))
            dat[base:base + self.stride] = stripped
        p = self.player_indices[int(rng.integers(len(self.player_indices)))]
        base = t * self.stride
        words = dat[base:base + self.stride]
        _set_bit(words, p)
        dat[base:base + self.stride] = words
        return dat

    # Backwards-compat alias for any callers; same semantics now.
    _force_one_player = _ensure_at_least_one_player

    def obj_to_layer_idx(self, obj_idx: int) -> int:
        """Return collision-layer index containing ``obj_idx``, or -1."""
        for li, lmask in enumerate(self.layer_masks):
            if _bit_set(lmask, obj_idx):
                return li
        return -1

    # ----- mask-based per-tile predicates -----

    def _tile_has_any(self, dat: list[int], tile: int, obj_set: frozenset[int]) -> bool:
        base = tile * self.stride
        words = dat[base:base + self.stride]
        for o in obj_set:
            if _bit_set(words, o):
                return True
        return False

    def _tiles_with_mask(self, dat: list[int], n_tiles: int,
                         obj_set: frozenset[int]) -> list[int]:
        return [t for t in range(n_tiles) if self._tile_has_any(dat, t, obj_set)]

    def _drop_mask_at_tile(self, dat: list[int], tile: int,
                           obj_set: frozenset[int]) -> None:
        """Clear all bits of ``obj_set`` from this tile."""
        base = tile * self.stride
        for o in obj_set:
            word_i, bit = divmod(o, 32)
            if word_i < self.stride:
                dat[base + word_i] = _to_i32(dat[base + word_i] & ~(1 << bit))

    # ----- win-constraint corrector -----

    def _apply_win_constraints(
        self, dat: list[int], width: int, height: int, rng: np.random.Generator,
    ) -> list[int] | None:
        """Drop surplus tiles to satisfy ``count_eq`` and ``count_min``.

        We never *add* objects (would need to know layer-safe placement);
        if a count_min can't be satisfied because no tile carries that
        object after sampling, this returns None and the caller resamples.
        Constraints involving the player object's layer are skipped to
        avoid clashing with the exactly-one-player invariant.
        """
        n_tiles = width * height
        # count_eq: equalize cell counts of two object groups by dropping
        # surplus from the larger side.
        for mask_a, mask_b in self.win_constraints["count_eq"]:
            tiles_a = self._tiles_with_mask(dat, n_tiles, mask_a)
            tiles_b = self._tiles_with_mask(dat, n_tiles, mask_b)
            if len(tiles_a) == len(tiles_b):
                continue
            if len(tiles_a) > len(tiles_b):
                surplus, mask = tiles_a, mask_a
                target = len(tiles_b)
            else:
                surplus, mask = tiles_b, mask_b
                target = len(tiles_a)
            n_drop = len(surplus) - target
            if n_drop <= 0:
                continue
            # Don't drop tiles that overlap the player layer (could remove the
            # player or alter player-layer-tied semantics in unexpected ways).
            # In practice mask_a/mask_b are usually different layers (e.g.
            # target on layer 1, crate on layer 2), so this is rare.
            drop_idxs = list(rng.choice(len(surplus), size=n_drop, replace=False))
            for di in drop_idxs:
                self._drop_mask_at_tile(dat, surplus[int(di)], mask)
        # count_max: drop random surplus tiles to keep |A| ≤ K.
        # Used for "no A" win conditions where unconstrained random sampling
        # produces too many A-tiles for BFS to clear within budget.
        for mask, n_max in self.win_constraints.get("count_max", []):
            tiles = self._tiles_with_mask(dat, n_tiles, mask)
            if len(tiles) <= n_max:
                continue
            n_drop = len(tiles) - n_max
            drop_idxs = list(rng.choice(len(tiles), size=n_drop, replace=False))
            for di in drop_idxs:
                self._drop_mask_at_tile(dat, tiles[int(di)], mask)
        # count_min: if any group has zero, we can't synthesize one safely.
        # Just signal failure to the caller for resampling.
        for mask, n_min in self.win_constraints["count_min"]:
            if len(self._tiles_with_mask(dat, n_tiles, mask)) < n_min:
                return None
        return dat

    # ----- structural validity (post-construction) -----

    def count_players(self, dat: list[int], width: int, height: int) -> int:
        n_tiles = width * height
        c = 0
        for tile in range(n_tiles):
            base = tile * self.stride
            words = dat[base:base + self.stride]
            for p in self.player_indices:
                if _bit_set(words, p):
                    c += 1
                    break
        return c

    def structurally_valid(self, dat: list[int], width: int, height: int) -> bool:
        """Player-count check, regime-aware (set in ``__init__`` from authored
        levels). Single-player games require exactly 1; swarm games require
        ≥ 1. ``_ensure_at_least_one_player`` will normally have made this
        already true, but the check guards against caller bypass."""
        n = self.count_players(dat, width, height)
        if self.single_player_game:
            return n == 1
        return n >= 1


# ---------------------------------------------------------------------------
# Search-based validation + transition collection
# ---------------------------------------------------------------------------

def _extract_bfs_solution(states, actions, next_states, wons) -> list[int]:
    """Reconstruct the shortest solution-action sequence from transitions
    that ``collect_transitions_bfs`` already collected — without re-running
    search.

    Approach: build a forward adjacency `(state) → list of (next_state,
    action, won)` from the transitions, then do a Python-side BFS from the
    initial state (= the source state of the first transition) until we
    reach a winning state. This is robust to whatever order the C++
    collector emits transitions in (the previous backward-walk approach
    failed when transitions weren't strictly BFS-depth-ordered, producing
    apparent cycles in the parent map).

    Returns the shortest action sequence to win, or ``[]`` if no winning
    state is reachable in the collected transitions.
    """
    if not any(w for w in wons):
        return []
    if not states:
        return []
    # Forward adjacency: state_tuple → list of (next_state_tuple, action, won)
    adj: dict[tuple, list[tuple]] = {}
    for s, a, ns, w in zip(states, actions, next_states, wons):
        s_key = tuple(int(x) for x in s)
        ns_key = tuple(int(x) for x in ns)
        adj.setdefault(s_key, []).append((ns_key, int(a), bool(w)))
    initial = tuple(int(x) for x in states[0])
    # BFS forward from initial; for each state we record the first (parent,
    # action) that reached it so we can reconstruct the path.
    from collections import deque
    parent_back: dict[tuple, tuple] = {initial: (None, None)}
    queue = deque([initial])
    win_state: tuple | None = None
    win_action: int | None = None
    while queue:
        cur = queue.popleft()
        for ns_key, a, w in adj.get(cur, ()):
            if ns_key not in parent_back:
                parent_back[ns_key] = (cur, a)
                if w:
                    win_state = cur
                    win_action = a
                    queue.clear()
                    break
                queue.append(ns_key)
    if win_state is None:
        return []
    # Walk back from win_state's recorded parent chain (acyclic by BFS
    # construction here, since parent_back is filled at first visit only).
    chain: list[int] = [int(win_action)]
    cur = win_state
    while parent_back.get(cur, (None, None))[0] is not None:
        parent, a = parent_back[cur]
        chain.append(int(a))
        cur = parent
    chain.reverse()
    return chain


def search_validate(
    engine: Engine,
    dat: list[int],
    width: int,
    height: int,
    *,
    max_iters: int = 5000,
    timeout_ms: int = 2000,
    min_states: int = 20,
    require_solvable: bool = False,
    track_rules_fired: bool = False,
) -> dict | None:
    """Restore ``dat`` into ``engine`` and run BFS-based transition collection.

    Returns the TransitionData-as-dict on accept, ``None`` on reject.

    Required: not initially winning, ≥``min_states`` reachable BFS states,
    no timeout. If ``require_solvable`` is True, also reject levels where
    BFS observed no winning transition.
    """
    backup = LevelBackup(dat, width, height)
    engine.restore_level(backup)
    if engine.check_win():
        return None
    result = collect_transitions_bfs(
        engine, max_iters=max_iters, timeout_ms=timeout_ms,
        track_rules_fired=track_rules_fired,
    )
    iterations = int(result.iterations)
    if iterations < min_states:
        return None
    if result.timeout:
        # Bounded BFS gives no guarantee about the rest of the state space;
        # don't trust it for "small synthetic level".
        return None
    wons = list(result.wons)
    is_solvable = any(w for w in wons)
    if require_solvable and not is_solvable:
        return None
    # Extract the optimal solution path from the transitions BFS already
    # collected — no second search call. The transitions come back in
    # BFS-visit order so the first occurrence of each ``next_state`` in the
    # list is on the shortest path from the root. We build a
    # ``next_state → (parent_state, action)`` map and walk back from the
    # earliest winning transition to the initial state.
    solution_actions: list[int] = []
    if is_solvable:
        solution_actions = _extract_bfs_solution(
            result.states, result.actions, result.next_states, wons,
        )
    rules_fired_per_t = (
        [list(rf) for rf in result.rules_fired] if track_rules_fired else []
    )
    rules_fired_union: list[int] = sorted(set().union(*rules_fired_per_t)) if rules_fired_per_t else []
    return {
        "states": result.states,
        "actions": list(result.actions),
        "next_states": result.next_states,
        "wons": wons,
        "iterations": iterations,
        "time": float(result.time),
        "id_dict": list(result.id_dict),
        "width": int(result.width),
        "height": int(result.height),
        "solution_actions": solution_actions,
        "rules_fired": rules_fired_per_t,
        "rules_fired_union": rules_fired_union,
        "n_rules": int(result.n_rules) if track_rules_fired else 0,
    }


# ---------------------------------------------------------------------------
# Pool generation + dataset assembly
# ---------------------------------------------------------------------------

def _bitpacked_to_multihot(
    dat_per_tile: list,
    width: int,
    height: int,
    n_objs: int,
    stride_obj: int,
    raw_to_canonical: dict[int, int] | None = None,
    n_canonical: int | None = None,
) -> np.ndarray:
    """Convert (column-major) tile bitmask flat list to (C, H, W) uint8 multihot.

    Layout matches what the C++ collector / engine returns: flat int32 list
    of length ``w * h * stride_obj`` with ``tile_idx = x * height + y``.
    """
    out_C = n_canonical if raw_to_canonical is not None else n_objs
    obs = np.zeros((out_C, height, width), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            base = (x * height + y) * stride_obj
            for obj_i in range(n_objs):
                word_i = obj_i // 32
                bit = obj_i % 32
                if int(dat_per_tile[base + word_i]) & (1 << bit):
                    c = raw_to_canonical[obj_i] if raw_to_canonical is not None else obj_i
                    obs[c, y, x] = 1
    return obs


def _build_dataset_from_transitions(
    accepted: list[dict],
    n_objs: int,
    stride_obj: int,
    width: int,
    height: int,
) -> dict:
    all_states: list[np.ndarray] = []
    all_next: list[np.ndarray] = []
    all_actions: list[int] = []
    all_wons: list[int] = []
    level_offsets: list[int] = [0]

    for entry in accepted:
        states_raw = entry["states"]
        next_raw = entry["next_states"]
        for s in states_raw:
            all_states.append(
                _bitpacked_to_multihot(s, width, height, n_objs, stride_obj)
            )
        for s in next_raw:
            all_next.append(
                _bitpacked_to_multihot(s, width, height, n_objs, stride_obj)
            )
        all_actions.extend(int(a) for a in entry["actions"])
        all_wons.extend(int(w) for w in entry["wons"])
        level_offsets.append(len(all_actions))

    if not all_states:
        return {
            "states": np.empty((0, n_objs, height, width), dtype=np.uint8),
            "actions": np.empty((0,), dtype=np.int32),
            "next_states": np.empty((0, n_objs, height, width), dtype=np.uint8),
            "wons": np.empty((0,), dtype=np.uint8),
            "level_offsets": np.array([0], dtype=np.int32),
        }
    return {
        "states": np.stack(all_states).astype(np.uint8),
        "actions": np.array(all_actions, dtype=np.int32),
        "next_states": np.stack(all_next).astype(np.uint8),
        "wons": np.array(all_wons, dtype=np.uint8),
        "level_offsets": np.array(level_offsets, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Evolutionary loop: hill-climbs on BFS-iterations fitness to find valid
# levels in games where rejection sampling has very low acceptance.
# Reuses LevelMutator from evolve_level_cpp.py so the mutation primitives
# (swap/place/remove/move + player-count-respecting rejection) stay in one
# place.
# ---------------------------------------------------------------------------

def _eval_candidate_fitness(
    engine: Engine,
    dat: list[int],
    width: int,
    height: int,
    *,
    max_iters: int,
    timeout_ms: int,
    track_rules_fired: bool = False,
    rule_coverage_weight: float = 0.0,
) -> tuple[float, dict | None]:
    """Score a candidate level. Returns (fitness, accepted_payload_or_None).

    Fitness encoding:
      -inf : invalid (already winning, BFS timeout, or 0 iterations)
      iterations + rule_coverage_weight * unique_rules_fired
                  + (1e6 if solvable else 0)

    The rule-coverage term biases evolution toward levels whose reachable
    state space exercises more of the game's rule set — important for games
    like sokoban_match3 where a level can have a huge state space without
    ever triggering the match-3 rule.

    Caller decides via ``require_solvable`` whether non-solvable scoring
    counts toward acceptance.
    """
    backup = LevelBackup(dat, width, height)
    engine.restore_level(backup)
    if engine.check_win():
        return -float("inf"), None
    result = collect_transitions_bfs(
        engine, max_iters=max_iters, timeout_ms=timeout_ms,
        track_rules_fired=track_rules_fired,
    )
    iterations = int(result.iterations)
    if iterations <= 0 or result.timeout:
        return -float("inf"), None
    wons = list(result.wons)
    is_solvable = bool(any(w for w in wons))
    rules_fired_per_t = (
        [list(rf) for rf in result.rules_fired] if track_rules_fired else []
    )
    rules_fired_union: set[int] = set()
    for rf in rules_fired_per_t:
        rules_fired_union.update(rf)
    payload = {
        "states": result.states,
        "actions": list(result.actions),
        "next_states": result.next_states,
        "wons": wons,
        "iterations": iterations,
        "time": float(result.time),
        "id_dict": list(result.id_dict),
        "width": int(result.width),
        "height": int(result.height),
        "is_solvable": is_solvable,
        "rules_fired": rules_fired_per_t,
        "rules_fired_union": sorted(rules_fired_union),
        "n_rules": int(result.n_rules) if track_rules_fired else 0,
    }
    fitness = (
        float(iterations)
        + rule_coverage_weight * len(rules_fired_union)
        + (1e6 if is_solvable else 0.0)
    )
    return fitness, payload


def _compute_pairwise_hamming(
    pop_arr: np.ndarray,
    archive_arr: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (P×N) distance matrix and the N×L matrix it indexes into.

    Distance is per-tile-word Hamming on int64-cast dat sequences (each tile
    holds one or more int32 layer-bitmask words; differing word counts as 1).
    """
    if archive_arr is not None and len(archive_arr) > 0:
        all_arr = np.concatenate([pop_arr, archive_arr], axis=0)
    else:
        all_arr = pop_arr
    P = pop_arr.shape[0]
    # Distances: pop (P, L) vs all (N, L) → (P, N)
    diffs = np.sum(pop_arr[:, None, :] != all_arr[None, :, :], axis=2)
    # Mark self-distance (i, i) so it doesn't enter the kNN.
    for i in range(P):
        diffs[i, i] = np.iinfo(diffs.dtype).max
    return diffs, all_arr


def _nslc_scores(
    pop_dats: list[list[int]],
    pop_fits: list[float],
    archive_dats: list[list[int]],
    archive_fits: list[float],
    k: int,
) -> tuple[list[float], list[int]]:
    """Compute (novelty, local_competition) per pop member.

    novelty[i]            = mean Hamming distance to i's k nearest neighbors
                            in (pop ∪ archive)
    local_competition[i]  = count of those k neighbors that i beats on fitness
    """
    if not pop_dats:
        return [], []
    pop_arr = np.asarray(pop_dats, dtype=np.int64)
    arch_arr = (np.asarray(archive_dats, dtype=np.int64)
                if archive_dats else None)
    diffs, _ = _compute_pairwise_hamming(pop_arr, arch_arr)
    all_fits = np.array(list(pop_fits) + list(archive_fits), dtype=np.float64)
    P = pop_arr.shape[0]
    # Number of available neighbours = N - 1 (excluding self).
    k_eff = min(k, diffs.shape[1] - 1)
    novelties: list[float] = []
    local_comps: list[int] = []
    for i in range(P):
        if k_eff <= 0:
            novelties.append(0.0)
            local_comps.append(0)
            continue
        row = diffs[i]
        idx = np.argpartition(row, k_eff - 1)[:k_eff] if k_eff > 1 \
            else np.array([int(np.argmin(row))])
        novelties.append(float(np.mean(row[idx])))
        local_comps.append(int(np.sum(all_fits[idx] < pop_fits[i])))
    return novelties, local_comps


def _nsga2_2obj_select(scores: list[tuple[float, float]], n: int) -> list[int]:
    """Select ``n`` indices from a list of 2-objective scores (both maximized)
    by non-dominated sorting; the final front is trimmed by crowding distance.
    """
    if not scores or n <= 0:
        return []
    n = min(n, len(scores))
    fronts: list[list[int]] = []
    remaining = set(range(len(scores)))
    while remaining:
        front: list[int] = []
        for i in remaining:
            ai0, ai1 = scores[i]
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                aj0, aj1 = scores[j]
                if (aj0 >= ai0 and aj1 >= ai1
                        and (aj0 > ai0 or aj1 > ai1)):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        if not front:
            front = list(remaining)
        fronts.append(front)
        remaining -= set(front)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= n:
            selected.extend(front)
            continue
        # Crowding distance within this front
        slots = n - len(selected)
        cd = {i: 0.0 for i in front}
        for obj_idx in (0, 1):
            sorted_front = sorted(front, key=lambda i: scores[i][obj_idx])
            lo = scores[sorted_front[0]][obj_idx]
            hi = scores[sorted_front[-1]][obj_idx]
            span = hi - lo if hi > lo else 1.0
            cd[sorted_front[0]] = float("inf")
            cd[sorted_front[-1]] = float("inf")
            for k_i in range(1, len(sorted_front) - 1):
                cd[sorted_front[k_i]] += (
                    scores[sorted_front[k_i + 1]][obj_idx]
                    - scores[sorted_front[k_i - 1]][obj_idx]
                ) / span
        front_by_cd = sorted(front, key=lambda i: -cd[i])
        selected.extend(front_by_cd[:slots])
        break
    return selected


def _evolve_levels(
    engine: Engine,
    json_state: dict,
    gen: "LevelGenerator",
    width: int,
    height: int,
    n_target: int,
    *,
    rng: np.random.Generator,
    require_solvable: bool,
    min_states: int,
    max_iters_search: int,
    timeout_ms_search: int,
    pop_size: int = 64,
    n_mutations_min: int = 1,
    n_mutations_max: int = 3,
    elite_frac: float = 0.5,
    max_generations: int = 200,
    init_dats: list[list[int]] | None = None,
    track_rules_fired: bool = False,
    rule_coverage_weight: float = 0.0,
    coverage_select_topk: bool = False,
    selection: str = "fitness",
    nslc_k: int = 5,
    nslc_archive_size: int = 500,
    verbose: bool = True,
) -> tuple[list[list[int]], list[dict]]:
    """Population-based GA that finds valid levels via fitness =
    BFS-iterations (with a +1e6 bonus for solvability).

    Each generation: evaluate the population, harvest any candidate that
    meets the validity criteria into the accept pool (deduped), then
    select top-K parents and fill the remaining slots with mutated
    offspring. Stops when ``n_target`` accepted candidates exist or the
    generation budget is exhausted.

    Mutations come from evolve_level_cpp.LevelMutator, which guarantees a
    valid player count after each step.
    """
    from evolve_level_cpp import LevelMutator

    mutator = LevelMutator(
        json_state, width=width, height=height,
        allowed_tile_patterns=gen.patterns,
    )
    n_elites = max(1, int(pop_size * elite_frac))

    # Initial population: optionally seeded from authored levels (rotated to
    # fill pop_size); the rest are random tile-pattern samples.
    pop: list[list[int]] = []
    if init_dats:
        for i in range(min(len(init_dats), pop_size)):
            pop.append(list(init_dats[i]))
    while len(pop) < pop_size:
        appended = False
        for _ in range(20):
            d = gen.random_dat(rng, width, height)
            if d is not None and gen.structurally_valid(d, width, height):
                pop.append(d)
                appended = True
                break
        if not appended:
            # Couldn't sample fresh; duplicate an existing pop member.
            if pop:
                pop.append(list(pop[int(rng.integers(len(pop)))]))
            else:
                break
    if not pop:
        return [], []

    accepted_dats: list[list[int]] = []
    accepted_payloads: list[dict] = []
    seen: set[tuple] = set()

    # NSLC archive (only populated when selection == "nslc"): tracks
    # behaviourally-novel pop members from prior generations so novelty is
    # measured against a wider reference set than the current population.
    archive_dats: list[list[int]] = []
    archive_fits: list[float] = []

    t0 = time.time()
    for gen_idx in range(max_generations):
        scored: list[tuple[float, dict | None, list[int]]] = []
        for dat in pop:
            fit, payload = _eval_candidate_fitness(
                engine, dat, width, height,
                max_iters=max_iters_search,
                timeout_ms=timeout_ms_search,
                track_rules_fired=track_rules_fired,
                rule_coverage_weight=rule_coverage_weight,
            )
            scored.append((fit, payload, dat))
            # Acceptance criteria for this level:
            #   - finite fitness (valid dynamics, no timeout)
            #   - >= min_states reachable
            #   - solvable iff require_solvable
            if payload is None:
                continue
            if payload["iterations"] < min_states:
                continue
            if require_solvable and not payload["is_solvable"]:
                continue
            key = tuple(dat)
            if key in seen:
                continue
            seen.add(key)
            accepted_dats.append(list(dat))
            accepted_payloads.append(payload)
            # Early-stop only when not in coverage-select mode. In coverage
            # mode we keep evolving for the full budget so selection pressure
            # has time to push toward higher-coverage levels, then pick the
            # top-K at the end.
            if not coverage_select_topk and len(accepted_dats) >= n_target:
                break

        if not coverage_select_topk and len(accepted_dats) >= n_target:
            break

        # Selection.
        if selection == "nslc":
            # Novelty Search with Local Competition. Elites are the
            # non-dominated front in (novelty, local_competition) space —
            # rewards both structural diversity and per-neighbourhood
            # competence. -inf-fitness candidates can still survive if their
            # behaviour is novel, which is desirable when the population
            # collapses around a single solvable-but-narrow region.
            pop_fits_all = [s[0] for s in scored]
            pop_dats_all = [s[2] for s in scored]
            novelties, local_comps = _nslc_scores(
                pop_dats_all, pop_fits_all,
                archive_dats, archive_fits,
                k=nslc_k,
            )
            scores2obj = list(zip(novelties, local_comps))
            elite_idx = _nsga2_2obj_select(scores2obj, n_elites)
            elites = [pop_dats_all[i] for i in elite_idx]
            # Archive update: append every pop member whose novelty is above
            # the population median this gen (cheap heuristic — keeps the
            # archive biased toward outliers); cap at nslc_archive_size by
            # dropping oldest entries.
            if novelties:
                median_nov = float(np.median(novelties))
                for i, nov in enumerate(novelties):
                    if nov >= median_nov:
                        archive_dats.append(list(pop_dats_all[i]))
                        archive_fits.append(float(pop_fits_all[i]))
                if len(archive_dats) > nslc_archive_size:
                    archive_dats[:] = archive_dats[-nslc_archive_size:]
                    archive_fits[:] = archive_fits[-nslc_archive_size:]
        else:
            # Top-K elites by raw fitness; -inf goes last.
            scored.sort(key=lambda x: x[0], reverse=True)
            elites = [s[2] for s in scored[:n_elites]]
        # If every scored candidate this gen has -inf fitness (total population
        # collapse — no valid dynamics anywhere), reseed with fresh random
        # levels to escape. NSLC keeps novel-but-invalid layouts on purpose, so
        # this safety net only triggers on full collapse, not per-elite.
        if all(s[0] == -float("inf") for s in scored):
            elites = []
            for _ in range(n_elites):
                for _ in range(20):
                    d = gen.random_dat(rng, width, height)
                    if d is not None and gen.structurally_valid(d, width, height):
                        elites.append(d)
                        break

        # Children: mutate random elites
        children: list[list[int]] = []
        n_children = pop_size - len(elites)
        for _ in range(n_children):
            parent = elites[int(rng.integers(len(elites)))]
            n_muts = int(rng.integers(n_mutations_min, n_mutations_max + 1))
            child = mutator.mutate(
                list(parent), rng, n_mutations=n_muts,
                required_player_count=1,
            )
            children.append(child)
        pop = elites + children

        if verbose and (gen_idx % 5 == 0 or len(accepted_dats) >= n_target):
            best = scored[0][0]
            elapsed = time.time() - t0
            extras = ""
            if track_rules_fired:
                # Best level's rule coverage (top-of-population payload).
                top_payload = scored[0][1]
                if top_payload is not None:
                    cov = len(top_payload.get("rules_fired_union", []))
                    n_rules = top_payload.get("n_rules", 0)
                    extras = f"  best_rules={cov}/{n_rules}"
            print(
                f"[evolve] gen {gen_idx}/{max_generations}  "
                f"best_fit={best:.0f}  accepted={len(accepted_dats)}/{n_target}{extras}  "
                f"({elapsed:.1f}s)"
            )

    # Coverage-select-topk: greedy selection over the full pool to maximize
    # *union* coverage (each pick is the level adding the most new rules to
    # the running union, tiebroken by per-level coverage then iterations).
    # This explicitly targets the diversity goal — maximizing distinct rules
    # the dataset exercises — rather than picking by individual fitness.
    if coverage_select_topk and track_rules_fired and len(accepted_dats) > n_target:
        n_pre = len(accepted_dats)
        # Build (idx, fired_set, iters) tuples
        cands = [
            (i,
             frozenset(p.get("rules_fired_union", [])),
             int(p.get("iterations", 0)))
            for i, p in enumerate(accepted_payloads)
        ]
        chosen: list[int] = []
        running_union: set[int] = set()
        remaining = list(cands)
        while len(chosen) < n_target and remaining:
            # Score each remaining cand by (new_rules_added, individual_cov, iters)
            best_idx = -1
            best_key = (-1, -1, -1)
            for ri, (i, fired, iters) in enumerate(remaining):
                new_rules = len(fired - running_union)
                key = (new_rules, len(fired), iters)
                if key > best_key:
                    best_key = key
                    best_idx = ri
            if best_idx < 0:
                break
            i, fired, _ = remaining.pop(best_idx)
            chosen.append(i)
            running_union |= fired
        accepted_dats = [accepted_dats[i] for i in chosen]
        accepted_payloads = [accepted_payloads[i] for i in chosen]
        if verbose:
            print(f"[evolve] coverage-select: kept top-{len(chosen)}/{n_pre} by greedy union coverage")

    if verbose:
        elapsed = time.time() - t0
        cov_summary = ""
        if track_rules_fired and accepted_payloads:
            unions = [set(p.get("rules_fired_union", [])) for p in accepted_payloads]
            n_rules = accepted_payloads[0].get("n_rules", 0)
            mean_cov = (sum(len(u) for u in unions) / max(1, len(unions))) if unions else 0
            global_union = set().union(*unions) if unions else set()
            cov_summary = (
                f" rules-covered: mean-per-level={mean_cov:.1f}/{n_rules},"
                f" union={len(global_union)}/{n_rules}"
            )
        print(
            f"[evolve] DONE: {len(accepted_dats)}/{n_target} accepted "
            f"in {elapsed:.1f}s after {gen_idx+1} generations.{cov_summary}"
        )
    return accepted_dats, accepted_payloads


# ---------------------------------------------------------------------------
# Top-level entry point: generate + collect + cache
# ---------------------------------------------------------------------------

CACHE_VERSION = 10  # bumped: optional rule-firing telemetry + rule-coverage fitness term


def collect_synthetic_dataset(
    game_name: str = "sokoban_basic",
    n_levels: int = 64,
    width: int = 7,
    height: int = 7,
    *,
    seed: int = 0,
    max_iters_search: int = 5000,
    timeout_ms_search: int = 2000,
    min_states: int = 20,
    mode: str = "tile_pattern_empirical",
    require_solvable: bool = False,
    max_attempts_per_level: int = 1000,
    evolve_pop_size: int = 64,
    evolve_max_generations: int = 200,
    evolve_n_mutations_min: int = 1,
    evolve_n_mutations_max: int = 3,
    seed_from_authored: bool = False,
    seed_level_indices: Optional[list[int]] = None,
    fallback_dynamics: bool = False,
    no_a_count_max: int = 3,
    track_rules_fired: bool = False,
    rule_coverage_weight: float = 0.0,
    coverage_select_topk: bool = False,
    selection: str = "fitness",
    nslc_k: int = 5,
    nslc_archive_size: int = 500,
    cache_root: str = "rollout_data",
    verbose: bool = True,
) -> dict:
    """Generate ``n_levels`` valid synthetic levels, collect their transitions,
    cache, and return a dict with:

      states, actions, next_states, wons   — same shape as collect_unique_transitions
      level_offsets                         — int32[N+1] offsets into actions per level
      level_dats                            — (N, w*h*stride) int64 bitfields
      gen_stats                             — JSON-encoded acceptance/timing stats

    If ``require_solvable=True`` produces zero levels (e.g. Zen / kettle at
    small grids — random layouts never reach a winning state within budget)
    and ``fallback_dynamics=True``, retry once with require_solvable=False
    so the multi-game pipeline still gets dynamics-only transitions for the
    hard game instead of silently dropping it.
    """
    cache_dir = os.path.join(cache_root, game_name, f"synthetic_{width}x{height}")
    os.makedirs(cache_dir, exist_ok=True)
    mode_tag = mode if mode != "evolve" else (
        f"evolve-p{evolve_pop_size}-g{evolve_max_generations}"
    )
    if seed_from_authored:
        if seed_level_indices is not None:
            seed_tag = "-sa[" + ",".join(str(i) for i in seed_level_indices) + "]"
        else:
            seed_tag = "-sa"
    else:
        seed_tag = ""
    # Only suffix the cache key with K when it's not the default, to keep
    # existing v8 caches at K=3 readable without rename.
    k_tag = f"_k{no_a_count_max}" if no_a_count_max != 3 else ""
    # Rule-coverage tag: only present when active, so old call sites still hit
    # their existing cache files.
    rc_tag = (
        f"_rc{rule_coverage_weight:g}"
        if (track_rules_fired and rule_coverage_weight != 0.0)
        else ""
    )
    if coverage_select_topk:
        rc_tag += "_cstop"
    sel_tag = (
        f"_sel-nslc-k{nslc_k}-a{nslc_archive_size}"
        if (mode == "evolve" and selection == "nslc") else ""
    )
    cache_path = os.path.join(
        cache_dir,
        f"seed{seed}_n{n_levels}_v{CACHE_VERSION}"
        f"_mode-{mode_tag}{seed_tag}{sel_tag}_solv{int(require_solvable)}"
        f"_mi{max_iters_search}_tmo{timeout_ms_search}_ms{min_states}{k_tag}{rc_tag}.npz",
    )
    if os.path.isfile(cache_path):
        if verbose:
            print(f"[synth] loading cached dataset {cache_path}")
        npz = np.load(cache_path, allow_pickle=True)
        return {k: npz[k] for k in npz.files}

    if verbose:
        print(f"[synth] compiling game={game_name}")
    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    json_str = backend.compile_and_serialize(parser, game_name)
    json_state = json.loads(json_str)

    # The construction sub-mode (only relevant when not in pure-evolve mode).
    construction_mode = (
        "tile_pattern_empirical" if mode == "evolve"
        else mode
    )
    gen = LevelGenerator(json_state, mode=construction_mode,
                         no_a_count_max=no_a_count_max)
    n_objs = gen.n_objs
    stride_obj = gen.stride
    rng = np.random.default_rng(seed)

    engine = Engine()
    engine.load_from_json(json_str)
    engine.load_level(0)  # immediately overwritten by restore_level

    n_attempts = 0
    n_struct_rejects = 0
    n_search_rejects = 0
    t0 = time.time()

    init_dats: list[list[int]] | None = None
    if seed_from_authored:
        seeds = extract_authored_dats(json_state, level_indices=seed_level_indices)
        init_dats = []
        # Detect swarm-style games (authored levels with >1 player tile) so we
        # know whether to enforce "exactly 1 player" on seeds or not.
        max_authored_players = 0
        for src_dat, sw, sh in seeds:
            n_p = gen.count_players(list(src_dat), sw, sh)
            if n_p > max_authored_players:
                max_authored_players = n_p
        is_swarm = max_authored_players > 1
        for src_dat, sw, sh in seeds:
            cropped = crop_dat(
                src_dat, sw, sh, width, height,
                stride_obj=stride_obj, rng=rng,
                bg_idx=int(json_state.get("backgroundid", 0)),
            )
            # If cropping might have removed all players from a swarm-style
            # game, this addition is fine; if cropped landed with 0 players in
            # any game, we add one. We *never* strip extras when the source
            # already has them (would break swarm puzzles).
            if not is_swarm:
                cropped = gen._force_one_player(rng, cropped, width, height, choices=None)
            else:
                if gen.count_players(cropped, width, height) == 0:
                    cropped = gen._force_one_player(rng, cropped, width, height, choices=None)
            # Skip win-constraint correction on authored-derived seeds: the
            # authored level was already designed with valid counts and our
            # corrector may unnecessarily drop tiles after cropping.
            if cropped is not None:
                init_dats.append(cropped)
        if verbose:
            print(f"[synth] seeded with {len(init_dats)}/{len(seeds)} authored-level "
                  f"crops at ({width}x{height})"
                  f"{' [swarm-style game detected]' if is_swarm else ''}.")

    if mode == "evolve":
        accepted_dats, accepted_results = _evolve_levels(
            engine, json_state, gen, width, height, n_levels,
            rng=rng,
            require_solvable=require_solvable,
            min_states=min_states,
            max_iters_search=max_iters_search,
            timeout_ms_search=timeout_ms_search,
            pop_size=evolve_pop_size,
            n_mutations_min=evolve_n_mutations_min,
            n_mutations_max=evolve_n_mutations_max,
            max_generations=evolve_max_generations,
            init_dats=init_dats,
            track_rules_fired=track_rules_fired,
            rule_coverage_weight=rule_coverage_weight,
            coverage_select_topk=coverage_select_topk,
            selection=selection,
            nslc_k=nslc_k,
            nslc_archive_size=nslc_archive_size,
            verbose=verbose,
        )
        # Bookkeeping: in evolve mode "attempts" tracks total BFS calls.
        n_attempts = evolve_pop_size * evolve_max_generations
    else:
        accepted_dats = []
        accepted_results = []
        seen_dat_hashes: set[tuple] = set()
        target_attempts_total = max(n_levels, 1) * max_attempts_per_level
        while len(accepted_dats) < n_levels and n_attempts < target_attempts_total:
            n_attempts += 1
            dat = gen.random_dat(rng, width, height)
            if dat is None or not gen.structurally_valid(dat, width, height):
                n_struct_rejects += 1
                continue
            h_key = tuple(dat)
            if h_key in seen_dat_hashes:
                continue
            result = search_validate(
                engine, dat, width, height,
                max_iters=max_iters_search,
                timeout_ms=timeout_ms_search,
                min_states=min_states,
                require_solvable=require_solvable,
                track_rules_fired=track_rules_fired,
            )
            if result is None:
                n_search_rejects += 1
                continue
            seen_dat_hashes.add(h_key)
            accepted_dats.append(dat)
            accepted_results.append(result)
            if verbose and len(accepted_dats) % max(1, n_levels // 10) == 0:
                elapsed = time.time() - t0
                print(
                    f"[synth] {len(accepted_dats)}/{n_levels} accepted "
                    f"({n_attempts} attempts, {n_struct_rejects} struct-rej, "
                    f"{n_search_rejects} search-rej, {elapsed:.1f}s)"
                )

    elapsed = time.time() - t0
    if len(accepted_dats) < n_levels:
        print(
            f"[synth] WARNING: only got {len(accepted_dats)}/{n_levels} valid levels "
            f"in {elapsed:.1f}s. "
            "Consider raising max_attempts_per_level (rejection mode), "
            "evolve_max_generations (evolve mode), or relaxing min_states."
        )

    dataset = _build_dataset_from_transitions(
        accepted_results,
        n_objs=n_objs,
        stride_obj=stride_obj,
        width=width,
        height=height,
    )
    dat_arr = (np.array(accepted_dats, dtype=np.int64)
               if accepted_dats
               else np.zeros((0, width * height * stride_obj), dtype=np.int64))
    dataset["level_dats"] = dat_arr

    # Per-level optimal solution actions (BFS-found during search_validate
    # for solvable levels; empty list for dynamics-only levels). Stored as
    # an object array of variable-length int lists — np.savez handles this.
    sol_acts = [
        np.array(r.get("solution_actions", []), dtype=np.int32)
        for r in accepted_results
    ]
    dataset["level_solutions"] = (
        np.array(sol_acts, dtype=object) if sol_acts
        else np.zeros((0,), dtype=object)
    )

    n_solvable_levels = sum(
        1 for r in accepted_results if any(int(w) for w in r["wons"])
    )
    # Rule-coverage rollup (when tracking on).
    rule_coverage_stats: dict = {}
    if track_rules_fired and accepted_results:
        per_level = [set(r.get("rules_fired_union", [])) for r in accepted_results]
        n_rules_total = int(accepted_results[0].get("n_rules", 0))
        global_union = set().union(*per_level) if per_level else set()
        rule_coverage_stats = {
            "track_rules_fired": True,
            "rule_coverage_weight": rule_coverage_weight,
            "n_rules_total": n_rules_total,
            "n_rules_covered_dataset_union": len(global_union),
            "rules_covered_dataset_union": sorted(global_union),
            "n_rules_covered_per_level_mean": (
                sum(len(s) for s in per_level) / max(1, len(per_level))
            ),
            "n_rules_covered_per_level_min": min((len(s) for s in per_level), default=0),
        }
    gen_stats = {
        "game_name": game_name,
        "n_levels_target": n_levels,
        "n_levels_accepted": len(accepted_dats),
        "n_levels_solvable": n_solvable_levels,
        "n_attempts": n_attempts,
        "n_struct_rejects": n_struct_rejects,
        "n_search_rejects": n_search_rejects,
        "elapsed_s": elapsed,
        "n_transitions": int(dataset["actions"].shape[0]),
        "n_winning_transitions": int(dataset["wons"].sum()) if dataset["wons"].size else 0,
        "id_dict": list(json_state["idDict"]),
        "stride_obj": stride_obj,
        "n_objs": n_objs,
        "mode": mode,
        "min_states": min_states,
        "require_solvable": require_solvable,
        "max_iters_search": max_iters_search,
        "timeout_ms_search": timeout_ms_search,
        "seed": seed,
        **rule_coverage_stats,
    }
    dataset["gen_stats"] = np.array([json.dumps(gen_stats)], dtype=object)
    if verbose:
        print(
            f"[synth] DONE: {gen_stats['n_levels_accepted']}/{n_levels} levels "
            f"({n_solvable_levels} solvable), "
            f"{gen_stats['n_transitions']:,} transitions "
            f"({gen_stats['n_winning_transitions']:,} winning) in {elapsed:.1f}s "
            f"(struct-rej={n_struct_rejects}, search-rej={n_search_rejects})"
        )

    np.savez_compressed(cache_path, **dataset)
    if verbose:
        print(f"[synth] cached -> {cache_path}")

    # Optional fallback: if require_solvable=True yielded zero levels, retry
    # once with require_solvable=False to at least surface dynamics-only data
    # for downstream multi-game training.
    if (require_solvable and fallback_dynamics
            and len(accepted_dats) == 0):
        if verbose:
            print(
                f"[synth] {game_name}: 0 levels with require_solvable=True; "
                f"falling back to dynamics-only (require_solvable=False)"
            )
        return collect_synthetic_dataset(
            game_name=game_name,
            n_levels=n_levels, width=width, height=height,
            seed=seed,
            max_iters_search=max_iters_search,
            timeout_ms_search=timeout_ms_search,
            min_states=min_states,
            mode=mode,
            require_solvable=False,  # the fallback
            max_attempts_per_level=max_attempts_per_level,
            evolve_pop_size=evolve_pop_size,
            evolve_max_generations=evolve_max_generations,
            evolve_n_mutations_min=evolve_n_mutations_min,
            evolve_n_mutations_max=evolve_n_mutations_max,
            seed_from_authored=seed_from_authored,
            seed_level_indices=seed_level_indices,
            fallback_dynamics=False,  # don't recurse
            no_a_count_max=no_a_count_max,
            track_rules_fired=track_rules_fired,
            rule_coverage_weight=rule_coverage_weight,
            coverage_select_topk=coverage_select_topk,
            selection=selection,
            nslc_k=nslc_k,
            nslc_archive_size=nslc_archive_size,
            cache_root=cache_root,
            verbose=verbose,
        )
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", default="sokoban_basic")
    ap.add_argument("--n_levels", type=int, default=64)
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--height", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--timeout_ms", type=int, default=2000)
    ap.add_argument("--min_states", type=int, default=20)
    ap.add_argument("--mode", default="tile_pattern_empirical",
                    choices=["tile_pattern_empirical", "tile_pattern_uniform", "evolve"])
    ap.add_argument("--require_solvable", action="store_true",
                    help="Reject levels with no winning transition reachable within budget")
    ap.add_argument("--max_attempts_per_level", type=int, default=1000)
    ap.add_argument("--evolve_pop_size", type=int, default=64)
    ap.add_argument("--evolve_max_generations", type=int, default=200)
    ap.add_argument("--evolve_n_mutations_min", type=int, default=1)
    ap.add_argument("--evolve_n_mutations_max", type=int, default=3)
    ap.add_argument("--seed_from_authored", action="store_true",
                    help="Seed initial population from cropped authored levels.")
    ap.add_argument("--no_a_count_max", type=int, default=3,
                    help="Cap count(A) at start for 'no A' (num=-1) win conditions.")
    ap.add_argument("--track_rules_fired", action="store_true",
                    help="Collect per-transition rule-firing telemetry from the engine.")
    ap.add_argument("--rule_coverage_weight", type=float, default=0.0,
                    help="Fitness term: weight per unique rule fired in a level's reachable "
                         "state space. Implies --track_rules_fired when > 0.")
    ap.add_argument("--coverage_select_topk", action="store_true",
                    help="Disable early stop: evolve full budget then greedily select top-K "
                         "by union rule coverage. Implies --track_rules_fired.")
    args = ap.parse_args()
    track_rules = (
        args.track_rules_fired
        or args.rule_coverage_weight > 0.0
        or args.coverage_select_topk
    )
    collect_synthetic_dataset(
        game_name=args.game,
        n_levels=args.n_levels,
        width=args.width,
        height=args.height,
        seed=args.seed,
        max_iters_search=args.max_iters,
        timeout_ms_search=args.timeout_ms,
        min_states=args.min_states,
        mode=args.mode,
        require_solvable=args.require_solvable,
        max_attempts_per_level=args.max_attempts_per_level,
        evolve_pop_size=args.evolve_pop_size,
        evolve_max_generations=args.evolve_max_generations,
        evolve_n_mutations_min=args.evolve_n_mutations_min,
        evolve_n_mutations_max=args.evolve_n_mutations_max,
        seed_from_authored=args.seed_from_authored,
        no_a_count_max=args.no_a_count_max,
        track_rules_fired=track_rules,
        rule_coverage_weight=args.rule_coverage_weight,
        coverage_select_topk=args.coverage_select_topk,
        verbose=True,
    )


if __name__ == "__main__":
    _main()

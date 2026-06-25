"""Engine-backed synthetic world family for the active-learning probe.

A *world* theta is one of a few PuzzleScript rulesets ("mechanisms") applied to a
shared object family and a shared set of level layouts. The mechanisms differ
ONLY in dynamics, never in objects or initial observation, so theta is hidden and
must be inferred by interaction — the PuzzleScript analog of the doc's TV world.

Mechanisms (each fires every turn; validated to diverge in check_double_step):
  - noop : ``[ ObjA ] -> [ ObjA ]``          identity; nothing changes
  - det  : ``[ ObjA ] -> [ ObjB ]``          ALL seeds sprout (deterministic)
  - rand : ``random [ ObjA ] -> [ ObjB ]``   ONE random seed sprouts (stochastic)

ObjA == "seed", ObjB == "sprout" in the canonical vocab family.

Compilation goes through the slow JS->C++ path once per mechanism (all layouts
baked into one game's LEVELS), then the serialized JSON is cached to disk; all
rollout/stepping is pure C++ (seed_rng + backup/restore for i.i.d. redraws).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nca_wm import game_curriculum as gc
from nca_wm import rule_game, rule_gp
from nca_wm.active_learning import vocab as V

_REPO = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO / "nca_wm" / "active_learning" / "_world_cache"
_MAT_DIR = _REPO / "nca_wm" / "active_learning" / "_world_games"

# Map engine id_dict names -> canonical family names (vocab.CANONICAL_OBJECTS).
_ENGINE_TO_CANON = {
    "background": "background",
    "wall": "wall",
    "player": "player",
    "obja": "seed",
    "objb": "sprout",
}

MECHANISMS = ("noop", "det", "rand")

# Sokoban variants: theta = which push rule. Box == ObjA (canonical "seed" slot);
# Wall/Player/Background are rule_game boilerplate. Validated by
# check_sokoban_variants.py.
SOKOBAN_VARIANTS = ("classic", "inert", "slide", "swap", "chaos")
_SOKOBAN_RULES = {
    "classic": ["[ > Player | ObjA ] -> [ > Player | > ObjA ]"],
    "inert": [],
    "slide": ["[ > Player | ObjA ] -> [ > Player | > ObjA ]",
              "[ > ObjA | no Wall no ObjA ] -> [ | > ObjA ] again"],
    "swap": ["[ > Player | ObjA ] -> [ ObjA | Player ]"],
    "chaos": ["[ > Player | ObjA ] -> [ Player | randomDir ObjA ]"],
}


class _Raw:
    """Minimal rule with .unparse() so raw PuzzleScript strings pass to assemble."""
    def __init__(self, s: str):
        self.s = s
    def unparse(self) -> str:
        return self.s


def form_mechanisms(form: str) -> tuple[str, ...]:
    return SOKOBAN_VARIANTS if form == "sokoban" else MECHANISMS


def _mech_rules(mech: str, form: str = "every_turn"):
    if form == "sokoban":
        return [_Raw(s) for s in _SOKOBAN_RULES[mech]]
    return _mech_rules_grid(mech, form)


def _mech_rules_grid(mech: str, form: str = "every_turn") -> list[rule_gp.Rule]:
    """Rules for `mech` in either `every_turn` or `adjacency` form.

    every_turn: ``[ ObjA ] -> ...`` fires regardless of the player (random policy
        already covers all informative transitions — for IG-probe validation).
    adjacency: ``[ Player | ObjA ] -> ...`` fires only when the player is next to
        a seed (informative transitions are sparse — for the navigation /
        online-vs-offline experiment).
    """
    def cell(*objs: str) -> rule_gp.Cell:
        return rule_gp.Cell([rule_gp.CellContent(o) for o in objs])

    if form == "every_turn":
        lhs = [rule_gp.RulePart(cells=[cell("ObjA")])]
        rhs_a = [rule_gp.RulePart(cells=[cell("ObjA")])]
        rhs_b = [rule_gp.RulePart(cells=[cell("ObjB")])]
    elif form == "adjacency":
        lhs = [rule_gp.RulePart(cells=[cell("Player"), cell("ObjA")])]
        rhs_a = [rule_gp.RulePart(cells=[cell("Player"), cell("ObjA")])]
        rhs_b = [rule_gp.RulePart(cells=[cell("Player"), cell("ObjB")])]
    elif form == "dir_adjacency":
        # Fires only when the player MOVES RIGHT into a seed (`>` = rightward
        # motion), so a single informative action exists per state — a greedy
        # depth-1 IG planner can pick it without multi-step lookahead.
        def rcell(*objs):
            return rule_gp.Cell([rule_gp.CellContent(objs[0], ">")] +
                                [rule_gp.CellContent(o) for o in objs[1:]])
        lhs = [rule_gp.RulePart(cells=[rcell("Player"), cell("ObjA")])]
        rhs_a = [rule_gp.RulePart(cells=[rcell("Player"), cell("ObjA")])]
        rhs_b = [rule_gp.RulePart(cells=[rcell("Player"), cell("ObjB")])]
    else:
        raise ValueError(form)

    if mech == "noop":
        return [rule_gp.Rule(lhs=lhs, rhs_parts=rhs_a)]
    if mech == "det":
        return [rule_gp.Rule(lhs=lhs, rhs_parts=rhs_b)]
    if mech == "rand":
        return [rule_gp.set_prefix(rule_gp.Rule(lhs=lhs, rhs_parts=rhs_b), "random")]
    raise ValueError(mech)


def _grid_to_str(grid: list[str], W: int) -> str:
    H = len(grid) // W
    return "\n".join("".join(grid[y * W:(y + 1) * W]) for y in range(H)) + "\n"


def make_layouts(n_layouts: int, n_seeds: int, seed: int,
                 style: str = "scatter") -> list[str]:
    """`n_layouts` fixed level strings (shared across mechanisms).

    style=scatter: Player + `n_seeds` ObjA at random distinct cells.
    style=far_cluster: Player in one corner region, a contiguous cluster of
        `n_seeds` ObjA in the opposite region — so reaching a seed requires
        navigation (informative transitions sparse under a random policy).
    """
    r = random.Random(seed)
    H, W = V.GRID_H, V.GRID_W
    layouts = []
    for _ in range(n_layouts):
        grid = ["."] * (H * W)
        if style == "scatter":
            cells = r.sample(range(H * W), 1 + n_seeds)
            grid[cells[0]] = "P"
            for c in cells[1:]:
                grid[c] = "A"
        elif style == "box_pushable":
            # Bordered room; player and one box on the same row, player >=2 cells
            # left of the box, with >=2 open cells right of the box (so slide is
            # distinguishable and identification needs navigation). Box maps to
            # the canonical "seed" slot.
            for c in range(H * W):
                gx, gy = c % W, c // W
                if gx == 0 or gy == 0 or gx == W - 1 or gy == H - 1:
                    grid[c] = "#"
            y = r.randint(1, H - 2)
            box_x = r.randint(3, W - 3)          # >=2 open cells to the right
            player_x = r.randint(1, box_x - 2)   # >=2 cells left, not adjacent
            grid[y * W + player_x] = "P"
            grid[y * W + box_x] = "A"
        elif style == "left_of_cluster":
            # Player immediately left of a horizontal seed cluster (same row), so
            # one RIGHT move triggers `[ > Player | Seed ]`. For dir_adjacency.
            y = r.randrange(H)
            cx0 = r.randint(2, max(2, W - n_seeds))
            grid[y * W + (cx0 - 1)] = "P"
            for k in range(n_seeds):
                if cx0 + k < W:
                    grid[y * W + (cx0 + k)] = "A"
        elif style == "far_cluster":
            # Player in left third, seed cluster in right third.
            px = r.randrange(0, max(1, W // 3))
            py = r.randrange(H)
            grid[py * W + px] = "P"
            placed = 0
            cx0 = W - max(1, W // 3)
            cand = [(x, y) for y in range(H) for x in range(cx0, W)]
            r.shuffle(cand)
            for (x, y) in cand:
                if placed >= n_seeds:
                    break
                grid[y * W + x] = "A"
                placed += 1
        else:
            raise ValueError(style)
        layouts.append(_grid_to_str(grid, W))
    return layouts


def _compile_mechanism(mech: str, layouts: list[str], form: str) -> str:
    """Return serialized engine JSON for `mech` with all `layouts` baked in."""
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_jax.utils import init_ps_lark_parser

    code = rule_game.assemble_game(
        _mech_rules(mech, form), title=f"al_{mech}", objects=["ObjA", "ObjB"],
        levels=layouts)
    gc._set_materialize_dir(_MAT_DIR)
    gc._materialize_game(f"al_world_{form}_{mech}", code)
    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    return backend.compile_and_serialize(parser, f"al_world_{form}_{mech}")


@dataclass
class WorldFamily:
    """Compiled, cached family: per-mechanism JSON + shared layouts."""
    jsons: dict[str, str]
    n_layouts: int
    form: str
    grid_h: int
    grid_w: int

    @property
    def mechanisms(self) -> tuple[str, ...]:
        return tuple(self.jsons.keys())

    def activate_geometry(self) -> None:
        """Set the global vocab geometry to match this family."""
        V.set_geometry(self.grid_h, self.grid_w)


def build_family(n_layouts: int = 16, n_seeds: int = 3, seed: int = 0,
                 form: str = "every_turn", grid_h: int = 3, grid_w: int = 5,
                 style: str = "scatter", rebuild: bool = False,
                 custom_levels: list[str] | None = None,
                 tag_suffix: str = "") -> WorldFamily:
    """Compile (or load cached) the world family and activate its geometry.

    `custom_levels` overrides the generated layouts (each a GRID_H x GRID_W grid
    string of '.', 'P', 'A'); `tag_suffix` disambiguates the on-disk cache.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    V.set_geometry(grid_h, grid_w)
    tag = f"{form}_{style}_L{n_layouts}_S{n_seeds}_s{seed}_g{grid_h}x{grid_w}{tag_suffix}"
    layouts = custom_levels if custom_levels is not None \
        else make_layouts(n_layouts, n_seeds, seed, style=style)
    jsons: dict[str, str] = {}
    for mech in form_mechanisms(form):
        path = _CACHE_DIR / f"{mech}_{tag}.json"
        if path.exists() and not rebuild:
            jsons[mech] = path.read_text()
        else:
            js = _compile_mechanism(mech, layouts, form)
            path.write_text(js)
            jsons[mech] = js
    return WorldFamily(jsons=jsons, n_layouts=len(layouts), form=form,
                       grid_h=grid_h, grid_w=grid_w)


# --- Engine instantiation + observation -----------------------------------
def _new_engine(json_str: str, level_i: int):
    from puzzlescript_cpp._puzzlescript_cpp import Engine
    e = Engine()
    e.load_from_json(json_str)
    e.load_level(level_i)
    return e


def _engine_id_to_canon_bit(engine) -> dict[int, int]:
    """engine object id -> canonical bit, by name (skip unmapped)."""
    out = {}
    for oid, name in enumerate(engine.get_id_dict()):
        canon = _ENGINE_TO_CANON.get(name)
        if canon is not None:
            out[oid] = V.NAME_TO_BIT[canon]
    return out


def read_obs(engine, id_to_bit: dict[int, int]) -> list[int]:
    """(W,H,1) engine bitmask grid -> row-major list of canonical cell masks."""
    a = np.asarray(engine.get_objects_2d(), dtype=np.int64)  # (W,H,stride)
    assert a.shape[-1] == 1, f"expected stride 1, got {a.shape}"
    grid_wh = a[:, :, 0]  # (W,H)
    H, W = V.GRID_H, V.GRID_W
    assert grid_wh.shape == (W, H), (grid_wh.shape, (W, H))
    masks = []
    for y in range(H):
        for x in range(W):
            engine_mask = int(grid_wh[x, y])
            canon = 0
            for oid, bit in id_to_bit.items():
                if engine_mask & (1 << oid):
                    canon |= (1 << bit)
            masks.append(canon)
    return masks


def step_engine(engine, action: str, seed: str | None = None,
                max_again: int = 50) -> None:
    """Apply one action (resolving `again`), optionally re-seeding the RNG first."""
    if seed is not None:
        engine.seed_rng(seed)
    engine.process_input(V.ACTION_TO_INPUT[action])
    n = 0
    while engine.is_againing() and n < max_again:
        engine.process_input(-1)
        n += 1

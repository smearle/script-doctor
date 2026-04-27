"""Tokenize a PuzzleScript game's mechanics into a domain-general integer sequence.

Converts collision layers, legend groups, rules, and win conditions into a
flat token sequence suitable for encoding by a transformer or RNN.

The vocabulary describes *typed grid rewrite rules* — a general formalism
that covers cellular automata, push puzzles, match-3, gravity games, etc.
The tokenization is name-invariant: objects are referenced by their channel
index in the multihot state (ch0, ch1, ...), and legend OR/AND groups get
their own group indices (g0, g1, ...).

Usage:
    from tokenize_game import tokenize_game, VOCAB
    tokens = tokenize_game(tree, canonical_ids)
    # tokens is a list of int token IDs
"""

from puzzlescript_jax.ps_game import PSGameTree, Rule, RuleBlock, LegendEntry

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Special tokens
_SPECIAL = [
    "PAD",          # 0 — padding
    "SEP",          # section/rule separator
]

# Structure tokens
_STRUCTURE = [
    "LAYER",        # collision layer declaration
    "GROUP_OR",     # legend OR group definition
    "GROUP_AND",    # legend AND group definition
]

# Rule tokens
_RULE = [
    "RULE",         # rule start
    "LHS",          # left-hand side
    "RHS",          # right-hand side
    "CELL_SEP",     # | between cells
    "ELLIPSIS",     # ... (wildcard gap)
    "LOOP_START",
    "LOOP_END",
]

# Direction prefixes
_DIRECTIONS = [
    "DIR_UP", "DIR_DOWN", "DIR_LEFT", "DIR_RIGHT",
    "DIR_HORIZONTAL", "DIR_VERTICAL",
    "DIR_PERPENDICULAR", "DIR_PARALLEL",
]

# Other rule prefixes
_PREFIXES = [
    "LATE", "RANDOM", "RIGID",
]

# Object modifiers (within a cell)
_MODIFIERS = [
    "MOD_PUSH",       # > (moving in rule direction)
    "MOD_UP",         # ^
    "MOD_DOWN",       # v
    "MOD_LEFT",       # <
    "MOD_NO",         # no (absence check)
    "MOD_RANDOM",     # random
    "MOD_RANDOMDIR",  # randomdir
    "MOD_STATIONARY", # stationary
]

# Commands
_COMMANDS = [
    "CMD_AGAIN",
    "CMD_CANCEL",
    "CMD_CHECKPOINT",
    "CMD_RESTART",
    "CMD_WIN",
]

# Win condition tokens
_WINCOND = [
    "WIN",          # win condition start
    "WC_ALL",
    "WC_SOME",
    "WC_NO",
    "WC_ANY",
    "WC_ON",
]

# Prelude flags
_PRELUDE = [
    "PRE_NOACTION",
    "PRE_REQUIRE_PLAYER_MOVEMENT",
    "PRE_RUN_RULES_ON_LEVEL_START",
]

# Channel references: ch0..ch_MAX
MAX_CHANNELS = 64
_CHANNELS = [f"CH{i}" for i in range(MAX_CHANNELS)]

# Group references: g0..g_MAX
MAX_GROUPS = 32
_GROUPS = [f"G{i}" for i in range(MAX_GROUPS)]

# ---- Sprite/visual tokens (opt-in via `encode_sprites=True`) ----
# Each PuzzleScript object declares:
#   - a list of 1..N colors (the object's local palette). We resolve named
#     colors + hex codes to RGB then quantize to a 4x4x4 = 64-bucket grid.
#   - a 5x5 sprite grid of digits 0-9 (indexing into that palette) or '.'
# These tokens let us fold that info into the sequence with a bounded vocab.
_SPRITE_STRUCT = [
    "OBJ_START",        # start of an object's palette+sprite block
    "OBJ_END",          # end of the block
    "PALETTE_START",    # begin palette-color list
    "SPRITE_START",     # begin 5x5 grid
    "PIXEL_ROW_SEP",    # end-of-row separator inside sprite grid
]

# Palette-index tokens: 0-9 plus TRANSPARENT ('.')
_SPRITE_PIX = [f"PIX{i}" for i in range(10)] + ["PIX_TRANSPARENT"]

# Color tokens: RGB quantized to 2 bits per channel → 64 buckets.
# (alpha handled via the separate PIX_TRANSPARENT palette-index token.)
COLOR_Q_BITS = 2
COLOR_Q_LEVELS = 1 << COLOR_Q_BITS          # 4 levels per channel
COLOR_Q_TOTAL = COLOR_Q_LEVELS ** 3         # 64 buckets
_SPRITE_CLRS = [f"COLOR_Q{i:02d}" for i in range(COLOR_Q_TOTAL)]

# Build vocabulary in two stacked layers:
#   1. _ALL_TOKENS_BASE (141): mechanics-only. Token IDs identical to
#      pre-sprite-tokens tokenizer — existing checkpoints stay compatible.
#   2. Sprite tokens appended at the end so their IDs sit above
#      VOCAB_SIZE_BASE. Only emitted when `encode_sprites=True`.
# Callers that know they want the extended vocab pass VOCAB_SIZE_EXT.
_ALL_TOKENS_BASE = (
    _SPECIAL + _STRUCTURE + _RULE + _DIRECTIONS + _PREFIXES +
    _MODIFIERS + _COMMANDS + _WINCOND + _PRELUDE + _CHANNELS + _GROUPS
)
_ALL_TOKENS_EXT = _ALL_TOKENS_BASE + _SPRITE_STRUCT + _SPRITE_PIX + _SPRITE_CLRS

VOCAB = {tok: i for i, tok in enumerate(_ALL_TOKENS_EXT)}
VOCAB_SIZE_BASE = len(_ALL_TOKENS_BASE)          # 141 — legacy compat
VOCAB_SIZE_EXT = len(_ALL_TOKENS_EXT)             # ~183 — with sprite tokens
VOCAB_SIZE = VOCAB_SIZE_BASE                      # default for legacy callers
INV_VOCAB = {i: tok for tok, i in VOCAB.items()}


# ---------------------------------------------------------------------------
# Modifier / prefix mapping
# ---------------------------------------------------------------------------

_MODIFIER_MAP = {
    ">": "MOD_PUSH",
    "^": "MOD_UP",
    "v": "MOD_DOWN",
    "<": "MOD_LEFT",
    "up": "MOD_UP",
    "down": "MOD_DOWN",
    "left": "MOD_LEFT",
    "right": "MOD_PUSH",
    "no": "MOD_NO",
    "random": "MOD_RANDOM",
    "randomdir": "MOD_RANDOMDIR",
    "stationary": "MOD_STATIONARY",
    "horizontal": "DIR_HORIZONTAL",
    "vertical": "DIR_VERTICAL",
    "perpendicular": "DIR_PERPENDICULAR",
    "parallel": "DIR_PARALLEL",
    "moving": "MOD_PUSH",         # alias used in some games
    "orthogonal": "DIR_HORIZONTAL",  # rare alias
}

_PREFIX_MAP = {
    "up": "DIR_UP",
    "down": "DIR_DOWN",
    "left": "DIR_LEFT",
    "right": "DIR_RIGHT",
    "horizontal": "DIR_HORIZONTAL",
    "vertical": "DIR_VERTICAL",
    "perpendicular": "DIR_PERPENDICULAR",
    "parallel": "DIR_PARALLEL",
    "late": "LATE",
    "random": "RANDOM",
    "rigid": "RIGID",
}

_QUANTIFIER_MAP = {
    "all": "WC_ALL",
    "some": "WC_SOME",
    "no": "WC_NO",
    "any": "WC_ANY",
}


# ---------------------------------------------------------------------------
# Color resolution + quantization for sprite tokens
# ---------------------------------------------------------------------------
# All name→hex resolution goes through puzzlescript_jax.colors, which parses
# the JS source of truth (PuzzleScript/src/js/colors.js) at import time.
from puzzlescript_jax.colors import resolve_color_to_rgb as _resolve_name_or_hex


def _resolve_color_to_rgb(s: str, palette_name: str | None = None):
    """Thin wrapper around puzzlescript_jax.colors.resolve_color_to_rgb."""
    return _resolve_name_or_hex(s, palette_name=palette_name)


def _quantize_rgb(rgb: tuple[int, int, int]) -> int:
    """Map (r,g,b) ∈ 0..255 to a bucket index in [0, COLOR_Q_TOTAL)."""
    shift = 8 - COLOR_Q_BITS   # e.g. 6 for 2-bit quantization
    r_q = rgb[0] >> shift
    g_q = rgb[1] >> shift
    b_q = rgb[2] >> shift
    return (r_q * COLOR_Q_LEVELS + g_q) * COLOR_Q_LEVELS + b_q


def color_q_to_rgb(bucket: int) -> tuple[int, int, int]:
    """Inverse of _quantize_rgb, returning the bucket's CENTER RGB in 0..255."""
    levels = COLOR_Q_LEVELS
    r_q = (bucket // (levels * levels)) % levels
    g_q = (bucket // levels) % levels
    b_q = bucket % levels
    # Map quantized coord 0..L-1 to center of its bucket on 0..255 range
    step = 256 // levels
    return (r_q * step + step // 2,
            g_q * step + step // 2,
            b_q * step + step // 2)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize_game(
    tree: PSGameTree,
    canonical_ids: list[str],
    encode_sprites: bool = False,
) -> list[int]:
    """Tokenize a PSGameTree into a sequence of integer token IDs.

    Args:
        tree: Parsed game tree (from js_bridge.parsed_state_to_tree or GenPSTree).
        canonical_ids: list of canonical object names in channel order,
            as returned by CppPuzzleScriptEnv._canonical_ids.
            canonical_ids[i] is the object name for multihot channel i.
        encode_sprites: if True, prepend each object's palette + 5x5 sprite
            grid to the token sequence. See _SPRITE_STRUCT / _SPRITE_PIX /
            _SPRITE_CLRS for the vocab extension. Uses VOCAB_SIZE_EXT.

    Returns:
        List of integer token IDs.
    """
    tokens: list[int] = []
    V = VOCAB  # shorthand

    # Build name → channel index mapping
    name_to_ch: dict[str, int] = {}
    for ch_i, name in enumerate(canonical_ids):
        name_to_ch[name.lower()] = ch_i

    # Build legend name → group index mapping, and identify which legend
    # entries are OR/AND groups (not simple aliases)
    legend: dict[str, LegendEntry] = {}
    if isinstance(tree.legend, dict):
        legend = tree.legend
    elif isinstance(tree.legend, list):
        for le in tree.legend:
            if isinstance(le, LegendEntry):
                legend[le.key] = le

    group_map: dict[str, int] = {}  # legend key → group index
    group_idx = 0
    for key, le in legend.items():
        if le.operator is not None:  # OR or AND group
            group_map[key] = group_idx
            group_idx += 1

    # Helper to resolve a name to channel token(s) or a group token
    def _resolve_name(name: str) -> list[int]:
        """Resolve an object/legend name to token(s)."""
        name_l = name.lower()
        # Direct channel reference
        if name_l in name_to_ch:
            ch = name_to_ch[name_l]
            if ch < MAX_CHANNELS:
                return [V[f"CH{ch}"]]
            return []  # skip if beyond max
        # Legend group reference
        if name_l in group_map:
            gi = group_map[name_l]
            if gi < MAX_GROUPS:
                return [V[f"G{gi}"]]
            return []
        # Legend alias (single object synonym) — resolve to channel
        if name_l in legend:
            le = legend[name_l]
            if le.operator is None and len(le.obj_names) == 1:
                return _resolve_name(le.obj_names[0])
        # Unknown — skip
        return []

    # --- Prelude flags ---
    if tree.prelude.noaction:
        tokens.append(V["PRE_NOACTION"])
    if tree.prelude.require_player_movement:
        tokens.append(V["PRE_REQUIRE_PLAYER_MOVEMENT"])
    if tree.prelude.run_rules_on_level_start:
        tokens.append(V["PRE_RUN_RULES_ON_LEVEL_START"])
    if tokens:
        tokens.append(V["SEP"])

    # --- Sprites (opt-in) ---
    if encode_sprites:
        # One block per canonical object, in channel order. Each block:
        #   OBJ_START CH{i} PALETTE_START COLOR_Q* ... SPRITE_START PIX* ... OBJ_END
        for ch_i, name in enumerate(canonical_ids):
            if ch_i >= MAX_CHANNELS:
                break
            obj = tree.objects.get(name) or tree.objects.get(name.lower())
            if obj is None:
                continue
            tokens.append(V["OBJ_START"])
            tokens.append(V[f"CH{ch_i}"])

            # Palette: list of quantized color tokens.
            tokens.append(V["PALETTE_START"])
            colors = obj.colors if obj.colors is not None else []
            for c in colors:
                rgb = _resolve_color_to_rgb(c)
                if rgb is None:
                    # Transparent or unresolvable — reuse PIX_TRANSPARENT
                    # at the palette level rather than adding more vocab.
                    tokens.append(V["PIX_TRANSPARENT"])
                else:
                    tokens.append(V[f"COLOR_Q{_quantize_rgb(rgb):02d}"])

            # Sprite grid (5x5 typically; shorter/odd sprites handled).
            tokens.append(V["SPRITE_START"])
            sprite = obj.sprite if obj.sprite is not None else []
            n_rows = len(sprite)
            for r in range(n_rows):
                row = sprite[r]
                for cell in row:
                    c = str(cell).strip()
                    if c == "." or c == "":
                        tokens.append(V["PIX_TRANSPARENT"])
                    elif c.isdigit():
                        d = int(c)
                        if 0 <= d <= 9:
                            tokens.append(V[f"PIX{d}"])
                        else:
                            tokens.append(V["PIX_TRANSPARENT"])
                    else:
                        tokens.append(V["PIX_TRANSPARENT"])
                if r < n_rows - 1:
                    tokens.append(V["PIXEL_ROW_SEP"])
            tokens.append(V["OBJ_END"])
        tokens.append(V["SEP"])

    # --- Collision layers ---
    for layer in tree.collision_layers:
        tokens.append(V["LAYER"])
        for obj_name in layer:
            tokens.extend(_resolve_name(obj_name))
        tokens.append(V["SEP"])

    # --- Legend groups (OR and AND only) ---
    for key, le in legend.items():
        if le.operator is None:
            continue  # skip simple aliases
        if le.operator.lower() == "or":
            tokens.append(V["GROUP_OR"])
        elif le.operator.lower() == "and":
            tokens.append(V["GROUP_AND"])
        else:
            continue
        gi = group_map[key]
        if gi < MAX_GROUPS:
            tokens.append(V[f"G{gi}"])
        # List member channels
        for obj_name in le.obj_names:
            tokens.extend(_resolve_name(obj_name))
        tokens.append(V["SEP"])

    # --- Rules ---
    def _tokenize_rules(rules, in_loop=False):
        for r in rules:
            if isinstance(r, RuleBlock):
                if r.looping:
                    tokens.append(V["LOOP_START"])
                _tokenize_rules(r.rules, in_loop=r.looping)
                if r.looping:
                    tokens.append(V["LOOP_END"])
            elif isinstance(r, Rule):
                _tokenize_rule(r)

    def _tokenize_rule(rule: Rule):
        tokens.append(V["RULE"])
        # Prefixes
        for prefix in rule.prefixes:
            prefix_l = prefix.lower()
            if prefix_l in _PREFIX_MAP:
                tokens.append(V[_PREFIX_MAP[prefix_l]])
        # LHS
        tokens.append(V["LHS"])
        if rule.left_kernels:
            for part in rule.left_kernels:
                _tokenize_kernel(part)
        # RHS
        tokens.append(V["RHS"])
        if rule.right_kernels:
            for part in rule.right_kernels:
                _tokenize_kernel(part)
        # Command
        if rule.command:
            cmd = rule.command.lower()
            cmd_tok = {
                "again": "CMD_AGAIN",
                "cancel": "CMD_CANCEL",
                "checkpoint": "CMD_CHECKPOINT",
                "restart": "CMD_RESTART",
                "win": "CMD_WIN",
            }.get(cmd)
            if cmd_tok:
                tokens.append(V[cmd_tok])
        tokens.append(V["SEP"])

    def _tokenize_kernel(cells: list):
        """Tokenize a kernel (list of cells, each cell is a list of token strings)."""
        for cell_i, cell in enumerate(cells):
            if cell_i > 0:
                tokens.append(V["CELL_SEP"])
            if isinstance(cell, str):
                cell_tokens = cell.split()
            elif isinstance(cell, list):
                cell_tokens = []
                for t in cell:
                    if isinstance(t, str):
                        cell_tokens.extend(t.split())
                    elif isinstance(t, list):
                        for tt in t:
                            cell_tokens.extend(str(tt).split())
            else:
                continue

            for tok in cell_tokens:
                tok_l = tok.lower()
                if tok_l == "...":
                    tokens.append(V["ELLIPSIS"])
                elif tok_l in _MODIFIER_MAP:
                    tokens.append(V[_MODIFIER_MAP[tok_l]])
                else:
                    # Object or legend reference
                    tokens.extend(_resolve_name(tok_l))

    _tokenize_rules(tree.rules)

    # --- Win conditions ---
    for wc in tree.win_conditions:
        tokens.append(V["WIN"])
        q = wc.quantifier.lower()
        if q in _QUANTIFIER_MAP:
            tokens.append(V[_QUANTIFIER_MAP[q]])
        # Source object
        if wc.src_obj:
            tokens.extend(_resolve_name(wc.src_obj))
        # Target object
        if wc.trg_obj:
            tokens.append(V["WC_ON"])
            tokens.extend(_resolve_name(wc.trg_obj))
        tokens.append(V["SEP"])

    return tokens


def tokens_to_str(token_ids: list[int]) -> str:
    """Convert token IDs back to readable string for debugging."""
    return " ".join(INV_VOCAB.get(t, f"?{t}") for t in token_ids)


def get_game_tree_from_js(ps_parser, game_name: str) -> tuple[PSGameTree, list[str]]:
    """Parse a game via the JS bridge and return (tree, canonical_ids).

    This is the recommended way to get a PSGameTree with proper legend entries.
    """
    import json
    from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
    from puzzlescript_jax.js_bridge import parsed_state_to_tree
    from puzzlescript_nodejs.utils import compile_game

    backend = CppPuzzleScriptBackend()
    backend._ensure_js_engine()
    compile_game(ps_parser, backend._js_engine, game_name, 0)
    parsed_dict = json.loads(str(backend._js_engine.serializeParsedStateJSON()))
    tree = parsed_state_to_tree(parsed_dict)

    # Get canonical_ids from env
    json_str = backend.compile_and_serialize(ps_parser, game_name)
    env = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
    canonical_ids = env._canonical_ids

    return tree, canonical_ids

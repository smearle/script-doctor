"""Bridge between the NodeJS PuzzleScript compiler and the JAX environment.

Converts the pre-compilation parsed state (captured by the monkey-patched
``serializeParsedState()`` in ``engine.js``) into a ``PSGameTree`` that
can be fed to ``PuzzleJaxEnv.__init__``.

This module itself does NOT depend on NodeJS — it only processes the
plain JSON dict that the JS side produces.

Usage
-----
**Quick start** (one-liner via ``utils``)::

    from puzzlescript_jax.utils import init_ps_env_from_js
    env = init_ps_env_from_js('Microban', level_i=0, max_episode_steps=200)

**Explicit two-step** (useful when you already have a JS engine)::

    import json
    from javascript import require
    from puzzlescript_jax.env import PuzzleJaxEnv

    engine = require('./puzzlescript_nodejs/puzzlescript/engine.js').createFreshApi()
    with open('data/scraped_games/Microban.txt') as f:
        game_text = f.read()
    engine.compile(['restart'], game_text)
    parsed = json.loads(str(engine.serializeParsedStateJSON()))

    env = PuzzleJaxEnv.from_js_parsed_state(parsed, level_i=0)

**From a saved JSON file** (no NodeJS needed at load time)::

    import json
    from puzzlescript_jax.env import PuzzleJaxEnv

    with open('compiled_game.json') as f:
        parsed = json.load(f)
    env = PuzzleJaxEnv.from_js_parsed_state(parsed, level_i=0)

The Lark-based path (``init_ps_env`` / ``PuzzleJaxEnv(tree)``) remains
available and does not require NodeJS.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from puzzlescript_jax.ps_game import (
    LegendEntry,
    Prelude,
    PSGameTree,
    PSObject,
    Rule,
    RuleBlock,
    WinCondition,
)

logger = logging.getLogger(__name__)

# Modifiers that can precede an object name in a rule cell.
_RULE_MODIFIERS = frozenset({
    '>', '<', 'v', '^',
    'up', 'down', 'left', 'right',
    'moving', 'stationary',
    'no',
    'action',
    'random', 'randomdir',
    'horizontal', 'vertical',
    'orthogonal', 'perpendicular', 'parallel',
})

# Prefixes that can appear before the first `[` in a rule line.
_RULE_PREFIXES = frozenset({
    'late', 'random', 'rigid',
    'up', 'down', 'left', 'right',
    'horizontal', 'vertical',
})

# Commands that can appear after `->` instead of a right pattern.
_RULE_COMMANDS = frozenset({
    'again', 'win', 'cancel', 'restart', 'checkpoint',
})


# ---------------------------------------------------------------------------
# Rule text parsing
# ---------------------------------------------------------------------------

def _parse_rule_text(text: str) -> Optional[Rule]:
    """Parse a single rule text line into a Rule object.

    Expected format (from the JS parser's mixed-case rule string):
        [prefix]* [cell | cell | ...] [+ [cell | ...]]* -> [cell | ...] [command]
    """
    text = text.strip()
    if not text:
        return None

    # Split on '->' to separate left and right sides
    arrow_idx = text.find('->')
    if arrow_idx < 0:
        logger.warning("Rule missing '->': %s", text)
        return None

    left_text = text[:arrow_idx].strip()
    right_text = text[arrow_idx + 2:].strip()

    # Extract prefixes (tokens before the first '[')
    prefixes = []
    bracket_idx = left_text.find('[')
    if bracket_idx > 0:
        prefix_text = left_text[:bracket_idx].strip()
        for token in prefix_text.split():
            tok_lower = token.lower()
            if tok_lower.startswith('sfx'):
                continue
            prefixes.append(tok_lower)
        left_text = left_text[bracket_idx:]
    elif bracket_idx < 0:
        # No brackets at all on left side
        logger.warning("Rule has no brackets on left side: %s", text)
        return None

    # Parse bracket blocks
    left_patterns = _parse_rule_side(left_text)
    if left_patterns is None:
        return None

    # Check for command-only right side
    command = None
    right_stripped = right_text.strip()

    # Extract trailing commands (after the last ']')
    last_bracket = right_stripped.rfind(']')
    if last_bracket >= 0:
        trailing = right_stripped[last_bracket + 1:].strip()
        if trailing:
            for token in trailing.split():
                tok_lower = token.lower()
                if tok_lower in _RULE_COMMANDS:
                    command = tok_lower
                # Ignore sfx and other tokens
        right_text_brackets = right_stripped[:last_bracket + 1].strip()
    else:
        right_text_brackets = right_stripped

    # Check if right side is just a command (no brackets)
    if '[' not in right_text_brackets:
        cmd_candidate = right_text_brackets.strip().lower()
        if cmd_candidate in _RULE_COMMANDS:
            command = cmd_candidate
            return Rule(
                left_patterns=left_patterns,
                right_patterns=None,
                prefixes=prefixes,
                command=command,
            )
        elif not right_text_brackets:
            # Empty right side with trailing command already captured
            if command:
                return Rule(
                    left_patterns=left_patterns,
                    right_patterns=None,
                    prefixes=prefixes,
                    command=command,
                )
        # else: unexpected right side
        logger.warning("Unexpected right side in rule: %s", text)
        return None

    right_patterns = _parse_rule_side(right_text_brackets)

    return Rule(
        left_patterns=left_patterns,
        right_patterns=right_patterns if right_patterns else [],
        prefixes=prefixes,
        command=command,
    )


def _parse_rule_side(text: str) -> Optional[List[List[List[str]]]]:
    """Parse one side of a rule (left or right) into a list of rule parts.

    Each rule part corresponds to a [...] block.
    Each rule part is a list of cells (separated by |).
    Each cell is a list of entity strings (e.g. '> player', 'crate').

    Returns list of rule_parts, or None on error.
    """
    parts = []
    # Find all [...] blocks
    i = 0
    while i < len(text):
        open_idx = text.find('[', i)
        if open_idx < 0:
            break
        close_idx = text.find(']', open_idx)
        if close_idx < 0:
            logger.warning("Unmatched '[' in rule side: %s", text)
            return None
        block_content = text[open_idx + 1:close_idx].strip()
        parts.append(_parse_rule_block(block_content))
        i = close_idx + 1

    return parts if parts else None


def _parse_rule_block(content: str) -> List[List[str]]:
    """Parse the content inside a [...] block into cells.

    Cells are separated by '|'. Within each cell, tokens are grouped
    into entities: consecutive modifiers followed by an object name.
    """
    cells = []
    for cell_text in content.split('|'):
        cell_text = cell_text.strip()
        if not cell_text:
            cells.append([])
            continue
        cells.append(_parse_cell_tokens(cell_text))
    return cells


def _parse_cell_tokens(cell_text: str) -> List[str]:
    """Parse tokens in a rule cell into entity strings.

    Groups modifier tokens with the following object name.
    E.g. '> Player Crate' -> ['> player', 'crate']
    """
    tokens = cell_text.split()
    entities = []
    current_mods = []

    for token in tokens:
        tok_lower = token.lower()
        if tok_lower in _RULE_MODIFIERS:
            current_mods.append(tok_lower)
        elif tok_lower == '...':
            # Ellipsis — flush any pending mods, then add as standalone
            if current_mods:
                entities.append(' '.join(current_mods))
                current_mods = []
            entities.append('...')
        else:
            # Object name — combine with preceding modifiers
            current_mods.append(tok_lower)
            entities.append(' '.join(current_mods))
            current_mods = []

    # Flush trailing modifiers (shouldn't happen in valid rules, but be safe)
    if current_mods:
        entities.append(' '.join(current_mods))

    return entities


# ---------------------------------------------------------------------------
# Level parsing
# ---------------------------------------------------------------------------

def _parse_levels(raw_levels: list) -> List[np.ndarray]:
    """Convert JS pre-compilation level entries to numpy arrays.

    JS format: [[lineNumber, 'row1', 'row2', ...], ...] for normal levels.
    Message entries: ['\\n', 'message text', lineNumber] — skipped.
    Empty entries: [] — skipped.
    """
    levels = []
    for entry in raw_levels:
        if not entry:
            continue
        # Message entry
        if isinstance(entry[0], str) and entry[0] == '\n':
            continue
        # Normal level: [lineNumber, row1, row2, ...]
        rows = []
        for item in entry:
            if isinstance(item, str):
                rows.append(list(item.lower()))
            # Skip the lineNumber (int)

        if not rows:
            continue

        # Pad rows to same width if needed
        max_width = max(len(r) for r in rows)
        for r in rows:
            while len(r) < max_width:
                r.append('.')

        # Shape (1, H, W) to match Lark pipeline output
        level_arr = np.array(rows, dtype='<U1')[np.newaxis, :, :]
        levels.append(level_arr)

    return levels


# ---------------------------------------------------------------------------
# Win condition parsing
# ---------------------------------------------------------------------------

def _parse_win_conditions(raw_wcs: list) -> List[WinCondition]:
    """Convert JS pre-compilation win condition entries to WinCondition objects.

    JS format: ['quantifier', 'src_obj', optional 'on', optional 'trg_obj', ..., lineNumber]
    """
    result = []
    for entry in raw_wcs:
        if not entry:
            continue
        # Filter out the line number (last element, always an int)
        tokens = [t for t in entry if isinstance(t, str)]
        if len(tokens) < 2:
            continue

        quantifier = tokens[0].lower()
        src_obj = tokens[1].lower()
        trg_obj = None

        if len(tokens) >= 4 and tokens[2].lower() == 'on':
            trg_obj = tokens[3].lower()

        result.append(WinCondition(
            quantifier=quantifier,
            src_obj=src_obj,
            trg_obj=trg_obj,
        ))

    return result


# ---------------------------------------------------------------------------
# Object conversion
# ---------------------------------------------------------------------------

def _convert_objects(raw_objects: dict) -> Dict[str, PSObject]:
    """Convert JS object entries to PSObject instances.

    JS format: {name: {colors: [...], spritematrix: ['..0..', ...]}}
    """
    objects = {}
    for name, obj_data in raw_objects.items():
        colors = obj_data.get('colors', [])
        raw_matrix = obj_data.get('spritematrix', [])

        # Convert spritematrix to 2D array matching Lark output.
        # JS stores rows as arrays of integers (-1 = transparent, 0+ = color index).
        # Lark stores rows as strings where each char is '.' (transparent) or a
        # digit ('0', '1', ...) indexing into the colors array.
        if raw_matrix:
            rows = []
            for row in raw_matrix:
                if isinstance(row, list):
                    # Array of ints from JS: -1 -> '.', N -> str(N)
                    rows.append([('.' if v == -1 else str(v)) for v in row])
                elif isinstance(row, str):
                    # Already a string like '..0..' from Lark-style storage
                    rows.append(list(row))
                else:
                    rows.append(['.'] * 5)
            sprite = np.array(rows)
        else:
            sprite = None

        objects[name.lower()] = PSObject(
            name=name.lower(),
            alt_names=None,
            colors=colors,
            sprite=sprite,
            legend_key=None,
        )

    return objects


# ---------------------------------------------------------------------------
# Legend conversion
# ---------------------------------------------------------------------------

def _convert_legend(
    synonyms: list,
    aggregates: list,
    properties: list,
) -> Dict[str, LegendEntry]:
    """Convert JS legend entries to a dict of LegendEntry objects."""
    legend = {}

    for entry in synonyms:
        if len(entry) < 2:
            continue
        key = entry[0].lower()
        obj_name = entry[1].lower()
        legend[key] = LegendEntry(key=key, obj_names=[obj_name], operator=None)

    for entry in aggregates:
        if len(entry) < 2:
            continue
        key = entry[0].lower()
        obj_names = [e.lower() for e in entry[1:]]
        legend[key] = LegendEntry(key=key, obj_names=obj_names, operator='and')

    for entry in properties:
        if len(entry) < 2:
            continue
        key = entry[0].lower()
        obj_names = [e.lower() for e in entry[1:]]
        legend[key] = LegendEntry(key=key, obj_names=obj_names, operator='or')

    return legend


# ---------------------------------------------------------------------------
# Collision layers
# ---------------------------------------------------------------------------

def _convert_collision_layers(raw_layers: list) -> List[List[str]]:
    """Convert JS collision layers to list of lists of lowercase names."""
    return [[name.lower() for name in layer] for layer in raw_layers]


# ---------------------------------------------------------------------------
# Rule grouping into RuleBlocks
# ---------------------------------------------------------------------------

def _group_rules_into_blocks(
    raw_rules: list,
    raw_loops: list,
) -> List[RuleBlock]:
    """Convert JS pre-compilation rules + loop markers into RuleBlocks.

    raw_rules: [[ruleText, lineNumber, mixedCase], ...]
        startloop/endloop appear as rule text entries in this list.
    raw_loops: [[lineNumber, bracketValue], ...]
        Only populated after rulesToArray (which runs after our snapshot),
        so this is typically empty. We detect loops from raw_rules instead.
    """
    # Walk rules in order, detecting startloop/endloop as text entries.
    blocks = []
    current_rules = []
    loop_stack = []  # Stack of lists for nested loops

    for entry in raw_rules:
        if len(entry) < 2:
            continue
        # Use the mixed-case version (index 2) if available, else index 0
        rule_text = entry[2] if len(entry) > 2 else entry[0]
        rule_text_stripped = rule_text.strip().lower()

        if rule_text_stripped == 'startloop':
            # Flush current non-loop rules into a block
            if current_rules:
                blocks.append(RuleBlock(rules=current_rules, looping=False))
                current_rules = []
            loop_stack.append([])
        elif rule_text_stripped == 'endloop':
            if loop_stack:
                loop_rules = loop_stack.pop()
                blocks.append(RuleBlock(rules=loop_rules, looping=True))
            else:
                logger.warning("endloop without matching startloop")
        else:
            rule = _parse_rule_text(rule_text)
            if rule is not None:
                if loop_stack:
                    loop_stack[-1].append(rule)
                else:
                    current_rules.append(rule)

    # Also process any explicit loop markers from raw_loops (if populated).
    # These would come from state.loops after rulesToArray, but our snapshot
    # is taken before that, so this is typically empty.
    if raw_loops:
        # Re-process using the timeline approach as a fallback.
        # Only needed if loops weren't detected from rule text above.
        pass

    # Flush remaining rules
    if current_rules:
        blocks.append(RuleBlock(rules=current_rules, looping=False))

    # Handle unclosed loops
    for remaining in loop_stack:
        if remaining:
            blocks.append(RuleBlock(rules=remaining, looping=True))

    return blocks


# ---------------------------------------------------------------------------
# Prelude / metadata
# ---------------------------------------------------------------------------

def _convert_prelude(metadata: dict, title_fallback: str = 'Untitled') -> Prelude:
    """Convert JS metadata dict to a Prelude object."""
    flickscreen = None
    zoomscreen = None

    raw_flick = metadata.get('flickscreen')
    if raw_flick is not None:
        if isinstance(raw_flick, list) and len(raw_flick) == 2:
            flickscreen = (int(raw_flick[0]), int(raw_flick[1]))
        elif isinstance(raw_flick, str):
            match = re.fullmatch(r'\s*(\d+)\s*[xX]\s*(\d+)\s*', raw_flick)
            if match:
                flickscreen = (int(match.group(1)), int(match.group(2)))

    raw_zoom = metadata.get('zoomscreen')
    if raw_zoom is not None:
        if isinstance(raw_zoom, list) and len(raw_zoom) == 2:
            zoomscreen = (int(raw_zoom[0]), int(raw_zoom[1]))
        elif isinstance(raw_zoom, str):
            match = re.fullmatch(r'\s*(\d+)\s*[xX]\s*(\d+)\s*', raw_zoom)
            if match:
                zoomscreen = (int(match.group(1)), int(match.group(2)))

    return Prelude(
        title=metadata.get('title', title_fallback),
        author=metadata.get('author'),
        homepage=metadata.get('homepage'),
        flickscreen=flickscreen,
        zoomscreen=zoomscreen,
        verbose_logging=bool(metadata.get('verbose_logging', False)),
        require_player_movement=bool(metadata.get('require_player_movement', False)),
        run_rules_on_level_start=bool(metadata.get('run_rules_on_level_start', False)),
        noaction=bool(metadata.get('noaction', False)),
    )


# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------

def parsed_state_to_tree(parsed_json: dict) -> PSGameTree:
    """Convert the JS serializeParsedState() output to a PSGameTree.

    This function does NOT depend on NodeJS — it processes a plain dict.
    """
    objects = _convert_objects(parsed_json.get('objects', {}))
    legend = _convert_legend(
        parsed_json.get('legend_synonyms', []),
        parsed_json.get('legend_aggregates', []),
        parsed_json.get('legend_properties', []),
    )
    collision_layers = _convert_collision_layers(
        parsed_json.get('collisionLayers', [])
    )
    rules = _group_rules_into_blocks(
        parsed_json.get('rules', []),
        parsed_json.get('loops', []),
    )
    win_conditions = _parse_win_conditions(
        parsed_json.get('winconditions', [])
    )
    levels = _parse_levels(parsed_json.get('levels', []))
    prelude = _convert_prelude(parsed_json.get('metadata', {}))

    return PSGameTree(
        prelude=prelude,
        objects=objects,
        legend=legend,
        collision_layers=collision_layers,
        rules=rules,
        win_conditions=win_conditions,
        levels=levels,
    )

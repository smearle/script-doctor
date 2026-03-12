import unicodedata


def get_extended_chars():
    return set(
        chr(i)
        for i in range(160, 592)
        if unicodedata.category(chr(i))[0] in {"L", "S", "P"}
    )


def get_box_drawing_and_shapes():
    return {chr(i) for i in range(0x2500, 0x2600)}


def get_braille_chars():
    return {chr(i) for i in range(0x2800, 0x28FF + 1)}


def get_llm_friendly_chars():
    chars = set()
    for i in range(33, 0x2B00):
        c = chr(i)
        cat = unicodedata.category(c)
        if cat[0] in {"L", "S", "P"} and not c.isspace():
            chars.add(c)
    return chars


def get_available_render_chars():
    ascii_chars = set(chr(i) for i in range(32, 127))
    all_chars = (
        ascii_chars
        | get_extended_chars()
        | get_box_drawing_and_shapes()
        | get_braille_chars()
        | get_llm_friendly_chars()
    )
    all_chars.discard(" ")
    return all_chars


class ASCIIStateFormatter:
    def __init__(self, legend_mapping=None):
        self.combo_to_char = {}
        self.available_chars = get_available_render_chars()
        if legend_mapping:
            self.seed_from_legend_mapping(legend_mapping)

    @staticmethod
    def normalize_combo(objs):
        combo = tuple(obj for obj in objs if str(obj).lower() != "background")
        if not combo:
            combo = ("background",)
        return combo

    def seed_from_legend_mapping(self, legend_mapping):
        for char, objs in legend_mapping.items():
            if len(char) != 1 or char.isspace():
                continue
            self.assign_char(self.normalize_combo(objs), char)

    def seed_from_char_mapping(self, char_mapping):
        for char, objs in char_mapping.items():
            if len(char) != 1 or char.isspace():
                continue
            self.assign_char(self.normalize_combo(objs), char)

    def assign_char(self, combo, char):
        existing = self.combo_to_char.get(combo)
        if existing is not None:
            return existing
        self.combo_to_char[combo] = char
        self.available_chars.discard(char)
        return char

    def char_for_combo(self, combo):
        combo = self.normalize_combo(combo)
        if combo not in self.combo_to_char:
            if not self.available_chars:
                raise ValueError("Not enough characters to represent all object combinations.")
            self.combo_to_char[combo] = self.available_chars.pop()
        return self.combo_to_char[combo]

    def render_from_name_grid(self, name_grid):
        used_chars = []
        seen_chars = set()
        rows = []

        for row in name_grid:
            row_chars = []
            for combo in row:
                char = self.char_for_combo(combo)
                if char not in seen_chars:
                    seen_chars.add(char)
                    used_chars.append(char)
                row_chars.append(char)
            rows.append("".join(row_chars))

        legend_lines = []
        for char in used_chars:
            combo = next(combo for combo, combo_char in self.combo_to_char.items() if combo_char == char)
            legend_lines.append(f"{char}: {', '.join(combo)}")

        return "LEGEND:\n" + "\n".join(legend_lines) + "\n\nMAP:\n" + "\n".join(rows)

    def render_map_only(self, name_grid):
        """Render just the map lines (no legend header). Registers any new combos."""
        rows = []
        for row in name_grid:
            row_chars = []
            for combo in row:
                char = self.char_for_combo(combo)
                row_chars.append(char)
            rows.append("".join(row_chars))
        return "\n".join(rows)

    def get_legend_text(self):
        """Return the full legend string for all combos seen so far."""
        legend_lines = []
        for combo, char in self.combo_to_char.items():
            legend_lines.append(f"{char}: {', '.join(combo)}")
        return "LEGEND:\n" + "\n".join(legend_lines)


def build_game_action_prompt(
    ascii_map,
    rules,
    action_space,
    action_meanings,
    think_aloud,
    memory,
    state_history,
):
    action_space_str = ", ".join(str(a) for a in action_space)
    action_map_str = ", ".join(f"{k}={v}" for k, v in action_meanings.items())

    prompt_parts = [f"Game description and rules:\n{rules}"]

    if memory > 0 and state_history:
        recent_states = state_history[-memory:]
        history_lines = ["Previous turns:"]
        for turn_i, (prev_ascii_state, prev_action_id) in enumerate(recent_states, start=1):
            history_lines.append(f"Turn -{len(recent_states) - turn_i + 1} state:")
            history_lines.append(prev_ascii_state)
            history_lines.append(f"Turn -{len(recent_states) - turn_i + 1} action: {prev_action_id}")
            history_lines.append("")
        prompt_parts.append("\n".join(history_lines).rstrip())

    prompt_parts.append(f"Game state (ASCII map) and Legend:\n{ascii_map}")
    prompt_parts.append(f"Available actions (action_space): {action_space_str}\nAction mapping: {action_map_str}")

    if not think_aloud:
        prompt_parts.append(
            "Please select the best action and ONLY return the action id (an integer from action_space)."
        )
    else:
        prompt_parts.append(
            "You may first reason about the problem, then output `ACTION: ID`, where `ID` is an integer from action_space."
        )

    return "\n\n".join(prompt_parts)


def build_human_like_prompt(
    *,
    title,
    author,
    legend_text,
    ascii_map,
    action_space,
    action_meanings,
    state_history,
    history_limit,
    scratchpad,
    messages=None,
    level_number=None,
):
    """Build a human-like game prompt: no rules, just title/author/legend/map/history/scratchpad."""
    action_space_str = ", ".join(str(a) for a in action_space)
    action_map_str = ", ".join(f"{k}={v}" for k, v in action_meanings.items())

    prompt_parts = []

    # Game identity
    header = f"Game: {title}"
    if author:
        header += f"\nAuthor: {author}"
    if level_number is not None:
        header += f"\nLevel: {level_number}"
    prompt_parts.append(header)

    # Messages (shown between levels)
    if messages:
        prompt_parts.append("Messages:\n" + "\n".join(messages))

    # Legend (consistent across all turns)
    prompt_parts.append(legend_text)

    # History of recent turns
    if state_history:
        recent = state_history[-history_limit:]
        history_lines = [f"Recent history (last {len(recent)} turns):"]
        for turn_i, (prev_map, prev_action_id) in enumerate(recent, start=1):
            turn_label = f"Turn -{len(recent) - turn_i + 1}"
            history_lines.append(f"{turn_label} map:\n{prev_map}")
            history_lines.append(f"{turn_label} action: {prev_action_id} ({action_meanings.get(prev_action_id, '?')})")
            history_lines.append("")
        prompt_parts.append("\n".join(history_lines).rstrip())

    # Current state
    prompt_parts.append(f"Current map:\n{ascii_map}")

    # Scratchpad
    prompt_parts.append(f"Your scratchpad (you may update this):\n{scratchpad if scratchpad else '(empty)'}")

    # Action selection
    prompt_parts.append(
        f"Available actions: {action_space_str}\nAction mapping: {action_map_str}\n\n"
        "First reason about the game. You can observe the effects of your actions by "
        "comparing the current map to the history. Figure out the game mechanics and "
        "how to win by experimentation.\n"
        "Then, optionally update your scratchpad by writing `SCRATCHPAD: <your notes>` "
        "(everything after SCRATCHPAD: until the next section is saved).\n"
        "Finally, select your action by writing `ACTION: <id>`."
    )

    return "\n\n".join(prompt_parts)

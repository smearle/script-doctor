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

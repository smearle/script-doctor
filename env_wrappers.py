import copy
from itertools import product, combinations
import itertools
import unicodedata

import jax
from lark import Lark
import numpy as np
from env import PSEnv, PSParams, PSState, multihot_to_desc
from preprocess_games import PS_LARK_GRAMMAR_PATH, get_tree_from_txt


def vec_to_obj_names(vec, idxs_to_objs):
    """Convert a binary vector to a list of object names."""
    obj_names = []
    for i, val in enumerate(vec):
        if val:
            obj_name = idxs_to_objs.get(i)
            if obj_name:
                obj_names.append(obj_name)
    return obj_names

def get_extended_chars():
    return set(chr(i) for i in range(160, 592)  # U+00A0 to U+024F
            if unicodedata.category(chr(i))[0] in {'L', 'S', 'P'})

def get_box_drawing_and_shapes():
    return set([chr(i) for i in range(0x2500, 0x2600)])

def get_braille_chars():
    return set(chr(i) for i in range(0x2800, 0x28FF + 1))

def get_llm_friendly_chars():
    chars = set()
    for i in range(33, 0x2B00):  # Skip control chars
        c = chr(i)
        cat = unicodedata.category(c)
        if cat[0] in {'L', 'S', 'P'} and not c.isspace():
            chars.add(c)
    return chars

class RepresentationWrapper(PSEnv):
    """Log the episode returns and lengths."""


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Generate a set of all possible ASCII characters
        ascii_chars = set(chr(i) for i in range(32, 127))
        extended_chars = get_extended_chars()
        box_drawing_chars = get_box_drawing_and_shapes()
        braille_chars = get_braille_chars()
        llm_friendly_chars = get_llm_friendly_chars()
        all_chars = ascii_chars | extended_chars | box_drawing_chars | braille_chars | llm_friendly_chars

        all_chars.remove(' ')

        self.idxs_to_chars = {v: k for k, v in self.chars_to_idxs.items()}
        
        # If any atomic object is not already mapped to an ASCII character, assign it to one
        for obj, idx in self.objs_to_idxs.items():
            if idx not in self.idxs_to_chars:
                self.idxs_to_chars[idx] = all_chars.pop()

        # By default, background is always overlapping with everything
        background_idx = self.objs_to_idxs['background']
        user_assigned_chars = set(self.idxs_to_chars.values())
        if background_idx not in self.idxs_to_chars:
            background_char = '.'
            if background_char in user_assigned_chars:
                background_char = all_chars.pop()
            else:
                all_chars.remove(background_char)
        else:
            background_char = self.idxs_to_chars[background_idx]

        obj_vecs = self.obj_vecs
        obj_vecs[:, background_idx] = 1

        # It's technically possible for the background to disappear, but we'll ignore this for now, or else we'd have to
        # define a version of each object without background behind it.
        # empty_vec = np.zeros(self.n_objs, dtype=int)
        # obj_vecs = np.vstack([obj_vecs, empty_vec])

        # Now define the mapping of from all binary (multi-)object vectors to ASII characters as has been defined by 
        # the user.
        vecs = [tuple([int(i) for i in v]) for v in obj_vecs]
        # vecs = [tuple([int(i) for i in v]) for v in obj_vecs[:-1]]
        self.vecs_to_chars = {vec: self.idxs_to_chars.get(i) for i, vec in enumerate(vecs)}

        # Remove all used ASCII characters from the set
        all_chars -= set(self.vecs_to_chars.values())

        # self.vecs_to_chars[tuple([int(i) for i in empty_vec])] = " "

        # Now, get all possible combinations of non-colliding objects and assign them ASCII characters if not already
        # present in our mapping.

        # Generate all possible non-colliding combinations including same-layer objects
        all_combinations_ints = []
        
        # First handle combinations where we pick one object from each collision layer
        for layer in self.collision_layers:
            layer_objs = [self.objs_to_idxs[obj] for obj in layer] + [-1]
            if layer_objs:
                all_combinations_ints.append(layer_objs)
        all_combinations_ints = list(itertools.product(*all_combinations_ints))
        
        # Generate combinations across layers
        all_combinations = []
        for combo in all_combinations_ints:
            combo = [i for i in combo if i != -1]
            vec = np.zeros(self.n_objs, dtype=int)
            vec[list(combo)] = 1
            all_combinations.append(vec)

        # Add background to all combinations
        background_idx = self.objs_to_idxs['background']
        for vec in all_combinations:
            vec[background_idx] = 1

        # Now assign ASCII characters to all combinations
        for vec in all_combinations:
            vec = tuple([int(i) for i in vec])
            if vec not in self.vecs_to_chars:
                if all_chars:
                    self.vecs_to_chars[vec] = all_chars.pop()
                else:
                    raise ValueError("Not enough ASCII characters to represent all object combinations.")

        # Handle empty cells (no objects except background)
        empty_vec = tuple([1 if i == background_idx else 0 for i in range(self.n_objs)])
        self.vecs_to_chars[empty_vec] = background_char

        self.chars_to_vecs = {v: k for k, v in self.vecs_to_chars.items()}
        self.ascii_legend, legend_lines = self.get_ascii_legend()
        self.full_ascii_legend_str = "\n".join(legend_lines)


    def get_ascii_legend(self) -> str:
        """Return a human-readable legend mapping ASCII chars to object combinations."""
        legend_lines = ["ASCII legend:"]
        legend = {}
        for char, vec in self.chars_to_vecs.items():
            obj_idxs = [i for i, val in enumerate(vec) if val]
            if not obj_idxs:
                obj_names = ["empty"]
            else:
                obj_names = [obj for obj, idx in self.objs_to_idxs.items() if idx in obj_idxs and obj != "background"]
            if not obj_names:
                obj_names = ["background"]
            legend[char] = obj_names
            legend_lines.append(f"{char}: {', '.join(obj_names)}")
        return legend, legend_lines

    def get_ascii_mapping(self):
        """
        Return a dictionary mapping each ASCII character to a list of object names.
        Example: {'@': ['player'], '#': ['wall'], ...}
        """
        mapping = {}
        for char, vec in self.chars_to_vecs.items():
            obj_idxs = [i for i, val in enumerate(vec) if val]
            if not obj_idxs:
                obj_names = ["empty"]
            else:
                obj_names = [obj for obj, idx in self.objs_to_idxs.items() if idx in obj_idxs and obj != "background"]
            if not obj_names:
                obj_names = ["background"]
            mapping[char] = obj_names
        return mapping

    def get_action_meanings(self):
        """
        Return a dictionary mapping action index to its meaning.
        Example: {0: "left", 1: "down", 2: "right", 3: "up", 4: "action"}
        """
        return {0: "left", 1: "down", 2: "right", 3: "up", 4: "action"}


    def print_full_ascii_legend(self):
        print(self.full_ascii_legend_str)


    def render_ascii(self, state: PSState):
        """Render the game state as ASCII art."""
        map_arr = state.multihot_level
        ascii_map = np.full(map_arr.shape[1:], " ", dtype="<U1")
        for i in range(map_arr.shape[1]):
            for j in range(map_arr.shape[2]):
                vec = tuple([int(i) for i in map_arr[:, i, j]])
                ascii_map[i, j] = self.vecs_to_chars[vec]
        return "\n".join("".join(row) for row in ascii_map)


    def render_ascii_and_legend(self, state: PSState):
        """Render the game state as ASCII art."""
        map_arr = state.multihot_level
        ascii_map = np.full(map_arr.shape[1:], " ", dtype="<U1")
        partial_legend = {}
        partial_legend_lines = []
        for i in range(map_arr.shape[1]):
            for j in range(map_arr.shape[2]):
                vec = tuple([int(i) for i in map_arr[:, i, j]])
                char = self.vecs_to_chars[vec]
                ascii_map[i, j] = char
                if char not in partial_legend:
                    obj_name = self.ascii_legend[char]
                    partial_legend[char] = obj_name
                    partial_legend_lines.append(f"{char}: {', '.join(obj_name)}")

        legend_str = "\n".join(partial_legend_lines)
        map_str = "\n".join("".join(row) for row in ascii_map)
        return f"LEGEND:\n{legend_str}\n\nMAP:\n{map_str}"


    def render_text(self, state: PSState):
        return multihot_to_desc(state.multihot_level, self.objs_to_idxs, self.n_objs, obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs, show_background=False)


def test_log_wrapper():
    game = 'Take_Heart_Lass'
    with open(PS_LARK_GRAMMAR_PATH, 'r', encoding='utf-8') as f:
        puzzlescript_grammar = f.read()
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    tree, success, err_msg = get_tree_from_txt(parser, game, test_env_init=False)
    print('Initializing environment')
    env = RepresentationWrapper(tree, debug=False, print_score=False)
    level = env.get_level(0)
    env_params = PSParams(
        level=level,
    )
    rng = jax.random.PRNGKey(0)
    print(f'Resetting')
    obs, state = env.reset(rng=rng, params=env_params)

    env.print_full_ascii_legend()

    for i in range(100):
        action = env.action_space.sample(rng)
        rng, _rng = jax.random.split(rng)
        obs, state, rew, done, info = env.step_env(rng=rng, action=action, state=state, params=env_params)

        # print(f"ASCII Game State:\n{env.render_ascii(state)}")
        print(f"ASCII Game State:\n{env.render_ascii_and_legend(state)}")
        print(f"Text Game State:\n{env.render_text(state)}")

        print(f"Reward: {rew}")
        print(f"Heuristic: {state.heuristic}")
        print(f"Score: {state.score}")
        print(f"Win: {state.win}")


if __name__ == "__main__":
    test_log_wrapper()

from itertools import product, combinations

from gymnax.environments import environment
import jax
from lark import Lark
import numpy as np
from env import PSEnv, PSParams, PSState, multihot_to_desc
from parse_lark import PS_LARK_GRAMMAR_PATH, get_tree_from_txt


# class RepresentationWrapper(GymnaxWrapper, PSEnv):
class RepresentationWrapper(PSEnv):
    """Log the episode returns and lengths."""


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Generate a set of all possible ASCII characters
        ascii_chars = set(chr(i) for i in range(32, 127))

        self.idxs_to_chars = {v: k for k, v in self.chars_to_idxs.items()}

        # By default, background is always overlapping with everything
        background_idx = self.objs_to_idxs['background']
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
        self.vecs_to_chars = {vec: self.idxs_to_chars.get(i, '?') for i, vec in enumerate(vecs)}

        # Remove all used ASCII characters from the set
        ascii_chars -= set(self.vecs_to_chars.values())

        # self.vecs_to_chars[tuple([int(i) for i in empty_vec])] = " "

        # Now, get all possible combinations of non-colliding objects and assign them ASCII characters if not already
        # present in our mapping.

        # Generate all possible non-colliding combinations including same-layer objects
        all_combinations = []
        
        # First handle combinations where we pick one object from each collision layer
        for layer in self.collision_layers:
            layer_objs = [self.objs_to_idxs[obj] for obj in layer]
            if layer_objs:
                all_combinations.append(layer_objs)
        
        # Generate combinations across layers
        for combo in product(*all_combinations):
            vec = np.zeros(self.n_objs, dtype=int)
            vec[list(combo)] = 1
            all_combinations.append(vec)

        # Now handle combinations within layers (multiple objects from same layer)
        for layer in self.collision_layers:
            layer_objs = [self.objs_to_idxs[obj] for obj in layer]
            # Generate all non-empty subsets of the layer
            for r in range(1, len(layer_objs)+1):
                for subset in combinations(layer_objs, r):
                    vec = np.zeros(self.n_objs, dtype=int)
                    vec[list(subset)] = 1
                    all_combinations.append(vec)

        # Add background to all combinations
        background_idx = self.objs_to_idxs['background']
        for vec in all_combinations:
            vec[background_idx] = 1

        # Now process all unique combinations
        unique_vecs = set(tuple(v) for v in all_combinations)
        for vec_tuple in unique_vecs:
            if vec_tuple not in self.vecs_to_chars:
                if not ascii_chars:
                    # Fallback to '?' for unhandled combinations
                    self.vecs_to_chars[vec_tuple] = '?'
                else:
                    self.vecs_to_chars[vec_tuple] = ascii_chars.pop()

        # Handle empty cells (no objects except background)
        empty_vec = tuple([1 if i == background_idx else 0 for i in range(self.n_objs)])
        self.vecs_to_chars[empty_vec] = ' '

        self.chars_to_vecs = {v: k for k, v in self.vecs_to_chars.items()}
        self.ascii_legend_str = self.get_ascii_legend()

    def get_ascii_legend(self) -> str:
        """Return a human-readable legend mapping ASCII chars to object combinations."""
        legend_lines = ["ASCII legend:"]
        for char, vec in self.chars_to_vecs.items():
            obj_idxs = [i for i, val in enumerate(vec) if val]
            if not obj_idxs:
                obj_names = ["empty"]
            else:
                obj_names = [obj for obj, idx in self.objs_to_idxs.items() if idx in obj_idxs and obj != "background"]
            if not obj_names:
                obj_names = ["background"]
            legend_lines.append(f"{repr(char)}: {', '.join(obj_names)}")
        return "\n".join(legend_lines)

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


    def print_ascii_legend(self):
        print(self.ascii_legend_str)


    def render_ascii(self, state: PSState):
        """Render the game state as ASCII art."""
        map_arr = state.multihot_level
        ascii_map = np.full(map_arr.shape[1:], " ", dtype="<U1")
        for i in range(map_arr.shape[1]):
            for j in range(map_arr.shape[2]):
                vec = tuple([int(i) for i in map_arr[:, i, j]])
                ascii_map[i, j] = self.vecs_to_chars[vec]
        return "\n".join("".join(row) for row in ascii_map)


    def render_text(self, state: PSState):
        return multihot_to_desc(state.multihot_level, self.objs_to_idxs, self.n_objs, show_background=False)


def test_log_wrapper():
    game = 'sokoban_basic'
    with open(PS_LARK_GRAMMAR_PATH, 'r') as f:
        puzzlescript_grammar = f.read()
    parser = Lark(puzzlescript_grammar, start="ps_game", maybe_placeholders=False)
    tree, success, err_msg = get_tree_from_txt(parser, game, test_env_init=False)
    env = RepresentationWrapper(tree, debug=False, print_score=False)
    level = env.get_level(0)
    env_params = PSParams(
        level=level,
    )
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng=rng, params=env_params)

    env.print_ascii_legend()

    for i in range(10):
        action = env.action_space.sample(rng)
        rng, _rng = jax.random.split(rng)
        obs, state, rew, done, info = env.step_env(rng=rng, action=action, state=state, params=env_params)

        print(f"ASCII Game State:\n{env.render_ascii(state)}")
        print(f"Text Game State:\n{env.render_text(state)}")

        print(f"Reward: {rew}")
        print(f"Heuristic: {state.heuristic}")
        print(f"Score: {state.score}")
        print(f"Win: {state.win}")


if __name__ == "__main__":
    test_log_wrapper()

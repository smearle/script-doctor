import gym

from gymnax.environments import environment
import jax
from lark import Lark
from env import PSEnv, PSParams, PSState, multihot_to_desc
from parse_lark import PS_LARK_GRAMMAR_PATH, get_tree_from_txt
from purejaxrl.wrappers import GymnaxWrapper


# class RepresentationWrapper(GymnaxWrapper, PSEnv):
class RepresentationWrapper(PSEnv):
    """Log the episode returns and lengths."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render_ascii(self, state: PSState):
        pass

    def render_text(self, state: PSState):
        return multihot_to_desc(state.multihot_level, self.obj_to_idxs, self.n_objs, show_background=False)


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

    for i in range(200):
        action = env.action_space.sample(rng)
        rng, _rng = jax.random.split(rng)
        obs, state, rew, done, info = env.step_env(rng=rng, action=action, state=state, params=env_params)

        # print(f"ASCII Game State:\n{env.render_ascii(state)}")
        print(f"Text Game State:\n{env.render_text(state)}")

        print(f"Reward: {rew}")
        print(f"Heuristic: {state.heuristic}")
        print(f"Score: {state.score}")
        print(f"Win: {state.win}")


if __name__ == "__main__":
    test_log_wrapper()
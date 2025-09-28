from functools import partial
import chex
import jax
import jax.numpy as jnp
from puxle import Puzzle, state_dataclass, FieldDescriptor

from puzzlejax.env import PuzzleJaxEnv, PJState
from puzzlejax.env_utils import multihot_to_desc
from puzzlejax.utils import init_ps_env


class PuzzleJaxPuxleEnv(Puzzle):
    action_size = 5

    def __init__(self, game: str, level_i: int):
        self.env: PuzzleJaxEnv = init_ps_env(game, level_i, max_episode_steps=1000)

    def get_solve_config(self):
        return Puzzle.SolveConfig()

    def define_state_class(self):
        return PJState

    def get_initial_state(self, solve_config, key=None, data=None):
        obs, state = self.env.reset(key)
        return state

    def get_neighbours(self, solve_config, state, filled=True):
        actions = jnp.arange(self.action_size)
        next_rets = jax.vmap(lambda a: self.env.step(state, a)[1])(actions)
        next_states = next_rets[1]

        # All moves have cost 1
        costs = jnp.ones(self.action_size)

        return next_states, costs

    def is_solved(self, solve_config, state: PJState):
        return state.win

    def get_string_parser(self):
        # Required: Return function to convert state to string
        _multi_hot_to_desc = partial(multihot_to_desc, objs_to_idxs=self.objs_to_idxs, n_objs=len(self.objs_to_idxs), obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs, show_background=False)
        def string_parser(state: PJState):
            return _multi_hot_to_desc(state.multihot_level)

        return string_parser

    def get_img_parser(self):
        def img_parser(state):
            return self.env.render(state, cv2=False)

        return img_parser


class PuzzleJaxHeuristic():
    puzzle: Puzzle  # The puzzle rule object

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

    def batched_distance(
        self, solve_config: Puzzle.SolveConfig, current: Puzzle.State
    ) -> chex.Array:
        """
        This function should return the distance between the state and the target.
        """
        return jax.vmap(self.distance, in_axes=(None, 0))(solve_config, current)

    def distance(self, solve_config: Puzzle.SolveConfig, current: PJState) -> float:
        """
        This function should return the distance between the state and the target.

        Args:
            solve_config: The solve config.
            current: The current state.

        Returns:
            The distance between the state and the target.
            shape : single scalar
        """
        return current.heuristic
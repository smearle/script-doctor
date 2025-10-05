from functools import partial
from typing import Any
import chex
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle, PuzzleState, state_dataclass, FieldDescriptor

from puzzlejax.env import PuzzleJaxEnv, PJState, PJParams
from puzzlejax.env_utils import multihot_to_desc
from puzzlejax.utils import init_ps_env


class PuzzleJaxPuxleEnv(Puzzle):
    action_size = 5

    def __init__(self, game: str, level_i: int):
        self.game = game
        self.level_i = level_i

        self.env: PuzzleJaxEnv = init_ps_env(game, level_i, max_episode_steps=1000)
        level = self.env.get_level(level_i)
        self.params = PJParams(level=level)

        _, init_state = self.env.reset(jax.random.PRNGKey(0), self.params)
        self._init_state_template = jax.tree_util.tree_map(jnp.asarray, init_state)

        # Cache commonly accessed metadata from the underlying environment.
        self.objs_to_idxs = self.env.objs_to_idxs
        self.obj_idxs_to_force_idxs = self.env.obj_idxs_to_force_idxs

        super().__init__()

    def define_state_class(self):
        template = self._init_state_template
        string_parser = self.get_string_parser()
        multihot_shape = template.multihot_level.shape
        rng_shape = template.rng.shape

        @state_dataclass
        class State(PuzzleState):
            multihot_level: FieldDescriptor[jnp.bool_, multihot_shape, False]
            win: FieldDescriptor[jnp.bool_, (), False]
            heuristic: FieldDescriptor[jnp.int32, (), 0]
            rng: FieldDescriptor[jnp.uint32, rng_shape, 0]
            prev_heuristic: FieldDescriptor[jnp.int32, (), 0]
            init_heuristic: FieldDescriptor[jnp.int32, (), 0]
            restart: FieldDescriptor[jnp.bool_, (), False]
            step_i: FieldDescriptor[jnp.int32, (), 0]
            score: FieldDescriptor[jnp.int32, (), 0]

            @classmethod
            def default(cls, shape: Any = ...):
                del shape
                return State(
                    multihot_level=template.multihot_level,
                    win=template.win,
                    heuristic=template.heuristic,
                    rng=template.rng,
                    prev_heuristic=template.prev_heuristic,
                    init_heuristic=template.init_heuristic,
                    restart=template.restart,
                    step_i=template.step_i,
                    score=template.score,
                )

            def __str__(self, **kwargs):
                return string_parser(self, **kwargs)

        return State

    def get_solve_config(self, key=None, data=None):
        return self.SolveConfig(TargetState=self.State.default())

    def get_initial_state(self, solve_config, key=None, data=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        _, state = self.env.reset(key, self.params)
        return self._pj_to_state(state)

    def get_neighbours(self, solve_config, state, filled=True):
        actions = jnp.arange(self.action_size)
        rng = state.rng
        pj_state = self._state_to_pj(state)

        def step_action(action):
            _, next_state, _, _, _ = self.env.step_env(rng, pj_state, action, self.params)
            return next_state

        next_states = jax.vmap(step_action)(actions)
        next_states = self.State(
            multihot_level=next_states.multihot_level,
            win=next_states.win,
            heuristic=next_states.heuristic,
            rng=next_states.rng,
            prev_heuristic=next_states.prev_heuristic,
            init_heuristic=next_states.init_heuristic,
            restart=next_states.restart,
            step_i=next_states.step_i,
            score=next_states.score,
        )

        # All moves have cost 1
        costs = jnp.ones(self.action_size)

        return next_states, costs

    def _state_to_pj(self, state: Puzzle.State) -> PJState:
        return PJState(
            multihot_level=state.multihot_level,
            win=state.win,
            score=state.score,
            heuristic=state.heuristic,
            restart=state.restart,
            init_heuristic=state.init_heuristic,
            prev_heuristic=state.prev_heuristic,
            step_i=state.step_i,
            rng=state.rng,
        )

    def _pj_to_state(self, state: PJState) -> Puzzle.State:
        return self.State(
            multihot_level=state.multihot_level,
            win=state.win,
            heuristic=state.heuristic,
            rng=state.rng,
            prev_heuristic=state.prev_heuristic,
            init_heuristic=state.init_heuristic,
            restart=state.restart,
            step_i=state.step_i,
            score=state.score,
        )

    def is_solved(self, solve_config, state: Puzzle.State):
        return state.win

    def get_string_parser(self):
        # Required: Return function to convert state to string
        _multi_hot_to_desc = partial(
            multihot_to_desc, objs_to_idxs=self.objs_to_idxs, n_objs=len(self.objs_to_idxs),
            obj_idxs_to_force_idxs=self.obj_idxs_to_force_idxs, show_background=False, show_forces=False)
        def string_parser(state: Puzzle.State, solve_config=None, **kwargs):
            del solve_config, kwargs
            return _multi_hot_to_desc(state.multihot_level)

        return string_parser

    def get_img_parser(self):
        def img_parser(state, **kwargs):
            pj_state = self._state_to_pj(state)
            im = self.env.render(pj_state, cv2=False)
            return np.array(im)

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

    def distance(self, solve_config: Puzzle.SolveConfig, current: Puzzle.State) -> float:
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

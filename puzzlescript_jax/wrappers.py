from functools import partial
from typing import Any
from importlib import import_module
import math
import chex
import jax
import jax.numpy as jnp
import numpy as np
from puxle import Puzzle, PuzzleState, state_dataclass, FieldDescriptor

from puzzlescript_jax.env import PuzzleJaxEnv, PJState, PJParams
from puzzlescript_jax.env_utils import multihot_to_desc
from puzzlescript_jax.utils import init_ps_env


def _ensure_jaxtar_current_padding_compat():
    try:
        Current = import_module("JAxtar.stars.search_base").Current
    except Exception:
        return

    if hasattr(Current, "padding_as_batch"):
        return

    def _padding_as_batch(self, shape):
        batch_size = shape[0] if isinstance(shape, tuple) else int(shape)
        padded = Current.default((batch_size,))
        return padded.at[0].set(self[0])

    Current.padding_as_batch = _padding_as_batch


_ensure_jaxtar_current_padding_compat()


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

    def get_actions(
        self,
        solve_config: Puzzle.SolveConfig,
        state: Puzzle.State,
        action: chex.Array,
        filled: bool = True,
    ) -> tuple[Puzzle.State, chex.Array]:
        del solve_config
        pj_state = self._state_to_pj(state)
        _, next_state, _, _, _ = self.env.step_env(state.rng, pj_state, action, self.params)
        next_state = self._pj_to_state(next_state)

        changed = jnp.any(next_state.multihot_level != state.multihot_level)
        cost = jnp.where(changed, jnp.array(1.0, dtype=jnp.float32), jnp.array(jnp.inf))

        next_state = jax.lax.cond(
            filled,
            lambda: jax.tree.map(
                lambda new_v, old_v: jnp.where(changed, new_v, old_v),
                next_state,
                state,
            ),
            lambda: next_state,
        )

        return next_state, cost

    def define_state_class(self):
        template = self._init_state_template
        string_parser = self.get_string_parser()
        multihot_shape = template.multihot_level.shape
        rng_shape = template.rng.shape
        view_bounds_shape = template.view_bounds.shape

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
            view_bounds: FieldDescriptor[jnp.int32, view_bounds_shape, 0]

            @classmethod
            def default(cls, shape: Any = ...):
                if shape is ... or shape == ():
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
                        view_bounds=template.view_bounds,
                    )

                if isinstance(shape, int):
                    batch_shape = (shape,)
                else:
                    batch_shape = tuple(shape)

                return State(
                    multihot_level=jnp.broadcast_to(
                        template.multihot_level, batch_shape + template.multihot_level.shape
                    ),
                    win=jnp.broadcast_to(template.win, batch_shape),
                    heuristic=jnp.broadcast_to(template.heuristic, batch_shape),
                    rng=jnp.broadcast_to(template.rng, batch_shape + template.rng.shape),
                    prev_heuristic=jnp.broadcast_to(template.prev_heuristic, batch_shape),
                    init_heuristic=jnp.broadcast_to(template.init_heuristic, batch_shape),
                    restart=jnp.broadcast_to(template.restart, batch_shape),
                    step_i=jnp.broadcast_to(template.step_i, batch_shape),
                    score=jnp.broadcast_to(template.score, batch_shape),
                    view_bounds=jnp.broadcast_to(template.view_bounds, batch_shape + view_bounds_shape),
                )

            def flatten(self):
                flat_batch = (math.prod(self.win.shape),)
                return State(
                    multihot_level=jnp.reshape(self.multihot_level, flat_batch + multihot_shape),
                    win=jnp.reshape(self.win, flat_batch),
                    heuristic=jnp.reshape(self.heuristic, flat_batch),
                    rng=jnp.reshape(self.rng, flat_batch + rng_shape),
                    prev_heuristic=jnp.reshape(self.prev_heuristic, flat_batch),
                    init_heuristic=jnp.reshape(self.init_heuristic, flat_batch),
                    restart=jnp.reshape(self.restart, flat_batch),
                    step_i=jnp.reshape(self.step_i, flat_batch),
                    score=jnp.reshape(self.score, flat_batch),
                    view_bounds=jnp.reshape(self.view_bounds, flat_batch + view_bounds_shape),
                )

            def reshape(self, shape):
                if isinstance(shape, int):
                    batch_shape = (shape,)
                else:
                    batch_shape = tuple(shape)
                return State(
                    multihot_level=jnp.reshape(self.multihot_level, batch_shape + multihot_shape),
                    win=jnp.reshape(self.win, batch_shape),
                    heuristic=jnp.reshape(self.heuristic, batch_shape),
                    rng=jnp.reshape(self.rng, batch_shape + rng_shape),
                    prev_heuristic=jnp.reshape(self.prev_heuristic, batch_shape),
                    init_heuristic=jnp.reshape(self.init_heuristic, batch_shape),
                    restart=jnp.reshape(self.restart, batch_shape),
                    step_i=jnp.reshape(self.step_i, batch_shape),
                    score=jnp.reshape(self.score, batch_shape),
                    view_bounds=jnp.reshape(self.view_bounds, batch_shape + view_bounds_shape),
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
            view_bounds=state.view_bounds,
        )

    def _pj_to_state(self, state: PJState) -> Puzzle.State:
        level = state.multihot_level
        win, _, heuristic = self.env.check_win(level)
        return self.State(
            multihot_level=level,
            win=win,
            heuristic=heuristic,
            rng=self._init_state_template.rng,
            prev_heuristic=heuristic,
            init_heuristic=heuristic,
            restart=jnp.array(False),
            step_i=jnp.array(0, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            view_bounds=state.view_bounds,
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
            visible_level = self.env.get_visible_multihot_level(
                level=state.multihot_level, view_bounds=state.view_bounds
            )
            return _multi_hot_to_desc(visible_level)

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
            shape : single scalar (non-negative)
        """
        # state.heuristic is stored as a negative value (e.g. -manhattan_dist)
        # by the env's check_win functions.  Negate so callers get a proper
        # non-negative distance estimate.
        return -current.heuristic

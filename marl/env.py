from collections import OrderedDict
import math
import os
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial

from marl.spaces import Box
from spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from typing import Optional, List, Sequence, Tuple, TypeVar, Union, Dict
from env import PSState, PSParams
from marl.model import ScannedRNN

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
from jax_utils import stack_leaves

TEnvState = TypeVar("TEnvState", bound="State")
TEnvParams = TypeVar("TEnvParams", bound="EnvParams")

DEBUG = False

class EnvParams:
    max_steps_in_episode: int = 1


@struct.dataclass
class State:
    done: chex.Array
    step: int


class MultiAgentEnv(object):
    """Jittable abstract base class for all jaxmarl Environments."""

    def __init__(
        self,
        num_agents: int,
    ) -> None:
        """
        num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Optional[TEnvParams]) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        raise NotImplementedError

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents.

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


# @struct.dataclass
# class MultiAgentLogState(LogEnvState):
#     pass


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    # def _batchify(self, x: dict):
    #     x = jnp.stack([x[a] for a in self._env.agents])
    #     return x.reshape((self._N_AGENTS, -1))

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


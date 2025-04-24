import functools
import math
from typing import Sequence
from envs.pcgrl_env import PCGRLObs
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from jax import numpy as jnp
import jax
import numpy as np

from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict

from conf.config import MultiAgentConfig
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
# from flax.training import orbax_utils
import numpy as np
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
import distrax


class ActorCategorical(nn.Module):
    action_dim: Sequence[int]
    subnet: nn.Module

    @nn.compact
    def __call__(self, hidden, x):
        action_logits = self.subnet.__call__(hidden, x)
        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: MultiAgentConfig

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config.hidden_dims[0], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config.hidden_dims[0], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = (nn.sigmoid(actor_mean) - 0.5) * 2
        unavail_actions = 1 - avail_actions

        actor_mean = actor_mean - (unavail_actions * 1e10)

        return actor_mean


class ActorMLP(nn.Module):
    action_dim: Sequence[int]
    config: MultiAgentConfig

    @nn.compact
    def __call__(self, _, x):
        obs, dones, avail_actions = x
        x = Dense(self.action_dim, self.config.hidden_dims[0], 'relu')(x)
        actor_mean = (nn.sigmoid(x) - 0.5) * 2
        unavail_actions = 1 - avail_actions

        actor_mean = actor_mean - (unavail_actions * 1e10)

        return actor_mean


class ActorBox(nn.Module):
    action_dim: Sequence[int]
    subnet: nn.Module

    @nn.compact
    def __call__(self, hidden, x):
        actor_mean = self.subnet.__call__(hidden, x)
        # action_logits = actor_mean - (unavail_actions * 1e10)
        # pi = distrax.Categorical(logits=action_logits)
        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return hidden, pi


class CriticRNN(nn.Module):
    config: MultiAgentConfig
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config.hidden_dims[0], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(self.config.hidden_dims[0], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        #print('ins', ins)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))




class Dense(nn.Module):
    action_dim: Sequence[int]
    hidden_dim: int = 64
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        act = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        act = activation(act)
        act = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(act)
        act = activation(act)
        act = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class MAConvForward2(nn.Module):
    """The way we crop out actions and values in ConvForward1 results in 
    values skipping conv layers, which is not what we intended. This matches
    the conv-dense model in the original paper without accounting for arf or 
    vrf."""
    action_dim: Sequence[int]
    act_shape: Tuple[int, int]
    hidden_dims: Tuple[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        flat_action_dim = self.action_dim * math.prod(self.act_shape)
        h1, h2 = self.hidden_dims

        if h2 == -1:
            raise Exception(
                f"The second hidden dimension has undefined size of {h2}. "
                """If running a single job from the command line with `python train_ma.py`, be sure to specify `hidden_dims="[X, Y]"). """
                "If running a sweep with `python sweep.py`, be sure that hidden_dims are specified in your sweep config, or loaded from `conf/.*hid_params.json file."
            )

        map_x = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(map_x)
        map_x = activation(map_x)
        map_x = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(map_x)
        map_x = activation(map_x)

        map_x = map_x.reshape((*map_x.shape[:-3], -1))
        x = jnp.concatenate((map_x, flat_x), axis=-1)

        x = nn.Dense(
            h2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        x = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        act, critic = x, x

        act = nn.Dense(
            flat_action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class ActorCriticPS(nn.Module):
    """Transform the action output into a distribution. Do some pre- and post-processing specific to the 
    PCGRL environments."""
    subnet: nn.Module
    act_shape: Tuple[int, int]
    n_agents: int
    n_ctrl_metrics: int

    @nn.compact
    def __call__(self, x: PCGRLObs, avail_actions):
        map_obs = x.map_obs
        ctrl_obs = x.flat_obs   

        # Hack. We had to put dummy ctrl obs's here to placate jax tree map during minibatch creation (FIXME?)
        # Now we need to remove them :)
        ctrl_obs = ctrl_obs[..., :self.n_ctrl_metrics]

        # n_gpu = x.shape[0]
        # n_envs = x.shape[1]
        # x_shape = x.shape[2:]
        # x = x.reshape((n_gpu * n_envs, *x_shape)) 

        actor_mean, val = self.subnet(map_obs, ctrl_obs)
        # actor_mean = (nn.sigmoid(actor_mean) - 0.5) * 2

        unavail_actions = 1 - avail_actions

        actor_mean = actor_mean - (unavail_actions * 1e10)

        actor_mean = actor_mean.reshape((actor_mean.shape[0], *self.act_shape, -1))
        # breakpoint()

        # actor_mean = actor_mean[None]

        pi = distrax.Categorical(logits=actor_mean)

        return pi, val


import math
from functools import partial
from timeit import default_timer as timer
from typing import Sequence, Tuple

from puzzlejax.categorical import Categorical
from flax.linen.initializers import constant, orthogonal
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp

from puzzlescript_jax.env import PSObs

def crop_rf(x, rf_size):
    mid_x = x.shape[1] // 2
    mid_y = x.shape[2] // 2
    return x[:, mid_x-math.floor(rf_size/2):mid_x+math.ceil(rf_size/2),
             mid_y-math.floor(rf_size/2):mid_y+math.ceil(rf_size/2)]

def crop_arf_vrf(x, arf_size, vrf_size):
    return crop_rf(x, arf_size), crop_rf(x, vrf_size)


class Dense(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    hidden_dim: int = 700

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        map_x = map_x.reshape((map_x.shape[0], -1))
        if flat_x is not None:
            x = jnp.concatenate((map_x, flat_x), axis=-1)
        else:
            x = map_x
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


class ConvForward2(nn.Module):
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

        # obs is stored CHW; Flax Conv expects NHWC → transpose
        map_x = jnp.transpose(map_x, (0, 2, 3, 1))

        map_x = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(map_x)
        map_x = activation(map_x)
        map_x = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(map_x)
        map_x = activation(map_x)

        map_x = map_x.reshape((map_x.shape[0], -1))
        if flat_x is not None:
            x = jnp.concatenate((map_x, flat_x), axis=-1)
        else:
            x = map_x

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


class ConvForward(nn.Module):
    action_dim: Sequence[int]
    act_shape: Tuple[int, int]
    arf_size: int
    vrf_size: int
    hidden_dims: Tuple[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        h1, h2 = self.hidden_dims

        flat_action_dim = self.action_dim * math.prod(self.act_shape)

        act, critic = crop_arf_vrf(map_x, self.arf_size, self.vrf_size)

        act = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)
        act = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)

        act = act.reshape((act.shape[0], -1))
        act = jnp.concatenate((act, flat_x), axis=-1)

        act = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(act)
        act = activation(act)

        act = nn.Dense(
            flat_action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = critic.reshape((critic.shape[0], -1))
        critic = jnp.concatenate((critic, flat_x), axis=-1)

        critic = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class SeqNCA(nn.Module):
    action_dim: Sequence[int]
    act_shape: Tuple[int, int]
    arf_size: int
    vrf_size: int
    hidden_dims: Tuple[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        h1 = self.hidden_dims[0]

        hid = nn.Conv(
            features=h1, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        )(map_x)
        hid = activation(hid)

        act, critic = crop_arf_vrf(hid, self.arf_size, self.vrf_size)

        flat_action_dim = self.action_dim * math.prod(self.act_shape)

        act = act.reshape((act.shape[0], -1))
        act = jnp.concatenate((act, flat_x), axis=-1)
        act = nn.Dense(
            h1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(act)
        act = nn.Dense(
            flat_action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = critic.reshape((critic.shape[0], -1))
        critic = jnp.concatenate((critic, flat_x), axis=-1)
        critic = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class NCA(nn.Module):
    representation: str
    tile_action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Tile the flat observations to match the map dimensions
        flat_x = jnp.tile(flat_x[:, None, None, :], (1, map_x.shape[1], map_x.shape[2], 1))

        # Concatenate the map and flat observations along the channel dimension
        x = jnp.concatenate((map_x, flat_x), axis=-1)

        x = nn.Conv(features=256, kernel_size=(9, 9), padding="SAME")(x)
        x = activation(x)
        x = nn.Conv(features=256, kernel_size=(5, 5), padding="SAME")(x)
        x = activation(x)
        x = nn.Conv(features=self.tile_action_dim,
                      kernel_size=(3, 3), padding="SAME")(x)

        if self.representation == 'wide':
            act = x.reshape((x.shape[0], -1))

        elif self.representation == 'nca':
            act = x

        else:
            raise NotImplementedError(f"Representation {self.representation} not implemented for NCA model.")

        # Generate random binary mask
        # mask = jax.random.uniform(rng[0], shape=actor_mean.shape) > 0.9
        # Apply mask to logits
        # actor_mean = actor_mean * mask
        # actor_mean = (actor_mean + x) / 2

        # actor_mean *= 10
        # actor_mean = nn.softmax(actor_mean, axis=-1)

        # critic = nn.Conv(features=256, kernel_size=(3,3), padding="SAME")(x)
        # critic = activation(critic)
        # # actor_mean = nn.Conv(
        #       features=256, kernel_size=(3,3), padding="SAME")(actor_mean)
        # # actor_mean = activation(actor_mean)
        # critic = nn.Conv(
        #       features=1, kernel_size=(1,1), padding="SAME")(critic)

        # return act, critic

        critic = activation(x)
        critic = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding="SAME")(x)
        critic = activation(critic)
        critic = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding="SAME")(x)
        critic = activation(critic)
        critic = critic.reshape((critic.shape[0], -1))
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class AutoEncoder(nn.Module):
    representation: str
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        act = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2),
                      padding="SAME")(x)
        act = activation(act)
        act = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2),
                      padding="SAME")(act)
        act = activation(act)
        act = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2),
                               padding="SAME")(act)
        act = activation(act)
        act = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2),
                               padding="SAME")(act)
        act = activation(act)
        act = nn.Conv(features=self.action_dim,
                      kernel_size=(3, 3), padding="SAME")(act)

        if self.representation == 'wide':
            act = act.reshape((x.shape[0], -1))

        critic = x.reshape((x.shape[0], -1))
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class ActorCriticPS(nn.Module):
    """Transform the action output into a distribution. Do some pre- and post-processing specific to the
    PS environments."""
    subnet: nn.Module

    @nn.compact
    def __call__(self, x: PSObs):
        obs_2d = x.multihot_level
        obs_1d = x.flat_obs
        act, val = self.subnet(obs_2d, obs_1d)
        pi = Categorical(logits=act)

        return pi, val


class ConvEncoder(nn.Module):
    """Spatial encoder shared by the recurrent actor-critic. Maps a batch of
    multi-hot levels (N, C, H, W) to a fixed-size embedding (N, hidden_dim)."""
    hidden_dim: int = 256
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        # obs stored CHW; Flax Conv expects NHWC.
        x = jnp.transpose(map_x, (0, 2, 3, 1)).astype(jnp.float32)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME",
                    kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = activation(x)
        return x


class ScannedRNN(nn.Module):
    """GRU scanned over a (time, batch, feat) sequence with per-step carry
    resets at episode boundaries (PureJaxRL-style)."""

    @partial(
        nn.scan,
        variable_broadcast="params",
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        # Reset the carry to zeros wherever an episode has just ended.
        rnn_state = jnp.where(
            resets[:, None],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    """Recurrent actor-critic for PuzzleScript. Takes a hidden state and a
    tuple (PSObs, dones) where the leading axis of each leaf is time."""
    action_dim: int
    hidden_dim: int = 256
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        map_x = obs.multihot_level  # (T, B, C, H, W)
        T, B = map_x.shape[0], map_x.shape[1]
        flat = map_x.reshape((T * B,) + map_x.shape[2:])
        emb = ConvEncoder(self.hidden_dim, self.activation)(flat)
        emb = emb.reshape((T, B, self.hidden_dim))

        rnn_in = (emb, dones)
        hidden, emb = ScannedRNN()(hidden, rnn_in)

        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                         bias_init=constant(0.0))(emb)
        actor = activation(actor)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(actor)
        pi = Categorical(logits=logits)

        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(emb)
        critic = activation(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(value, axis=-1)


class ActorCriticRNNMulti(nn.Module):
    """Generalist recurrent actor-critic shared across games. Observations are
    multi-hot levels padded to a common (C_max, H_max, W_max) shape; a learned
    per-game embedding (looked up from a game id carried in obs.flat_obs)
    conditions the policy. The id embedding is a lookup-table stand-in for a
    richer rule encoding (rule_attn / FiLM)."""
    action_dim: int
    n_games: int
    hidden_dim: int = 128
    game_embed_dim: int = 32
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        map_x = obs.multihot_level          # (T, B, C, H, W)
        game_id = obs.flat_obs              # (T, B) int game index
        T, B = map_x.shape[0], map_x.shape[1]
        flat = map_x.reshape((T * B,) + map_x.shape[2:])
        emb = ConvEncoder(self.hidden_dim, self.activation)(flat)   # (T*B, hidden)

        gid = game_id.reshape((T * B,)).astype(jnp.int32)
        gemb = nn.Embed(self.n_games, self.game_embed_dim)(gid)     # (T*B, ge)

        activation = nn.relu if self.activation == "relu" else nn.tanh
        emb = jnp.concatenate([emb, gemb], axis=-1)
        emb = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                       bias_init=constant(0.0))(emb)
        emb = activation(emb)
        emb = emb.reshape((T, B, self.hidden_dim))

        hidden, emb = ScannedRNN()(hidden, (emb, dones))

        actor = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                         bias_init=constant(0.0))(emb)
        actor = activation(actor)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(actor)
        pi = Categorical(logits=logits)

        critic = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(emb)
        critic = activation(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(value, axis=-1)

if __name__ == '__main__':
    n_trials = 100
    rng = jax.random.PRNGKey(42)
    start_time = timer()
    for _ in range(n_trials):
        rng, _rng = jax.random.split(rng)
        data = jax.random.normal(rng, (4, 256, 2))
        print('data', data)
        dist = Categorical(data)
        sample = dist.sample(seed=rng)
        print('sample', sample)
        log_prob = dist.log_prob(sample)
        print('log_prob', log_prob)
    time = timer() - start_time
    print(f'Average time per sample: {time / n_trials}')

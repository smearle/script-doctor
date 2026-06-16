import jax
import jax.numpy as jnp

from puzzlescript_jax.env import PJParams
from puzzlescript_jax.utils import init_ps_env


def test_padded_cells_are_excluded_from_run_rules_on_level_start():
    env = init_ps_env(game="test_padding_mask", level_i=-1, max_episode_steps=1, vmap=False)
    params = PJParams(level=env.get_level(0), level_i=0)

    _, state = env.reset(jax.random.PRNGKey(0), params)

    player_idx = env.objs_to_idxs["player"]
    invalid_players = state.multihot_level[player_idx] & ~state.valid_mask

    assert int(state.level_height) == 1
    assert int(state.level_width) == 1
    assert jnp.sum(state.valid_mask) == 1
    assert jnp.sum(state.multihot_level[player_idx]) == 1
    assert not jnp.any(invalid_players)

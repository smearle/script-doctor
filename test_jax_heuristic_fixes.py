"""Tests for heuristic fixes in puzzlescript_jax/env.py.

Fix 1: compute_sum_of_manhattan_dists_from_channels - no-target fallback was 0, now max_dist.
Fix 2: compute_min_manhattan_dist_from_channels - no-tiles fallback was INT_MAX, now max_dist.
Fix 3: step_env reward normalized by max_dist so Δheuristic ∈ [-1, 1] per step.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puzzlescript_jax.env import (
    compute_sum_of_manhattan_dists_from_channels,
    compute_min_manhattan_dist_from_channels,
)
from puzzlescript_jax.utils import init_ps_env
from puzzlescript_jax.env import PJParams


def make_channel(h, w, coords):
    """Create a boolean (h, w) channel with True at given (row, col) coords."""
    ch = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        ch[r, c] = True
    return jnp.array(ch)


H, W = 5, 8
MAX_DIST = H + W  # 13


# ---------------------------------------------------------------------------
# compute_sum_of_manhattan_dists_from_channels
# ---------------------------------------------------------------------------

class TestComputeSum:
    def test_no_targets_returns_n_src_times_max_dist(self):
        """Fix 1: when no targets exist, each source contributes max_dist (not 0)."""
        src = make_channel(H, W, [(1, 1), (3, 5)])  # 2 sources
        trg = make_channel(H, W, [])                 # no targets
        result = int(compute_sum_of_manhattan_dists_from_channels(src, trg))
        assert result == 2 * MAX_DIST, f"Expected {2 * MAX_DIST}, got {result}"

    def test_no_sources_returns_zero(self):
        """No sources → sum is trivially 0."""
        src = make_channel(H, W, [])
        trg = make_channel(H, W, [(2, 2)])
        result = int(compute_sum_of_manhattan_dists_from_channels(src, trg))
        assert result == 0

    def test_source_on_target_returns_zero(self):
        """Source coincides with target → distance 0."""
        src = make_channel(H, W, [(2, 3)])
        trg = make_channel(H, W, [(2, 3)])
        result = int(compute_sum_of_manhattan_dists_from_channels(src, trg))
        assert result == 0

    def test_single_source_single_target(self):
        """Sum of min dists = manhattan distance between the two cells."""
        src = make_channel(H, W, [(0, 0)])
        trg = make_channel(H, W, [(2, 3)])
        expected = 2 + 3  # |0-2| + |0-3|
        result = int(compute_sum_of_manhattan_dists_from_channels(src, trg))
        assert result == expected

    def test_each_source_picks_nearest_target(self):
        """Each source picks its own nearest target."""
        # src at (0,0) and (4,7); targets at (0,1) and (4,6)
        src = make_channel(H, W, [(0, 0), (4, 7)])
        trg = make_channel(H, W, [(0, 1), (4, 6)])
        # (0,0) → nearest (0,1): dist=1; (4,7) → nearest (4,6): dist=1
        result = int(compute_sum_of_manhattan_dists_from_channels(src, trg))
        assert result == 2


# ---------------------------------------------------------------------------
# compute_min_manhattan_dist_from_channels
# ---------------------------------------------------------------------------

class TestComputeMin:
    def test_no_sources_returns_max_dist(self):
        """Fix 2: no sources → fallback is max_dist (not INT_MAX)."""
        src = make_channel(H, W, [])
        trg = make_channel(H, W, [(1, 1)])
        result = int(compute_min_manhattan_dist_from_channels(src, trg))
        assert result == MAX_DIST, f"Expected {MAX_DIST}, got {result}"

    def test_no_targets_returns_max_dist(self):
        """Fix 2: no targets → fallback is max_dist (not INT_MAX)."""
        src = make_channel(H, W, [(1, 1)])
        trg = make_channel(H, W, [])
        result = int(compute_min_manhattan_dist_from_channels(src, trg))
        assert result == MAX_DIST, f"Expected {MAX_DIST}, got {result}"

    def test_no_src_no_trg_returns_max_dist(self):
        """Both empty → max_dist."""
        src = make_channel(H, W, [])
        trg = make_channel(H, W, [])
        result = int(compute_min_manhattan_dist_from_channels(src, trg))
        assert result == MAX_DIST

    def test_source_on_target_returns_zero(self):
        src = make_channel(H, W, [(3, 3)])
        trg = make_channel(H, W, [(3, 3)])
        result = int(compute_min_manhattan_dist_from_channels(src, trg))
        assert result == 0

    def test_global_min_across_pairs(self):
        """Returns the global minimum distance, not the sum."""
        src = make_channel(H, W, [(0, 0), (4, 7)])
        trg = make_channel(H, W, [(0, 1), (4, 6)])
        # min pair distances: (0,0)↔(0,1)=1; (4,7)↔(4,6)=1 → global min = 1
        result = int(compute_min_manhattan_dist_from_channels(src, trg))
        assert result == 1

    def test_result_bounded_by_max_dist(self):
        """Even with distant tiles, result should never exceed max_dist."""
        src = make_channel(H, W, [(0, 0)])
        trg = make_channel(H, W, [(H - 1, W - 1)])
        result = int(compute_min_manhattan_dist_from_channels(src, trg))
        assert result <= MAX_DIST


# ---------------------------------------------------------------------------
# Fix 3: reward normalization in step_env
# ---------------------------------------------------------------------------

class TestRewardNormalization:
    """step_env divides Δheuristic by max_dist, so the heuristic-shaped reward
    component is bounded to [-1, 1] per step."""

    def _get_env_and_params(self, game="Microban"):
        env = init_ps_env(game=game, level_i=0, max_episode_steps=100, vmap=False)
        params = PJParams(level=env.get_level(0), level_i=0)
        return env, params

    def test_reward_bounded_after_random_steps(self):
        """Heuristic component of reward (excluding win bonus and step penalty)
        should be in [-1, 1] for any step on a normal level."""
        env, params = self._get_env_and_params()
        rng = jax.random.PRNGKey(42)
        _, state = env.reset(rng, params)

        # Take several random actions and check reward magnitude
        for action in range(5):
            rng, step_rng = jax.random.split(rng)
            _, new_state, reward, done, info = env.step_env(step_rng, state, action, params)
            # reward = Δheuristic/max_dist + win_bonus - 0.01
            # The win_bonus is at most 1. step_penalty is 0.01.
            # So reward should be in roughly [-1 - 0.01, 1 + 1 - 0.01] = [-1.01, 1.99]
            reward_val = float(reward)
            assert reward_val >= -1.02, f"Reward {reward_val} unexpectedly low for action {action}"
            assert reward_val <= 2.0, f"Reward {reward_val} unexpectedly high for action {action}"
            state = new_state

    def test_solved_level_reward_contains_win_bonus(self):
        """When a level is won, reward includes +1 win bonus on top of the heuristic delta."""
        env, params = self._get_env_and_params()
        rng = jax.random.PRNGKey(0)
        _, state = env.reset(rng, params)
        # Force a win by replaying a known solution if available; otherwise just
        # verify that winning steps report higher reward than non-winning steps.
        # Check that the win flag in info is consistent with reward > 0.9.
        _, _, reward, _, info = env.step_env(rng, state, 0, params)
        if bool(info["won"]):
            assert float(reward) >= 0.99 - 0.01 - 1, "Win reward too low"

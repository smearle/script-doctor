"""Nearest-target heuristics must retain exact values under JIT and vmap."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from puzzlescript_jax.env import (
    compute_min_manhattan_dist_from_channels,
    compute_sum_of_manhattan_dists_from_channels,
)


def numpy_reference(src, trg):
    sources, targets = np.argwhere(src), np.argwhere(trg)
    fallback = sum(src.shape)
    if len(sources) == 0:
        return 0, fallback
    if len(targets) == 0:
        return len(sources) * fallback, fallback
    nearest = np.abs(sources[:, None] - targets[None]).sum(axis=-1).min(axis=1)
    return nearest.sum(), nearest.min()


def metrics(src, trg):
    return (
        compute_sum_of_manhattan_dists_from_channels(src, trg),
        compute_min_manhattan_dist_from_channels(src, trg),
    )


def assert_batch_matches(sources, targets):
    expected = np.asarray([numpy_reference(src, trg) for src, trg in zip(sources, targets)])
    actual = jax.jit(jax.vmap(metrics))(jnp.asarray(sources), jnp.asarray(targets))
    for i, values in enumerate(actual):
        assert values.dtype == jnp.int32
        np.testing.assert_array_equal(values, expected[:, i])


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1), (2, 3)])
def test_exhaustive_small_grids(shape):
    masks = np.asarray(list(itertools.product([False, True], repeat=np.prod(shape))))
    masks = masks.reshape(-1, *shape)
    assert_batch_matches(np.repeat(masks, len(masks), axis=0), np.tile(masks, (len(masks), 1, 1)))


@pytest.mark.parametrize("shape", [(5, 8), (8, 5), (16, 16), (31, 17)])
def test_rectangular_and_sparse_grids(shape):
    rng = np.random.default_rng(42)
    densities = np.asarray([0, 0.01, 0.1, 0.5, 1])
    sources = rng.random((100, *shape)) < rng.choice(densities, (100, 1, 1))
    targets = rng.random((100, *shape)) < rng.choice(densities, (100, 1, 1))
    assert_batch_matches(sources, targets)
    # Exercise eager execution as well as the batched, compiled path.
    for src, trg in zip(sources[:5], targets[:5]):
        np.testing.assert_array_equal(metrics(jnp.asarray(src), jnp.asarray(trg)), numpy_reference(src, trg))

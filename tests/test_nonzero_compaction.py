"""Fixed-capacity coordinate collection must retain scan order and padding."""

import itertools

import jax
import numpy as np
import pytest

from puzzlescript_jax.env import _compact_nonzero_coords


def check_masks(masks, sizes):
    for size in sizes:
        expected = np.full((len(masks), size, masks.ndim - 1), -1, dtype=np.int32)
        for i, mask in enumerate(masks):
            coords = np.argwhere(mask)[:size]
            expected[i, :len(coords)] = coords
        actual = jax.jit(jax.vmap(lambda mask: _compact_nonzero_coords(mask, size)))(masks)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(7,), (2, 3), (2, 2, 2), (1, 3, 2), (0, 3)])
def test_exhaustive_masks_and_capacities(shape):
    masks = np.array(list(itertools.product((False, True), repeat=np.prod(shape))), dtype=bool)
    masks = masks.reshape(len(masks), *shape)
    check_masks(masks, range(np.prod(shape) + 3))


@pytest.mark.parametrize("shape", [(6, 7, 12), (13, 11, 12), (12, 12, 12)])
def test_real_movement_shapes_sparse_dense_and_truncated(shape):
    rng = np.random.default_rng(42)
    masks = rng.random((24, *shape)) < np.linspace(0, 1, 24).reshape(-1, 1, 1, 1)
    check_masks(masks, [1, np.prod(shape) // 4 + 1, np.prod(shape), np.prod(shape) + 1])

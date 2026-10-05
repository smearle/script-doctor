"""Scan order matters when earlier matches invalidate later replacements."""

import itertools

import jax
import numpy as np
import pytest

from puzzlescript_jax.env import _ordered_match_coords


def assert_coordinate_order(masks):
    # Independent kernels with different masks and mixed scan directions.
    kernels = np.stack([masks, ~masks, masks[:, ::-1]], axis=1)
    for orders in itertools.product((False, True), repeat=3):
        expected = np.full((*kernels.shape[:2], masks[0].size, 2), -1, np.int32)
        for batch, boards in enumerate(kernels):
            for k, (mask, col_major) in enumerate(zip(boards, orders)):
                height, width = mask.shape
                positions = ([(r, c) for c in range(width) for r in range(height)]
                             if col_major else
                             [(r, c) for r in range(height) for c in range(width)])
                matches = [xy for xy in positions if mask[xy]]
                expected[batch, k, :len(matches)] = np.asarray(matches).reshape(-1, 2)
        result = jax.jit(jax.vmap(lambda activations: _ordered_match_coords(activations, orders)))(kernels)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (2, 3), (3, 3)])
def test_exhaustive_masks(shape):
    masks = np.array(list(itertools.product((False, True), repeat=np.prod(shape))))
    assert_coordinate_order(masks.reshape(-1, *shape))


def test_sparse_dense_and_empty_rectangular_masks():
    rng = np.random.default_rng(42)
    masks = rng.random((30, 7, 11)) < np.linspace(0, 1, 30)[:, None, None]
    assert_coordinate_order(masks)

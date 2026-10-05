"""Parallel cleanup must preserve the selected coordinates and other channels."""

import itertools

import jax
import numpy as np
import pytest

from puzzlescript_jax.env import _remove_invalid_forces
from puzzlescript_jax.env_utils import N_FORCES, N_MOVEMENTS


def oracle(level, coords, masks):
    result = level.copy()
    n_objs = masks.shape[1]
    for y, x, channel in coords:
        if y < 0:
            break
        layer = channel // N_MOVEMENTS
        if not np.any(level[0, :n_objs, x, y] & masks[layer]):
            first = n_objs + layer * N_FORCES
            result[0, first:first + N_FORCES, x, y] = False
    return result


def check(levels, masks, capacity):
    n_layers, n_objs = masks.shape
    coords = np.full((len(levels), capacity, 3), -1, dtype=np.int32)
    for i, level in enumerate(levels):
        forces = level[0, n_objs:n_objs + n_layers * N_FORCES]
        movement = forces.reshape(n_layers, N_FORCES, *level.shape[2:])[:, :N_MOVEMENTS]
        movement = movement.reshape(-1, *level.shape[2:]).transpose(2, 1, 0)
        selected = np.argwhere(movement)[:capacity]
        coords[i, :len(selected)] = selected
    expected = np.stack([oracle(level, coord, masks) for level, coord in zip(levels, coords)])
    actual = jax.jit(jax.vmap(lambda level, coord: _remove_invalid_forces(level, coord, masks)))(levels, coords)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("capacity", [0, 1, 2, 4, 6])
def test_all_single_cell_object_and_force_combinations(capacity):
    # Every object-occupancy/force-bit combination, including ACTION-only,
    # multiple simultaneous directions, no objects, and no movement forces.
    levels = np.array(list(itertools.product((False, True), repeat=7)), dtype=bool)
    levels = levels.reshape(-1, 1, 7, 1, 1)
    levels = np.concatenate([levels, np.ones((len(levels), 1, 2, 1, 1), dtype=bool)], axis=2)
    check(levels, np.ones((1, 2), dtype=bool), capacity)


@pytest.mark.parametrize("shape", [(1, 5), (4, 3), (11, 13)])
def test_multiple_layers_padding_and_truncation(shape):
    # Include an object in multiple masks; cleanup must use actual masks.
    masks = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [1, 0, 0, 0, 1]], dtype=bool)
    rng = np.random.default_rng(19)
    channels = masks.shape[1] + len(masks) * N_FORCES + 2
    levels = rng.random((24, 1, channels, *shape)) < 0.4
    levels[0] = False
    levels[1] = True
    levels[2, 0, :masks.shape[1]] = False
    levels[3, 0, :masks.shape[1]] = True
    for capacity in [0, 1, 3, np.prod(shape) * len(masks) + 1,
                     np.prod(shape) * len(masks) * N_MOVEMENTS + 1]:
        check(levels, masks, capacity)

"""Compiler/layout experiments must preserve every force coordinate."""

import jax
import numpy as np
import pytest

from scripts.benchmarks.benchmark_movement_refinements import (
    compact_coords, prefix_barrier_coords, mask_prefix_barrier_coords,
    gather_force_array, sliced_force_array,
)


@pytest.mark.parametrize("shape", [(0, 3), (2, 3), (7, 11, 12)])
@pytest.mark.parametrize("fn", [prefix_barrier_coords, mask_prefix_barrier_coords])
def test_compaction_boundaries(shape, fn):
    rng = np.random.default_rng(31)
    masks = rng.random((12, *shape)) < np.linspace(0, 1, 12).reshape((-1,) + (1,) * len(shape))
    for size in (0, 1, int(np.prod(shape)) // 4 + 1, int(np.prod(shape)) + 1):
        before = jax.jit(jax.vmap(lambda mask: compact_coords(mask, size)))(masks)
        after = jax.jit(jax.vmap(lambda mask: fn(mask, size)))(masks)
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("shape", [(5, 1, 1), (15, 7, 11), (45, 12, 8)])
def test_force_layout_preserves_direction_and_spatial_order(shape):
    # Distinct integer labels expose axis permutations, even for equal bits.
    forces = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    before = jax.jit(gather_force_array)(forces)
    after = jax.jit(sliced_force_array)(forces)
    np.testing.assert_array_equal(before, after)

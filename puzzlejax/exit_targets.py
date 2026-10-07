"""Parallel extraction of ExIt targets with the original backup semantics."""

import jax
import jax.numpy as jnp


@jax.jit
def compute_targets_and_weights(g_all, expanded_mask, env_h_all, parent_indices, solved_flag):
    """Match the original descending-cost, single-pass backup on device.

    Inputs are fixed-size search-table arrays (including unused slots). Only
    expanded nodes with finite costs participate. Environment heuristics use
    the positive distance convention, as in the original target extractor.

    A node forwards the value it has when visited by the original stable
    descending-g pass. Values received *after* that visit remain at the node
    but are never forwarded. This distinction matters for cost ties and for
    parent pointers whose costs have changed since expansion.

    Edges to later visits form a forest, since visit ranks strictly increase.
    Pointer doubling aggregates that forest in O(log(depth + 1)) rounds with
    O(N) storage. After k rounds a node has received values from descendants
    up to 2**k - 1 edges away. The final scatter along every valid edge adds
    late arrivals, preserving the scalar pass even for malformed input cycles.
    The stable sort is retained; total work is not logarithmic in N.
    """
    n = g_all.shape[0]
    idx = jnp.arange(n, dtype=jnp.int32)
    finite_mask = jnp.isfinite(g_all) & expanded_mask
    min_desc_f = jnp.where(finite_mask, g_all + env_h_all, jnp.inf)

    order = jnp.argsort(jnp.where(finite_mask, -g_all, jnp.inf), stable=True)
    rank = jnp.zeros(n, dtype=jnp.int32).at[order].set(idx)
    safe_parent = jnp.clip(parent_indices, 0, n - 1)
    valid_edge = (
        finite_mask
        & (parent_indices >= 0)
        & (parent_indices < n)
        & finite_mask[safe_parent]
    )
    destinations = jnp.where(valid_edge, parent_indices, n)
    forward_edge = valid_edge & (rank < rank[safe_parent])
    jump = jnp.where(forward_edge, parent_indices, n)

    def cond(state):
        _, ancestor = state
        return jnp.any(ancestor < n)

    def body(state):
        values, ancestor = state
        # Repeated destinations must reduce with min, not overwrite with set.
        # The out-of-bounds sentinel drops invalid edges instead of aliasing
        # the last table slot (as a negative array index would).
        values_next = values.at[ancestor].min(values, mode="drop")
        ancestor_with_sentinel = jnp.concatenate((ancestor, jnp.full((1,), n, ancestor.dtype)))
        ancestor_next = ancestor_with_sentinel[ancestor]
        return values_next, ancestor_next

    min_desc_f, _ = jax.lax.while_loop(cond, body, (min_desc_f, jump))
    min_desc_f = min_desc_f.at[destinations].min(min_desc_f, mode="drop")

    h_targets = jnp.maximum(jnp.where(finite_mask, min_desc_f - g_all, 0.0), 0.0)
    expanded_indices = jnp.where(finite_mask, size=n, fill_value=-1)[0]
    n_expanded = jnp.sum(finite_mask)
    global_min_f = jnp.min(jnp.where(finite_mask, min_desc_f, jnp.inf))
    path_quality = min_desc_f - global_min_f
    solved_weights = jnp.where(path_quality < 1.0, 3.0, 1.0)
    unsolved_weights = jnp.where(path_quality < 1.0, 1.5, 0.5)
    weights = jnp.where(solved_flag, solved_weights, unsolved_weights).astype(jnp.float32)
    return expanded_indices, n_expanded, h_targets, weights

"""Low-level state-representation helpers for the NCA world model.

Pure-numpy leaf utilities shared across data collection, the training loop,
evaluation, and rendering. This module deliberately has no dependency on
``train.py`` (or jax / the C++ backend) so any of those can import it without
creating an import cycle.

Two coordinate conventions appear here:

* Multihot observations are ``(..., C, H, W)`` uint8 (channel = canonical
  object index).
* Bitpacked states are ``(..., C, H, ceil(W/8))`` uint8 (``_pack_states`` /
  ``_unpack_states``), used to keep gallery-scale datasets in RAM.
"""
import numpy as np


def _pack_states(arr: np.ndarray) -> np.ndarray:
    """Bitpack a (..., H, W) uint8 multihot along W → (..., H, ceil(W/8)) uint8.

    Pads the last axis up to a multiple of 8 with zeros (`np.packbits` does
    this implicitly; we just need to remember the original W to unpack).
    Yields ~8x reduction in both on-disk and in-RAM size for the per-game
    state arrays — needed to keep a gallery-scale dataset in RAM without
    OOMing the worker.
    """
    return np.packbits(np.ascontiguousarray(arr, dtype=np.uint8), axis=-1)


def _unpack_states(packed: np.ndarray, W: int) -> np.ndarray:
    """Reverse of `_pack_states` — returns (..., H, W) uint8."""
    return np.unpackbits(packed, axis=-1, count=W)


def _pad_offsets(src: int, dst: int) -> tuple[int, int]:
    """(before, after) amounts to center ``src`` within ``dst`` (extra pixel goes after)."""
    delta = max(0, dst - src)
    before = delta // 2
    after = delta - before
    return before, after


def _pad_obs(obs: np.ndarray, target_C: int, target_H: int, target_W: int) -> np.ndarray:
    """Pad (N, C, H, W) observations to (N, target_C, target_H, target_W) with zeros.

    Channels are top-aligned (semantic, not spatial). Spatial dims are centered.
    """
    N, C, H, W = obs.shape
    if C == target_C and H == target_H and W == target_W:
        return obs
    oy, _ = _pad_offsets(H, target_H)
    ox, _ = _pad_offsets(W, target_W)
    padded = np.zeros((N, target_C, target_H, target_W), dtype=obs.dtype)
    padded[:, :C, oy:oy+H, ox:ox+W] = obs
    return padded


def _pad_packed(
    packed: np.ndarray, src_W: int, target_C: int, target_H: int, target_W: int,
) -> np.ndarray:
    """Pad bitpacked observations (N, C, H, ceil(src_W/8)) to packed
    (N, target_C, target_H, ceil(target_W/8)).

    Bit-alignment makes raw byte padding ambiguous (an offset that isn't a
    multiple of 8 would split a packed byte across cells), so we unpack →
    pad → repack. This is per-level, so the temporary unpacked tensor is
    bounded by the per-level transition count, never the per-game total.
    """
    N = packed.shape[0]
    if N == 0:
        return _pack_states(
            np.zeros((0, target_C, target_H, target_W), dtype=np.uint8)
        )
    unpacked = _unpack_states(packed, src_W)  # (N, C, H, src_W)
    padded = _pad_obs(unpacked, target_C, target_H, target_W)
    return _pack_states(padded)


def _dats_to_multihot_batch(
    dats, n_objs: int, width: int, height: int,
    raw_to_canonical: dict[int, int] | None = None,
    n_canonical: int | None = None,
) -> np.ndarray:
    """Batch convert bitpacked states to (n_states, out_C, H, W) uint8 multihot.

    Replaces a Python triple-loop over (state, cell, object) with a single
    vectorized pass per raw object index. Numpy does the per-cell bit-test
    across all states in C, so a workload that previously took hours of
    Python time finishes in seconds.
    """
    stride_obj = (n_objs + 31) // 32
    if raw_to_canonical is not None:
        assert n_canonical is not None
        out_C = n_canonical
    else:
        out_C = n_objs

    n_states = len(dats)
    if n_states == 0:
        return np.zeros((0, out_C, height, width), dtype=np.uint8)

    # Stack & reshape into (n_states, width, height, stride_obj) uint32.
    # The flat layout matches `(x * height + y) * stride_obj + word` from the
    # original loop, so reshape((-1, w, h, stride_obj)) lines up directly.
    # The C++ backend returns words as signed int32 — np.asarray(..., dtype=
    # np.uint32) rejects those with the high bit set. Cast to int32 first
    # (which fits the negative range), then reinterpret-view as uint32 so the
    # bit pattern is preserved.
    arr = np.asarray(dats, dtype=np.int32).view(np.uint32).reshape(
        n_states, width, height, stride_obj
    )

    out = np.zeros((n_states, out_C, height, width), dtype=np.uint8)
    for raw_i in range(n_objs):
        c = raw_to_canonical[raw_i] if raw_to_canonical is not None else raw_i
        word = raw_i // 32
        bit_mask = np.uint32(1 << (raw_i % 32))
        # mask shape: (n_states, width, height) bool. Transpose to (..., h, w)
        # to match the original `obs[c, y, x] = 1` ordering, then OR-merge
        # bits into the canonical channel.
        mask = (arr[..., word] & bit_mask) != 0
        out[:, c] |= mask.transpose(0, 2, 1).astype(np.uint8)
    return out


def _multihot_to_objects(obs: np.ndarray) -> np.ndarray:
    """Convert (n_objs, H, W) multihot to flat objects array for CPP renderer."""
    n_objs, grid_h, grid_w = obs.shape
    stride_obj = (n_objs + 31) // 32
    # Use uint32 for bitwise ops, then view as int32 for C++ compatibility
    objects = np.zeros(grid_w * grid_h * stride_obj, dtype=np.uint32)
    for x in range(grid_w):
        for y in range(grid_h):
            flat_idx = (x * grid_h + y) * stride_obj
            for obj_i in range(n_objs):
                if obs[obj_i, y, x]:
                    word = obj_i // 32
                    bit = obj_i % 32
                    objects[flat_idx + word] |= np.uint32(1 << bit)
    return objects.view(np.int32)

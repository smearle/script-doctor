"""Exact raw/normalized scores and wins for packed, multiword object masks."""
import json

import numpy as np
import pytest

from puzzlescript_cpp import _cpp


def compiled_board(width, height, objects, conditions):
    stride = objects.shape[1]
    return json.dumps({
        "objectCount": stride * 32, "layerCount": 1, "STRIDE_OBJ": stride, "STRIDE_MOV": 1,
        "rigid": False, "idDict": [f"object{i}" for i in range(stride * 32)],
        "playerMask": [False, [0] * stride], "layerMasks": [[-1] * stride],
        "rules": [], "lateRules": [], "winconditions": conditions,
        "levels": [{"type": "level", "index": 0, "lineNumber": 1, "width": width,
                    "height": height, "layerCount": 1, "objects": objects.reshape(-1).tolist()}],
        "metadata": {},
    })


def reference(objects, width, height, conditions):
    def matches(mask, aggregate):
        if mask is None:
            return np.ones(len(objects), bool)
        bits = np.asarray(mask, dtype=np.int32)
        return np.all((objects & bits) == bits, axis=1) if aggregate else np.any(objects & bits, axis=1)

    score = denominator = 0
    won = bool(conditions)
    for wc in conditions:
        source, target = matches(wc["mask1"], wc["aggr1"]), matches(wc["mask2"], wc["aggr2"])
        if wc["num"] == -1:
            count = int(np.count_nonzero(source & target))
            score += count
            denominator += count * (width + height)
            won &= count == 0
            continue
        distances = [min((abs(i // height - j // height) + abs(i % height - j % height)
                          for j in np.flatnonzero(target)), default=width + height)
                     for i in np.flatnonzero(source)]
        if wc["num"] == 0:
            score += min(distances, default=width + height)
            denominator += width + height
            won &= bool(np.any(source & target))
        else:
            score += sum(distances)
            denominator += len(distances) * (width + height)
            won &= not bool(np.any(source & ~target))
    return score, 1 - score / denominator if denominator else 0, won


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (4, 6)])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_score_and_win_match_packed_mask_reference(shape, stride, seed):
    width, height = shape
    rng = np.random.default_rng(seed)
    objects = rng.integers(0, 2**32, (width * height, stride), dtype=np.uint32).view(np.int32)
    if seed == 0:
        objects[:] = 0
    if seed == 1:
        objects[:] = -1
    masks = [[-2147483648] * stride, [3] * stride, [0] * stride]
    conditions = []
    for number in (-1, 0, 1):
        for aggregate1 in (False, True):
            for aggregate2 in (False, True):
                for target in [None, *masks]:
                    wc = {"num": number, "mask1": masks[(number + seed) % 3], "mask2": target,
                          "aggr1": aggregate1, "aggr2": aggregate2, "lineNumber": 1}
                    conditions.append(wc)
                    engine = _cpp.Engine()
                    assert engine.load_from_json(compiled_board(width, height, objects, [wc]))
                    engine.load_level(0)
                    raw, normalized, won = reference(objects, width, height, [wc])
                    assert engine.get_score() == raw
                    assert engine.get_score_normalized() == normalized
                    assert engine.check_win() == won
    # Accumulating mixed conditions must retain the original normalization.
    engine = _cpp.Engine()
    assert engine.load_from_json(compiled_board(width, height, objects, conditions))
    engine.load_level(0)
    raw, normalized, won = reference(objects, width, height, conditions)
    assert engine.get_score() == raw
    assert engine.get_score_normalized() == normalized
    assert engine.check_win() == won

"""Original JS snapshots may need more than two 32-bit words per cell."""
import numpy as np
import pytest

from puzzlejax.validate_sols_jax import multihot_level_from_js_state


@pytest.mark.parametrize('n_objects', [1, 32, 64, 97, 160])
@pytest.mark.parametrize('dictionary_data', [False, True])
def test_word_boundaries_signed_words_and_column_major_layout(n_objects, dictionary_data):
    expected = np.zeros((n_objects, 2, 3), dtype=bool)
    for channel in range(n_objects):
        expected[channel, channel % 2, (channel // 2) % 3] = True
    words = []
    for x in range(3):
        for y in range(2):
            for start in range(0, n_objects, 32):
                word = sum(1 << (c - start) for c in range(start, min(start + 32, n_objects))
                           if expected[c, y, x])
                words.append(word - 2**32 if word >= 2**31 else word)
    data = {str(i): word for i, word in enumerate(words)} if dictionary_data else words
    names = [f'object{i}' for i in range(n_objects)]
    actual = multihot_level_from_js_state({'width': 3, 'height': 2, 'dat': data}, names)
    np.testing.assert_array_equal(actual, expected)


def test_wide_aliases_are_merged_before_target_channel_reordering():
    names = ['player'] + [f'object{i}' for i in range(1, 96)] + ['PLAYER']
    state = {'width': 2, 'height': 1, 'dat': [1, 0, 0, 0, 0, 0, 0, 1]}
    actual = multihot_level_from_js_state(state, names, ['missing', 'Player', 'object63'])
    np.testing.assert_array_equal(actual, [[[False, False]], [[True, True]], [[False, False]]])


def test_legacy_packed_snapshot_still_maps_low_and_high_bits():
    actual = multihot_level_from_js_state(np.array([[1], [2**63]], dtype=np.uint64),
                                        [f'object{i}' for i in range(64)], ['object63', 'object0'])
    np.testing.assert_array_equal(actual, [[[False, True]], [[True, False]]])

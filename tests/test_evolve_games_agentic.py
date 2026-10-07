"""compile_game_text / evaluate_game: the Node engine's console verdict decides
compile success, and only real levels (not message entries) are searched."""
from pathlib import Path

from puzzlejax.evolve_games_agentic import compile_game_text, evaluate_game

# Byte-identical to gallery_games/sokoban_basic.txt, but tracked in git.
SOKOBAN_BASIC = (
    Path(__file__).resolve().parents[1] / "data" / "scraped_games" / "sokoban_basic.txt"
).read_text()
RULE = "[ > Player | Crate ] -> [ > Player | > Crate ]"
LEVELS_HEADER = "=======\nLEVELS\n=======\n"


def _with_levels(levels_section: str) -> str:
    head, _ = SOKOBAN_BASIC.split(LEVELS_HEADER)
    return head + LEVELS_HEADER + levels_section


def test_known_good_game_compiles():
    cr = compile_game_text(SOKOBAN_BASIC)
    assert cr.success, cr.error
    assert cr.error == ""
    assert cr.level_indices == [0, 1]
    assert cr.warnings == []


def test_garbage_fails():
    cr = compile_game_text("this is not a PuzzleScript game {{{ ]]]")
    assert not cr.success
    assert "Unrecognised stuff in the prelude" in cr.error
    assert cr.n_levels == 0


def test_real_compile_error_fails():
    # An undefined map key is a real error, but the engine still salvages both
    # levels, so only its "Errors detected" verdict rejects the game.
    text = SOKOBAN_BASIC.replace("#@P..#", "#@PQ.#")
    assert text != SOKOBAN_BASIC
    cr = compile_game_text(text)
    assert not cr.success
    assert 'Key "Q" not found' in cr.error
    assert "Errors detected" not in cr.error
    assert cr.n_levels == 0


def test_warning_is_informational():
    assert RULE in SOKOBAN_BASIC
    cr = compile_game_text(SOKOBAN_BASIC.replace(RULE, RULE + " sfx0"))
    assert cr.success, cr.error
    assert any('Sound effect "sfx0" not defined' in w for w in cr.warnings)


def test_message_entries_are_not_levels():
    _, levels = SOKOBAN_BASIC.split(LEVELS_HEADER)
    cr = compile_game_text(_with_levels("message hi\n\n" + levels))
    assert cr.success, cr.error
    assert cr.level_indices == [1, 2]

    cr = compile_game_text(_with_levels("message hi\n"))
    assert not cr.success
    assert cr.n_levels == 0

    # The search must load engine indices 1 and 2; index 0 is the message.
    ev = evaluate_game(_with_levels("message hi\n\n" + levels), search_timeout_ms=10_000)
    assert [lsr.level_i for lsr in ev.level_results] == [0, 1]
    assert all(lsr.solved and not lsr.error for lsr in ev.level_results)
    assert ev.all_solvable

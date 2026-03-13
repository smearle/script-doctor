from pathlib import Path

from ascii_prompting import ASCIIStateFormatter, build_human_like_prompt
from backends.nodejs import NodeJSPuzzleScriptBackend
from llm_agent_loop_nodejs import _extract_char_mapping_from_state, _get_level_message, collect_game_info


def test_ascii_state_formatter_filters_legend_to_visible_maps():
    formatter = ASCIIStateFormatter(
        {
            "@": ["player"],
            "#": ["wall"],
            "*": ["player", "box"],
            ".": ["background"],
        }
    )

    first_map = formatter.render_map_only(
        [
            [("player",), ("background",)],
            [("wall",), ("background",)],
        ]
    )
    assert first_map == "@.\n#."
    assert formatter.render_map_only([[("box", "player")]]) == "*"

    first_legend = formatter.get_legend_text([first_map])
    assert "@: player" in first_legend
    assert "#: wall" in first_legend
    assert "*: box, player" not in first_legend

    second_legend = formatter.get_legend_text([first_map, "*"])
    assert "*: box, player" in second_legend

    trimmed_legend = formatter.get_legend_text([first_map])
    assert "*: box, player" not in trimmed_legend


def test_extract_char_mapping_from_state_reads_compiled_glyph_dict():
    backend = NodeJSPuzzleScriptBackend()
    try:
        game_text = Path("custom_games/test_dot.txt").read_text(encoding="utf-8")
        backend.engine.compile(["restart"], game_text)
        char_mapping = _extract_char_mapping_from_state(backend.engine.getState())
    finally:
        backend.unload_game()

    assert char_mapping["."] == ["background"]
    assert char_mapping["@"] == ["Crate", "Target"]
    assert char_mapping["p"] == ["Player"]


def test_collect_game_info_uses_original_game_text_with_messages():
    game_info = collect_game_info("___Sixty-Five")

    assert game_info is not None
    assert game_info["game_text"].startswith("title    Sixty-Five")
    assert any(str(level["type"]) == "message" for level in game_info["level_info"])


def test_get_level_message_reads_js_proxy_message_levels():
    game_info = collect_game_info("___Sixty-Five")

    message_levels = [level for level in game_info["level_info"] if str(level["type"]) == "message"]
    assert message_levels
    assert _get_level_message(message_levels[0]) != ""


def test_build_human_like_prompt_can_skip_scratchpad():
    prompt = build_human_like_prompt(
        title="Test",
        author="Author",
        legend_text="LEGEND:\n@: player",
        ascii_map="@",
        action_space=[0, 1],
        action_meanings={0: "left", 1: "right"},
        state_history=[],
        history_limit=0,
        scratchpad="note",
        include_scratchpad=False,
    )

    assert "Your scratchpad" not in prompt
    assert "SCRATCHPAD:" not in prompt
    assert "ACTION: <id>" in prompt

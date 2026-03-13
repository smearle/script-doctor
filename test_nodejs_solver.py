from backends.nodejs import NodeJSPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser


def test_bfs_find_girlfriend_level_14_replayable_solution():
    backend = NodeJSPuzzleScriptBackend()
    parser = init_ps_lark_parser()

    try:
        game_text = backend.compile_game(parser, "Find_Girlfriend!")
        result = backend.run_search(
            "bfs",
            game_text=game_text,
            level_i=14,
            n_steps=10_000,
            timeout_ms=10_000,
        )

        assert result.solved
        assert len(result.actions) >= 2

        backend.load_level(game_text, 14)
        engine = backend.engine
        first_action = result.actions[0]
        changed = engine.processInput(first_action)
        while engine.getAgaining():
            changed = engine.processInput(-1) or changed
        assert changed
        assert not engine.getWinning()

        for action in result.actions[1:]:
            changed = engine.processInput(action)
            while engine.getAgaining():
                changed = engine.processInput(-1) or changed

        assert engine.getWinning()
    finally:
        backend.unload_game()

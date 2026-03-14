from backends.nodejs import NodeJSPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser, level_to_int_arr
from puzzlescript_nodejs.utils import replay_actions_js


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


def test_successful_solver_state_matches_replayed_actions():
    backend = NodeJSPuzzleScriptBackend()
    parser = init_ps_lark_parser()

    def assert_state_matches(algo_name, raw_result, game_text, level_i):
        assert raw_result[0], f"{algo_name} did not solve the level"
        actions = list(raw_result[1])
        returned_state = raw_result[5]
        obj_list = list(raw_result[7])

        _, js_states = replay_actions_js(
            backend.engine,
            backend.solver,
            actions,
            game_text,
            level_i,
        )

        returned_arr = level_to_int_arr(returned_state, len(obj_list))
        replayed_arr = level_to_int_arr(js_states[-1], len(obj_list))
        assert (returned_arr == replayed_arr).all(), algo_name

    try:
        game_text = backend.compile_game(parser, "Slidings")
        level_i = 0

        backend.load_level(game_text, level_i)
        assert_state_matches(
            "astar",
            backend.solver.solveAStar(backend.engine, 10_000),
            game_text,
            level_i,
        )

        backend.load_level(game_text, level_i)
        assert_state_matches(
            "gbfs",
            backend.solver.solveGBFS(backend.engine, 10_000),
            game_text,
            level_i,
        )

        backend.load_level(game_text, level_i)
        assert_state_matches(
            "mcts",
            backend.solver.solveMCTS(
                backend.engine,
                {"max_iterations": 5_000, "max_sim_length": 50},
            ),
            game_text,
            level_i,
        )
    finally:
        backend.unload_game()

import glob
import json
import math
import os
import traceback
from typing import List, Optional

import dotenv
import hydra
import submitit

from conf.config import SearchNodeJSConfig
from puzzlejax.backends import NodeJSPuzzleScriptBackend
from puzzlejax.globals import JS_SOLS_DIR
from puzzlejax.utils import get_list_of_games_for_testing, init_ps_lark_parser


dotenv.load_dotenv()


@hydra.main(version_base="1.3", config_path='conf', config_name='search_nodejs_config')
def main_launch(cfg: SearchNodeJSConfig):
    if cfg.slurm:
        games = get_list_of_games_for_testing(all_games=cfg.all_games, random_order=cfg.random_order)
        # Get sub-lists of games to distribute across nodes.
        n_jobs = math.ceil(len(games) / cfg.n_games_per_job)
        game_sublists = [games[i::n_jobs] for i in range(n_jobs)]
        assert sum(len(g) for g in game_sublists) == len(games), "Not all games are assigned to a job."
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "render_js_sols"))
        executor.update_parameters(
            slurm_job_name="render_js_sols",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
            slurm_array_parallelism=n_jobs,
            slurm_account=os.environ.get("SLURM_ACCOUNT"),
            slurm_setup=["export JAX_PLATFORMS=cpu"],
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(
    cfg: SearchNodeJSConfig,
    games: Optional[List[str]] = None,
    backend: Optional[NodeJSPuzzleScriptBackend] = None,
):
    parser = init_ps_lark_parser()
    backend = backend or NodeJSPuzzleScriptBackend()
    if cfg.slurm:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"
    if games is not None:
        game_sols_dirs = [os.path.join(JS_SOLS_DIR, game) for game in games]
    elif cfg.game is None:
        if cfg.all_games:
            game_sols_dirs = glob.glob(f"{JS_SOLS_DIR}/*")
        else:
            game_sols_dirs = get_list_of_games_for_testing(
                all_games=cfg.all_games, random_order=cfg.random_order
            )
            game_sols_dirs = [os.path.join(JS_SOLS_DIR, game) for game in game_sols_dirs]
    else:
        game_sols_dirs = [os.path.join(JS_SOLS_DIR, cfg.game)]
    for game_dir in game_sols_dirs:
        game_name = game_dir.split(os.path.sep)[-1]
        level_sol_jsons = glob.glob(f"{game_dir}/*.json")
        game_text = None
        for level_sol_json in level_sol_jsons:
            level_i = level_sol_json.split(os.path.sep)[-1].split(".")[0].split('-')[-1]
            if cfg.level is not None and int(level_i) != cfg.level:
                continue
            if '-steps' in level_sol_json:
                n_steps = level_sol_json.split(os.path.sep)[-1].split(".")[0].split('-steps')[0]
            else:
                n_steps = 10_000
            print(f"Game: {game_name}, Level: {level_i}, Steps: {n_steps}")
            level_sol_gif_path = os.path.splitext(level_sol_json)[0] + "_sol.gif"
            if os.path.isfile(level_sol_gif_path) and not cfg.overwrite:
                print(f"Level {level_i} already rendered")
                continue

            if game_text is None:
                try:
                    backend.unload_game()
                    game_text = backend.compile_game(parser, game_name)
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error compiling game {game_name} level {level_i}: {e}")
                    continue

            with open(level_sol_json, "r") as f:
                data = json.load(f)
            if 'actions' in data:
                actions = data['actions']
            elif 'sol' in data:
                actions = data['sol']
            else:
                print(f"No actions found in {level_sol_json}")
                continue

            try:
                backend.render_gif(
                    game_text=game_text,
                    level_i=int(level_i),
                    actions=actions,
                    gif_path=level_sol_gif_path,
                    frame_duration_s=1.0,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error rendering game {game_name} level {level_i}: {e}")
                continue

            # jax_val_game_dir = os.path.join(JAX_VALIDATED_JS_SOLS_DIR, game_name)
            # os.makedirs(jax_val_game_dir, exist_ok=True)
            # level_sol_jax_val_path = os.path.join(jax_val_game_dir, f"level-{level_i}_js.gif")
            # shutil.copyfile(level_sol_gif_path, level_sol_jax_val_path)
            # print(f"Level {level_i} rendered and saved to {level_sol_gif_path}, and copied to {level_sol_jax_val_path}")
            print(f"Level {level_i} rendered and saved to {level_sol_gif_path}")

            

if __name__ == '__main__':
    main_launch()

    

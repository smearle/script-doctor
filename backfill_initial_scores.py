"""Backfill score_initial into existing search result JSONs.

For each result JSON in js_sols/ (or cpp_sols/), loads the game and level,
computes the initial heuristic score, and writes it into the result file.
Existing fields are preserved. Results that already have score_initial are
skipped unless --overwrite is passed.

Can be parallelized on SLURM with slurm=True.

Usage:
    python backfill_initial_scores.py                           # js backend, pedro dataset
    python backfill_initial_scores.py backend=cpp dataset=increpare
    python backfill_initial_scores.py slurm=True n_games_per_job=50
"""

import glob
import json
import os
import re
import traceback
from typing import List, Optional

import hydra
import submitit
from dataclasses import dataclass
from tqdm import tqdm

from hydra.core.config_store import ConfigStore
from puzzlescript_jax.globals import JS_SOLS_DIR, CPP_SOLS_DIR
from puzzlescript_jax.utils import (
    get_list_of_games_for_testing,
    init_ps_lark_parser,
    distribute_slurm_jobs,
)


@dataclass
class BackfillConfig:
    dataset: str = "pedro"
    backend: str = "js"  # "js" or "cpp"
    overwrite: bool = False
    trajectories: bool = True  # also backfill score_trajectory
    slurm: bool = False
    n_games_per_job: int = 1
    slurm_timeout_min: int = 120


cs = ConfigStore.instance()
cs.store(name="backfill_config", node=BackfillConfig)


RESULT_FILENAME_RE = re.compile(
    r'^(astar|bfs|gbfs|mcts|random)_(\d+)-steps_level-(\d+)\.json$'
)


def _init_backend(backend_name: str):
    if backend_name == "js":
        from backends.nodejs import NodeJSPuzzleScriptBackend
        return NodeJSPuzzleScriptBackend()
    elif backend_name == "cpp":
        from puzzlescript_cpp import CppPuzzleScriptBackend
        return CppPuzzleScriptBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def _sols_dir(backend_name: str) -> str:
    return JS_SOLS_DIR if backend_name == "js" else CPP_SOLS_DIR


@hydra.main(version_base="1.3", config_path="./", config_name="backfill_config")
def main_launch(cfg: BackfillConfig):
    if cfg.slurm:
        games = get_list_of_games_for_testing(dataset=cfg.dataset)
        game_sublists = distribute_slurm_jobs(games, cfg.n_games_per_job)
        n_jobs = len(game_sublists)
        executor = submitit.AutoExecutor(
            folder=os.path.join("submitit_logs", "backfill_scores")
        )
        executor.update_parameters(
            slurm_job_name="backfill_scores",
            mem_gb=16,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=cfg.slurm_timeout_min,
            slurm_array_parallelism=n_jobs,
            slurm_account=os.environ.get("SLURM_ACCOUNT"),
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: BackfillConfig, games: Optional[List[str]] = None):
    parser = init_ps_lark_parser()
    backend = _init_backend(cfg.backend)
    sols_dir = _sols_dir(cfg.backend)

    if games is None:
        games = get_list_of_games_for_testing(dataset=cfg.dataset)

    n_updated = 0
    n_skipped = 0
    n_errors = 0
    error_games: dict[str, str] = {}  # game -> first error message

    pbar = tqdm(games, desc="Backfilling scores", unit="game")
    for game in pbar:
        game_dir = os.path.join(sols_dir, game)
        if not os.path.isdir(game_dir):
            continue

        result_files = glob.glob(os.path.join(game_dir, "*.json"))
        if not result_files:
            continue

        # Group by level to avoid recompiling per-file
        levels_to_files: dict[int, list[str]] = {}
        for path in result_files:
            match = RESULT_FILENAME_RE.match(os.path.basename(path))
            if not match:
                continue
            level_i = int(match.group(3))
            levels_to_files.setdefault(level_i, []).append(path)

        if not levels_to_files:
            continue

        # Check if all files already have the fields we want (skip whole game)
        if not cfg.overwrite:
            all_have = True
            for paths in levels_to_files.values():
                for path in paths:
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        if 'score_initial' not in data:
                            all_have = False
                            break
                        if cfg.trajectories and 'score_trajectory' not in data:
                            all_have = False
                            break
                    except Exception:
                        pass
                if not all_have:
                    break
            if all_have:
                n_skipped += len(result_files)
                continue

        # Compile game once
        try:
            backend.unload_game()
            game_text = backend.compile_game(parser, game)
        except Exception as e:
            err_msg = str(e).split('\n')[0][:200]
            error_games.setdefault(game, f"compile: {err_msg}")
            n_errors += 1
            continue

        game_had_error = False
        for level_i, paths in sorted(levels_to_files.items()):
            try:
                score_initial = backend.get_initial_score(game_text, level_i)
            except (BrokenPipeError, ConnectionError, OSError) as e:
                # Backend bridge is dead — abort entirely
                err_msg = str(e).split('\n')[0][:200]
                error_games.setdefault(game, f"bridge died at level {level_i}: {err_msg}")
                print(f"\nBackend bridge died on {game} level {level_i}, aborting.")
                n_errors += 1
                game_had_error = True
                break
            except Exception as e:
                err_msg = str(e).split('\n')[0][:200]
                if 'BrokenBarrier' in repr(e) or 'BrokenBarrier' in str(type(e)):
                    error_games.setdefault(game, f"bridge died at level {level_i}: {err_msg}")
                    print(f"\nBackend bridge died on {game} level {level_i}, aborting.")
                    n_errors += 1
                    game_had_error = True
                    break
                error_games.setdefault(game, f"level {level_i}: {err_msg}")
                n_errors += 1
                continue

            for path in paths:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    n_errors += 1
                    continue

                needs_initial = cfg.overwrite or 'score_initial' not in data
                needs_trajectory = cfg.trajectories and (
                    cfg.overwrite or 'score_trajectory' not in data
                )
                if not needs_initial and not needs_trajectory:
                    n_skipped += 1
                    continue

                if needs_initial:
                    data['score_initial'] = score_initial

                if needs_trajectory:
                    actions = data.get('actions')
                    if actions and not data.get('error'):
                        try:
                            trajectory = backend.replay_score_trajectory(
                                game_text, level_i, actions,
                            )
                            data['score_trajectory'] = trajectory
                        except Exception as e:
                            err_msg = str(e).split('\n')[0][:200]
                            error_games.setdefault(
                                game, f"trajectory level {level_i}: {err_msg}")

                with open(path, 'w') as f:
                    json.dump(data, f, indent=4)
                n_updated += 1

        if game_had_error and 'BrokenBarrier' in error_games.get(game, ''):
            # Try to reinitialize the backend
            print("Attempting to reinitialize backend...")
            try:
                backend = _init_backend(cfg.backend)
            except Exception as e:
                print(f"Failed to reinitialize backend: {e}. Stopping.")
                break

        pbar.set_postfix(updated=n_updated, skipped=n_skipped, errors=len(error_games))

    print(f"\nBackfill complete: {n_updated} updated, {n_skipped} skipped, {n_errors} errors")

    if error_games:
        log_path = os.path.join(sols_dir, 'backfill_errors.json')
        # Merge with any existing error log
        existing_errors = {}
        if os.path.isfile(log_path):
            try:
                with open(log_path, 'r') as f:
                    existing_errors = json.load(f)
            except Exception:
                pass
        existing_errors.update(error_games)
        with open(log_path, 'w') as f:
            json.dump(existing_errors, f, indent=2, sort_keys=True)
        print(f"Error log ({len(error_games)} games) saved to {log_path}")
        for game, err in sorted(error_games.items()):
            print(f"  {game}: {err}")


if __name__ == "__main__":
    main_launch()

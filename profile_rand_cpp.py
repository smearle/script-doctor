"""Profile C++ PuzzleScript engine speed while taking random actions in parallel."""
import json
import logging
import math
import os
import random
import traceback
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, List, Optional

import cpuinfo
import hydra
import multiprocessing as mp
import numpy as np
import submitit
from javascript import require

from conf.config import ProfileRandCppConfig
from puzzlescript_jax.globals import CPP_PROFILING_RESULTS_DIR
from puzzlescript_jax.utils import get_list_of_games_for_testing, init_ps_lark_parser
from puzzlescript_cpp import CppBatchedPuzzleScriptEnv, CppPuzzleScriptEngine
from puzzlescript_nodejs.utils import compile_game


BATCH_SIZES = [
    1,
    2,
    4,
    8,
    16,
    32,
    36,
    40,
    48,
]
INCLUDED_CPP_FIXED_EXECUTION_MODES = [
    # "cpp_native",
    "cpp_native_multiprocess",
]
INCLUDED_CPP_SWEEP_EXECUTION_MODES = [
    "cpp_batched",
]
MAX_AGAIN = 50
WORKER_TIMEOUT_GRACE_SECONDS = 15.0
WORKER_TIMEOUT_MULTIPLIER = 2.0
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_JS_PATH = os.path.join(ROOT_DIR, "puzzlescript_nodejs", "puzzlescript", "engine.js")


def get_step_str(n_steps: int) -> str:
    return f"{n_steps}-step_rollout"


def get_level_str(level_i: int) -> str:
    return f"level-{level_i}"


def get_stats_key(n_envs: int, execution_mode: str) -> str:
    return f"{n_envs}-{execution_mode}"


def _get_fixed_run_specs() -> list[tuple[int, str]]:
    run_specs = []
    for execution_mode in INCLUDED_CPP_FIXED_EXECUTION_MODES:
        if execution_mode == "cpp_native":
            run_specs.append((1, execution_mode))
        elif execution_mode == "cpp_native_multiprocess":
            run_specs.extend((n_envs, execution_mode) for n_envs in BATCH_SIZES if n_envs > 1)
        else:
            raise ValueError(f"Unsupported C++ fixed execution mode: {execution_mode}")
    return run_specs


def save_results(results: dict, results_path: str) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def _best_fps(stats: dict) -> float:
    fpss = stats.get("fps", ())
    if not fpss:
        return 0.0
    return float(max(fpss))


def _compile_game_for_cpp(parser: Any, game: str) -> str:
    js_engine = require(ENGINE_JS_PATH)
    compile_game(parser, js_engine, game, 0)
    return str(js_engine.serializeCompiledStateJSON())


def _run_cpp_random_rollout(
    engine: CppPuzzleScriptEngine,
    *,
    n_steps: int,
    timeout_ms: int,
    initial_backup: Any,
) -> dict[str, float | int | bool]:
    engine.restore_level(initial_backup)
    start_time = timer()
    completed_steps = 0
    timeout = False
    wins = 0

    for completed_steps in range(n_steps):
        if timeout_ms > 0 and completed_steps % 1_000 == 0:
            if (timer() - start_time) * 1_000 > timeout_ms:
                timeout = True
                break

        action = random.randint(0, 4)
        engine.process_input(action)

        again_count = 0
        while engine.againing and again_count < MAX_AGAIN:
            engine.process_input(-1)
            again_count += 1

        if engine.winning:
            wins += 1
            engine.restore_level(initial_backup)
    else:
        completed_steps = n_steps

    elapsed = timer() - start_time
    return {
        "iterations": completed_steps,
        "time": elapsed,
        "timeout": timeout,
        "wins": wins,
    }


def _cpp_rollout_worker(serialized_json: str, level_i: int, n_steps: int, timeout_ms: int) -> dict:
    engine = CppPuzzleScriptEngine()
    engine.load_from_json(serialized_json)
    engine.load_level(level_i)
    initial_backup = engine.backup_level()
    return _run_cpp_random_rollout(
        engine,
        n_steps=n_steps,
        timeout_ms=timeout_ms,
        initial_backup=initial_backup,
    )


def _get_worker_timeout_seconds(timeout_ms: int) -> float | None:
    if timeout_ms <= 0:
        return None
    return max((timeout_ms / 1_000) * WORKER_TIMEOUT_MULTIPLIER + WORKER_TIMEOUT_GRACE_SECONDS, 1.0)


def _persistent_rollout_worker(
    worker_id: int,
    serialized_json: str,
    level_i: int,
    conn: Any,
) -> None:
    try:
        engine = CppPuzzleScriptEngine()
        engine.load_from_json(serialized_json)
        engine.load_level(level_i)
        initial_backup = engine.backup_level()
    except Exception as exc:
        conn.send({
            "worker_id": worker_id,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "error_traceback": traceback.format_exc(),
            "startup_failed": True,
        })
        conn.close()
        return

    while True:
        message = conn.recv()
        cmd = message.get("cmd")
        if cmd == "close":
            conn.close()
            return
        if cmd != "run":
            conn.send({
                "worker_id": worker_id,
                "ok": False,
                "error_type": "ValueError",
                "error": f"Unknown worker command: {cmd}",
                "error_traceback": None,
            })
            continue

        try:
            result = _run_cpp_random_rollout(
                engine,
                n_steps=message["n_steps"],
                timeout_ms=message["timeout_ms"],
                initial_backup=initial_backup,
            )
            result["worker_id"] = worker_id
            result["ok"] = True
        except Exception as exc:
            result = {
                "worker_id": worker_id,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "error_traceback": traceback.format_exc(),
            }
        conn.send(result)


@dataclass
class PersistentRolloutWorkers:
    ctx: mp.context.BaseContext
    processes: list[mp.Process]
    conns: list[Any]

    @classmethod
    def start(cls, *, serialized_json: str, level_i: int, n_envs: int) -> "PersistentRolloutWorkers":
        ctx = mp.get_context("spawn")
        processes = []
        conns = []

        for worker_id in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_persistent_rollout_worker,
                args=(worker_id, serialized_json, level_i, child_conn),
            )
            proc.start()
            child_conn.close()
            processes.append(proc)
            conns.append(parent_conn)

        return cls(ctx=ctx, processes=processes, conns=conns)

    def run_batch(self, *, n_steps: int, timeout_ms: int) -> dict:
        for conn in self.conns:
            conn.send({"cmd": "run", "n_steps": n_steps, "timeout_ms": timeout_ms})

        start = timer()
        worker_timeout_s = _get_worker_timeout_seconds(timeout_ms)
        worker_results = []
        failed_results = []
        pending_conns = set(self.conns)
        expected_results = len(self.conns)
        deadline = None if worker_timeout_s is None else start + worker_timeout_s

        while pending_conns:
            timeout = None if deadline is None else max(deadline - timer(), 0.0)
            ready_conns = mp.connection.wait(pending_conns, timeout=timeout)
            if not ready_conns:
                raise TimeoutError(
                    f"Timed out waiting for C++ worker results after {worker_timeout_s:.1f}s."
                )
            for conn in ready_conns:
                message = conn.recv()
                pending_conns.remove(conn)
                if message.get("ok", True):
                    worker_results.append(message)
                else:
                    failed_results.append(message)

        wall_time = timer() - start

        if not worker_results:
            sample_error = failed_results[0] if failed_results else None
            if sample_error is None:
                raise RuntimeError("All C++ workers failed without returning results.")
            raise RuntimeError(
                f"All {expected_results} C++ workers failed. "
                f"Sample {sample_error.get('error_type', 'error')}: {sample_error.get('error', '')}"
            )

        total_iterations = sum(result["iterations"] for result in worker_results)
        total_reported_worker_time = sum(result["time"] for result in worker_results)
        timeouts = sum(int(result["timeout"]) for result in worker_results)
        total_wins = sum(int(result["wins"]) for result in worker_results)
        requested_iterations = expected_results * n_steps
        total_fps = total_iterations / wall_time if wall_time > 0 else 0.0
        mean_worker_fps = (
            total_iterations / total_reported_worker_time if total_reported_worker_time > 0 else 0.0
        )

        return {
            "n_envs": expected_results,
            "total_iterations": total_iterations,
            "requested_iterations": requested_iterations,
            "completed_ratio": total_iterations / requested_iterations if requested_iterations > 0 else 0.0,
            "successful_workers": len(worker_results),
            "failed_workers": len(failed_results),
            "wall_time": wall_time,
            "fps": total_fps,
            "mean_worker_fps": mean_worker_fps,
            "wins": total_wins,
            "timeouts": timeouts,
            "timed_out": timeouts > 0,
            "had_worker_failures": bool(failed_results),
            "sample_worker_error": (
                {
                    "error_type": failed_results[0]["error_type"],
                    "error": failed_results[0]["error"],
                }
                if failed_results
                else None
            ),
            "execution_mode": "cpp_native_multiprocess",
        }

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send({"cmd": "close"})
            except Exception:
                pass
            conn.close()
        for proc in self.processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)


def _profile_single_process_rollout(
    serialized_json: str,
    *,
    level_i: int,
    n_steps: int,
    timeout_ms: int,
) -> dict:
    start = timer()
    result = _cpp_rollout_worker(serialized_json, level_i, n_steps, timeout_ms)
    wall_time = timer() - start

    return {
        "n_envs": 1,
        "total_iterations": result["iterations"],
        "requested_iterations": n_steps,
        "completed_ratio": result["iterations"] / n_steps if n_steps > 0 else 0.0,
        "successful_workers": 1,
        "failed_workers": 0,
        "wall_time": wall_time,
        "fps": result["iterations"] / wall_time if wall_time > 0 else 0.0,
        "mean_worker_fps": result["iterations"] / result["time"] if result["time"] > 0 else 0.0,
        "wins": result["wins"],
        "timeouts": int(result["timeout"]),
        "timed_out": bool(result["timeout"]),
        "had_worker_failures": False,
        "sample_worker_error": None,
        "execution_mode": "cpp_native",
    }


def _profile_batched_rollout(
    env: CppBatchedPuzzleScriptEnv,
    *,
    n_steps: int,
    timeout_ms: int,
) -> dict:
    env.reset()
    start = timer()
    completed_steps = 0
    wins = 0
    timeout = False

    for completed_steps in range(n_steps):
        if timeout_ms > 0 and completed_steps % 1_000 == 0:
            if (timer() - start) * 1_000 > timeout_ms:
                timeout = True
                break

        actions = np.random.randint(0, env.num_actions, size=env.batch_size, dtype=np.int32)
        _, _, _, _, infos = env.step(actions)
        wins += int(np.sum(infos["won"]))
    else:
        completed_steps = n_steps

    wall_time = timer() - start
    total_iterations = completed_steps * env.batch_size
    requested_iterations = n_steps * env.batch_size
    fps = total_iterations / wall_time if wall_time > 0 else 0.0

    return {
        "n_envs": env.batch_size,
        "total_iterations": total_iterations,
        "requested_iterations": requested_iterations,
        "completed_ratio": total_iterations / requested_iterations if requested_iterations > 0 else 0.0,
        "successful_workers": env.batch_size,
        "failed_workers": 0,
        "wall_time": wall_time,
        "fps": fps,
        "mean_worker_fps": fps / env.batch_size if env.batch_size > 0 else 0.0,
        "wins": wins,
        "timeouts": int(timeout),
        "timed_out": timeout,
        "had_worker_failures": False,
        "sample_worker_error": None,
        "execution_mode": "cpp_batched",
    }


@hydra.main(version_base="1.3", config_path="./", config_name="profile_rand_cpp_config")
def main_launch(cfg: ProfileRandCppConfig):
    if cfg.slurm:
        if cfg.game is None:
            games = get_list_of_games_for_testing(
                all_games=cfg.all_games,
                include_random=cfg.include_randomness,
                random_order=cfg.random_order,
            )
        else:
            games = [cfg.game]
        if not games:
            return

        n_jobs = math.ceil(len(games) / cfg.n_games_per_job)
        game_sublists = [games[i::n_jobs] for i in range(n_jobs)]
        assert sum(len(game_list) for game_list in game_sublists) == len(games), (
            "Not all games are assigned to a job."
        )
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "profile_rand_cpp"))
        executor.update_parameters(
            slurm_job_name="profile_rand_cpp",
            mem_gb=30,
            tasks_per_node=1,
            cpus_per_task=1,
            timeout_min=180,
            slurm_array_parallelism=n_jobs,
            slurm_account=os.environ.get("SLURM_ACCOUNT"),
        )
        executor.map_array(main, [cfg] * n_jobs, game_sublists)
    else:
        main(cfg)


def main(cfg: ProfileRandCppConfig, games: Optional[List[str]] = None):
    logging.getLogger().setLevel(logging.WARNING)
    cpu_name = cpuinfo.get_cpu_info()["brand_raw"].replace(" ", "_")
    step_str = get_step_str(cfg.n_steps)
    device_dir = os.path.join(CPP_PROFILING_RESULTS_DIR, cpu_name, step_str)

    if games is not None:
        games = list(games)
    elif cfg.game is None:
        games = get_list_of_games_for_testing(
            all_games=cfg.all_games,
            include_random=cfg.include_randomness,
            random_order=cfg.random_order,
        )
    else:
        games = [cfg.game]

    parser = init_ps_lark_parser()
    serialized_by_game = {}
    fixed_run_specs = _get_fixed_run_specs()

    for game in games:
        if game not in serialized_by_game:
            serialized_by_game[game] = _compile_game_for_cpp(parser, game)
        serialized_json = serialized_by_game[game]

        for level_i in range(1):
            results_path = os.path.join(device_dir, game, f"{get_level_str(level_i)}.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    n_envs_to_stats = json.load(f)
            else:
                n_envs_to_stats = {}

            timeout_ms = cfg.timeout * 1_000 if cfg.timeout > 0 else -1

            for n_envs, execution_mode in fixed_run_specs:
                print(f"\nGame: {game}, n_envs: {n_envs}, mode: {execution_mode}.")
                stats_key = get_stats_key(n_envs, execution_mode)
                if not cfg.overwrite and stats_key in n_envs_to_stats:
                    print(
                        f"Skipping {game} level {level_i} with n_envs={n_envs} mode={execution_mode} "
                        f"as results already exist."
                    )
                    continue

                try:
                    persistent_workers = None
                    iterations = []
                    fpss = []
                    last_stats = None
                    try:
                        if execution_mode == "cpp_native":
                            stats_fn = lambda: _profile_single_process_rollout(
                                serialized_json,
                                level_i=level_i,
                                n_steps=cfg.n_steps,
                                timeout_ms=timeout_ms,
                            )
                        else:
                            persistent_workers = PersistentRolloutWorkers.start(
                                serialized_json=serialized_json,
                                level_i=level_i,
                                n_envs=n_envs,
                            )
                            stats_fn = lambda: persistent_workers.run_batch(
                                n_steps=cfg.n_steps,
                                timeout_ms=timeout_ms,
                            )

                        for run_i in range(3):
                            stats = stats_fn()
                            iterations.append(stats["total_iterations"])
                            fpss.append(stats["fps"])
                            last_stats = stats
                            print(
                                f"Loop {run_i} ran {stats['total_iterations']} steps in "
                                f"{stats['wall_time']:.3f} seconds. FPS: {stats['fps']:,.2f}"
                            )
                    finally:
                        if persistent_workers is not None:
                            persistent_workers.close()

                    n_envs_to_stats[stats_key] = {
                        "fps": tuple(fpss),
                        "iterations": tuple(iterations),
                        "total_iterations": last_stats["total_iterations"],
                        "requested_iterations": last_stats["requested_iterations"],
                        "completed_ratio": last_stats["completed_ratio"],
                        "successful_workers": last_stats["successful_workers"],
                        "failed_workers": last_stats["failed_workers"],
                        "wall_time": last_stats["wall_time"],
                        "mean_worker_fps": last_stats["mean_worker_fps"],
                        "wins": last_stats["wins"],
                        "timeouts": last_stats["timeouts"],
                        "timed_out": last_stats["timed_out"],
                        "had_worker_failures": last_stats["had_worker_failures"],
                        "sample_worker_error": last_stats["sample_worker_error"],
                        "execution_mode": last_stats["execution_mode"],
                    }
                except Exception as exc:
                    err_msg = traceback.format_exc()
                    print(
                        f"Error profiling {game} level {level_i} with n_envs={n_envs} "
                        f"mode={execution_mode}: {err_msg}"
                    )
                    n_envs_to_stats[stats_key] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "error_traceback": err_msg,
                        "execution_mode": execution_mode,
                    }

                save_results(n_envs_to_stats, results_path)

            if "cpp_batched" in INCLUDED_CPP_SWEEP_EXECUTION_MODES:
                prev_batched_best_fps = None
                n_envs = 1
                while True:
                    execution_mode = "cpp_batched"
                    stats_key = get_stats_key(n_envs, execution_mode)
                    print(f"\nGame: {game}, n_envs: {n_envs}, mode: {execution_mode}.")

                    if not cfg.overwrite and stats_key in n_envs_to_stats:
                        print(
                            f"Skipping {game} level {level_i} with n_envs={n_envs} mode={execution_mode} "
                            f"as results already exist."
                        )
                        stats_entry = n_envs_to_stats[stats_key]
                    else:
                        try:
                            batched_env = CppBatchedPuzzleScriptEnv(
                                serialized_json,
                                batch_size=n_envs,
                                level_indices=[level_i] * n_envs,
                                max_episode_steps=max(cfg.n_steps, 1),
                            )
                            iterations = []
                            fpss = []
                            last_stats = None
                            for run_i in range(3):
                                stats = _profile_batched_rollout(
                                    batched_env,
                                    n_steps=cfg.n_steps,
                                    timeout_ms=timeout_ms,
                                )
                                iterations.append(stats["total_iterations"])
                                fpss.append(stats["fps"])
                                last_stats = stats
                                print(
                                    f"Loop {run_i} ran {stats['total_iterations']} steps in "
                                    f"{stats['wall_time']:.3f} seconds. FPS: {stats['fps']:,.2f}"
                                )

                            stats_entry = {
                                "fps": tuple(fpss),
                                "iterations": tuple(iterations),
                                "total_iterations": last_stats["total_iterations"],
                                "requested_iterations": last_stats["requested_iterations"],
                                "completed_ratio": last_stats["completed_ratio"],
                                "successful_workers": last_stats["successful_workers"],
                                "failed_workers": last_stats["failed_workers"],
                                "wall_time": last_stats["wall_time"],
                                "mean_worker_fps": last_stats["mean_worker_fps"],
                                "wins": last_stats["wins"],
                                "timeouts": last_stats["timeouts"],
                                "timed_out": last_stats["timed_out"],
                                "had_worker_failures": last_stats["had_worker_failures"],
                                "sample_worker_error": last_stats["sample_worker_error"],
                                "execution_mode": last_stats["execution_mode"],
                            }
                            n_envs_to_stats[stats_key] = stats_entry
                        except Exception as exc:
                            err_msg = traceback.format_exc()
                            print(
                                f"Error profiling {game} level {level_i} with n_envs={n_envs} "
                                f"mode={execution_mode}: {err_msg}"
                            )
                            stats_entry = {
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "error_traceback": err_msg,
                                "execution_mode": execution_mode,
                            }
                            n_envs_to_stats[stats_key] = stats_entry

                        save_results(n_envs_to_stats, results_path)

                    if "error_type" in stats_entry:
                        break

                    current_batched_best_fps = _best_fps(stats_entry)
                    if (
                        prev_batched_best_fps is not None
                        and current_batched_best_fps < prev_batched_best_fps
                    ):
                        print(
                            f"Stopping cpp_batched sweep for {game} level {level_i}: "
                            f"best FPS dropped from {prev_batched_best_fps:,.2f} "
                            f"to {current_batched_best_fps:,.2f}."
                        )
                        break

                    prev_batched_best_fps = current_batched_best_fps
                    n_envs *= 2


if __name__ == "__main__":
    main_launch()

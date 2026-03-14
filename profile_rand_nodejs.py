"""Profile original PuzzleScript engine speed while taking random actions in parallel."""
import itertools
import json
import logging
import math
import os
import subprocess
import traceback
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import Any, List, Optional

import cpuinfo
import hydra
import multiprocessing as mp
import numpy as np
import submitit

from conf.config import ProfileRandNodeJSConfig
from backends import NodeJSPuzzleScriptBackend
from puzzlescript_jax.globals import GAMES_DIR, NODEJS_PROFILING_RESULTS_DIR, SIMPLIFIED_GAMES_DIR
from puzzlescript_jax.utils import get_list_of_games_for_testing
from puzzlescript_nodejs.rl_env import NodeJSBatchedPuzzleEnv


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
INCLUDED_NODEJS_EXECUTION_MODES = [
    # "single_process",
    # "nodejs_native",
    "nodejs_batched",
]
WORKER_TIMEOUT_GRACE_SECONDS = 15.0
WORKER_TIMEOUT_MULTIPLIER = 2.0


def get_step_str(n_steps: int) -> str:
    return f"{n_steps}-step_rollout"


def get_level_str(level_i: int) -> str:
    return f"level-{level_i}"


def get_stats_key(n_envs: int, execution_mode: str) -> str:
    return f"{n_envs}-{execution_mode}"


def _get_run_specs() -> list[tuple[int, str]]:
    run_specs = []
    for execution_mode in INCLUDED_NODEJS_EXECUTION_MODES:
        if execution_mode in {"single_process", "nodejs_native"}:
            run_specs.append((1, execution_mode))
        elif execution_mode == "nodejs_batched":
            run_specs.extend((n_envs, execution_mode) for n_envs in BATCH_SIZES)
        else:
            raise ValueError(f"Unsupported NodeJS execution mode: {execution_mode}")
    return run_specs


def save_results(results: dict, results_path: str) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)


def _strip_level_messages(game_text: str) -> str:
    lines = game_text.splitlines()
    output_lines = []
    in_levels_section = False

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper == "LEVELS":
            in_levels_section = True
            output_lines.append(line)
            continue

        if in_levels_section:
            is_section_header = (
                stripped
                and stripped == upper
                and any(ch.isalpha() for ch in stripped)
                and " " not in stripped
            )
            if is_section_header and upper != "LEVELS":
                in_levels_section = False

        if in_levels_section and stripped.lower().startswith("message "):
            continue

        output_lines.append(line)

    return "\n".join(output_lines) + ("\n" if game_text.endswith("\n") else "")


def _load_original_game_text(game: str) -> str:
    game_path = os.path.join(GAMES_DIR, f"{game}.txt")
    with open(game_path, "r", encoding="utf-8") as f:
        return _strip_level_messages(f.read())


def _load_nodejs_native_game_text(game: str) -> str:
    simplified_path = os.path.join(SIMPLIFIED_GAMES_DIR, f"{game}_simplified.txt")
    if os.path.isfile(simplified_path):
        with open(simplified_path, "r", encoding="utf-8") as f:
            return f.read()
    return _load_original_game_text(game)


def _get_nodejs_native_game_path(game: str) -> str:
    simplified_path = os.path.join(SIMPLIFIED_GAMES_DIR, f"{game}_simplified.txt")
    if os.path.isfile(simplified_path):
        return simplified_path
    return os.path.join(GAMES_DIR, f"{game}.txt")


def _random_rollout_worker(game_text: str, level_i: int, n_steps: int, timeout_ms: int) -> dict:
    backend = NodeJSPuzzleScriptBackend()
    return _run_random_rollout(backend, game_text=game_text, level_i=level_i, n_steps=n_steps, timeout_ms=timeout_ms)


def _run_random_rollout(
    backend: NodeJSPuzzleScriptBackend,
    *,
    game_text: str,
    level_i: int,
    n_steps: int,
    timeout_ms: int,
) -> dict:
    result = backend.run_random_rollout_raw(
        game_text=game_text,
        level_i=level_i,
        n_steps=n_steps,
        timeout_ms=timeout_ms,
    )
    return {
        "iterations": result["iterations"],
        "time": result["time"],
        "timeout": result["timeout"],
    }


def _get_worker_timeout_seconds(timeout_ms: int) -> float | None:
    if timeout_ms <= 0:
        return None
    return max((timeout_ms / 1_000) * WORKER_TIMEOUT_MULTIPLIER + WORKER_TIMEOUT_GRACE_SECONDS, 1.0)


def _persistent_rollout_worker(
    worker_id: int,
    game_text: str,
    level_i: int,
    conn: Any,
) -> None:
    try:
        backend = NodeJSPuzzleScriptBackend()
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
            result = _run_random_rollout(
                backend,
                game_text=game_text,
                level_i=level_i,
                n_steps=message["n_steps"],
                timeout_ms=message["timeout_ms"],
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
    def start(cls, *, game_text: str, level_i: int, n_envs: int) -> "PersistentRolloutWorkers":
        ctx = mp.get_context("spawn")
        processes = []
        conns = []

        for worker_id in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_persistent_rollout_worker,
                args=(worker_id, game_text, level_i, child_conn),
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
                    f"Timed out waiting for NodeJS worker results after {worker_timeout_s:.1f}s."
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
                raise RuntimeError("All NodeJS workers failed without returning results.")
            raise RuntimeError(
                f"All {expected_results} NodeJS workers failed. "
                f"Sample {sample_error.get('error_type', 'error')}: {sample_error.get('error', '')}"
            )

        total_iterations = sum(result["iterations"] for result in worker_results)
        total_reported_worker_time = sum(result["time"] for result in worker_results)
        timeouts = sum(int(result["timeout"]) for result in worker_results)
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
            "execution_mode": "multiprocess",
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
    backend: NodeJSPuzzleScriptBackend,
    *,
    game_text: str,
    level_i: int,
    n_steps: int,
    timeout_ms: int,
) -> dict:
    start = timer()
    result = _run_random_rollout(
        backend,
        game_text=game_text,
        level_i=level_i,
        n_steps=n_steps,
        timeout_ms=timeout_ms,
    )
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
        "timeouts": int(result["timeout"]),
        "timed_out": bool(result["timeout"]),
        "had_worker_failures": False,
        "sample_worker_error": None,
        "execution_mode": "single_process",
    }


def _profile_nodejs_native_rollout(
    backend: NodeJSPuzzleScriptBackend,
    *,
    game_text: str,
    level_i: int,
    n_steps: int,
    timeout_ms: int,
) -> dict:
    start = timer()
    result = backend.run_search(
        "random",
        game_text=game_text,
        level_i=level_i,
        n_steps=n_steps,
        timeout_ms=timeout_ms,
        warmup=False,
    )
    wall_time = timer() - start

    return {
        "n_envs": 1,
        "total_iterations": result.iterations,
        "requested_iterations": n_steps,
        "completed_ratio": result.iterations / n_steps if n_steps > 0 else 0.0,
        "successful_workers": 1,
        "failed_workers": 0,
        "wall_time": wall_time,
        "fps": result.fps,
        "mean_worker_fps": result.fps,
        "engine_time": result.time,
        "timeouts": int(result.timeout),
        "timed_out": bool(result.timeout),
        "had_worker_failures": False,
        "sample_worker_error": None,
        "execution_mode": "nodejs_native",
    }


def _profile_nodejs_batched_rollout(
    env: NodeJSBatchedPuzzleEnv,
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
        _, _, dones, truncated, infos = env.step(actions)
        del dones, truncated
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
        "execution_mode": "nodejs_batched",
    }


def _run_nodejs_native_pool(
    *,
    game_path: str,
    level_i: int,
    n_envs: int,
    n_steps: int,
    timeout_ms: int,
    repeats: int,
    execution_mode: str,
) -> list[dict]:
    pool_script_path = os.path.join(
        os.path.dirname(__file__),
        "puzzlescript_nodejs",
        "puzzlescript",
        "random_rollout_pool.js",
    )
    payload = json.dumps({
        "gamePath": game_path,
        "levelI": level_i,
        "nEnvs": n_envs,
        "nSteps": n_steps,
        "timeoutMs": timeout_ms,
        "repeats": repeats,
    })
    result = subprocess.run(
        ["node", pool_script_path, payload],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Node rollout pool failed.")

    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError("Node rollout pool returned no JSON output.")

    payload = json.loads(stdout_lines[-1])
    runs = payload["runs"]
    for run in runs:
        run["execution_mode"] = execution_mode
    return runs


@hydra.main(version_base="1.3", config_path="./", config_name="profile_rand_nodejs_config")
def main_launch(cfg: ProfileRandNodeJSConfig):
    if cfg.slurm:
        if cfg.game is None:
            games = get_list_of_games_for_testing(
                dataset=cfg.dataset,
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
        executor = submitit.AutoExecutor(folder=os.path.join("submitit_logs", "profile_rand_nodejs"))
        executor.update_parameters(
            slurm_job_name="profile_rand_nodejs",
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


def main(cfg: ProfileRandNodeJSConfig, games: Optional[List[str]] = None):
    logging.getLogger().setLevel(logging.WARNING)
    cpu_name = cpuinfo.get_cpu_info()["brand_raw"].replace(" ", "_")
    step_str = get_step_str(cfg.n_steps)
    device_dir = os.path.join(NODEJS_PROFILING_RESULTS_DIR, cpu_name, step_str)

    if games is not None:
        games = list(games)
    elif cfg.game is None:
        games = get_list_of_games_for_testing(
            dataset=cfg.dataset,
            include_random=cfg.include_randomness,
            random_order=cfg.random_order,
        )
    else:
        games = [cfg.game]

    run_specs = _get_run_specs()

    for game, (n_envs, execution_mode) in itertools.product(games, run_specs):
        print(f"\nGame: {game}, n_envs: {n_envs}, mode: {execution_mode}.")
        for level_i in range(1):
            results_path = os.path.join(device_dir, game, f"{get_level_str(level_i)}.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    n_envs_to_stats = json.load(f)
            else:
                n_envs_to_stats = {}

            stats_key = get_stats_key(n_envs, execution_mode)
            legacy_stats_key = str(n_envs) if execution_mode == "multiprocess" else None
            if not cfg.overwrite and (
                stats_key in n_envs_to_stats or (legacy_stats_key is not None and legacy_stats_key in n_envs_to_stats)
            ):
                print(
                    f"Skipping {game} level {level_i} with n_envs={n_envs} mode={execution_mode} "
                    f"as results already exist."
                )
                continue

            timeout_ms = cfg.timeout * 1_000 if cfg.timeout > 0 else -1

            try:
                single_process_stats_fn = None
                persistent_workers = None
                batched_env = None
                game_text = _load_original_game_text(game)
                native_runs = None
                if execution_mode == "single_process":
                    backend = NodeJSPuzzleScriptBackend()
                    single_process_stats_fn = lambda: _profile_single_process_rollout(
                        backend,
                        game_text=game_text,
                        level_i=level_i,
                        n_steps=cfg.n_steps,
                        timeout_ms=timeout_ms,
                    )
                elif execution_mode == "nodejs_native":
                    native_runs = _run_nodejs_native_pool(
                        game_path=_get_nodejs_native_game_path(game),
                        level_i=level_i,
                        n_envs=1,
                        n_steps=cfg.n_steps,
                        timeout_ms=timeout_ms,
                        repeats=3,
                        execution_mode="nodejs_native",
                    )
                elif execution_mode == "nodejs_batched":
                    batched_env = NodeJSBatchedPuzzleEnv(
                        game=game,
                        level_i=level_i,
                        batch_size=n_envs,
                        max_episode_steps=max(cfg.n_steps, 1),
                    )
                else:
                    persistent_workers = PersistentRolloutWorkers.start(
                        game_text=game_text,
                        level_i=level_i,
                        n_envs=n_envs,
                    )

                iterations = []
                fpss = []
                last_stats = None
                try:
                    run_count = 3 if native_runs is None else len(native_runs)
                    for run_i in range(run_count):
                        if native_runs is not None:
                            stats = native_runs[run_i]
                        elif single_process_stats_fn is not None:
                            stats = single_process_stats_fn()
                        elif batched_env is not None:
                            stats = _profile_nodejs_batched_rollout(
                                batched_env,
                                n_steps=cfg.n_steps,
                                timeout_ms=timeout_ms,
                            )
                        else:
                            stats = persistent_workers.run_batch(
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
                finally:
                    if batched_env is not None:
                        batched_env.close()
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
                    "engine_time": last_stats.get("engine_time"),
                    "wins": last_stats.get("wins"),
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


if __name__ == "__main__":
    main_launch()

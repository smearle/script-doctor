#!/usr/bin/env python3
"""Launch vLLM server + llm_agent_loop_nodejs.py jobs via submitit (SLURM or local).

Usage examples
--------------
# Local (server + 1 agent, both as subprocesses):
    python launch_llm_agent.py

# Local, specific game:
    python launch_llm_agent.py --game Microban

# SLURM (GPU node for vLLM, CPU node(s) for agents):
    python launch_llm_agent.py --slurm

# Sweep over multiple models:
    python launch_llm_agent.py --slurm --models vllm-qwen3.5-9b vllm-qwen3.5-27b-fp8
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import submitit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PYTHON = os.path.join(".venv", "bin", "python3")
SUBMITIT_LOG_DIR = os.path.join("submitit_logs", "llm_agent")

# ── Defaults matching Makefile ──────────────────────────────────────────────
VLLM_DEFAULTS = dict(
    model="Qwen/Qwen3.5-27B-FP8",
    host="0.0.0.0",
    port=8000,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=32768,
    enforce_eager=True,
    enable_prefix_caching=True,
    max_num_batched_tokens=4096,
)


# ── vLLM server callable (submitted via submitit) ──────────────────────────
class VLLMServer:
    """Callable submitted as a SLURM/local job that runs the vLLM server."""

    def __init__(self, vllm_model: str, port: int, extra_args: list[str] | None = None):
        self.vllm_model = vllm_model
        self.port = port
        self.extra_args = extra_args or []

    def __call__(self):
        cmd = [
            PYTHON, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.vllm_model,
            "--served-model-name", self.vllm_model,
            "--host", VLLM_DEFAULTS["host"],
            "--port", str(self.port),
            "--tensor-parallel-size", str(VLLM_DEFAULTS["tensor_parallel_size"]),
            "--gpu-memory-utilization", str(VLLM_DEFAULTS["gpu_memory_utilization"]),
            "--max-model-len", str(VLLM_DEFAULTS["max_model_len"]),
            "--max-num-batched-tokens", str(VLLM_DEFAULTS["max_num_batched_tokens"]),
        ]
        if VLLM_DEFAULTS["enforce_eager"]:
            cmd.append("--enforce-eager")
        if VLLM_DEFAULTS["enable_prefix_caching"]:
            cmd.append("--enable-prefix-caching")
        cmd.extend(self.extra_args)

        logger.info("Starting vLLM server: %s", " ".join(cmd))
        proc = subprocess.Popen(cmd)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
        return proc.returncode

    def checkpoint(self):
        """submitit requeue hook (no-op for the server)."""
        return submitit.helpers.DelayedSubmission(self)


# ── Agent callable (submitted via submitit) ─────────────────────────────────
class LLMAgentJob:
    """Callable submitted as a SLURM/local job that runs llm_agent_loop_nodejs.py."""

    def __init__(self, agent_args: list[str]):
        self.agent_args = agent_args

    def __call__(self):
        cmd = [PYTHON, "llm_agent_loop_nodejs.py"] + self.agent_args
        logger.info("Starting agent: %s", " ".join(cmd))
        return subprocess.call(cmd)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _wait_for_server(host: str, port: int, timeout: int = 300, poll: float = 5.0,
                     proc: subprocess.Popen | None = None):
    """Block until the vLLM health endpoint responds or *timeout* seconds elapse.

    If *proc* is provided (local mode), also checks whether the server process
    has died — raising immediately instead of waiting the full timeout.
    """
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/health"
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        # Detect early server crash in local mode
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited with code {proc.returncode} "
                f"before becoming healthy. Check logs above for details."
            )
        try:
            urllib.request.urlopen(url, timeout=5)
            logger.info("vLLM server at %s is ready.", url)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(poll)
    raise TimeoutError(f"vLLM server at {url} did not become ready within {timeout}s")


def _get_slurm_job_hostname(job: submitit.Job, timeout: int = 600, poll: float = 10.0):
    """Wait for a SLURM job to start running, then return its first node hostname."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        info = job.get_info()
        state = info.get("State", "UNKNOWN")
        node = info.get("NodeList", "")
        if state == "RUNNING" and node:
            logger.info("vLLM SLURM job %s running on node: %s", job.job_id, node)
            return node
        logger.info("Waiting for vLLM job %s (state=%s)...", job.job_id, state)
        time.sleep(poll)
    raise TimeoutError(f"vLLM SLURM job did not start within {timeout}s")


def _build_agent_args(args, model: str, vllm_base_url: str) -> list[str]:
    """Build CLI args for llm_agent_loop_nodejs.py."""
    agent_args = [
        "--model", model,
        "--vllm_base_url", vllm_base_url,
        "--num_runs", str(args.num_runs),
        "--max_steps", str(args.max_steps),
        "--workers", str(args.agent_workers),
        "--history_limit", str(args.history_limit),
    ]
    if args.game:
        agent_args.extend(["--game", args.game])
    if args.enable_thinking:
        agent_args.append("--enable_thinking")
    if args.action_only:
        agent_args.append("--action_only")
    if args.no_render:
        agent_args.append("--no_render")
    if args.force:
        agent_args.append("--force")
    if args.save_dir:
        agent_args.extend(["--save_dir", args.save_dir])
    return agent_args


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM server + LLM agent jobs (SLURM or local)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Execution mode
    parser.add_argument("--slurm", action="store_true",
                        help="Submit jobs to SLURM (otherwise run locally as subprocesses)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be launched without running anything")

    # Model(s)
    parser.add_argument("--models", nargs="+", default=["vllm-qwen3.5-9b"],
                        help="Model alias(es) for the agent. Each gets its own agent job.")
    parser.add_argument("--vllm_model", type=str, default="Qwen/Qwen3.5-9B",
                        help="HuggingFace model ID for the vLLM server")

    # vLLM server
    parser.add_argument("--vllm_port", type=int, default=VLLM_DEFAULTS["port"])
    parser.add_argument("--vllm_base_url", type=str, default="",
                        help="If set, skip launching a vLLM server and connect to this URL")
    parser.add_argument("--server_startup_timeout", type=int, default=600,
                        help="Seconds to wait for vLLM server to become healthy")

    # Agent args (forwarded to llm_agent_loop_nodejs.py)
    parser.add_argument("--game", type=str, default="")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--agent_workers", type=int, default=4,
                        help="--workers passed to llm_agent_loop_nodejs.py")
    parser.add_argument("--history_limit", type=int, default=10)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--action_only", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save_dir", type=str, default="")

    # SLURM resource params
    parser.add_argument("--slurm_server_partition", type=str, default="",
                        help="SLURM partition for the GPU server job")
    parser.add_argument("--slurm_server_gpus", type=int, default=1)
    parser.add_argument("--slurm_server_mem_gb", type=int, default=64)
    parser.add_argument("--slurm_server_time_min", type=int, default=480,
                        help="Time limit for the vLLM server job (minutes)")
    parser.add_argument("--slurm_agent_mem_gb", type=int, default=16)
    parser.add_argument("--slurm_agent_cpus", type=int, default=8)
    parser.add_argument("--slurm_agent_time_min", type=int, default=480)

    args = parser.parse_args()

    # ── Resolve vLLM base URL ───────────────────────────────────────────
    skip_server = bool(args.vllm_base_url)
    vllm_base_url = args.vllm_base_url  # may be overwritten below

    if args.dry_run:
        print("=== DRY RUN ===")
        if not skip_server:
            print(f"  vLLM server: model={args.vllm_model}, port={args.vllm_port}")
        for model in args.models:
            agent_args = _build_agent_args(args, model, vllm_base_url or "<TBD>")
            print(f"  Agent job: {PYTHON} llm_agent_loop_nodejs.py {' '.join(agent_args)}")
        return

    # ── LOCAL mode ──────────────────────────────────────────────────────
    if not args.slurm:
        server_proc = None
        if not skip_server:
            vllm_base_url = f"http://localhost:{args.vllm_port}/v1"
            # Start server as background subprocess
            server_cmd = [
                PYTHON, "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.vllm_model,
                "--served-model-name", args.vllm_model,
                "--host", VLLM_DEFAULTS["host"],
                "--port", str(args.vllm_port),
                "--tensor-parallel-size", str(VLLM_DEFAULTS["tensor_parallel_size"]),
                "--gpu-memory-utilization", str(VLLM_DEFAULTS["gpu_memory_utilization"]),
                "--max-model-len", str(VLLM_DEFAULTS["max_model_len"]),
                "--max-num-batched-tokens", str(VLLM_DEFAULTS["max_num_batched_tokens"]),
            ]
            if VLLM_DEFAULTS["enforce_eager"]:
                server_cmd.append("--enforce-eager")
            if VLLM_DEFAULTS["enable_prefix_caching"]:
                server_cmd.append("--enable-prefix-caching")
            logger.info("Starting local vLLM server...")
            server_proc = subprocess.Popen(server_cmd)
            _wait_for_server("localhost", args.vllm_port, timeout=args.server_startup_timeout,
                            proc=server_proc)

        # Run agent jobs sequentially (each already uses --workers internally)
        try:
            for model in args.models:
                agent_args = _build_agent_args(args, model, vllm_base_url)
                cmd = [PYTHON, "llm_agent_loop_nodejs.py"] + agent_args
                logger.info("Running agent: %s", " ".join(cmd))
                subprocess.call(cmd)
        finally:
            if server_proc is not None:
                logger.info("Shutting down vLLM server (pid=%d)...", server_proc.pid)
                server_proc.send_signal(signal.SIGTERM)
                server_proc.wait(timeout=30)
        return

    # ── SLURM mode ──────────────────────────────────────────────────────
    os.makedirs(SUBMITIT_LOG_DIR, exist_ok=True)

    if not skip_server:
        # Submit GPU server job
        server_executor = submitit.AutoExecutor(
            folder=os.path.join(SUBMITIT_LOG_DIR, "server"))
        server_params = dict(
            slurm_job_name="vllm-server",
            mem_gb=args.slurm_server_mem_gb,
            tasks_per_node=1,
            cpus_per_task=4,
            gpus_per_node=args.slurm_server_gpus,
            timeout_min=args.slurm_server_time_min,
        )
        if args.slurm_server_partition:
            server_params["slurm_partition"] = args.slurm_server_partition
        slurm_account = os.environ.get("SLURM_ACCOUNT")
        if slurm_account:
            server_params["slurm_account"] = slurm_account
        server_executor.update_parameters(**server_params)

        server_job = server_executor.submit(
            VLLMServer(args.vllm_model, args.vllm_port))
        logger.info("Submitted vLLM server job: %s", server_job.job_id)

        # Wait for the server node to be allocated, then build the URL
        server_node = _get_slurm_job_hostname(server_job, timeout=args.server_startup_timeout)
        vllm_base_url = f"http://{server_node}:{args.vllm_port}/v1"

        # Wait for the server process itself to be healthy
        _wait_for_server(server_node, args.vllm_port, timeout=args.server_startup_timeout)

    # Submit agent jobs (CPU-only)
    agent_executor = submitit.AutoExecutor(
        folder=os.path.join(SUBMITIT_LOG_DIR, "agents"))
    agent_params = dict(
        slurm_job_name="llm-agent-nodejs",
        mem_gb=args.slurm_agent_mem_gb,
        tasks_per_node=1,
        cpus_per_task=args.slurm_agent_cpus,
        gpus_per_node=0,
        timeout_min=args.slurm_agent_time_min,
    )
    slurm_account = os.environ.get("SLURM_ACCOUNT")
    if slurm_account:
        agent_params["slurm_account"] = slurm_account
    if args.slurm_server_partition:
        # Agent jobs can usually go to any partition; skip setting this
        pass
    agent_executor.update_parameters(**agent_params)

    agent_jobs = []
    for model in args.models:
        agent_args = _build_agent_args(args, model, vllm_base_url)
        job = agent_executor.submit(LLMAgentJob(agent_args))
        logger.info("Submitted agent job for model=%s: %s", model, job.job_id)
        agent_jobs.append(job)

    print(f"\n{'='*60}")
    if not skip_server:
        print(f"  vLLM server job:  {server_job.job_id}  (node: {server_node})")
        print(f"  vLLM base URL:    {vllm_base_url}")
    print(f"  Agent job(s):     {', '.join(j.job_id for j in agent_jobs)}")
    print(f"  Logs:             {SUBMITIT_LOG_DIR}/")
    print(f"{'='*60}")
    print("\nTo monitor: sacct -j " + ",".join(
        ([server_job.job_id] if not skip_server else [])
        + [j.job_id for j in agent_jobs]))


if __name__ == "__main__":
    main()

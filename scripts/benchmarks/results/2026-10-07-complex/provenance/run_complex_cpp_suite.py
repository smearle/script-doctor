"""Run each C++ curve in isolation with explicit memory and wall-time limits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.benchmarks.complex_games import COMPLEX_GAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", nargs="+", default=COMPLEX_GAMES)
    parser.add_argument("--memory-limit-gb", type=int, default=32)
    parser.add_argument("--game-timeout", type=int, default=900)
    parser.add_argument("--max-threads", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "launch-status.json"
    result = {"memory_limit_gb": args.memory_limit_gb, "game_timeout_s": args.game_timeout,
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(), "games": []}
    for game in args.games:
        cmd = ["prlimit", f"--as={args.memory_limit_gb * 1024**3}", "--", sys.executable, "-u", "-m",
               "scripts.benchmarks.benchmark_cpp_throughput", "--library", str(args.library.resolve()),
               "--compiled-dir", str(args.compiled_dir), "--output-dir", str(args.output_dir),
               "--games", game, "--max-threads", str(args.max_threads)]
        start = time.monotonic()
        with (args.output_dir / f"{game}.log").open("w") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.game_timeout)
                status = "complete" if code == 0 else "error"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
                status = "timeout"
        row = {"game": game, "status": status, "exit_code": code, "elapsed_s": time.monotonic() - start}
        result["games"].append(row)
        status_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

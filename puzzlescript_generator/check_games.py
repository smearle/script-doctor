"""Run ps_check.js over many games in parallel Node workers.

A game that stalls a worker for --stall-s seconds is recorded as a timeout and the
worker restarts at the next game. Results come back in input order.

    python check_games.py IN.jsonl OUT.jsonl --engine-dir DIR [--workers N] [--dynamics]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _run_shard(items, engine_dir, dynamics, stall_s, node, results, idx_map, script="ps_check.js"):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
        shard = f.name
    try:
        start = 0
        while start < len(items):
            cmd = [node, str(HERE / script), str(engine_dir), shard, str(start)]
            if dynamics:
                cmd.append("--dynamics")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    text=True, bufsize=1)
            lines = queue.Queue()

            def reader(p=proc, q=lines):
                for line in p.stdout:
                    q.put(line)
                q.put(None)

            threading.Thread(target=reader, daemon=True).start()
            expect = start
            while expect < len(items):
                try:
                    line = lines.get(timeout=stall_s)
                except queue.Empty:  # stalled on items[expect]
                    proc.kill()
                    proc.wait()
                    results[idx_map[expect]] = {"i": expect, "id": items[expect]["id"], "ok": False,
                                                "compiled": None, "timeout": True}
                    expect += 1
                    break
                if line is None:  # worker died on items[expect]
                    proc.wait()
                    results[idx_map[expect]] = {"i": expect, "id": items[expect]["id"], "ok": False,
                                                "compiled": None, "crash": proc.returncode}
                    expect += 1
                    break
                rec = json.loads(line)
                assert rec["i"] == expect and rec["id"] == items[expect]["id"], (rec, expect)
                results[idx_map[expect]] = rec
                expect += 1
            else:
                proc.wait()
            start = expect
    finally:
        os.unlink(shard)


def check_texts(items, engine_dir, workers=8, dynamics=False, stall_s=60.0, node="node",
                script="ps_check.js", progress_s=None):
    """items: list of dicts with an "id" (plus "text" for ps_check.js, or the fields the
    given script reads); returns a list of result dicts in the same order. With
    progress_s, prints the number of finished games that often."""
    results = [None] * len(items)
    errors = []

    def shard(*a):
        try:
            _run_shard(*a)
        except BaseException as ex:  # surfaced below, e.g. node missing from PATH
            errors.append(ex)

    threads = []
    for w in range(workers):
        idx = list(range(w, len(items), workers))
        if not idx:
            continue
        shard_items = [items[i] for i in idx]
        t = threading.Thread(target=shard, args=(shard_items, engine_dir, dynamics, stall_s,
                                                 node, results, idx, script), daemon=True)
        t.start()
        threads.append(t)
    t0 = time.time()
    for t in threads:
        while t.is_alive():
            t.join(timeout=progress_s or None)
            if progress_s and t.is_alive():
                done = sum(r is not None for r in results)
                print(f"{script}: {done}/{len(items)} after {time.time() - t0:.0f}s", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} checker shard(s) failed") from errors[0]
    assert all(r is not None for r in results)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--dynamics", action="store_true")
    ap.add_argument("--stall-s", type=float, default=60.0)
    ap.add_argument("--node", default="node")
    args = ap.parse_args()
    items = [json.loads(line) for line in args.inp.open() if line.strip()]
    res = check_texts(items, args.engine_dir, args.workers, args.dynamics, args.stall_s, args.node)
    with args.out.open("w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    ok = sum(bool(r.get("ok")) for r in res)
    print(f"{ok}/{len(res)} pass ({ok / max(1, len(res)):.3f})")


if __name__ == "__main__":
    main()

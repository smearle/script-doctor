"""Hybrid ELM orchestrator: alternate Claude-inject and Qwen-amplify rounds into
one shared pool. Claude (Sonnet) injects new mechanic classes; Qwen amplifies them
for free (parent-sampling draws Claude variants as parents). Enforces a hard
cumulative Claude budget across all invocations (persisted in hybrid_cost.jsonl).

One parent process spawning sequential evolve.py subprocesses (robust here vs.
compound shell jobs). Reads the Claude round's cost from summary.json BEFORE the
Qwen round overwrites it.

Usage: PYTHONPATH=game_synth/autumn .venv/bin/python3 game_synth/autumn/hybrid_run.py \
         --rounds 3 --claude-iters 30 --qwen-iters 500 --budget 5.0
"""

import argparse
import json
import os
import subprocess

HERE = os.path.dirname(__file__)
VENV_PY = "/home/jupyter-smearle/script-doctor/.venv/bin/python3"
EVOLVE = os.path.join(HERE, "evolve.py")


def run_round(out, backend, model, iters, rng_seed):
    env = dict(os.environ)
    env["PYTHONPATH"] = "game_synth/autumn"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [VENV_PY, EVOLVE, "--seed-game", "mario", "--iters", str(iters),
           "--backend", backend, "--probe-seeds", "3", "--out", out,
           "--resume", "--rng-seed", str(rng_seed)]
    if model:
        cmd += ["--model", model]
    subprocess.run(cmd, env=env, cwd="/home/jupyter-smearle/script-doctor", check=True)
    return json.load(open(os.path.join(out, "summary.json")))


def prior_cost(out):
    log = os.path.join(out, "hybrid_cost.jsonl")
    if not os.path.exists(log):
        return 0.0
    return sum(json.loads(l).get("cost_usd", 0.0) for l in open(log) if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--claude-iters", type=int, default=30)
    ap.add_argument("--qwen-iters", type=int, default=500)
    ap.add_argument("--claude-model", default="claude-sonnet-4-6")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "mario_hybrid"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spent = prior_cost(args.out)
    cost_log = open(os.path.join(args.out, "hybrid_cost.jsonl"), "a")
    print(f"hybrid start: prior Claude spend ${spent:.4f}, budget ${args.budget}")

    for r in range(args.rounds):
        if spent >= args.budget:
            print(f"BUDGET REACHED (${spent:.4f} >= ${args.budget}); stopping before round {r}")
            break
        # 1) Claude inject
        s = run_round(args.out, "anthropic", args.claude_model, args.claude_iters, 100 + r)
        spent += s["cost_usd"]
        cost_log.write(json.dumps({"round": r, "phase": "claude",
                                   "cost_usd": s["cost_usd"], "accept": s["accept"],
                                   "cum_cost": round(spent, 4)}) + "\n")
        cost_log.flush()
        print(f"[round {r}] claude: +{s['accept'] - s['reloaded']} variants, "
              f"${s['cost_usd']:.4f}, cum ${spent:.4f}")
        # 2) Qwen amplify (free)
        s = run_round(args.out, "vllm", None, args.qwen_iters, 200 + r)
        print(f"[round {r}] qwen:  +{s['accept'] - s['reloaded']} variants (free), "
              f"pool now {s['n_variants']}")

    print(f"\nHYBRID DONE: total Claude spend ${spent:.4f}; pool in {args.out}/variants/")


if __name__ == "__main__":
    main()

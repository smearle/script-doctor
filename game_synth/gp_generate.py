"""Track B: hand-written GP + search generator of PuzzleScript environments.

Samples + mutates rule_gp rulesets, assembles a runnable game with rule_game,
compiles via the JS->C++ path, and validates by BFS search ("plays" each level).
Saves compiling, non-trivial (branching / solvable) games to game_synth/games/
with a JSONL manifest. No LLM / GPU needed.

    .venv/bin/python -u -m game_synth.gp_generate --target 50 --out game_synth/gp_run0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from nca_wm import rule_game, rule_gp
from puzzlescript_jax.utils import init_ps_lark_parser

_OBJSETS = [["ObjA", "ObjB"], ["ObjA", "ObjB", "ObjC"]]
_PREFIX_FLAVORS = ["", "", "", "late", "random"]


def _sample_rules(rng, max_rules=4):
    rules = []
    n_grow = rng.randint(2, max(3, max_rules + 2))
    for _ in range(n_grow):
        rules = rule_gp.mutate_ruleset(rules, rng, max_rules=max_rules)
    # extra per-rule diversification (prefixes / modifiers / commands / dirs)
    for i in range(len(rules)):
        if rng.random() < 0.6:
            rules[i] = rule_gp.random_mutate_rule(rules[i], rng)
    return rules or [rule_gp.base_two_cell_rule()]


def _sample_wins(rng):
    wins = []
    for _ in range(rng.randint(0, 3)):
        wins = rule_gp.mutate_wins(wins, rng, max_wins=2)
    return wins


# Optional fixed geometry / vocab for a clean multi-game world-model training set.
FIXED = {"w": None, "h": None, "n_levels": None, "objs": None}


def _sample_levels(rng):
    n = FIXED["n_levels"] or rng.randint(1, 3)
    out = []
    for _ in range(n):
        w = FIXED["w"] or rng.randint(5, 12)
        h = FIXED["h"] or rng.randint(5, 11)
        wd = rng.choice([0.0, 0.0, 0.0, 0.1, 0.2])
        out.append(rule_game.random_level_text(rng, w=w, h=h, wall_density=wd))
    return out


def sample_game(rng) -> str:
    objs = FIXED["objs"] or rng.choice(_OBJSETS)
    rules = _sample_rules(rng)
    wins = _sample_wins(rng)
    levels = _sample_levels(rng)
    title = f"gp_{rng.getrandbits(28):07x}"
    return rule_game.assemble_game(rules, title=title, objects=objs,
                                   levels=levels, wins=wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50, help="valid games to collect")
    ap.add_argument("--max-attempts", type=int, default=100000)
    ap.add_argument("--out", default="game_synth/gp_run0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-ms", type=int, default=4000)
    ap.add_argument("--n-steps", type=int, default=8000)
    ap.add_argument("--min-iters", type=int, default=15, help="min BFS iters = branching dynamics")
    ap.add_argument("--fixed-w", type=int, default=0, help="fixed level width (0=random)")
    ap.add_argument("--fixed-h", type=int, default=0, help="fixed level height (0=random)")
    ap.add_argument("--n-levels", type=int, default=0, help="levels per game (0=random 1-3)")
    ap.add_argument("--fixed-objs", action="store_true", help="always use ObjA,ObjB,ObjC")
    args = ap.parse_args()
    if args.fixed_w:
        FIXED["w"] = args.fixed_w
    if args.fixed_h:
        FIXED["h"] = args.fixed_h
    if args.n_levels:
        FIXED["n_levels"] = args.n_levels
    if args.fixed_objs:
        FIXED["objs"] = ["ObjA", "ObjB", "ObjC"]

    out = _REPO / args.out
    games_dir = out / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(games_dir)
    parser = init_ps_lark_parser()
    manifest = (out / "manifest.jsonl").open("a", encoding="utf-8")

    rng = random.Random(args.seed)
    run = f"gpB_{args.seed}_{int(time.time())}"
    seen = set()
    n_valid = n_solvable = n_attempt = 0
    t0 = time.time()
    while n_valid < args.target and n_attempt < args.max_attempts:
        n_attempt += 1
        try:
            code = sample_game(rng)
        except Exception:
            continue
        from game_synth.dedup import strip_noop_rules
        sig = hash(strip_noop_rules(code).split("RULES")[-1])  # crude dedup (no-op rules stripped)
        if sig in seen:
            continue
        seen.add(sig)
        name = f"{run}_{n_attempt:06d}"
        try:
            gc._materialize_game(name, code)
            ev = gc._search_materialized_game(
                parser, name, search_algo="bfs",
                search_timeout_ms=args.timeout_ms, search_n_steps=args.n_steps)
        except Exception:
            continue
        if not ev.compile_ok or ev.n_levels < 1:
            continue
        if not (ev.all_solvable or ev.max_search_iters >= args.min_iters):
            continue
        n_valid += 1
        n_solvable += int(ev.all_solvable)
        (games_dir / f"{name}.txt").write_text(code, encoding="utf-8")
        manifest.write(json.dumps({
            "name": name, "track": "gp", "n_levels": ev.n_levels,
            "all_solvable": ev.all_solvable, "max_iters": ev.max_search_iters,
        }) + "\n")
        manifest.flush()
        if n_valid % 10 == 0:
            rate = n_valid / max(n_attempt, 1)
            print(f"valid {n_valid}/{args.target} (solvable {n_solvable}) | "
                  f"attempts {n_attempt} yield {rate:.1%} | "
                  f"{n_valid/max(time.time()-t0,1e-9):.2f} games/s", flush=True)
    print(f"DONE valid={n_valid} solvable={n_solvable} attempts={n_attempt} "
          f"in {time.time()-t0:.0f}s -> {games_dir}", flush=True)
    manifest.close()


if __name__ == "__main__":
    main()

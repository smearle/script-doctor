"""Evolve rule-GP rulesets toward higher BFS state-space coverage.

Pop = K rulesets. Each generation, every parent spawns ``--children`` mutated
copies; we score each candidate by reachable-state count + longest BFS path
through the state graph, gated by transition-diff non-no-op vs an empty-rules
baseline. Top-K survive.

Renders the longest-BFS-path GIF for the best ruleset in each generation
(plus the best across all generations).
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp._puzzlescript_cpp import collect_transitions_bfs
from puzzlescript_cpp import CppPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_jax.preprocessing import add_extra_games_dir

from nca_wm.rule_gp import (
    BASE_RULE_FACTORIES, Rule, WinCondition, mutate_ruleset, mutate_wins,
    base_two_cell_rule, base_push_rule, base_delete_rule,
)
from nca_wm.rule_game import assemble_game, random_level_text, DEFAULT_LEVEL


@dataclass
class Candidate:
    rules: list[Rule]
    wins: list[WinCondition] = field(default_factory=list)
    name: str = ""
    iters: int = 0
    n_transitions: int = 0
    novel_transitions: int = 0
    longest_path: list[int] = field(default_factory=list)
    win_path: list[int] = field(default_factory=list)
    won: bool = False
    no_op: bool = False
    score: float = -1.0
    best_level_i: int = -1
    error: str = ""


_GAMES_DIR: Path | None = None


def _set_games_dir(p: Path) -> None:
    global _GAMES_DIR
    _GAMES_DIR = p
    p.mkdir(parents=True, exist_ok=True)
    add_extra_games_dir(str(p))


def _materialize(name: str, rules: list[Rule],
                 wins: list[WinCondition],
                 levels: list[str] | None = None) -> Path:
    assert _GAMES_DIR is not None, "call _set_games_dir() before _materialize()"
    text = assemble_game(rules, title=name, wins=wins, levels=levels)
    p = _GAMES_DIR / f"{name}.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _bfs_one_level(backend, level_i: int, *, max_iters: int, timeout_ms: int):
    backend.cpp_engine.load_level(level_i)
    return collect_transitions_bfs(backend.cpp_engine._engine,
                                   max_iters, timeout_ms)


def _transition_set(result) -> set:
    states = list(result.states)
    actions = list(result.actions)
    next_states = list(result.next_states)
    return {
        (tuple(int(x) for x in s), int(a), tuple(int(x) for x in ns))
        for s, a, ns in zip(states, actions, next_states)
    }


def _path_extracts(result) -> tuple[list[int], list[int], bool]:
    """Return (longest_path, win_path, won_anywhere). win_path is the
    shortest BFS-discovered winning trajectory, or [] if none.
    """
    states = list(result.states)
    if not states:
        return [], [], False
    actions = list(result.actions)
    next_states = list(result.next_states)
    wons = list(result.wons)
    root = tuple(int(x) for x in states[0])
    forward: dict[tuple, list[tuple[int, tuple]]] = defaultdict(list)
    parent: dict[tuple, tuple[tuple, int]] = {}
    win_target: tuple | None = None
    for s, a, ns, w in zip(states, actions, next_states, wons):
        s_t = tuple(int(x) for x in s)
        ns_t = tuple(int(x) for x in ns)
        forward[s_t].append((int(a), ns_t))
        if ns_t not in parent and ns_t != root:
            parent[ns_t] = (s_t, int(a))
        if w and win_target is None:
            win_target = ns_t
    depths = {root: 0}
    q = deque([root])
    while q:
        s = q.popleft()
        for _, ns in forward.get(s, ()):
            if ns not in depths:
                depths[ns] = depths[s] + 1
                q.append(ns)
    longest: list[int] = []
    if depths and any(d > 0 for d in depths.values()):
        target = max(depths, key=depths.get)
        rev = []
        cur = target
        while cur != root and cur in parent:
            prev, a = parent[cur]
            rev.append(a)
            cur = prev
        longest = list(reversed(rev))
    win: list[int] = []
    if win_target is not None and win_target != root:
        rev = []
        cur = win_target
        while cur != root and cur in parent:
            prev, a = parent[cur]
            rev.append(a)
            cur = prev
        win = list(reversed(rev))
    return longest, win, win_target is not None


def _score_one(cand: Candidate, r, baseline_set: set) -> tuple[float, dict]:
    """Score a single (level) BFS result. Returns (score, payload)."""
    ts = _transition_set(r)
    novel = ts - baseline_set
    longest, win, won = _path_extracts(r)
    payload = {
        "iters": int(r.iterations),
        "n_transitions": len(ts),
        "novel_transitions": len(novel),
        "longest_path": longest,
        "win_path": win,
        "won": won,
    }
    if not novel:
        return -1.0, payload
    depth = len(longest)
    win_bonus = 0.0
    if won and win:
        wlen = len(win)
        win_bonus = 1.0 + min(1.0, wlen / 20.0)
        if wlen <= 1:
            win_bonus = -0.5
    score = (math.log1p(payload["iters"])
             + 0.5 * math.log1p(payload["novel_transitions"])
             + 0.3 * depth
             - 0.05 * max(0, len(cand.rules) - 1)
             + win_bonus)
    return score, payload


def _evaluate(parser, cand: Candidate, baseline_set: set, *,
              max_iters: int, timeout_ms: int,
              n_levels: int) -> Candidate:
    """Sample n_levels random levels, embed all in one game, BFS each, take best.

    The compiled engine is shared across all n_levels (cheap level swap), so
    cost per candidate ≈ 1 compile + n_levels BFS calls.
    """
    try:
        backend = CppPuzzleScriptBackend()
        backend.compile_game(parser, cand.name)
        actual_n = int(backend.get_num_levels())
        best_score = -float("inf")
        best_payload = None
        best_li = -1
        for li in range(actual_n):
            r = _bfs_one_level(backend, li,
                               max_iters=max_iters, timeout_ms=timeout_ms)
            s, payload = _score_one(cand, r, baseline_set)
            if s > best_score:
                best_score = s
                best_payload = payload
                best_li = li
        if best_payload is None or best_score == -float("inf"):
            cand.score = -1.0
            cand.no_op = True
            return cand
        cand.score = best_score
        cand.iters = best_payload["iters"]
        cand.n_transitions = best_payload["n_transitions"]
        cand.novel_transitions = best_payload["novel_transitions"]
        cand.longest_path = best_payload["longest_path"]
        cand.win_path = best_payload["win_path"]
        cand.won = best_payload["won"]
        cand.no_op = (best_payload["novel_transitions"] == 0)
        cand.best_level_i = best_li
        return cand
    except Exception as e:
        cand.error = f"{type(e).__name__}: {str(e)[:120]}"
        cand.score = -1.0
        return cand


def _initial_population(rng) -> list[tuple[list[Rule], list[WinCondition]]]:
    """Hand-pick seed (rules, wins) pairs."""
    return [
        ([base_two_cell_rule()],   []),                                     # convert, no win
        ([base_two_cell_rule()],   [WinCondition("no", "ObjA")]),           # convert, win when A gone
        ([base_push_rule("ObjA")], []),                                     # push, no win
        ([base_delete_rule("ObjA")], [WinCondition("no", "ObjA")]),         # destroy + win
        ([base_push_rule("ObjA"), base_push_rule("ObjB")], []),
        ([base_push_rule("ObjA"), base_push_rule("ObjB")], [WinCondition("no", "ObjC")]),
    ]


def _name_for(gen: int, idx: int, rules: list[Rule],
              wins: list[WinCondition]) -> str:
    fingerprint = abs(hash(
        tuple(r.unparse() for r in rules) + tuple(w.unparse() for w in wins),
    )) % 0xFFFFFF
    return f"rule_gp_evo_g{gen:02d}_c{idx:02d}_{fingerprint:06x}"


def _render_gif(parser, cand: Candidate, gif_path: Path, *,
                scale: int, frame_duration_s: float) -> None:
    # Prefer the winning path when there is one — that's the actual puzzle
    # solution. Fall back to the longest-explored path otherwise. Render on
    # the BFS-best level we picked during evaluation.
    actions = cand.win_path or cand.longest_path
    if not actions or cand.best_level_i < 0:
        return
    backend = CppPuzzleScriptBackend()
    backend.compile_game(parser, cand.name)
    backend.render_gif(
        game_text=cand.name, level_i=cand.best_level_i, actions=actions,
        gif_path=str(gif_path),
        frame_duration_s=frame_duration_s, scale=scale,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop_size", type=int, default=4)
    ap.add_argument("--n_generations", type=int, default=6)
    ap.add_argument("--children", type=int, default=4,
                    help="Children per parent per generation.")
    ap.add_argument("--bfs_max_iters", type=int, default=8000)
    ap.add_argument("--bfs_timeout_ms", type=int, default=4000)
    ap.add_argument("--levels_per_candidate", type=int, default=8,
                    help="N random levels embedded in each candidate game; "
                         "the candidate is scored on its best-BFS level. "
                         "Always includes the canonical default template + "
                         "(N-1) random samples.")
    ap.add_argument("--run_dir", default=None,
                    help="Per-run output dir. Defaults to "
                         "nca_wm/logs/rule_gp_evo_<unix_ts>_s<seed>/.")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--frame_duration_s", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.run_dir is None:
        import time as _time
        args.run_dir = f"nca_wm/logs/rule_gp_evo_{int(_time.time())}_s{args.seed}"
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_games_dir(run_dir / "games")
    parser = init_ps_lark_parser()
    print(f"[evolve] run_dir={run_dir}")
    print(f"[evolve] games  -> {run_dir / 'games'}")
    print(f"[evolve] gifs   -> {out_dir}")

    def _sample_levels(local_rng: random.Random) -> list[str]:
        """Default template first (so canonical scoring is comparable across
        runs), then N-1 randomized samples."""
        levels = [DEFAULT_LEVEL]
        for _ in range(max(0, args.levels_per_candidate - 1)):
            levels.append(random_level_text(local_rng))
        return levels

    print("[evolve] Materializing baseline (no rules, no win)...")
    base_name = "rule_gp_evo_baseline"
    _materialize(base_name, [], [], levels=[DEFAULT_LEVEL])
    base_backend = CppPuzzleScriptBackend()
    base_backend.compile_game(parser, base_name)
    base_backend.cpp_engine.load_level(0)
    base_r = collect_transitions_bfs(base_backend.cpp_engine._engine,
                                     args.bfs_max_iters, args.bfs_timeout_ms)
    baseline_set = _transition_set(base_r)
    print(f"[evolve]   baseline iters={int(base_r.iterations)}, "
          f"transitions={len(baseline_set)}")

    pop: list[Candidate] = []
    for i, (rules, wins) in enumerate(_initial_population(rng)):
        name = _name_for(0, i, rules, wins)
        _materialize(name, rules, wins, levels=_sample_levels(rng))
        c = _evaluate(parser,
                      Candidate(rules=rules, wins=wins, name=name),
                      baseline_set,
                      max_iters=args.bfs_max_iters,
                      timeout_ms=args.bfs_timeout_ms,
                      n_levels=args.levels_per_candidate)
        pop.append(c)

    pop.sort(key=lambda c: c.score, reverse=True)
    pop = pop[:args.pop_size]
    print(f"\n[evolve] gen 0 init pop:")
    for c in pop:
        wstr = ", ".join(w.unparse() for w in c.wins) or "(no win)"
        print(f"  {c.name}  score={c.score:.3f}  iters={c.iters}  "
              f"path={len(c.longest_path)}  win={len(c.win_path)}  "
              f"won={c.won}  wins=[{wstr}]  "
              f"rules={[r.unparse() for r in c.rules]}")

    history: list[dict] = []
    best_overall = max(pop, key=lambda c: c.score)
    for gen in range(1, args.n_generations + 1):
        children: list[Candidate] = []
        for parent in pop:
            for ci in range(args.children):
                child_rules = mutate_ruleset(
                    [copy.deepcopy(r) for r in parent.rules], rng,
                )
                # Mutate win conditions about 1/3 of the time (independent
                # of the rule mutation).
                if rng.random() < 0.35:
                    child_wins = mutate_wins(
                        [copy.deepcopy(w) for w in parent.wins], rng,
                    )
                else:
                    child_wins = [copy.deepcopy(w) for w in parent.wins]
                name = _name_for(gen, len(children), child_rules, child_wins)
                _materialize(name, child_rules, child_wins,
                             levels=_sample_levels(rng))
                c = _evaluate(parser,
                              Candidate(rules=child_rules, wins=child_wins,
                                        name=name),
                              baseline_set,
                              max_iters=args.bfs_max_iters,
                              timeout_ms=args.bfs_timeout_ms,
                              n_levels=args.levels_per_candidate)
                children.append(c)
        pool = pop + children
        pool.sort(key=lambda c: c.score, reverse=True)
        pop = pool[:args.pop_size]
        gen_best = pop[0]
        if gen_best.score > best_overall.score:
            best_overall = gen_best

        n_survivors = sum(1 for c in pool if c.score > 0)
        n_won = sum(1 for c in pool if c.won)
        print(f"\n[evolve] gen {gen}: best score={gen_best.score:.3f}  "
              f"iters={gen_best.iters}  path={len(gen_best.longest_path)}  "
              f"win_path={len(gen_best.win_path)}  won={gen_best.won}  "
              f"survivors={n_survivors}/{len(pool)}  won_in_pool={n_won}")
        for c in pop:
            wstr = ", ".join(w.unparse() for w in c.wins) or "(no win)"
            print(f"  {c.name}  score={c.score:.3f}  iters={c.iters}  "
                  f"path={len(c.longest_path)}  win={len(c.win_path)}  "
                  f"won={c.won}  wins=[{wstr}]  "
                  f"rules={[r.unparse() for r in c.rules]}")

        if gen_best.longest_path or gen_best.win_path:
            gif_path = out_dir / f"gen{gen:02d}_best.gif"
            _render_gif(parser, gen_best, gif_path,
                        scale=args.scale, frame_duration_s=args.frame_duration_s)
            print(f"  wrote {gif_path}")

        history.append({
            "generation": gen,
            "best_score": gen_best.score,
            "best_name": gen_best.name,
            "best_rules": [r.unparse() for r in gen_best.rules],
            "best_wins": [w.unparse() for w in gen_best.wins],
            "best_iters": gen_best.iters,
            "best_path_len": len(gen_best.longest_path),
            "best_win_path_len": len(gen_best.win_path),
            "best_won": gen_best.won,
        })

    if best_overall.longest_path or best_overall.win_path:
        gif_path = out_dir / "best_overall.gif"
        _render_gif(parser, best_overall, gif_path,
                    scale=args.scale, frame_duration_s=args.frame_duration_s)
        wstr = ", ".join(w.unparse() for w in best_overall.wins) or "(no win)"
        print(f"\n[evolve] best overall: {best_overall.name} "
              f"score={best_overall.score:.3f}  -> {gif_path}")
        print(f"  wins: {wstr}")
        for r in best_overall.rules:
            print(f"  rule: {r.unparse()}")


if __name__ == "__main__":
    main()

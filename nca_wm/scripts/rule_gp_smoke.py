"""Smoketest + GIF render for the rule-GP curriculum prototype.

For every enumerated ruleset:
  1. Materialize into ``custom_games/`` via the assembler.
  2. Compile and BFS-explore.
  3. Diff transition table against an empty-rules baseline → no-op flag.
  4. Extract the "best" BFS path: a winning path if BFS found one, else the
     longest reachable path through the BFS tree (the path the GIF replays).
  5. Render a GIF for any ruleset whose best path is non-trivial.

Reports per-ruleset: compile?, no-op?, BFS iterations, transitions vs
baseline, best-path length, won?, GIF path. Aggregates: compile rate,
no-op rate, "interesting" rate (non-no-op AND path-length >= min_len).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp._puzzlescript_cpp import Engine, collect_transitions_bfs
from puzzlescript_cpp import CppPuzzleScriptBackend
from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_jax.preprocessing import add_extra_games_dir

from nca_wm.rule_gp import enumerate_smoketest_rulesets, axis_label
from nca_wm.rule_game import assemble_game


_GAMES_DIR: Path | None = None


def _set_games_dir(p: Path) -> None:
    global _GAMES_DIR
    _GAMES_DIR = p
    p.mkdir(parents=True, exist_ok=True)
    add_extra_games_dir(str(p))


def _materialize(name: str, rules) -> Path:
    assert _GAMES_DIR is not None
    text = assemble_game(rules, title=name)
    p = _GAMES_DIR / f"{name}.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _run_bfs(eng: Engine, *, max_iters: int = 5000, timeout_ms: int = 3000):
    return collect_transitions_bfs(eng, max_iters, timeout_ms)


def _transition_set(result) -> set:
    states = list(result.states)
    actions = list(result.actions)
    next_states = list(result.next_states)
    return {
        (tuple(int(x) for x in s), int(a), tuple(int(x) for x in ns))
        for s, a, ns in zip(states, actions, next_states)
    }


def _extract_best_path(result, *, prefer_winning: bool = True) -> list[int]:
    """Either the winning path (if BFS observed a win) or the longest path
    through the BFS tree from the initial state.
    """
    states = list(result.states)
    if not states:
        return []
    actions = list(result.actions)
    next_states = list(result.next_states)
    wons = list(result.wons)
    root = tuple(int(x) for x in states[0])

    forward: dict[tuple, list[tuple[int, tuple]]] = defaultdict(list)
    parent: dict[tuple, tuple[tuple, int]] = {}
    for s, a, ns in zip(states, actions, next_states):
        s_t = tuple(int(x) for x in s)
        ns_t = tuple(int(x) for x in ns)
        forward[s_t].append((int(a), ns_t))
        if ns_t not in parent and ns_t != root:
            parent[ns_t] = (s_t, int(a))

    # BFS-depth from the root using the forward map (re-computed; cheap).
    depths: dict[tuple, int] = {root: 0}
    q = deque([root])
    while q:
        s = q.popleft()
        for _, ns in forward.get(s, ()):
            if ns not in depths:
                depths[ns] = depths[s] + 1
                q.append(ns)

    target: tuple | None = None
    if prefer_winning:
        for ns, w in zip(next_states, wons):
            if w:
                target = tuple(int(x) for x in ns)
                break
    if target is None:
        if not depths:
            return []
        target = max(depths, key=depths.get)
        if depths[target] == 0:
            return []
    if target == root:
        return []

    rev: list[int] = []
    cur = target
    while cur != root:
        if cur not in parent:
            return []
        prev, a = parent[cur]
        rev.append(a)
        cur = prev
    return list(reversed(rev))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="nca_wm/logs/rule_gp_smoke",
                    help="Per-run output dir. Holds games/ and gifs/ subdirs.")
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--frame_duration_s", type=float, default=0.18)
    ap.add_argument("--bfs_max_iters", type=int, default=5000)
    ap.add_argument("--bfs_timeout_ms", type=int, default=3000)
    ap.add_argument("--min_path_len", type=int, default=2,
                    help="Don't render GIFs with paths shorter than this.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_games_dir(run_dir / "games")
    out_dir = run_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)
    parser = init_ps_lark_parser()
    print(f"[rule_gp_smoke] run_dir={run_dir}")

    print("[rule_gp_smoke] Materializing baseline (empty rule set)...")
    base_name = "rule_gp_smoke_baseline"
    _materialize(base_name, [])
    base_backend = CppPuzzleScriptBackend()
    base_backend.compile_game(parser, base_name)
    base_backend.cpp_engine.load_level(0)
    base_result = _run_bfs(base_backend.cpp_engine._engine,
                           max_iters=args.bfs_max_iters,
                           timeout_ms=args.bfs_timeout_ms)
    base_set = _transition_set(base_result)
    base_path = _extract_best_path(base_result, prefer_winning=False)
    print(f"[rule_gp_smoke]   baseline: iters={int(base_result.iterations)}, "
          f"transitions={len(base_set)}, longest_path={len(base_path)}")

    rulesets = enumerate_smoketest_rulesets()
    print(f"[rule_gp_smoke] Enumerated {len(rulesets)} rulesets.\n")

    rows = []
    for axes, rules in rulesets:
        label = axis_label(axes)
        slug = (label.replace(" ", "_").replace("=", "-").replace(",", ""))
        name = f"rule_gp_smoke_{slug}"
        _materialize(name, rules)
        try:
            backend = CppPuzzleScriptBackend()
            backend.compile_game(parser, name)  # also loads renderer
            backend.cpp_engine.load_level(0)
            eng_text = name
            r = _run_bfs(backend.cpp_engine._engine,
                         max_iters=args.bfs_max_iters,
                         timeout_ms=args.bfs_timeout_ms)
            ts = _transition_set(r)
            no_op = (ts == base_set)
            path = _extract_best_path(r, prefer_winning=True)
            won = bool(any(list(r.wons)))
            row = {
                "axes": axes, "label": label, "name": name,
                "compiled": True, "iterations": int(r.iterations),
                "n_transitions": len(ts),
                "diff_size": len(ts.symmetric_difference(base_set)),
                "no_op": no_op, "path_len": len(path),
                "won": won, "rule_text": rules[0].unparse(),
                "gif": "",
            }
            if not no_op and len(path) >= args.min_path_len:
                gif_path = out_dir / f"{name}.gif"
                backend.render_gif(
                    game_text=eng_text, level_i=0, actions=path,
                    gif_path=str(gif_path),
                    frame_duration_s=args.frame_duration_s,
                    scale=args.scale,
                )
                row["gif"] = str(gif_path)
            rows.append(row)
        except Exception as e:
            rows.append({
                "axes": axes, "label": label, "name": name,
                "compiled": False, "error": str(e)[:120],
                "rule_text": rules[0].unparse(), "gif": "",
            })

    print(f"{'label':<28s}  cmpl  iters  trans  diff  no-op  path  won  rule")
    print("-" * 120)
    for r in rows:
        if r["compiled"]:
            print(f"{r['label']:<28s}  OK    "
                  f"{r['iterations']:5d}  {r['n_transitions']:5d}  "
                  f"{r['diff_size']:4d}  "
                  f"{('YES' if r['no_op'] else 'no'):<5s}  "
                  f"{r['path_len']:4d}  "
                  f"{('Y' if r['won'] else '.'):<3s}  {r['rule_text']}")
        else:
            print(f"{r['label']:<28s}  FAIL  "
                  f"{'-':>27s}  {r['rule_text']}  [{r['error']}]")

    n_total = len(rows)
    n_compiled = sum(1 for r in rows if r["compiled"])
    n_no_op = sum(1 for r in rows if r["compiled"] and r["no_op"])
    n_interesting = sum(
        1 for r in rows
        if r["compiled"] and not r["no_op"]
        and r["path_len"] >= args.min_path_len
    )
    n_won = sum(1 for r in rows if r["compiled"] and r["won"])
    n_gifs = sum(1 for r in rows if r.get("gif"))
    print()
    print(f"[rule_gp_smoke] compile rate:     {n_compiled}/{n_total}")
    print(f"[rule_gp_smoke] no-op (compiled): {n_no_op}/{n_compiled}")
    print(f"[rule_gp_smoke] won (compiled):   {n_won}/{n_compiled}")
    print(f"[rule_gp_smoke] interesting:      {n_interesting}/{n_total}")
    print(f"[rule_gp_smoke] GIFs written:     {n_gifs} -> {out_dir}/")
    if n_gifs:
        for r in rows:
            if r.get("gif"):
                print(f"  {Path(r['gif']).name}  "
                      f"(path_len={r['path_len']}, won={r['won']})")


if __name__ == "__main__":
    main()

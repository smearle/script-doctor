"""Render BFS-solution GIFs for every game in a curriculum run.

Walks every ``summary.json`` under a curriculum save_dir, collects unique
``compile_ok`` games (seeds + LLM-generated children), runs BFS on each
level, and writes a GIF per (game, level) pair where BFS found a win.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_jax.utils import init_ps_lark_parser
from puzzlescript_jax.preprocessing import add_extra_games_dir
from puzzlescript_cpp import CppPuzzleScriptBackend


def _collect_games(run_dir: Path) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    for summary in run_dir.glob("gen_*/set_*/summary.json"):
        s = json.loads(summary.read_text(encoding="utf-8"))
        for g in s["games"]:
            if g.get("compile_ok") and g["name"] not in seen:
                seen[g["name"]] = g
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Curriculum save_dir (contains gen_NNN/set_*/summary.json).")
    ap.add_argument("--out_dir", default=None,
                    help="Default: <run_dir>/gifs.")
    ap.add_argument("--search_n_steps", type=int, default=30000)
    ap.add_argument("--search_timeout_ms", type=int, default=10000)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--frame_duration_s", type=float, default=0.18)
    ap.add_argument("--max_levels_per_game", type=int, default=4)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Newer runs put materialized games under <run_dir>/games/. Register that
    # so the lookup chain finds them (older runs left them in custom_games/).
    games_subdir = run_dir / "games"
    if games_subdir.is_dir():
        add_extra_games_dir(str(games_subdir))

    games = _collect_games(run_dir)
    print(f"[curriculum_gifs] {len(games)} unique compiled games in {run_dir}")
    parser = init_ps_lark_parser()

    n_gifs = 0
    n_unsolved = 0
    n_errors = 0
    per_game: list[dict] = []
    for name in sorted(games.keys()):
        meta = games[name]
        try:
            backend = CppPuzzleScriptBackend()
            backend.compile_game(parser, name)
            n_levels = int(backend.get_num_levels())
        except Exception as e:
            print(f"  {name}: COMPILE FAIL {type(e).__name__}: {str(e)[:80]}")
            n_errors += 1
            per_game.append({"name": name, "compile": False})
            continue

        gifs_for_game = []
        for li in range(min(n_levels, args.max_levels_per_game)):
            try:
                backend.cpp_engine.load_level(li)
                sr = backend.cpp_engine.solve_bfs(
                    args.search_n_steps, args.search_timeout_ms,
                )
            except Exception as e:
                print(f"  {name} lvl {li}: SOLVE FAIL {type(e).__name__}: {str(e)[:80]}")
                continue

            if not sr.won or not list(sr.actions):
                n_unsolved += 1
                continue
            actions = list(sr.actions)
            gif_path = out_dir / f"{name}_lvl{li:02d}.gif"
            try:
                backend.render_gif(
                    game_text=name, level_i=li, actions=actions,
                    gif_path=str(gif_path),
                    frame_duration_s=args.frame_duration_s,
                    scale=args.scale,
                )
            except Exception as e:
                print(f"  {name} lvl {li}: RENDER FAIL {type(e).__name__}: {str(e)[:80]}")
                continue
            print(f"  {name} lvl {li}: {len(actions)} steps -> {gif_path.name}")
            n_gifs += 1
            gifs_for_game.append({
                "level": li, "n_actions": len(actions),
                "iterations": int(sr.iterations),
                "gif": str(gif_path),
            })

        per_game.append({
            "name": name,
            "mode": meta.get("mode", ""),
            "generation": meta.get("generation", -1),
            "all_solvable_per_summary": meta.get("all_solvable", False),
            "n_levels": n_levels,
            "n_gifs_rendered": len(gifs_for_game),
            "gifs": gifs_for_game,
        })

    index = {
        "run_dir": str(run_dir),
        "n_unique_games": len(games),
        "n_gifs": n_gifs,
        "n_unsolved_levels": n_unsolved,
        "n_errors": n_errors,
        "per_game": per_game,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\n[curriculum_gifs] wrote {n_gifs} GIFs into {out_dir}")
    print(f"[curriculum_gifs] {n_unsolved} levels unsolved-by-BFS, "
          f"{n_errors} games failed to compile fresh.")


if __name__ == "__main__":
    main()

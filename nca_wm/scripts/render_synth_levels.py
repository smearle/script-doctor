"""ASCII-render synth-cache levels and save per-level GIFs.

Usage:
    python -m nca_wm.scripts.render_synth_levels <cache.npz> [--n 8] [--no_gifs]

By default:
  - Prints up to ``--n`` levels as ASCII to stdout.
  - Saves per-level GIFs to ``<cache_dir>/<cache_stem>_gifs/L{idx}.gif``
    (alongside the cache file in rollout_data). Each GIF starts from the
    synth level's initial state and animates a short random-action rollout
    so you can eyeball the dynamics in addition to the static layout.

Pass ``--no_gifs`` to skip the (expensive) GIF rendering when you only
want the ASCII output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Generic glyphs for the most common PuzzleScript object names. Anything not
# in this map falls back to the first letter of the name (uppercased), with
# duplicates disambiguated by suffix.
GLYPH_MAP = {
    "background": ".",
    "wall": "#",
    "player": "P",
    "crate": "*",
    "target": "O",
    "fruit": "F",
    "exit": "E",
    "entrance": "S",
    "gate": "G",
}


def _build_glyphs(id_dict: list[str]) -> dict[int, str]:
    glyphs: dict[int, str] = {}
    used: set[str] = set()
    # First pass: known names
    for i, name in enumerate(id_dict):
        key = name.lower()
        if key in GLYPH_MAP:
            glyphs[i] = GLYPH_MAP[key]
            used.add(GLYPH_MAP[key])
    # Second pass: derive single-letter glyphs from name initials
    for i, name in enumerate(id_dict):
        if i in glyphs:
            continue
        for c in name:
            if c.isalpha():
                cu = c.upper()
                if cu not in used:
                    glyphs[i] = cu
                    used.add(cu)
                    break
        if i not in glyphs:
            glyphs[i] = "?"
    return glyphs


def _render_dat(
    dat: list[int], width: int, height: int, stride: int,
    id_dict: list[str], glyphs: dict[int, str],
) -> str:
    """Column-major: tile = x * height + y."""
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            tile = x * height + y
            base = tile * stride
            present = []
            for o in range(len(id_dict)):
                word_i, bit = divmod(o, 32)
                if word_i < stride and int(dat[base + word_i]) & (1 << bit):
                    present.append(o)
            # Skip pure-background tiles to use "."
            non_bg = [o for o in present if id_dict[o].lower() != "background"]
            if not non_bg:
                row.append(glyphs.get(0, "."))
            else:
                # Render the topmost non-background object (highest layer)
                # heuristically: prefer player > crate > target > wall > others
                priority_order = ["player", "crate", "target", "wall"]
                best = None
                for name in priority_order:
                    for o in non_bg:
                        if id_dict[o].lower() == name:
                            best = o
                            break
                    if best is not None:
                        break
                if best is None:
                    best = non_bg[0]
                row.append(glyphs.get(best, "?"))
        rows.append("".join(row))
    return "\n".join(rows)


def _save_level_gifs(
    cache_path: str,
    game_name: str,
    dats: np.ndarray,
    width: int, height: int,
    *,
    saved_solutions: list[list[int]] | None = None,
    n_levels: int,
    start: int = 0,
    max_action_steps: int = 50,
    scale: int = 4,
    seed: int = 0,
    frame_duration_s: float = 0.15,
) -> str:
    """For each of ``n_levels`` synth levels, render a GIF that replays the
    **BFS-optimal solution path saved during evolve** (cache v9+). For
    levels with no saved solution (e.g. dynamics-only fallback), a short
    random rollout is shown instead. No search is re-run during rendering.

    Saves to ``<cache_dir>/<cache_stem>_gifs/L{idx}[_solved|_dyn].gif``.
    """
    import imageio
    from puzzlescript_cpp import CppPuzzleScriptBackend
    from puzzlescript_cpp._puzzlescript_cpp import LevelBackup
    from puzzlescript_jax.utils import init_ps_lark_parser

    out_dir = Path(cache_path).with_suffix("").as_posix() + "_gifs"
    os.makedirs(out_dir, exist_ok=True)

    parser = init_ps_lark_parser()
    backend = CppPuzzleScriptBackend()
    backend.compile_game(parser, game_name)
    backend.cpp_engine.load_level(0)
    engine = backend.cpp_engine

    rng = np.random.default_rng(seed)
    end = min(start + n_levels, len(dats))
    n_actions = 5  # up/left/down/right/action
    n_with_solution = 0

    for i in range(start, end):
        backup = LevelBackup(list(int(v) for v in dats[i]), width, height)
        engine.restore_level(backup)
        # Pick action sequence: prefer cached solution, fall back to random.
        cached = (saved_solutions[i] if saved_solutions is not None
                  and i < len(saved_solutions) else None)
        if cached is not None and len(cached) > 0:
            actions = [int(a) for a in cached][:max_action_steps]
            n_with_solution += 1
            tag = "_solved"
        else:
            actions = [int(rng.integers(n_actions))
                       for _ in range(min(15, max_action_steps))]
            tag = "_dyn"
        frames = [backend.render_frame()]
        for a in actions:
            engine.process_input(int(a))
            again_steps = 0
            while engine.againing and again_steps < 8:
                engine.process_input(-1)
                again_steps += 1
            frames.append(backend.render_frame())
        if scale > 1:
            frames = [
                np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)
                for f in frames
            ]
        gif_path = os.path.join(out_dir, f"L{i:04d}{tag}.gif")
        imageio.mimsave(gif_path, frames, duration=frame_duration_s, loop=0)
    print(f"  GIF: {n_with_solution}/{end-start} levels played the cached "
          f"BFS solution; {(end-start) - n_with_solution} used a short random fallback.")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache", help="Path to synth cache .npz")
    ap.add_argument("--n", type=int, default=8, help="How many levels to render")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--legend", action="store_true", help="Print object→glyph legend")
    ap.add_argument("--no_gifs", action="store_true",
                    help="Skip GIF saving (ASCII only). Default behavior is to also save "
                         "per-level GIFs alongside the cache.")
    ap.add_argument("--gif_n_steps", type=int, default=10,
                    help="Random action steps per GIF (frames = steps + 1)")
    ap.add_argument("--gif_scale", type=int, default=4,
                    help="Pixel scaling for GIF frames (nearest-neighbor)")
    ap.add_argument("--gif_seed", type=int, default=0)
    args = ap.parse_args()

    npz = np.load(args.cache, allow_pickle=True)
    if "level_dats" not in npz.files or "gen_stats" not in npz.files:
        print(f"ERROR: {args.cache} doesn't contain level_dats / gen_stats")
        sys.exit(1)

    dats = npz["level_dats"]  # (N, w*h*stride)
    stats = json.loads(npz["gen_stats"][0])
    id_dict = stats["id_dict"]
    stride = stats["stride_obj"]

    # Derive width/height from the cache key embedded in path. Failing that,
    # try to infer from dat length and stride.
    parts = os.path.basename(os.path.dirname(args.cache))
    w, h = None, None
    if "x" in parts:
        try:
            wh = parts.split("_")[-1]  # e.g. "synthetic_7x7"
            w_str, h_str = wh.split("x")
            w, h = int(w_str), int(h_str)
        except Exception:
            pass
    if w is None or h is None:
        n_tiles = dats.shape[1] // stride
        w = h = int(n_tiles ** 0.5)

    glyphs = _build_glyphs(id_dict)
    if args.legend:
        print("Legend:")
        for i, name in enumerate(id_dict):
            print(f"  {glyphs[i]} = {name}")
        print()

    print(f"Cache: {args.cache}")
    print(f"  game={stats.get('game_name','?')}  n_levels={stats['n_levels_accepted']}/"
          f"{stats['n_levels_target']}  shape={w}x{h}  stride={stride}")
    print(f"  n_transitions={stats['n_transitions']:,}  "
          f"winning={stats.get('n_winning_transitions','?')}  "
          f"mode={stats.get('mode','?')}  require_solvable={stats.get('require_solvable','?')}")
    print()

    end = min(args.start + args.n, len(dats))
    if not args.no_gifs:
        # Pull cached BFS-optimal solution actions if the cache was generated
        # with v9+ (which saves them so we don't re-search during render).
        saved_solutions = None
        if "level_solutions" in npz.files:
            try:
                saved_solutions = [list(s) for s in npz["level_solutions"]]
            except Exception:
                saved_solutions = None
        if saved_solutions is None:
            print("  (cache predates v9: no saved solutions; GIFs will use random actions)")
        out_dir = _save_level_gifs(
            args.cache, stats.get("game_name", "?"),
            dats, w, h,
            saved_solutions=saved_solutions,
            n_levels=args.n, start=args.start,
            max_action_steps=args.gif_n_steps * 5 if args.gif_n_steps else 50,
            scale=args.gif_scale, seed=args.gif_seed,
        )
        print(f"GIFs saved to {out_dir}")
        print()
    for i in range(args.start, end):
        rendered = _render_dat(list(dats[i]), w, h, stride, id_dict, glyphs)
        # Player count for sanity
        n_players = 0
        for tile in range(w * h):
            base = tile * stride
            for o in range(len(id_dict)):
                if id_dict[o].lower() in ("player", "upolice", "dpolice", "lpolice", "rpolice"):
                    word_i, bit = divmod(o, 32)
                    if word_i < stride and int(dats[i][base + word_i]) & (1 << bit):
                        n_players += 1
                        break
        print(f"--- L{i} ({n_players} player tiles) ---")
        print(rendered)
        print()


if __name__ == "__main__":
    main()

"""LLM-driven curriculum over *sets* of PuzzleScript game descriptions.

This is the game-level analogue of ``nca_wm.curriculum``: instead of mutating
levels for one fixed game, it asks a local OpenAI-compatible vLLM server to
generate or edit complete PuzzleScript games, validates them, and evolves
fixed-size game sets. The output is intentionally training-ready: every kept set
gets materialized into ``custom_games/`` plus a ``train_command.sh`` that calls
``nca_wm.train --games name1,name2,...``.

The default score is a cheap proxy that does not train a world model:
solvability/search hardness + token/mechanic diversity + shape sanity. This
keeps the generator loop fast. Downstream runs can train/evaluate each set with
the emitted command, then feed those metrics back in a later scoring pass.

Example:
    python -m nca_wm.game_curriculum \\
        --seed_games sokoban_basic,nekopuzzle,blocks \\
        --model google/gemma-4-31B-it \\
        --vllm_base_url http://localhost:8000/v1 \\
        --pop_size 4 --set_size 4 --n_generations 3
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from puzzlescript_cpp import CppPuzzleScriptBackend, CppPuzzleScriptEnv
from puzzlescript_jax.globals import CUSTOM_GAMES_DIR, SIMPLIFIED_GAMES_DIR
from puzzlescript_jax.utils import init_ps_lark_parser

from nca_wm.tokenize_game import get_game_tree_from_js, tokenize_game

# Reuse the mature vLLM, code-extraction, Node compile, and BFS-search helpers.
from evolve_games_agentic import (
    CROSSOVER_PROMPT,
    GENERATE_PROMPT,
    MUTATE_PROMPT,
    build_system_prompt,
    extract_ps_code,
    query_vllm,
    sample_games,
)


TARGETED_EDIT_PROMPT = (
    "Consider this PuzzleScript game:\n"
    "```plaintext\n{parent_code}\n```\n\n"
    "Make a targeted curriculum edit for training a conditional world model. "
    "Preserve the core theme, but add one learnable mechanics variation such as "
    "a new interaction rule, a delayed/again dynamic, a global row/column rule, "
    "or an object whose behavior composes with an existing rule. Keep object "
    "count and level sizes modest.\n\n"
    "Return the complete edited game inside a ```plaintext code block.\n"
    "Constraints: no randomDir, no sound effects, at most {max_levels} levels.\n"
)

LEVEL_REPAIR_PROMPT = (
    "The following PuzzleScript game compiled, but its levels need curriculum "
    "repair:\n```plaintext\n{code}\n```\n\n"
    "Solver feedback:\n```\n{feedback}\n```\n\n"
    "Return the full game with deterministic level/rule fixes so every level is "
    "non-trivially solvable. Prefer editing LEVELS before changing mechanics.\n"
    "Return only one ```plaintext code block. At most {max_levels} levels.\n"
)


@dataclass
class GameCandidate:
    name: str
    code: str
    uid: str
    parent_uids: list[str] = field(default_factory=list)
    mode: str = "seed"
    generation: int = 0
    materialized: bool = False
    compile_ok: bool = False
    all_solvable: bool = False
    n_levels: int = 0
    n_tokens: int = 0
    n_objs: int = 0
    max_h: int = 0
    max_w: int = 0
    max_search_iters: int = 0
    score: float = -1.0
    error: str = ""


@dataclass
class GameSet:
    uid: str
    games: list[GameCandidate]
    generation: int = 0
    parent_set_uids: list[str] = field(default_factory=list)
    score: float = -1.0
    metrics: dict = field(default_factory=dict)


@dataclass
class LevelFeedback:
    level_i: int
    solved: bool
    n_iters: int
    solution_len: int
    error: str = ""


@dataclass
class EvalFeedback:
    compile_ok: bool
    error: str = ""
    n_levels: int = 0
    all_solvable: bool = False
    max_search_iters: int = 0
    levels: list[LevelFeedback] = field(default_factory=list)


def _safe_slug(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return (slug or "game")[:max_len]


def _hash_text(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def _split_games(raw: str) -> list[str]:
    presets = _load_train_presets()
    if raw in presets:
        return list(presets[raw])
    return [g.strip() for g in raw.split(",") if g.strip()]


def _load_train_presets() -> dict[str, list[str]]:
    """Read nca_wm.train.MULTI_GAME_PRESETS without importing train.py."""
    train_path = _REPO_ROOT / "nca_wm" / "train.py"
    tree = ast.parse(train_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MULTI_GAME_PRESETS":
                    return ast.literal_eval(node.value)
    return {}


def _source_for_game(parser, game_name: str) -> str:
    backend = CppPuzzleScriptBackend()
    return backend.compile_game(parser, game_name)


def _materialize_game(name: str, code: str) -> Path:
    """Write code into custom_games and clear any stale simplified cache."""
    custom_dir = Path(CUSTOM_GAMES_DIR)
    custom_dir.mkdir(parents=True, exist_ok=True)
    path = custom_dir / f"{name}.txt"
    path.write_text(code.strip() + "\n", encoding="utf-8")

    simplified_dir = Path(SIMPLIFIED_GAMES_DIR)
    for stale in (
        simplified_dir / f"{name}.txt",
        simplified_dir / f"{name}_simplified.txt",
    ):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    return path


def _trim_levels(code: str, max_levels: int) -> str:
    """Best-effort deterministic patch: keep only the first N level blocks."""
    marker = re.search(r"(?im)^={5,}\s*$\n^\s*levels\s*$", code)
    if not marker:
        return code
    head = code[: marker.end()]
    rest = code[marker.end():]
    parts = re.split(r"(?m)^\s*\n", rest.strip())
    kept = []
    for part in parts:
        if part.strip():
            kept.append(part.rstrip())
        if len(kept) >= max_levels:
            break
    return head.rstrip() + "\n\n" + "\n\n".join(kept).rstrip() + "\n"


def _deterministic_patch(code: str, max_levels: int) -> str:
    patched = code.replace("randomDir", "random")
    patched = re.sub(r"(?im)^={5,}\s*$\n^\s*sounds\s*$.*?(?=^={5,}\s*$|\Z)",
                     "", patched, flags=re.DOTALL)
    return _trim_levels(patched, max_levels)


def _search_materialized_game(
    parser,
    game_name: str,
    *,
    search_algo: str,
    search_timeout_ms: int,
    search_n_steps: int,
    min_solution_len: int = 5,
) -> EvalFeedback:
    backend = CppPuzzleScriptBackend()
    try:
        backend.compile_game(parser, game_name)
    except Exception as e:
        return EvalFeedback(compile_ok=False, error=f"{type(e).__name__}: {e}"[:500])

    n_levels = int(backend.get_num_levels())
    if n_levels < 1:
        return EvalFeedback(
            compile_ok=False,
            error="Compilation produced no playable levels.",
            n_levels=n_levels,
        )

    levels: list[LevelFeedback] = []
    all_solvable = True
    max_iters = 0
    for li in range(n_levels):
        try:
            backend.cpp_engine.load_level(li)
            if search_algo == "astar":
                sr = backend.cpp_engine.solve_astar(search_n_steps, search_timeout_ms)
            else:
                sr = backend.cpp_engine.solve_bfs(search_n_steps, search_timeout_ms)
            solved = bool(sr.won)
            sol_len = len(sr.actions)
            err = ""
            if solved and sol_len < min_solution_len:
                solved = False
                err = f"Solution is trivially short ({sol_len} moves)."
            if not solved:
                all_solvable = False
            max_iters = max(max_iters, int(sr.iterations))
            levels.append(LevelFeedback(
                level_i=li,
                solved=solved,
                n_iters=int(sr.iterations),
                solution_len=sol_len,
                error=err,
            ))
        except Exception as e:
            all_solvable = False
            levels.append(LevelFeedback(
                level_i=li,
                solved=False,
                n_iters=0,
                solution_len=0,
                error=f"{type(e).__name__}: {e}"[:300],
            ))
    return EvalFeedback(
        compile_ok=True,
        n_levels=n_levels,
        all_solvable=all_solvable,
        max_search_iters=max_iters,
        levels=levels,
    )


def _format_solver_feedback(ev: EvalFeedback) -> str:
    if not ev.compile_ok:
        return ev.error
    lines = []
    for lvl in ev.levels:
        if lvl.solved:
            lines.append(
                f"Level {lvl.level_i}: SOLVED in {lvl.n_iters} iterations, "
                f"solution length {lvl.solution_len}."
            )
        elif lvl.error:
            lines.append(f"Level {lvl.level_i}: FAILED - {lvl.error}")
        else:
            lines.append(
                f"Level {lvl.level_i}: NOT SOLVABLE within {lvl.n_iters} iterations."
            )
    return "\n".join(lines)


def _eval_candidate(
    parser,
    cand: GameCandidate,
    *,
    max_levels: int,
    search_algo: str,
    search_timeout_ms: int,
    search_n_steps: int,
    encode_sprites: bool,
) -> GameCandidate:
    cand.code = _deterministic_patch(cand.code, max_levels=max_levels)
    if cand.materialized:
        _materialize_game(cand.name, cand.code)
    try:
        ev = _search_materialized_game(
            parser,
            cand.name,
            search_algo=search_algo,
            search_timeout_ms=search_timeout_ms,
            search_n_steps=search_n_steps,
        )
        cand.compile_ok = ev.compile_ok
        cand.all_solvable = ev.all_solvable
        cand.n_levels = ev.n_levels
        cand.max_search_iters = ev.max_search_iters
        if not ev.compile_ok:
            cand.error = ev.error[:500]
            cand.score = -1.0
            return cand

        backend = CppPuzzleScriptBackend()
        json_str = backend.compile_and_serialize(parser, cand.name)
        env0 = CppPuzzleScriptEnv(json_str, level_i=0, max_episode_steps=10)
        cand.n_objs = int(env0.observation_shape[0])
        cand.max_h = int(env0.observation_shape[1])
        cand.max_w = int(env0.observation_shape[2])
        for li in range(int(env0.num_levels)):
            env_li = CppPuzzleScriptEnv(json_str, level_i=li, max_episode_steps=10)
            c, h, w = env_li.observation_shape
            cand.n_objs = max(cand.n_objs, int(c))
            cand.max_h = max(cand.max_h, int(h))
            cand.max_w = max(cand.max_w, int(w))

        tree, canonical_ids = get_game_tree_from_js(parser, cand.name)
        cand.n_tokens = len(tokenize_game(
            tree, canonical_ids, encode_sprites=encode_sprites,
        ))
        shape_penalty = max(0, cand.n_objs - 24) + max(0, cand.max_h * cand.max_w - 225) / 25
        solvable_bonus = 1.0 if cand.all_solvable else 0.25
        hardness = math.log1p(max(0, cand.max_search_iters))
        mechanic_size = math.log1p(max(0, cand.n_tokens))
        cand.score = solvable_bonus * (hardness + 0.35 * mechanic_size) - 0.25 * shape_penalty
    except Exception as e:
        cand.compile_ok = False
        cand.score = -1.0
        cand.error = f"{type(e).__name__}: {e}"[:500]
    return cand


def _token_jaccard(a: GameCandidate, b: GameCandidate) -> float:
    # Cheap name/code shingles; avoids re-tokenizing while still rewarding variety.
    def shingles(s: str) -> set[str]:
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\[\]\|\->]+", s.lower())
        return set(" ".join(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    sa, sb = shingles(a.code), shingles(b.code)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _score_set(gs: GameSet) -> GameSet:
    valid = [g for g in gs.games if g.compile_ok]
    if not valid:
        gs.score = -1.0
        gs.metrics = {"n_valid": 0}
        return gs
    base = float(np.mean([g.score for g in valid]))
    solvable_frac = float(np.mean([g.all_solvable for g in valid]))
    token_lens = np.array([g.n_tokens for g in valid], dtype=np.float64)
    size_lens = np.array([g.n_objs * max(1, g.max_h) * max(1, g.max_w) for g in valid],
                         dtype=np.float64)
    diversity = 0.0
    pairs = 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            diversity += 1.0 - _token_jaccard(valid[i], valid[j])
            pairs += 1
    diversity = diversity / pairs if pairs else 0.0
    balance_penalty = float(token_lens.std() / max(1.0, token_lens.mean()))
    shape_penalty = float(size_lens.std() / max(1.0, size_lens.mean()))
    gs.score = base + 1.5 * diversity + solvable_frac - 0.2 * balance_penalty - 0.1 * shape_penalty
    gs.metrics = {
        "n_valid": len(valid),
        "solvable_frac": solvable_frac,
        "mean_game_score": base,
        "diversity": diversity,
        "token_balance_penalty": balance_penalty,
        "shape_balance_penalty": shape_penalty,
    }
    return gs


def _make_candidate_name(run_tag: str, gen: int, mode: str, code: str) -> str:
    return _safe_slug(f"nca_gc_{run_tag}_g{gen:03d}_{mode}_{_hash_text(code, 8)}")


def _llm_game(
    *,
    model: str,
    base_url: str | None,
    mode: str,
    parents: list[GameCandidate],
    max_levels: int,
    fewshot_n: int,
    temperature: float,
    enable_thinking: bool | None,
) -> str | None:
    system = build_system_prompt(sample_games(n=fewshot_n))
    if mode == "init":
        prompt = GENERATE_PROMPT
    elif mode == "targeted" and parents:
        prompt = TARGETED_EDIT_PROMPT.format(parent_code=parents[0].code, max_levels=max_levels)
    elif mode == "mutate" and parents:
        prompt = MUTATE_PROMPT.format(parent_code=parents[0].code)
    elif mode == "crossover" and len(parents) >= 2:
        prompt = CROSSOVER_PROMPT.format(parent_a=parents[0].code, parent_b=parents[1].code)
    else:
        prompt = GENERATE_PROMPT
    text = query_vllm(
        system, prompt, model,
        base_url=base_url, temperature=temperature,
        enable_thinking=enable_thinking, max_tokens=8192,
    )
    return extract_ps_code(text or "")


def _repair_with_llm(
    code: str,
    *,
    model: str,
    base_url: str | None,
    feedback: str,
    max_levels: int,
    temperature: float,
    enable_thinking: bool | None,
) -> str | None:
    system = build_system_prompt([])
    prompt = LEVEL_REPAIR_PROMPT.format(code=code, feedback=feedback, max_levels=max_levels)
    text = query_vllm(
        system, prompt, model,
        base_url=base_url, temperature=temperature,
        enable_thinking=enable_thinking, max_tokens=8192,
    )
    return extract_ps_code(text or "")


def _write_set_artifacts(base_dir: Path, gs: GameSet, train_extra_args: str) -> None:
    set_dir = base_dir / f"gen_{gs.generation:03d}" / gs.uid
    set_dir.mkdir(parents=True, exist_ok=True)
    names = [g.name for g in gs.games if g.compile_ok]
    (set_dir / "games.txt").write_text(",".join(names) + "\n", encoding="utf-8")
    (set_dir / "summary.json").write_text(json.dumps({
        "uid": gs.uid,
        "generation": gs.generation,
        "score": gs.score,
        "metrics": gs.metrics,
        "parent_set_uids": gs.parent_set_uids,
        "games": [asdict(g) for g in gs.games],
    }, indent=2), encoding="utf-8")
    cmd = (
        ".venv/bin/python -m nca_wm.train "
        f"--games \"{','.join(names)}\" "
        f"--save_dir \"{set_dir / 'wm_run'}\" "
        f"{train_extra_args.strip()}\n"
    )
    train_sh = set_dir / "train_command.sh"
    train_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + cmd, encoding="utf-8")
    train_sh.chmod(0o755)


def _seed_population(parser, args, run_tag: str) -> list[GameSet]:
    seed_names = _split_games(args.seed_games)
    seed_games: list[GameCandidate] = []
    for name in seed_names:
        code = _source_for_game(parser, name)
        cand = GameCandidate(
            name=name,
            code=code,
            uid=f"seed_{_safe_slug(name)}",
            mode="seed",
            generation=0,
        )
        seed_games.append(_eval_candidate(
            parser, cand,
            max_levels=args.max_levels,
            search_algo=args.search_algo,
            search_timeout_ms=args.search_timeout_ms,
            search_n_steps=args.search_n_steps,
            encode_sprites=args.encode_sprites,
        ))

    sets = []
    for i in range(args.pop_size):
        games = random.sample(seed_games, k=min(args.set_size, len(seed_games)))
        uid = f"set_{i:03d}_{_hash_text(','.join(g.uid for g in games), 6)}"
        sets.append(_score_set(GameSet(uid=uid, games=list(games), generation=0)))
    return sets


def _breed_set(parser, args, run_tag: str, parents: list[GameSet], gen: int, child_i: int) -> GameSet:
    parent = random.choice(parents)
    games = list(parent.games)
    replace_i = random.randrange(len(games))
    mode = random.choices(
        ["targeted", "mutate", "crossover", "init"],
        weights=[0.45, 0.25, 0.20, 0.10],
        k=1,
    )[0]
    if mode == "crossover" and len(games) >= 2:
        parent_games = random.sample(games, 2)
    elif mode in {"targeted", "mutate"}:
        parent_games = [games[replace_i]]
    else:
        parent_games = []

    code = _llm_game(
        model=args.model,
        base_url=args.vllm_base_url or None,
        mode=mode,
        parents=parent_games,
        max_levels=args.max_levels,
        fewshot_n=args.fewshot_n,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking or None,
    )
    if not code:
        # Keep the parent slot when generation fails, but score the set honestly.
        new_game = games[replace_i]
    else:
        name = _make_candidate_name(run_tag, gen, mode, code)
        new_game = GameCandidate(
            name=name,
            code=code,
            uid=name,
            parent_uids=[g.uid for g in parent_games],
            mode=mode,
            generation=gen,
            materialized=True,
        )
        new_game = _eval_candidate(
            parser, new_game,
            max_levels=args.max_levels,
            search_algo=args.search_algo,
            search_timeout_ms=args.search_timeout_ms,
            search_n_steps=args.search_n_steps,
            encode_sprites=args.encode_sprites,
        )
        if args.repair_levels and new_game.compile_ok and not new_game.all_solvable:
            ev = _search_materialized_game(
                parser,
                new_game.name,
                search_algo=args.search_algo,
                search_timeout_ms=args.search_timeout_ms,
                search_n_steps=args.search_n_steps,
            )
            repaired = _repair_with_llm(
                new_game.code,
                model=args.model,
                base_url=args.vllm_base_url or None,
                feedback=_format_solver_feedback(ev),
                max_levels=args.max_levels,
                temperature=max(0.2, args.temperature * 0.7),
                enable_thinking=args.enable_thinking or None,
            )
            if repaired:
                new_game.code = repaired
                new_game.name = _make_candidate_name(run_tag, gen, "repair", repaired)
                new_game.uid = new_game.name
                new_game.mode = f"{mode}+repair"
                new_game.materialized = True
                new_game = _eval_candidate(
                    parser, new_game,
                    max_levels=args.max_levels,
                    search_algo=args.search_algo,
                    search_timeout_ms=args.search_timeout_ms,
                    search_n_steps=args.search_n_steps,
                    encode_sprites=args.encode_sprites,
                )

    games[replace_i] = new_game
    uid = f"set_g{gen:03d}_{child_i:03d}_{_hash_text(','.join(g.uid for g in games), 6)}"
    return _score_set(GameSet(
        uid=uid,
        games=games,
        generation=gen,
        parent_set_uids=[parent.uid],
    ))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evolve curricula of PuzzleScript game sets with a local vLLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seed_games", default="scaling_4",
                    help="Comma-separated names or an nca_wm.train preset.")
    ap.add_argument("--set_size", type=int, default=4)
    ap.add_argument("--pop_size", type=int, default=4)
    ap.add_argument("--n_generations", type=int, default=3)
    ap.add_argument("--n_children_per_gen", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", default=None)

    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--vllm_base_url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--fewshot_n", type=int, default=2)

    ap.add_argument("--max_levels", type=int, default=5)
    ap.add_argument("--repair_levels", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--search_algo", choices=["bfs", "astar"], default="bfs")
    ap.add_argument("--search_timeout_ms", type=int, default=10_000)
    ap.add_argument("--search_n_steps", type=int, default=30_000)
    ap.add_argument("--encode_sprites", action="store_true")

    ap.add_argument("--train_extra_args", default=(
        "--conditional --architecture rule_attn --n_updates 5000 "
        "--max_transitions_per_game 50000"
    ))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    run_tag = f"{_safe_slug(args.model, 24)}_{int(time.time())}_{args.seed}"
    base_dir = Path(args.save_dir or f"nca_wm/logs/game_curriculum_{run_tag}")
    base_dir.mkdir(parents=True, exist_ok=True)

    parser = init_ps_lark_parser()
    print(f"[game-curriculum] save_dir={base_dir}")
    print(f"[game-curriculum] model={args.model} base_url={args.vllm_base_url}")

    pop = _seed_population(parser, args, run_tag)
    for gs in pop:
        _write_set_artifacts(base_dir, gs, args.train_extra_args)
    (base_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    history = []
    for gen in range(1, args.n_generations + 1):
        pop.sort(key=lambda s: s.score, reverse=True)
        parents = pop[:args.pop_size]
        print(f"\n[game-curriculum] gen {gen}: parents "
              f"{[(p.uid, round(p.score, 3)) for p in parents]}")
        children = []
        for i in range(args.n_children_per_gen):
            child = _breed_set(parser, args, run_tag, parents, gen, i)
            children.append(child)
            _write_set_artifacts(base_dir, child, args.train_extra_args)
            print(f"  child {i}: score={child.score:.3f} "
                  f"valid={child.metrics.get('n_valid', 0)}/{len(child.games)} "
                  f"games={[g.name for g in child.games]}")
        pop = sorted(parents + children, key=lambda s: s.score, reverse=True)[:args.pop_size]
        row = {
            "generation": gen,
            "population": [
                {"uid": s.uid, "score": s.score, "metrics": s.metrics,
                 "games": [g.name for g in s.games]}
                for s in pop
            ],
        }
        history.append(row)
        (base_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    best = sorted(pop, key=lambda s: s.score, reverse=True)[0]
    best_dir = base_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(base_dir / f"gen_{best.generation:03d}" / best.uid, best_dir)
    print(f"\n[game-curriculum] DONE best={best.uid} score={best.score:.3f}")
    print(f"[game-curriculum] best artifacts: {best_dir}")


if __name__ == "__main__":
    main()

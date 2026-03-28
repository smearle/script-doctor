#!/usr/bin/env python3
"""Agentic LLM-driven PuzzleScript game evolution loop.

Uses a vLLM server to generate, compile, debug, and evaluate novel PuzzleScript
games. Random samples from data/scraped_games_increpare are injected as few-shot
context. Compilation errors are fed back to the LLM for iterative repair. Tree
search (BFS) is run on each level (max 5 levels per game) to evaluate solvability.

Usage examples
--------------
# Local (assumes vLLM server already running on localhost:8000):
    python evolve_games_agentic.py

# Specify model and base URL:
    python evolve_games_agentic.py --model vllm-qwen3.5-9b --vllm_base_url http://localhost:8000/v1

# Full evolution with custom params:
    python evolve_games_agentic.py --pop_size 4 --n_gens 10 --max_repair_attempts 10

# Single-shot generation (no evolution, just generate one game):
    python evolve_games_agentic.py --mode single --n_games 5
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
INCREPARE_DIR = DATA_DIR / "scraped_games_increpare"
DOCS_PATH = SCRIPT_DIR / "script_doctor" / "all_documentation.txt"
LOGS_ROOT = SCRIPT_DIR / "evo_agentic_logs"

logger = logging.getLogger("evo_agentic")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Lazy imports — heavy deps loaded only when needed
# ---------------------------------------------------------------------------
_backend_cache: dict[str, object] = {}


def _get_nodejs_backend():
    """Return a (possibly cached) NodeJSPuzzleScriptBackend instance."""
    if "nodejs" not in _backend_cache:
        from backends.nodejs import NodeJSPuzzleScriptBackend
        _backend_cache["nodejs"] = NodeJSPuzzleScriptBackend()
    return _backend_cache["nodejs"]


def _fresh_nodejs_backend():
    """Return a fresh NodeJSPuzzleScriptBackend (no shared state)."""
    from backends.nodejs import NodeJSPuzzleScriptBackend
    return NodeJSPuzzleScriptBackend()


# ---------------------------------------------------------------------------
# PuzzleScript documentation (loaded once)
# ---------------------------------------------------------------------------
_DOCS_CACHE: str | None = None


def _load_docs() -> str:
    global _DOCS_CACHE
    if _DOCS_CACHE is None:
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            _DOCS_CACHE = f.read()
    return _DOCS_CACHE


# ---------------------------------------------------------------------------
# Increpare game sampling
# ---------------------------------------------------------------------------
_INCREPARE_FILES: list[Path] | None = None


def _list_increpare_games() -> list[Path]:
    global _INCREPARE_FILES
    if _INCREPARE_FILES is None:
        _INCREPARE_FILES = sorted(INCREPARE_DIR.glob("*.txt"))
    return _INCREPARE_FILES


def sample_increpare_games(n: int = 3, max_chars: int = 12_000) -> list[str]:
    """Return up to *n* random increpare game texts, limited by total char count."""
    files = _list_increpare_games()
    if not files:
        logger.warning("No increpare games found in %s", INCREPARE_DIR)
        return []
    sampled: list[str] = []
    total_chars = 0
    candidates = random.sample(files, min(len(files), n * 4))
    for p in candidates:
        if len(sampled) >= n:
            break
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > max_chars:
            continue  # skip very large games
        if total_chars + len(text) > max_chars * 2:
            break
        sampled.append(text)
        total_chars += len(text)
    return sampled


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "You are a creative and resourceful indie puzzle game designer, expert in "
    "the PuzzleScript game description language. "
    "Recall that comments in PuzzleScript are enclosed in parentheses. "
    "(E.g. this is a comment.)\n\n"
    "Here are the PuzzleScript docs:\n{docs}\n\n"
    "{fewshot_section}"
)

FEWSHOT_HEADER = (
    "Here are some example games from the PuzzleScript community for "
    "inspiration. Do NOT reproduce these games exactly — use them to "
    "understand the language and generate original designs:\n\n"
)

GENERATE_PROMPT = (
    "Design and output the complete code for an original PuzzleScript game. "
    "The game should have inventive mechanics and be unlike the examples above. "
    "First, briefly reason about what kind of game you want to create and what "
    "mechanics to use. Then, write the full PuzzleScript code.\n\n"
    "IMPORTANT CONSTRAINTS:\n"
    "- Do NOT include more than 5 levels.\n"
    "- Do NOT use the `randomDir` keyword.\n"
    "- Do NOT include sound effects.\n"
    "- Return your code inside a ```plaintext code block.\n"
)

MUTATE_PROMPT = (
    "Consider the following PuzzleScript game:\n"
    "```plaintext\n{parent_code}\n```\n\n"
    "Create a variation on this game. You may change the mechanics, add new "
    "objects, redesign levels, or alter the win conditions — but keep the "
    "spirit of the original while making it more complex and interesting.\n\n"
    "First, reason about what changes to make. Then write the full code.\n\n"
    "IMPORTANT CONSTRAINTS:\n"
    "- Do NOT include more than 5 levels.\n"
    "- Do NOT use the `randomDir` keyword.\n"
    "- Do NOT include sound effects.\n"
    "- Return your code inside a ```plaintext code block.\n"
)

CROSSOVER_PROMPT = (
    "Consider the following two PuzzleScript games:\n\n"
    "Game A:\n```plaintext\n{parent_a}\n```\n\n"
    "Game B:\n```plaintext\n{parent_b}\n```\n\n"
    "Create a new game that creatively combines elements from both games. "
    "Merge their mechanics, objects, or themes into something novel.\n\n"
    "First, reason about what to combine. Then write the full code.\n\n"
    "IMPORTANT CONSTRAINTS:\n"
    "- Do NOT include more than 5 levels.\n"
    "- Do NOT use the `randomDir` keyword.\n"
    "- Do NOT include sound effects.\n"
    "- Return your code inside a ```plaintext code block.\n"
)

COMPILE_REPAIR_PROMPT = (
    "The following PuzzleScript game code:\n"
    "```plaintext\n{code}\n```\n\n"
    "produced this compilation error:\n"
    "```\n{error}\n```\n\n"
    "Return a repaired version of the full code that fixes these errors. "
    "First, reason about what went wrong. Then write the corrected code.\n\n"
    "Return your code inside a ```plaintext code block.\n"
    "Do NOT include more than 5 levels.\n"
)

SOLVABILITY_REPAIR_PROMPT = (
    "The following PuzzleScript game code:\n"
    "```plaintext\n{code}\n```\n\n"
    "compiled successfully, but a solvability check returned:\n"
    "```\n{solver_feedback}\n```\n\n"
    "Return a repaired version of the full code that makes all levels "
    "solvable with non-trivial solutions (at least 5 moves). "
    "You may redesign levels or adjust rules as needed.\n\n"
    "First, reason about why levels are unsolvable. Then write the corrected code.\n\n"
    "Return your code inside a ```plaintext code block.\n"
    "Do NOT include more than 5 levels.\n"
)

LEVEL_WARNING_PROMPT = (
    "\n\nWARNING: Your game has {n_levels} levels, but only the first 5 will "
    "be evaluated. Consider reducing the number of levels to 5 or fewer.\n"
)


def build_system_prompt(fewshot_games: list[str]) -> str:
    """Build the system prompt with docs and few-shot examples."""
    docs = _load_docs()
    if fewshot_games:
        fewshot_section = FEWSHOT_HEADER
        for i, game_text in enumerate(fewshot_games, 1):
            fewshot_section += f"--- Example {i} ---\n```plaintext\n{game_text}\n```\n\n"
    else:
        fewshot_section = ""
    return SYSTEM_PROMPT_TEMPLATE.format(docs=docs, fewshot_section=fewshot_section)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_ps_code(text: str) -> str | None:
    """Extract PuzzleScript code from a ```plaintext code block."""
    # Try with closing ```
    m = re.search(r"```plaintext\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try without closing ``` (truncated response)
    m = re.search(r"```plaintext\n(.*)$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# LLM query (vLLM)
# ---------------------------------------------------------------------------

def query_vllm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    enable_thinking: bool | None = None,
) -> str | None:
    """Query the vLLM server. Delegates to puzzlescript_jax.utils._vllm_text_query."""
    from puzzlescript_jax.utils import _vllm_text_query, resolve_vllm_model
    resolved = resolve_vllm_model(model)
    return _vllm_text_query(
        system_prompt,
        user_prompt,
        model_name=resolved,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )


# ---------------------------------------------------------------------------
# Game compilation and evaluation
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
    success: bool
    error: str = ""
    game_text: str = ""
    n_levels: int = 0


def compile_game_text(code: str) -> CompileResult:
    """Compile a PuzzleScript game from raw text using the Node.js backend."""
    try:
        backend = _fresh_nodejs_backend()
        backend.engine.compile(["restart"], code)
        n_levels = int(backend.get_num_levels())
        return CompileResult(success=True, game_text=code, n_levels=n_levels)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        return CompileResult(success=False, error=error_msg)


@dataclass
class LevelSearchResult:
    level_i: int
    solved: bool
    n_iters: int
    solution_len: int
    time_s: float
    actions: tuple[int, ...] = ()
    error: str = ""


@dataclass
class EvalResult:
    compile_result: CompileResult
    level_results: list[LevelSearchResult] = field(default_factory=list)
    all_solvable: bool = False
    max_search_iters: int = 0
    excess_levels_warning: str = ""

    @property
    def fitness(self) -> float:
        """Fitness = max search iterations across solvable levels. Higher = harder puzzles."""
        if not self.compile_result.success:
            return -1.0
        if not self.all_solvable:
            return 0.0
        return float(self.max_search_iters)


MAX_LEVELS_TO_SEARCH = 5
BFS_TIMEOUT_ITERS = 2**16  # 65536 iterations


def evaluate_game(code: str, search_algo: str = "bfs",
                  search_timeout_ms: int = 30_000,
                  search_n_steps: int = BFS_TIMEOUT_ITERS,
                  gif_dir: Path | str | None = None) -> EvalResult:
    """Compile and evaluate a PuzzleScript game: compile, then search each level.

    If *gif_dir* is provided, solution GIFs are automatically rendered for every
    level where the search finds a solution.
    """
    cr = compile_game_text(code)
    result = EvalResult(compile_result=cr)

    if not cr.success:
        return result

    n_levels = cr.n_levels
    if n_levels > MAX_LEVELS_TO_SEARCH:
        result.excess_levels_warning = (
            f"Game has {n_levels} levels. Only the first {MAX_LEVELS_TO_SEARCH} "
            f"will be evaluated by tree search. Consider reducing to {MAX_LEVELS_TO_SEARCH} or fewer."
        )
        logger.warning(result.excess_levels_warning)

    if gif_dir is not None:
        gif_dir = Path(gif_dir)
        gif_dir.mkdir(parents=True, exist_ok=True)

    levels_to_search = min(n_levels, MAX_LEVELS_TO_SEARCH)
    all_solvable = True

    for level_i in range(levels_to_search):
        try:
            backend = _fresh_nodejs_backend()
            sr = backend.run_search(
                search_algo,
                game_text=code,
                level_i=level_i,
                n_steps=search_n_steps,
                timeout_ms=search_timeout_ms,
            )
            lsr = LevelSearchResult(
                level_i=level_i,
                solved=sr.solved,
                n_iters=sr.iterations,
                solution_len=len(sr.actions),
                time_s=sr.time,
                actions=sr.actions,
            )
            if not sr.solved:
                all_solvable = False
            elif sr.solved and len(sr.actions) < 5:
                lsr.error = "Solution is trivially short (< 5 moves)."
                all_solvable = False

            # Render GIF immediately when search returns a solution
            if sr.solved and sr.actions and gif_dir is not None:
                try:
                    gif_backend = _fresh_nodejs_backend()
                    gif_path = str(gif_dir / f"search_level_{level_i}.gif")
                    gif_backend.render_gif(
                        game_text=code,
                        level_i=level_i,
                        actions=list(sr.actions),
                        gif_path=gif_path,
                        frame_duration_s=0.1,
                        scale=5,
                    )
                    logger.info("Rendered search GIF: %s", gif_path)
                except Exception as gif_err:
                    logger.warning(
                        "Failed to render GIF for level %d: %s", level_i, gif_err
                    )

            result.max_search_iters = max(result.max_search_iters, sr.iterations)
        except Exception as e:
            lsr = LevelSearchResult(
                level_i=level_i,
                solved=False,
                n_iters=0,
                solution_len=0,
                time_s=0.0,
                error=f"{type(e).__name__}: {e}",
            )
            all_solvable = False

        result.level_results.append(lsr)

    result.all_solvable = all_solvable
    return result


def build_solver_feedback(eval_result: EvalResult) -> str:
    """Build a human-readable solver feedback string from eval results."""
    lines = []
    for lsr in eval_result.level_results:
        if lsr.solved and not lsr.error:
            lines.append(
                f"Level {lsr.level_i}: SOLVED in {lsr.n_iters} search iterations, "
                f"solution length {lsr.solution_len} moves."
            )
        elif lsr.solved and lsr.error:
            lines.append(
                f"Level {lsr.level_i}: Solved but {lsr.error}"
            )
        elif lsr.error:
            lines.append(f"Level {lsr.level_i}: FAILED — {lsr.error}")
        else:
            lines.append(
                f"Level {lsr.level_i}: NOT SOLVABLE (searched {lsr.n_iters} iterations)."
            )
    if eval_result.excess_levels_warning:
        lines.append(f"\n{eval_result.excess_levels_warning}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Game individual (for evolution)
# ---------------------------------------------------------------------------

@dataclass
class GameIndividual:
    code: str
    fitness: float = -1.0
    eval_result: EvalResult | None = None
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)
    gen_mode: str = "init"  # init, mutate, crossover
    attempt_history: list[str] = field(default_factory=list)  # track repair attempts
    uid: int = 0


_UID_COUNTER = 0


def _next_uid() -> int:
    global _UID_COUNTER
    _UID_COUNTER += 1
    return _UID_COUNTER


# ---------------------------------------------------------------------------
# Core generation + repair loop
# ---------------------------------------------------------------------------

def generate_and_evaluate(
    model: str,
    base_url: str | None,
    gen_mode: str,
    parents: list[GameIndividual],
    save_dir: Path,
    max_repair_attempts: int = 15,
    fewshot_n: int = 3,
    temperature: float = 0.7,
    enable_thinking: bool | None = None,
) -> GameIndividual:
    """Generate a game via LLM, compile, search, repair iteratively.

    Returns a GameIndividual with the best code found.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    uid = _next_uid()

    # Sample fresh few-shot examples for this generation
    fewshot_games = sample_increpare_games(n=fewshot_n)
    system_prompt = build_system_prompt(fewshot_games)

    code: str | None = None
    compile_success = False
    solvable = False
    eval_result: EvalResult | None = None
    compile_error = ""
    solver_feedback = ""
    attempt_history: list[str] = []

    for attempt in range(max_repair_attempts):
        logger.info("Game uid=%d, attempt %d/%d (mode=%s)",
                     uid, attempt + 1, max_repair_attempts, gen_mode)

        # Build the user prompt
        if attempt == 0:
            if gen_mode == "init":
                user_prompt = GENERATE_PROMPT
            elif gen_mode == "mutate" and parents:
                user_prompt = MUTATE_PROMPT.format(parent_code=parents[0].code)
            elif gen_mode == "crossover" and len(parents) >= 2:
                user_prompt = CROSSOVER_PROMPT.format(
                    parent_a=parents[0].code,
                    parent_b=parents[1].code,
                )
            else:
                user_prompt = GENERATE_PROMPT
        elif not compile_success:
            user_prompt = COMPILE_REPAIR_PROMPT.format(
                code=code, error=compile_error
            )
        else:
            user_prompt = SOLVABILITY_REPAIR_PROMPT.format(
                code=code, solver_feedback=solver_feedback
            )

        # Save prompt (full system + user prompt for reproducibility)
        prompt_path = save_dir / f"{attempt:02d}_prompt.txt"
        prompt_path.write_text(
            f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{user_prompt}",
            encoding="utf-8",
        )

        # Query LLM
        t0 = time.time()
        response_text = query_vllm(
            system_prompt, user_prompt, model,
            base_url=base_url, temperature=temperature,
            enable_thinking=enable_thinking,
        )
        query_time = time.time() - t0

        if response_text is None:
            logger.error("LLM returned None on attempt %d", attempt + 1)
            attempt_history.append(f"attempt_{attempt}: LLM returned None")
            continue

        # Save response
        response_path = save_dir / f"{attempt:02d}_response.txt"
        response_path.write_text(response_text, encoding="utf-8")

        # Extract code
        extracted = extract_ps_code(response_text)
        if extracted is None:
            logger.warning("No code block found in LLM response (attempt %d)", attempt + 1)
            attempt_history.append(f"attempt_{attempt}: no code block in response")
            compile_error = "The LLM response did not contain a ```plaintext code block."
            compile_success = False
            continue

        # Skip if uses randomDir
        if "randomDir" in extracted:
            logger.warning("Game uses randomDir, skipping (attempt %d)", attempt + 1)
            attempt_history.append(f"attempt_{attempt}: uses randomDir")
            compile_error = "Game uses randomDir which is not supported. Remove all uses of randomDir."
            compile_success = False
            code = extracted
            continue

        code = extracted

        # Save extracted code
        code_path = save_dir / f"{attempt:02d}_code.txt"
        code_path.write_text(code, encoding="utf-8")

        # Evaluate (compile + search), rendering GIFs for solved levels
        t0 = time.time()
        eval_result = evaluate_game(code, gif_dir=save_dir / f"{attempt:02d}_gifs")
        eval_time = time.time() - t0

        # Save eval result
        eval_summary = {
            "attempt": attempt,
            "compile_success": eval_result.compile_result.success,
            "compile_error": eval_result.compile_result.error,
            "n_levels": eval_result.compile_result.n_levels,
            "all_solvable": eval_result.all_solvable,
            "max_search_iters": eval_result.max_search_iters,
            "fitness": eval_result.fitness,
            "query_time_s": query_time,
            "eval_time_s": eval_time,
            "levels": [
                {
                    "level_i": lsr.level_i,
                    "solved": lsr.solved,
                    "n_iters": lsr.n_iters,
                    "solution_len": lsr.solution_len,
                    "time_s": lsr.time_s,
                    "error": lsr.error,
                }
                for lsr in eval_result.level_results
            ],
        }
        eval_path = save_dir / f"{attempt:02d}_eval.json"
        eval_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

        compile_success = eval_result.compile_result.success
        if not compile_success:
            compile_error = eval_result.compile_result.error
            attempt_history.append(
                f"attempt_{attempt}: compile error: {compile_error[:200]}"
            )
            logger.info("Compilation failed: %s", compile_error[:200])
            continue

        solver_feedback = build_solver_feedback(eval_result)
        solvable = eval_result.all_solvable

        attempt_history.append(
            f"attempt_{attempt}: compiled, solvable={solvable}, "
            f"fitness={eval_result.fitness:.0f}"
        )

        if solvable:
            logger.info(
                "Game uid=%d SOLVED on attempt %d! fitness=%.0f",
                uid, attempt + 1, eval_result.fitness,
            )
            break
        else:
            logger.info(
                "Game compiled but not all levels solvable (attempt %d). "
                "Feeding solver feedback back to LLM.",
                attempt + 1,
            )

    # Build the individual
    ind = GameIndividual(
        code=code or "",
        fitness=eval_result.fitness if eval_result else -1.0,
        eval_result=eval_result,
        generation=0,
        parent_ids=[p.uid for p in parents],
        gen_mode=gen_mode,
        attempt_history=attempt_history,
        uid=uid,
    )

    # Save final summary
    summary = {
        "uid": uid,
        "gen_mode": gen_mode,
        "parent_ids": ind.parent_ids,
        "fitness": ind.fitness,
        "n_attempts": len(attempt_history),
        "compile_success": compile_success,
        "all_solvable": solvable,
        "attempt_history": attempt_history,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ind


# ---------------------------------------------------------------------------
# Evolution modes
# ---------------------------------------------------------------------------

def run_single_mode(args) -> None:
    """Generate N independent games (no evolution)."""
    base_dir = LOGS_ROOT / f"single_{args.model}_{int(time.time())}"
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Single mode: generating %d games, saving to %s", args.n_games, base_dir)

    results = []
    for i in range(args.n_games):
        save_dir = base_dir / f"game_{i:03d}"
        ind = generate_and_evaluate(
            model=args.model,
            base_url=args.vllm_base_url or None,
            gen_mode="init",
            parents=[],
            save_dir=save_dir,
            max_repair_attempts=args.max_repair_attempts,
            fewshot_n=args.fewshot_n,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking or None,
        )
        results.append({
            "uid": ind.uid,
            "fitness": ind.fitness,
            "solvable": ind.eval_result.all_solvable if ind.eval_result else False,
            "n_levels": ind.eval_result.compile_result.n_levels if ind.eval_result else 0,
        })
        logger.info(
            "Game %d/%d: uid=%d fitness=%.0f solvable=%s",
            i + 1, args.n_games, ind.uid, ind.fitness,
            ind.eval_result.all_solvable if ind.eval_result else False,
        )

    (base_dir / "all_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    logger.info("All results saved to %s/all_results.json", base_dir)


def run_evolution_mode(args) -> None:
    """Run a full evolutionary loop with mutation and crossover."""
    base_dir = LOGS_ROOT / f"evo_{args.model}_pop{args.pop_size}_gen{args.n_gens}_{int(time.time())}"
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Evolution mode: pop_size=%d, n_gens=%d, saving to %s",
        args.pop_size, args.n_gens, base_dir,
    )

    pop: list[GameIndividual] = []

    # Generation 0: initialize population (generate 2x pop_size, keep best)
    init_size = args.pop_size * 2
    logger.info("=== Generation 0: initializing %d candidates ===", init_size)

    for i in range(init_size):
        save_dir = base_dir / f"gen_000" / f"game_{i:03d}"
        ind = generate_and_evaluate(
            model=args.model,
            base_url=args.vllm_base_url or None,
            gen_mode="init",
            parents=[],
            save_dir=save_dir,
            max_repair_attempts=args.max_repair_attempts,
            fewshot_n=args.fewshot_n,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking or None,
        )
        ind.generation = 0
        pop.append(ind)
        logger.info(
            "Gen 0, game %d/%d: uid=%d fitness=%.0f",
            i + 1, init_size, ind.uid, ind.fitness,
        )

    # Log generation 0 results
    _log_generation(base_dir, 0, pop)

    # Evolution loop
    for gen in range(1, args.n_gens + 1):
        logger.info("=== Generation %d/%d ===", gen, args.n_gens)

        # Select top individuals as parents
        pop.sort(key=lambda x: x.fitness, reverse=True)
        ancestors = pop[:args.pop_size]

        logger.info(
            "Top %d ancestors: %s",
            args.pop_size,
            [(a.uid, f"{a.fitness:.0f}") for a in ancestors],
        )

        new_pop: list[GameIndividual] = []

        for i in range(args.pop_size):
            # Choose mutation or crossover
            if random.random() < 0.5 and len(ancestors) >= 2:
                gen_mode = "crossover"
                p1 = random.choice(ancestors)
                remaining = [a for a in ancestors if a is not p1]
                p2 = random.choice(remaining) if remaining else p1
                parents = [p1, p2]
            else:
                gen_mode = "mutate"
                parents = [random.choice(ancestors)]

            save_dir = base_dir / f"gen_{gen:03d}" / f"game_{i:03d}"
            ind = generate_and_evaluate(
                model=args.model,
                base_url=args.vllm_base_url or None,
                gen_mode=gen_mode,
                parents=parents,
                save_dir=save_dir,
                max_repair_attempts=args.max_repair_attempts,
                fewshot_n=args.fewshot_n,
                temperature=args.temperature,
                enable_thinking=args.enable_thinking or None,
            )
            ind.generation = gen
            new_pop.append(ind)
            logger.info(
                "Gen %d, game %d/%d: uid=%d mode=%s fitness=%.0f",
                gen, i + 1, args.pop_size, ind.uid, gen_mode, ind.fitness,
            )

        pop.extend(new_pop)
        _log_generation(base_dir, gen, pop)

    # Final summary
    pop.sort(key=lambda x: x.fitness, reverse=True)
    logger.info("=== Evolution complete ===")
    logger.info("Top 5 individuals:")
    for ind in pop[:5]:
        logger.info(
            "  uid=%d gen=%d mode=%s fitness=%.0f solvable=%s",
            ind.uid, ind.generation, ind.gen_mode, ind.fitness,
            ind.eval_result.all_solvable if ind.eval_result else False,
        )

    # Save best game
    if pop and pop[0].code:
        best_path = base_dir / "best_game.txt"
        best_path.write_text(pop[0].code, encoding="utf-8")
        logger.info("Best game saved to %s", best_path)


def _log_generation(base_dir: Path, gen: int, pop: list[GameIndividual]) -> None:
    """Log generation summary to JSON."""
    summary = []
    for ind in pop:
        summary.append({
            "uid": ind.uid,
            "generation": ind.generation,
            "gen_mode": ind.gen_mode,
            "fitness": ind.fitness,
            "solvable": ind.eval_result.all_solvable if ind.eval_result else False,
            "n_levels": (
                ind.eval_result.compile_result.n_levels if ind.eval_result else 0
            ),
            "parent_ids": ind.parent_ids,
        })
    gen_log_path = base_dir / f"gen_{gen:03d}_summary.json"
    gen_log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Agentic LLM-driven PuzzleScript game evolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["single", "evolve"], default="evolve",
        help="'single' generates N independent games; 'evolve' runs evolutionary loop",
    )

    # Model
    parser.add_argument("--model", type=str, default="vllm-qwen3.5-9b",
                        help="vLLM model alias")
    parser.add_argument("--vllm_base_url", type=str, default="",
                        help="vLLM server URL (default: VLLM_BASE_URL env or localhost:8000)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Enable Qwen3 thinking mode")

    # Evolution params
    parser.add_argument("--pop_size", type=int, default=3,
                        help="Population size per generation")
    parser.add_argument("--n_gens", type=int, default=10,
                        help="Number of generations")

    # Single-mode params
    parser.add_argument("--n_games", type=int, default=5,
                        help="Number of games to generate in single mode")

    # Shared params
    parser.add_argument("--max_repair_attempts", type=int, default=15,
                        help="Max LLM attempts per game (compile + solvability repair)")
    parser.add_argument("--fewshot_n", type=int, default=3,
                        help="Number of increpare games to sample as few-shot examples")
    parser.add_argument("--search_timeout_ms", type=int, default=30_000,
                        help="Tree search timeout per level in ms")

    args = parser.parse_args()

    if args.vllm_base_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_base_url

    if args.mode == "single":
        run_single_mode(args)
    else:
        run_evolution_mode(args)


if __name__ == "__main__":
    main()

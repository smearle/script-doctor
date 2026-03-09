"""
Repeatedly mutate a PuzzleScript game file using a trained diff language model,
solve each mutation with A*, and render solutions as GIFs.

Usage:
    # Using vLLM for fast inference (recommended):
    python mutate_and_solve.py --model_dir models/diff_lm/final --n_mutations 10

    # Using HuggingFace transformers directly (no vLLM needed):
    python mutate_and_solve.py --model_dir models/diff_lm/final --n_mutations 10 --no_vllm

    # Specify a game file:
    python mutate_and_solve.py --game_file puzzlescript_data/final_game_versions/sokoban.txt ...
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import tempfile
import traceback
from pathlib import Path

import imageio
import numpy as np

# Lazy imports for backends
_backend = None
_tokenizer = None
_model = None
_vllm_model = None

BOS_SOURCE = "<|source|>"
BOS_DIFF = "<|startdiff|>"
EOS_DIFF = "<|enddiff|>"

GAME_FILES_DIR = os.path.join("puzzlescript_data", "final_game_versions")
CLEAN_DATA_DIR = os.path.join("puzzlescript_data", "clean_data")
OUTPUT_DIR = os.path.join("data", "diff_mutations")


def get_backend():
    """Lazy-init NodeJS backend."""
    global _backend
    if _backend is None:
        from puzzlejax.backends import NodeJSPuzzleScriptBackend
        _backend = NodeJSPuzzleScriptBackend()
    return _backend


def load_model_vllm(model_dir: str):
    """Load model via vLLM for fast batched inference."""
    global _vllm_model
    if _vllm_model is None:
        from vllm import LLM, SamplingParams  # noqa: F811
        _vllm_model = LLM(model=model_dir, dtype="float16", max_model_len=2048)
    return _vllm_model


def load_model_hf(model_dir: str):
    """Load model via HuggingFace transformers."""
    global _tokenizer, _model
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForCausalLM.from_pretrained(model_dir)
        if torch.cuda.is_available():
            _model = _model.cuda().half()
        _model.eval()
    return _tokenizer, _model


def pick_random_game(game_file: str = None) -> tuple[str, str]:
    """Pick a random PuzzleScript game file and return (path, content)."""
    if game_file:
        with open(game_file, "r", errors="replace") as f:
            return game_file, f.read()

    game_files = glob.glob(os.path.join(GAME_FILES_DIR, "*.txt"))
    if not game_files:
        raise FileNotFoundError(f"No game files found in {GAME_FILES_DIR}")

    path = random.choice(game_files)
    with open(path, "r", errors="replace") as f:
        return path, f.read()


def build_diff_prompt(game_text: str) -> str:
    """Build a prompt matching the training format: <|source|> game <|startdiff|>."""
    return f"{BOS_SOURCE}\n{game_text.strip()}\n{BOS_DIFF}\n"


def generate_diff_vllm(model_dir: str, prompt: str, temperature: float = 0.8,
                       max_tokens: int = 512) -> str:
    """Generate a diff completion using vLLM."""
    from vllm import SamplingParams
    llm = load_model_vllm(model_dir)
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=[EOS_DIFF],
        top_p=0.95,
    )
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text


def generate_diff_hf(model_dir: str, prompt: str, temperature: float = 0.8,
                     max_tokens: int = 512) -> str:
    """Generate a diff completion using HuggingFace transformers."""
    import torch
    tokenizer, model = load_model_hf(model_dir)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    eos_diff_id = tokenizer.convert_tokens_to_ids(EOS_DIFF)
    eos_ids = [tokenizer.eos_token_id]
    if eos_diff_id != tokenizer.unk_token_id:
        eos_ids.append(eos_diff_id)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=False).split(EOS_DIFF)[0]


def apply_unified_diff(original_text: str, diff_text: str) -> str | None:
    """
    Apply a unified diff to original text using the `patch` command.
    Returns the patched text or None if patching fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Recreate the directory structure expected by the diff headers
        game_dir = os.path.join(tmpdir, "clean_data")
        os.makedirs(game_dir, exist_ok=True)
        orig_path = os.path.join(game_dir, "game.txt")

        with open(orig_path, "w") as f:
            f.write(original_text)

        # Normalize diff headers to match file path
        diff_lines = diff_text.strip().splitlines(keepends=True)
        normalized = []
        for line in diff_lines:
            if line.startswith("--- "):
                normalized.append("--- clean_data/game.txt\n")
            elif line.startswith("+++ "):
                normalized.append("+++ clean_data/game.txt\n")
            else:
                normalized.append(line if line.endswith("\n") else line + "\n")
        diff_text = "".join(normalized)

        diff_path = os.path.join(tmpdir, "mutation.diff")
        with open(diff_path, "w") as f:
            f.write(diff_text)

        result = subprocess.run(
            ["patch", "--no-backup-if-mismatch", "-p0", "-l", "-f", "--fuzz=3",
             "-i", diff_path],
            capture_output=True, text=True, cwd=tmpdir,
            timeout=10,
        )

        # Read result regardless of return code (partial patches are ok)
        try:
            with open(orig_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return None


def apply_diff_manual(original_text: str, diff_text: str) -> str | None:
    """
    Manually apply a unified diff by parsing +/- lines.
    More forgiving than `patch` for imperfect model outputs.
    """
    orig_lines = original_text.splitlines(keepends=True)
    result_lines = list(orig_lines)

    # Extract hunks from the diff
    hunk_pattern = re.compile(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@")
    diff_lines = diff_text.splitlines(keepends=True)

    i = 0
    offset = 0  # Track line number shifts from insertions/deletions

    while i < len(diff_lines):
        match = hunk_pattern.match(diff_lines[i])
        if match:
            orig_start = int(match.group(1)) - 1  # 0-indexed
            i += 1
            pos = orig_start + offset
            while i < len(diff_lines):
                line = diff_lines[i]
                if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
                    break
                if line.startswith("-"):
                    # Remove line
                    content = line[1:]
                    if pos < len(result_lines):
                        result_lines.pop(pos)
                        offset -= 1
                    i += 1
                elif line.startswith("+"):
                    # Add line
                    content = line[1:]
                    result_lines.insert(pos, content)
                    pos += 1
                    offset += 1
                    i += 1
                elif line.startswith(" ") or line.strip() == "":
                    # Context line
                    pos += 1
                    i += 1
                else:
                    i += 1
        else:
            i += 1

    if not result_lines:
        return None
    return "".join(result_lines)


def try_compile_game(game_text: str) -> bool:
    """Try to compile a game with the NodeJS engine. Returns True if successful."""
    backend = get_backend()
    try:
        backend.engine.compile(["restart"], game_text)
        n_levels = backend.engine.getNumLevels()
        return n_levels > 0
    except Exception:
        return False


def solve_game(game_text: str, level_i: int = 0, algo: str = "astar",
               n_steps: int = 5000, timeout_ms: int = 10000) -> dict | None:
    """Try to solve a game level. Returns solution dict or None."""
    backend = get_backend()
    try:
        result = backend.run_search(
            algo,
            game_text=game_text,
            level_i=level_i,
            n_steps=n_steps,
            timeout_ms=timeout_ms,
        )
        if result.solved:
            return {
                "solved": True,
                "actions": list(result.actions),
                "time": result.time,
                "iterations": result.iterations,
            }
        return {"solved": False, "time": result.time, "iterations": result.iterations}
    except Exception as e:
        return {"solved": False, "error": str(e)}


def render_solution_gif(game_text: str, actions: list[int], gif_path: str,
                        level_i: int = 0, frame_duration_s: float = 0.3) -> str | None:
    """Render a solution as a GIF."""
    backend = get_backend()
    try:
        return backend.render_gif(
            game_text=game_text,
            level_i=level_i,
            actions=actions,
            gif_path=gif_path,
            frame_duration_s=frame_duration_s,
        )
    except Exception as e:
        print(f"  Render error: {e}")
        return None


def run_mutation_loop(
    model_dir: str,
    game_file: str = None,
    n_mutations: int = 10,
    use_vllm: bool = True,
    algo: str = "astar",
    n_steps: int = 5000,
    timeout_ms: int = 10000,
    temperature: float = 0.8,
    max_tokens: int = 512,
    output_dir: str = OUTPUT_DIR,
    iterative: bool = False,
):
    """Main loop: mutate game, solve, render."""
    os.makedirs(output_dir, exist_ok=True)
    generate_fn = generate_diff_vllm if use_vllm else generate_diff_hf

    game_path, game_text = pick_random_game(game_file)
    game_name = Path(game_path).stem
    print(f"Base game: {game_name} ({len(game_text)} chars)")

    run_dir = os.path.join(output_dir, game_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save original game
    with open(os.path.join(run_dir, "original.txt"), "w") as f:
        f.write(game_text)

    current_text = game_text
    results = []

    for i in range(n_mutations):
        print(f"\n--- Mutation {i+1}/{n_mutations} ---")

        # Generate diff
        prompt = build_diff_prompt(current_text)
        print(f"  Generating diff ({len(prompt)} char prompt)...")
        try:
            diff_output = generate_fn(model_dir, prompt, temperature=temperature,
                                      max_tokens=max_tokens)
        except Exception as e:
            print(f"  Generation error: {e}")
            results.append({"mutation": i, "error": f"generation: {e}"})
            continue

        print(f"  Generated {len(diff_output)} chars of diff")

        # The model outputs hunk lines directly (@@, +, -, context)
        # Apply as a unified diff
        mutated = apply_diff_manual(current_text, diff_output)
        if mutated is None or mutated.strip() == current_text.strip():
            # Try with patch command as fallback (needs ---/+++ headers)
            full_diff = f"--- a/game.txt\n+++ b/game.txt\n{diff_output}"
            mutated = apply_unified_diff(current_text, full_diff)
        if mutated is None or len(mutated.strip()) < 10:
            print("  Failed to apply diff")
            results.append({"mutation": i, "error": "diff_apply_failed"})
            continue

        # Save mutated game
        mut_path = os.path.join(run_dir, f"mutation_{i:03d}.txt")
        with open(mut_path, "w") as f:
            f.write(mutated)

        # Save the generated diff
        diff_save_path = os.path.join(run_dir, f"mutation_{i:03d}.diff")
        with open(diff_save_path, "w") as f:
            f.write(diff_output)

        # Try to compile
        if not try_compile_game(mutated):
            print("  Failed to compile mutated game")
            results.append({
                "mutation": i, "compiled": False,
                "game_file": f"mutation_{i:03d}.txt",
            })
            continue

        print("  Compiled successfully!")

        # Solve
        sol = solve_game(mutated, level_i=0, algo=algo, n_steps=n_steps,
                         timeout_ms=timeout_ms)
        result_entry = {
            "mutation": i,
            "compiled": True,
            "game_file": f"mutation_{i:03d}.txt",
            "diff_file": f"mutation_{i:03d}.diff",
        }

        if sol and sol.get("solved"):
            print(f"  Solved! {len(sol['actions'])} actions in {sol['time']:.2f}s")

            # Render GIF
            gif_path = os.path.join(run_dir, f"mutation_{i:03d}_sol.gif")
            rendered = render_solution_gif(mutated, sol["actions"], gif_path)
            if rendered:
                print(f"  Rendered: {gif_path}")
            result_entry.update({
                "solved": True,
                "actions": sol["actions"],
                "time": sol["time"],
                "iterations": sol["iterations"],
                "gif": f"mutation_{i:03d}_sol.gif" if rendered else None,
            })
        else:
            reason = sol.get("error", "unsolved") if sol else "search_error"
            print(f"  Not solved: {reason}")
            result_entry.update({"solved": False, "reason": reason})

        results.append(result_entry)

        # For iterative mode, use the mutated game as basis for next mutation
        if iterative:
            current_text = mutated

    # Save results summary
    summary_path = os.path.join(run_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    compiled = sum(1 for r in results if r.get("compiled"))
    solved = sum(1 for r in results if r.get("solved"))
    print(f"\n=== Summary ===")
    print(f"Mutations attempted: {n_mutations}")
    print(f"Compiled successfully: {compiled}/{n_mutations}")
    print(f"Solved: {solved}/{n_mutations}")
    print(f"Results saved to: {run_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Mutate PuzzleScript games with a diff LM")
    parser.add_argument("--model_dir", type=str, default=os.path.join("models", "diff_lm", "final"),
                        help="Path to trained model")
    parser.add_argument("--game_file", type=str, default=None,
                        help="Specific game file to mutate (default: random)")
    parser.add_argument("--n_mutations", type=int, default=10,
                        help="Number of mutations to generate")
    parser.add_argument("--no_vllm", action="store_true",
                        help="Use HuggingFace transformers instead of vLLM")
    parser.add_argument("--algo", type=str, default="astar",
                        choices=["bfs", "astar", "gbfs"],
                        help="Search algorithm for solving")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Max search steps")
    parser.add_argument("--timeout_ms", type=int, default=10000,
                        help="Search timeout in milliseconds")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature for diff generation")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens to generate per diff")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--iterative", action="store_true",
                        help="Chain mutations: each mutation builds on the previous one")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    run_mutation_loop(
        model_dir=args.model_dir,
        game_file=args.game_file,
        n_mutations=args.n_mutations,
        use_vllm=not args.no_vllm,
        algo=args.algo,
        n_steps=args.n_steps,
        timeout_ms=args.timeout_ms,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        iterative=args.iterative,
    )


if __name__ == "__main__":
    main()

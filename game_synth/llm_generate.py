"""Live LLM-in-the-loop game generation via HF transformers (vLLM unavailable here).

A lightweight ELM/Picbreeder-style loop: few-shot-prompt a local Qwen3 with valid
example games to write a NEW complete PuzzleScript game, extract the code,
compile + BFS-search-validate it ("play" it), and add survivors to the seed pool
so later prompts evolve from accepted games. Saves valid fresh games + manifest.

    .venv/bin/python -u -m game_synth.llm_generate --target 30 --model Qwen/Qwen3-4B
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

import re as _re

from nca_wm import game_curriculum as gc
from puzzlescript_jax.utils import init_ps_lark_parser


def extract_ps_code(text: str) -> str | None:
    """Pull the PuzzleScript game out of an LLM reply (any fenced block, or bare)."""
    blocks = _re.findall(r"```(?:puzzlescript|plaintext|text)?\s*\n(.*?)```", text,
                         _re.DOTALL | _re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    # No fence: if it looks like a game (has OBJECTS + RULES), take from the title/OBJECTS on.
    up = text.upper()
    if "OBJECTS" in up and "RULES" in up:
        i = up.find("TITLE")
        return text[i if 0 <= i < up.find("OBJECTS") else up.find("OBJECTS"):].strip()
    return None

SYS = ("You are an expert PuzzleScript game designer. PuzzleScript games have "
       "sections: OBJECTS, LEGEND, SOUNDS, COLLISIONLAYERS, RULES, WINCONDITIONS, "
       "LEVELS. Rules look like `[ > Player | Crate ] -> [ > Player | > Crate ]`. "
       "You write small, novel, fully valid, non-trivial games.")

PROMPT = ("Here are example PuzzleScript games:\n\n{examples}\n\n"
          "Now design a NEW, different, creative PuzzleScript game with its own "
          "objects, mechanics, and at least one solvable LEVELS section. It must "
          "compile and be non-trivially playable. Output ONLY the complete game "
          "inside a ```puzzlescript code block.")


def _seed_examples(rng, k=2):
    pool = sorted((_REPO / "gallery_games").glob("*.txt"))
    picks = []
    for p in rng.sample(pool, min(k, len(pool))):
        txt = p.read_text(errors="ignore")
        if 200 < len(txt) < 2200:
            picks.append(txt)
    return picks or [p.read_text(errors="ignore")[:2000] for p in pool[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=30)
    ap.add_argument("--max-attempts", type=int, default=400)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="game_synth/llm_live")
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    print(f"[llm] loaded {args.model} in {time.time()-t0:.0f}s", flush=True)

    out = _REPO / args.out
    games_dir = out / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(out / "_scratch"); (out / "_scratch").mkdir(exist_ok=True)
    parser = init_ps_lark_parser()
    man = (out / "manifest.jsonl").open("a", encoding="utf-8")
    rng = random.Random(args.seed)

    accepted = []          # ELM pool of validated game texts (seeds future prompts)
    n_valid = n_attempt = 0
    while n_valid < args.target and n_attempt < args.max_attempts:
        n_attempt += 1
        # ELM: prefer accepted games as few-shot seeds once we have some.
        if len(accepted) >= 2 and rng.random() < 0.6:
            examples = rng.sample(accepted, 2)
        else:
            examples = _seed_examples(rng)
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": PROMPT.format(examples="\n\n---\n\n".join(examples))}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", enable_thinking=False).to("cuda:0")
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=0.95)
            text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            code = extract_ps_code(text)
        except Exception as e:
            print(f"  attempt {n_attempt}: gen/extract fail {type(e).__name__}", flush=True)
            continue
        if not code or "RULES" not in code.upper() or "LEVELS" not in code.upper():
            continue
        name = f"llm_{args.seed}_{n_attempt:04d}"
        try:
            gc._materialize_game(name, code)
            ev = gc._search_materialized_game(parser, name, search_algo="bfs",
                                              search_timeout_ms=6000, search_n_steps=10000)
        except Exception:
            continue
        if not ev.compile_ok or ev.n_levels < 1:
            continue
        if not (ev.all_solvable or ev.max_search_iters >= 15):
            continue
        n_valid += 1
        accepted.append(code)
        (games_dir / f"{name}.txt").write_text(code, encoding="utf-8")
        man.write(json.dumps({"name": name, "track": "llm_live", "n_levels": ev.n_levels,
                              "all_solvable": ev.all_solvable, "max_iters": ev.max_search_iters}) + "\n")
        man.flush()
        print(f"[llm] valid {n_valid}/{args.target} (attempt {n_attempt}, "
              f"solvable={ev.all_solvable}, iters={ev.max_search_iters})", flush=True)
    man.close()
    print(f"[llm] DONE valid={n_valid} attempts={n_attempt} in {time.time()-t0:.0f}s "
          f"-> {games_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Proper GP evolution with WM-loss as (non-stationary) fitness.

Steady-state GP over game genomes (rule_gp rules + wins + level). Fitness of a
game = the world model's LOSS on it (model_NLL): high = the WM predicts it poorly
= a hard, learnable-frontier game. Because the
WM is *training* each generation, fitness is NON-STATIONARY (a game gets easier as
the WM learns it), so we RE-EVALUATE the whole population every generation rather
than trusting stale scores. Selection favors high fitness; offspring (mutation +
crossover of high-fitness parents) replace the lowest-fitness (mastered) members.

Whether the frontier (max/mean fitness) is SUSTAINED or COLLAPSES tells us if GP
can out-innovate the WM in this game space.

    .venv/bin/python -u -m game_synth.gp_evolve --gens 30
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nca_wm import game_curriculum as gc
from nca_wm import rule_game, rule_gp
from game_synth import gp_generate as GP
from game_synth.dedup import token_key
from game_synth.code_cond_train import CodeCondWorldModel, mixture_nll, pad_tokens
from game_synth.engine_train import OBJ6, _engine, sample_traj
from game_synth.render_games import coverage_walk, game_quality
from nca_wm.tokenize_game import get_game_tree_from_js, tokenize_game

VOCAB = 1024
OBJS = ["ObjA", "ObjB", "ObjC"]


# ---------- genome ops ----------
def random_genome(rng, max_rules=4):
    return {"rules": GP._sample_rules(rng, max_rules=max_rules), "wins": GP._sample_wins(rng),
            "levels": GP._sample_levels(rng), "layers": rule_gp.sample_layers(rng, OBJS)}


def to_code(g):
    return rule_game.assemble_game(g["rules"], objects=OBJS, levels=g["levels"],
                                   wins=g["wins"], layers=g.get("layers"))


def mutate(g, rng, max_rules=4):
    c = {"rules": copy.deepcopy(g["rules"]), "wins": list(g["wins"]),
         "levels": list(g["levels"]), "layers": g.get("layers")}
    for _ in range(rng.randint(1, 3)):
        c["rules"] = rule_gp.mutate_ruleset(c["rules"], rng, max_rules=max_rules)
    if rng.random() < 0.5:
        for i in range(len(c["rules"])):
            if rng.random() < 0.4:
                c["rules"][i] = rule_gp.random_mutate_rule(c["rules"][i], rng)
    if rng.random() < 0.4:
        c["wins"] = rule_gp.mutate_wins(c["wins"], rng)
    if rng.random() < 0.3:
        c["levels"] = GP._sample_levels(rng)
    if rng.random() < 0.2:
        c["layers"] = rule_gp.sample_layers(rng, OBJS)
    return c


def crossover(a, b, rng, max_rules=4):
    ra, rb = a["rules"], b["rules"]
    rules = copy.deepcopy(ra[:rng.randint(0, len(ra))] + rb[rng.randint(0, len(rb)):])[:max_rules]
    if not rules:
        rules = copy.deepcopy(ra) or copy.deepcopy(rb)
    src = a if rng.random() < 0.5 else b
    return {"rules": rules, "wins": list(src["wins"]), "levels": list(src["levels"]),
            "layers": src.get("layers")}


# ---------- validate / fitness ----------
def validate(name, code, parser, gates=None):
    from puzzlescript_cpp import CppPuzzleScriptBackend
    try:
        GP.FIXED.update(w=8, h=8, n_levels=1, objs=OBJS)
        gc._materialize_game(name, code)
        js = CppPuzzleScriptBackend().compile_and_serialize(parser, name)
        e = _engine(js)
        if np.asarray(e.get_objects_2d()).shape != (8, 8, 1):
            return None
        idd = [s.lower() for s in e.get_id_dict()]
        if any(s not in OBJ6 for s in idd):
            return None
        tree, ids = get_game_tree_from_js(parser, name)
        tok = tokenize_game(tree, ids, encode_sprites=False, include_levels=False)[:256]
        if not tok or max(tok) >= VOCAB:
            return None
        # ACTIVATED mechanics: distinct rules that actually fire in a short
        # rule-coverage walk (a far better complexity signal than rule_count;
        # ~45% of GP games fire ZERO rules despite many compiled rules).
        et = _engine(js); et.set_track_rules_fired(True)
        _, fired = coverage_walk(et, idd, 5, 16, random.Random(0))
        # NON-TRIVIALITY gates from bounded BFS search (reject trivially-winnable,
        # tiny-state-space, and no-op-action games -- they game WM loss cheaply).
        # Small budget: enough to catch <=1-step wins and confirm >=min_states
        # reachable, while keeping the per-candidate cost low in the hot loop.
        q = game_quality(js, budget=400, timeout_ms=600)
        if gates:
            if q["solvable"] and 0 <= q["sol_len"] < gates["min_sol_len"]:
                return None
            if q["n_states"] < gates["min_states"]:
                return None
            if q["effect_rate"] < gates["min_action_effect"]:
                return None
        return js, idd, tok, len(fired), q
    except Exception:
        return None


@torch.no_grad()
def fitness(model, ent, device, n=6, seed=0):
    """WM loss (model_NLL) over sampled transitions (high = hard for WM).

    Plain loss, not excess-over-no-rule: subtracting the no-rule baseline created
    a dead-game attractor (a dead game's no-rule baseline is exactly right, so any
    residual makes excess>0, while learned mechanical games go excess<0). Plain
    model_NLL instead gives dead games the LOWEST fitness (trivial to predict)."""
    rng = random.Random(seed)
    jsons = {ent["name"]: (ent["json"], ent["idd"])}
    tok, mask = pad_tokens([ent["tok"]], device)
    m = []
    for _ in range(n):
        o, a, _ = sample_traj(jsons, ent["name"], 5, rng)
        for t in range(len(a)):
            s = torch.from_numpy(o[t][None]).to(device)
            nx = torch.from_numpy(o[t + 1][None]).to(device)
            lo, lp = model.logits(s, torch.tensor([a[t]], device=device), tok, mask)
            m.append(mixture_nll(lo, lp, nx).item())
    return float(np.mean(m))


def _lp_weight(e, default, floor):
    """Replay-sampling weight for an archived game: its (absolute) learning
    progress, or `default` if never measured (so fresh archive entries get
    sampled and measured), plus a uniform `floor` so nothing starves."""
    lp = e.get("lp")
    return (default if lp is None else lp) + floor


def train_batch(model, pop, archive, device, rng, bs=48, replay_frac=0.0,
                lp_default=1.0, lp_floor=0.05):
    """One training batch. `replay_frac` of the samples are drawn from the
    archive of ALL distinct past games (sampled proportional to |learning
    progress|), the rest from the current frontier pop. replay_frac=0 trains on
    pop only (old behavior)."""
    arch = list(archive.values()) if archive else []
    n_replay = int(round(bs * replay_frac)) if arch else 0
    samp = [rng.choice(pop) for _ in range(bs - n_replay)]
    if n_replay:
        w = [_lp_weight(e, lp_default, lp_floor) for e in arch]
        samp += rng.choices(arch, weights=w, k=n_replay)
    jsons = {e["name"]: (e["json"], e["idd"]) for e in samp}
    S, A, NX, codes = [], [], [], []
    for e in samp:
        o, a, _ = sample_traj(jsons, e["name"], rng.randint(2, 6), rng)
        t = rng.randrange(len(a))
        S.append(o[t]); A.append(a[t]); NX.append(o[t + 1]); codes.append(e["tok"])
    tok, mask = pad_tokens(codes, device)
    s = torch.from_numpy(np.stack(S)).to(device); nx = torch.from_numpy(np.stack(NX)).to(device)
    a = torch.tensor(A, device=device)
    lo, lp = model.logits(s, a, tok, mask)
    return mixture_nll(lo, lp, nx).mean()


# ---------- LLM mutation operator (ELM-style; pluggable backend) ----------
_LLM_SYS = (
    "You are an expert PuzzleScript game designer. You MUST keep the objects to "
    "EXACTLY: Background, Wall, Player, ObjA, ObjB, ObjC, and a SINGLE 8x8 LEVELS "
    "grid (8 rows x 8 columns using . # P A B C). Vary the RULES (and optionally "
    "WINCONDITIONS) to create different, non-trivial dynamics. Output ONLY the "
    "complete game inside a fenced code block.")


def _setup_llm(args):
    """Return a generate(prompt)->text callable for the chosen backend."""
    if args.llm_backend == "endpoint":
        from puzzlescript_jax.utils import _vllm_text_query
        def gen(prompt):
            return _vllm_text_query(_LLM_SYS, prompt, args.llm_model, base_url=args.llm_base_url)
        return gen
    import torch as _t
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_model)
    mdl = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=_t.bfloat16,
                                               device_map="cuda:0")
    print(f"[llm] loaded HF {args.hf_model}", flush=True)
    def gen(prompt):
        msgs = [{"role": "system", "content": _LLM_SYS}, {"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      enable_thinking=False).to("cuda:0")
        out = mdl.generate(ids, max_new_tokens=1800, do_sample=True, temperature=0.9, top_p=0.95)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return gen


def _llm_child(parents, generate, rng):
    from game_synth.llm_generate import extract_ps_code
    if len(parents) >= 2 and rng.random() < 0.5:
        prompt = (f"Here are two PuzzleScript games:\n\nGAME A:\n```\n{parents[0]['code']}\n```\n\n"
                  f"GAME B:\n```\n{parents[1]['code']}\n```\n\nCreate a NEW game that COMBINES "
                  "mechanics from both, keeping the same objects and an 8x8 level. Output only the game.")
    else:
        prompt = (f"Here is a PuzzleScript game:\n```\n{parents[0]['code']}\n```\n\nCreate a NEW "
                  "variant by adding/removing/changing its RULES to give different dynamics, "
                  "keeping the same objects and an 8x8 level. Output only the new game.")
    try:
        return extract_ps_code(generate(prompt))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--offspring", type=int, default=14)
    ap.add_argument("--updates-per-gen", type=int, default=250)
    ap.add_argument("--tournament", type=int, default=4)
    ap.add_argument("--max-rules", type=int, default=4,
                    help="genome source-rule cap (raises the activated-mechanics ceiling: ~4x compiled)")
    ap.add_argument("--no-random", action="store_true",
                    help="forbid stochastic constructs (random/randomDir) so games are deterministic "
                         "-- randomness is a cheap way to inflate WM loss without learnable mechanics")
    # non-triviality gates (bounded-BFS): reject degenerate games at validation
    ap.add_argument("--min-sol-len", type=int, default=2,
                    help="reject solvable games winnable in fewer than N moves (2 => exclude 1-step wins)")
    ap.add_argument("--min-states", type=int, default=6,
                    help="reject games whose BFS-reachable state-space is smaller than N")
    ap.add_argument("--min-action-effect", type=float, default=0.02,
                    help="reject games where < this fraction of explored actions change the state")
    ap.add_argument("--fitness", default="loss", choices=["loss", "progress"],
                    help="loss = WM model_NLL (high=hard); progress = per-gen DECREASE in loss "
                         "(learning progress; deprioritizes both mastered AND unlearnable games)")
    ap.add_argument("--activation-weight", type=float, default=0.0,
                    help="add lambda * (#activated mechanics) to the fitness")
    # archive + learning-progress replay (combat catastrophic forgetting of culled games)
    ap.add_argument("--replay-frac", type=float, default=0.5,
                    help="fraction of each training batch drawn from the archive of ALL "
                         "distinct past games (vs. the current frontier pop); 0 = no replay")
    ap.add_argument("--archive-cap", type=int, default=4000,
                    help="max games kept in the replay archive (evict lowest learning-progress)")
    ap.add_argument("--archive-refresh", type=int, default=64,
                    help="non-pop archived games to re-measure each gen to refresh learning progress")
    ap.add_argument("--signed-lp", action="store_true",
                    help="use signed loss-decrease for learning progress instead of |Δloss| "
                         "(default abs, per Oudeyer: catches FORGETTING where loss rises)")
    ap.add_argument("--operator", default="gp", choices=["gp", "llm", "mixed"])
    ap.add_argument("--llm-frac", type=float, default=0.5, help="(mixed) fraction of LLM offspring")
    ap.add_argument("--llm-backend", default="endpoint", choices=["endpoint", "hf"])
    ap.add_argument("--llm-model", default="Qwen/Qwen3-32B", help="served model name (endpoint)")
    ap.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B", help="local HF model (hf backend)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="game_synth/gp_evolve_run")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); random.seed(args.seed)
    rule_gp.ALLOW_RANDOM = not args.no_random
    out = _REPO / args.out; out.mkdir(parents=True, exist_ok=True)
    gc._set_materialize_dir(out / "_scratch"); (out / "_scratch").mkdir(exist_ok=True)
    GP.FIXED.update(w=8, h=8, n_levels=1, objs=OBJS)
    from puzzlescript_jax.utils import init_ps_lark_parser
    parser = init_ps_lark_parser()
    rng = random.Random(args.seed)

    model = CodeCondWorldModel(vocab=VOCAB).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"WM params {sum(p.numel() for p in model.parameters()):,}", flush=True)

    gates = {"min_sol_len": args.min_sol_len, "min_states": args.min_states,
             "min_action_effect": args.min_action_effect}
    pop, sigs = [], set()
    all_sigs = set()   # all-time DISTINCT games ever accepted (never decremented) -> novelty
    archive = {}       # sig -> entry, ALL distinct games ever accepted (survives culling)
    ctr = [0]

    def make_entry(code, genome, gen):
        ctr[0] += 1
        name = f"ev_{gen:03d}_{ctr[0]:05d}"
        sig = token_key(code)[0]
        if sig in sigs:
            return None
        r = validate(name, code, parser, gates=gates)
        if r is None:
            return None
        sigs.add(sig); all_sigs.add(sig)
        ent = {"name": name, "genome": genome, "code": code, "json": r[0], "idd": r[1],
               "tok": r[2], "activated": r[3], "quality": r[4], "sig": sig,
               "fitness": 0.0, "born": gen, "op": "init", "lp": None, "wm_loss": None}
        archive[sig] = ent   # same object as the pop entry; persists after culling
        return ent

    # variation operators -> child (code, genome). GP ops keep a rule_gp genome
    # (further GP-mutable); the LLM op returns code-only (genome=None).
    llm_gen = _setup_llm(args) if args.operator in ("llm", "mixed") else None

    mr = args.max_rules

    def vary(pick, gen, rng):
        if llm_gen is not None and (args.operator == "llm" or rng.random() < args.llm_frac):
            code = _llm_child([pick(), pick()], llm_gen, rng)
            return (code, None, "llm")
        p = pick()
        if p["genome"] is None:                 # LLM-born parent: no GP genome -> fresh random
            g = random_genome(rng, max_rules=mr)
            return to_code(g), g, "mutate"
        if rng.random() < 0.5:
            g = mutate(p["genome"], rng, max_rules=mr)
            return to_code(g), g, "mutate"
        q = pick()
        g = crossover(p["genome"], q["genome"] if q["genome"] else random_genome(rng, max_rules=mr),
                      rng, max_rules=mr)
        return to_code(g), g, "crossover"

    # init population
    tries = 0
    while len(pop) < args.pop and tries < args.pop * 8:
        tries += 1
        g = random_genome(rng, max_rules=mr)
        e = make_entry(to_code(g), g, 0)
        if e:
            pop.append(e)
    print(f"init pop {len(pop)} | operator={args.operator} | fitness={args.fitness} "
          f"| max_rules={mr} | random={'off' if args.no_random else 'on'} "
          f"| gates(min_sol_len={gates['min_sol_len']},min_states={gates['min_states']},"
          f"min_effect={gates['min_action_effect']})", flush=True)

    metrics = []
    log = (out / "evolve.jsonl").open("w")
    t0 = time.time()
    for gen in range(1, args.gens + 1):
        # LEARN
        model.train()
        for _ in range(args.updates_per_gen):
            loss = train_batch(model, pop, archive, device, rng, replay_frac=args.replay_frac)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # RE-EVALUATE fitness of the whole pop (non-stationary)
        model.eval()

        def score(e):
            """Re-measure loss; set e['wm_loss','lp'] and return fitness per the
            objective. loss: raw model_NLL (high=hard). progress: learning
            progress = |Δloss| across gens (signed loss-decrease under --signed-lp).
            Birth-gen entries (no prior loss) fall back to current loss as
            entry-potential so new games get measured rather than culled at zero."""
            prev = e.get("wm_loss")
            now = fitness(model, e, device)
            e["lp"] = (now if prev is None else
                       (prev - now) if args.signed_lp else abs(prev - now))
            if args.fitness == "progress" and prev is not None:
                base = (prev - now) if args.signed_lp else abs(prev - now)
            else:
                base = now
            e["wm_loss"] = now
            return base + args.activation_weight * e["activated"]

        for e in pop:
            e["fitness"] = score(e)
        # Refresh learning progress on a random slice of the ARCHIVE (culled games
        # not in the current pop) so replay weighting tracks forgetting/relearning.
        non_pop = [e for s, e in archive.items() if s not in sigs]
        if non_pop and args.archive_refresh:
            for e in random.sample(non_pop, min(args.archive_refresh, len(non_pop))):
                score(e)
        pop.sort(key=lambda e: e["fitness"], reverse=True)
        # VARY: tournament-select high-fitness parents -> offspring (GP or LLM op)
        accepted = 0
        from collections import Counter
        opc = Counter()
        def pick():
            return max(random.sample(pop, min(args.tournament, len(pop))), key=lambda e: e["fitness"])
        for _ in range(args.offspring):
            code, genome, op = vary(pick, gen, rng)
            if not code:
                continue
            e = make_entry(code, genome, gen)
            if e:
                e["op"] = op
                e["fitness"] = score(e)
                pop.append(e); accepted += 1; opc[op] += 1
        # CULL lowest-fitness back to pop size
        pop.sort(key=lambda e: e["fitness"], reverse=True)
        culled = pop[args.pop:]
        for e in culled:
            sigs.discard(e["sig"])   # leaves pop dedup; entry stays in archive
        pop = pop[:args.pop]
        # Evict lowest-learning-progress games if the archive exceeds its cap.
        if len(archive) > args.archive_cap:
            ranked = sorted(archive.values(), key=lambda e: _lp_weight(e, 1.0, 0.0))
            for e in ranked[:len(archive) - args.archive_cap]:
                archive.pop(e["sig"], None)
        fits = np.array([e["fitness"] for e in pop])
        act = np.array([e["activated"] for e in pop])
        nstates = np.array([e["quality"]["n_states"] for e in pop])
        effect = np.array([e["quality"]["effect_rate"] for e in pop])
        frac_solv = float(np.mean([e["quality"]["solvable"] for e in pop]))
        # LLM-born offspring carry no GP genome (genome=None); skip them here.
        _gen_rules = [len(e["genome"]["rules"]) for e in pop if e["genome"]]
        nrules = float(np.mean(_gen_rules)) if _gen_rules else 0.0
        row = {"gen": gen, "pop": len(pop), "max_fit": float(fits.max()),
               "mean_fit": float(fits.mean()), "min_fit": float(fits.min()),
               "frac_hard": float((fits > 0.2).mean()), "mean_rules": float(nrules),
               "mean_activated": float(act.mean()), "max_activated": int(act.max()),
               "frac_dead": float((act == 0).mean()),
               "mean_states": float(nstates.mean()), "mean_effect": float(effect.mean()),
               "frac_solvable": frac_solv,
               "accepted": accepted, "ops": dict(opc), "alltime": ctr[0],
               "distinct": len(all_sigs), "archive": len(archive),
               "mean_lp": float(np.mean([e["lp"] for e in archive.values()
                                         if e.get("lp") is not None]) if archive else 0.0),
               "t": time.time() - t0}
        metrics.append(row); log.write(json.dumps(row) + "\n"); log.flush()
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"gen {gen:2d} | pop {len(pop)} | fit mean {row['mean_fit']:+.2f} | "
              f"act mean {row['mean_activated']:.2f} max {row['max_activated']} | "
              f"states {row['mean_states']:.0f} effect {row['mean_effect']:.2f} "
              f"solv {row['frac_solvable']:.0%} | new {accepted} | distinct {len(all_sigs)} "
              f"| arch {len(archive)} lp {row['mean_lp']:.3f} | {row['t']:.0f}s", flush=True)
    log.close()
    torch.save({"model_state": model.state_dict()}, out / "wm.pt")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        g = [m["gen"] for m in metrics]
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(g, [m["max_fit"] for m in metrics], "-o", label="max fitness (hardest)")
        ax[0].plot(g, [m["mean_fit"] for m in metrics], "-s", label="mean fitness")
        ax[0].axhline(0, color="k", lw=0.6)
        ax[0].set_xlabel("generation"); ax[0].set_ylabel("WM excess-loss (fitness)")
        ax[0].set_title("Frontier: sustained or collapsed?"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(g, [m["mean_rules"] for m in metrics], "-o", color="tab:purple")
        ax[1].set_xlabel("generation"); ax[1].set_ylabel("mean #rules / game")
        ax[1].set_title("Pop complexity under selection"); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(f"nca_wm/figures/gp_evolve_frontier.{e}", dpi=140, bbox_inches="tight")
        print("saved nca_wm/figures/gp_evolve_frontier.{png,pdf}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)
    print(f"DONE | {ctr[0]} games evaluated | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

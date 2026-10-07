"""Generator for the PuzzleScript DT loop (protocol.md, stage 4), in its own Torch process.

    python generator.py panel --out DIR COMMON
    python generator.py serve --state DIR --seed S [--train] --panel DIR COMMON
    COMMON = --data DIR --run DIR --engine-dir DIR --cpp SO --level ID [--sampler full|none]
             [--temperature T] [--batch B] [--pool K] [--workers W]

The generator is the stage-3 level-first model writing the rules and win conditions of one fixed
level (its rules-only prompt: the level, flags, objects, legend and collision layers). A sample
is admitted when
  - level_first.to_standard accepts the text;
  - the reference JS engine compiles it with one playable level (ps_export.js);
  - it is deterministic, with no message or checkpoint command (engine.admission);
  - the 200-move seeded random rollout of ps_check.js changes the level (engine.rollout_changes,
    in the C++ engine); and
  - (serve) its mechanics differ from every held-out panel game's (canonicalize.mechanics_key).

panel: the held-out panel. PANEL_SAMPLES samples under PANEL_SEED, admitted as above; one game
per behaviour class (ps_probe.js fingerprints, as in level_eval.py), the first in sample order,
plus the level's human original. Writes panel.jsonl (gid 0..M-1) and summary.json.

serve: the round driver (loop.py) writes one JSON request per line to stdin,
    {"round": r, "out": DIR, "reward": x | null, "gid_base": n}
and this process answers one JSON line on stdout when round r's pool is in DIR:
  1. with --train and a reward for round r-1: one REINFORCE step on the joint log-likelihood,
     under the sampling temperature, of round r-1's action: every completion drawn up to the
     pool's K-th admission, kept or not (the stopping rule is a function of those samples, so
     their joint likelihood is the action's). The advantage is the reward minus the EMA
     baseline (0.9 / 0.1, updated in every arm as a diagnostic); the loss is scaled by the fixed
     constant 1 / LOSS_TOKENS, not by the realised token count; Adam 1e-5 after clipping the
     gradient norm at 1. Under masked sampling the log-likelihood is the model's unmasked one:
     it omits the mask normaliser, an approximation the recorded rejection rate bounds;
  2. samples round r's pool: batches of B rows until K games are admitted. DIR gets
     samples.jsonl (every row drawn, with its verdict and whether it belongs to the action),
     pool.jsonl (the K games, gid = gid_base + k; pool-texts.jsonl without the compiled JSON)
     and generator.json (the round's summary);
  3. saves its full state (weights and optimizer with --train, baseline, sampler RNG, round and
     the action's tokens) to STATE/rNNN.pt and keeps the newest two.
Sampling and the update run with deterministic algorithms (cuBLAS workspace, the math attention
kernel in the update), so a restored state reproduces the rounds that follow exactly.
Requests and answers are the only stdout traffic; progress goes to stderr.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts: deterministic cuBLAS
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from canonicalize import CanonError, from_engine, mechanics_key  # noqa: E402
from check_games import check_texts  # noqa: E402
from level_eval import prompt_until  # noqa: E402
from level_first import FormatError, to_standard  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
import engine as E  # noqa: E402

PANEL_SEED = 900001
PANEL_SAMPLES = 256
LOSS_TOKENS = 32768
LR = 1e-5
MAX_BATCHES = 32
MAX_NEW = 2048  # new tokens per row; stage 3's usable rules-only samples are at most 635 tokens


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Generator:
    def __init__(self, a, seed):
        from tokenizers import Tokenizer
        self.a, self.device = a, torch.device("cuda")
        self.tok = Tokenizer.from_file(str(a.data / "tokenizer.json"))
        prep = json.loads((a.data / "prep_report.json").read_text())
        self.bos, self.eos, self.pad = prep["bos"], prep["eos"], prep["pad"]
        ck = torch.load(a.run / "best.pt", map_location="cpu", weights_only=False)
        self.cfg = GPTConfig(**ck["config"])
        self.model = GPT(self.cfg).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.checkpoint_sha256 = sha256(a.run / "best.pt")
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        texts = {}
        for split in ("val", "test"):
            for line in open(a.data / f"{split}_texts.jsonl"):
                t = json.loads(line)
                texts[t["id"]] = t["text"]
        self.human_text = texts[a.level]
        self.prompt_text = prompt_until(self.human_text, "RULES")
        self.prompt_ids = self.tok.encode(self.prompt_text).ids
        self.max_new = min(MAX_NEW, self.cfg.max_seq_len - 1 - len(self.prompt_ids))
        if a.sampler != "none":
            from constrained import token_texts
            self.token_texts = token_texts(a.data / "tokenizer.json")
        self.cpp = E.load_cpp(a.cpp)
        self.train_keys = {json.loads(line)["mechanics_key"] for line in open(a.data / "train_docs.jsonl")}
        human = E.export_games([{"id": "human", "text": to_standard(self.human_text)}], a.engine_dir, workers=1)["human"]
        self.noaction = "noaction" in json.loads(human)["metadata"]
        reference = E.GamePool(self.cpp, self.noaction)
        reference.add(0, human)
        self.meta, self.human_json = reference.meta, human
        self.panel_keys = set()
        self.baseline, self.round, self.action = 0.0, 0, None

    # sampling -----------------------------------------------------------------------------
    def sample_batch(self, tag, start):
        a = self.a
        prompt = torch.tensor([[self.bos] + self.prompt_ids] * a.batch, dtype=torch.long, device=self.device)
        stats = [None] * a.batch
        t0 = time.time()
        with torch.no_grad():
            if a.sampler == "none":
                outs, done = self.model.generate(prompt, self.max_new, self.eos, temperature=a.temperature,
                                                 generator=self.rng)
            else:
                from constrained import RulesChecker, generate_constrained
                checkers = [RulesChecker(self.prompt_text, a.sampler, self.token_texts, self.eos, True)
                            for _ in range(a.batch)]
                outs, done, stats = generate_constrained(self.model, prompt, self.max_new, self.eos, checkers,
                                                         temperature=a.temperature, generator=self.rng)
        items = []
        for k, (o, d) in enumerate(zip(outs, done)):
            it = {"id": f"{tag}-{start + k:05d}", "row": start + k, "tokens": list(o) + ([self.eos] if d else []),
                  "n_new_tokens": len(o), "hit_eos": bool(d), "standard": None}
            if stats[k] is not None:
                it["constrain"] = stats[k]
            try:
                it["standard"] = to_standard(self.tok.decode(self.prompt_ids + list(o)))
            except (FormatError, CanonError) as ex:
                it["verdict"], it["format_error"] = "format", str(ex)[:200]
            items.append(it)
        log(f"{tag}: sampled rows {start}-{start + a.batch - 1} in {time.time() - t0:.0f}s")
        return items

    def screen(self, items, exclude_panel):
        """Fill each item's verdict: format, compile, random, message, checkpoint, meta, static,
        panel or admitted; admitted items get json, rollout, mechanics_key and n_rules."""
        a = self.a
        std = [it for it in items if it["standard"] is not None]
        js = E.export_games([{"id": it["id"], "text": it["standard"]} for it in std], a.engine_dir,
                            workers=a.workers) if std else {}
        live = []
        for it in std:
            j = js.get(it["id"])
            if j is None:
                it["verdict"] = "compile"
                continue
            ok, why = E.admission(j)
            if not ok:
                it["verdict"] = why
                continue
            pool = E.GamePool(self.cpp, self.noaction)
            try:
                pool.add(0, j)
            except ValueError:
                it["verdict"] = "meta"
                continue
            if pool.meta != self.meta:
                it["verdict"] = "meta"
                continue
            changed, won = E.rollout_changes(pool, 0)
            if not changed:
                it["verdict"] = "static"
                continue
            it["json"], it["rollout"] = j, {"changed": changed, "won": won}
            live.append(it)
        ext = check_texts([{"id": it["id"], "text": it["standard"]} for it in live], a.engine_dir,
                          workers=a.workers, script="ps_extract.js", stall_s=90.0) if live else []
        for it, e in zip(live, ext):
            try:
                c = from_engine(e) if e.get("ok") else None
            except CanonError:
                c = None
            it["mechanics_key"] = mechanics_key(c) if c is not None else None
            it["n_rules"] = len(c.rules) if c is not None else None
            it["train_copy"] = it["mechanics_key"] in self.train_keys
            if exclude_panel and it["mechanics_key"] is not None and it["mechanics_key"] in self.panel_keys:
                it["verdict"] = "panel"
            else:
                it["verdict"] = "admitted"

    def probe(self, items):
        probes = check_texts([{"id": it["id"], "text": it["standard"]} for it in items], self.a.engine_dir,
                             workers=self.a.workers, script="ps_probe.js", stall_s=90.0) if items else []
        for it, p in zip(items, probes):
            it["probe"] = {k: p.get(k) for k in ("ok", "probes", "distinct_states", "won", "again_capped",
                                                 "stochastic", "timeout", "exception")}

    def draw(self, tag, k, exclude_panel, max_batches=MAX_BATCHES):
        """Batches until k admissions: (every item drawn, the items up to the k-th admission)."""
        drawn, admitted = [], 0
        for b in range(max_batches):
            batch = self.sample_batch(tag, b * self.a.batch)
            self.screen(batch, exclude_panel)
            for it in batch:
                it["in_action"] = admitted < k
                admitted += it["verdict"] == "admitted" and it["in_action"]
            drawn += batch
            if admitted >= k:
                return drawn, [it for it in drawn if it["in_action"]]
        raise RuntimeError(f"{tag}: only {admitted} of {k} admissions in {max_batches} batches")

    # learning -----------------------------------------------------------------------------
    def _chunk_log_probs(self, chunk, autocast):
        """Per-completion log p_T(completion | prompt) for a few token lists, with gradients. The
        math attention kernel and cross-entropy keep the backward pass deterministic."""
        from torch.nn.attention import SDPBackend, sdpa_kernel
        prefix = [self.bos] + self.prompt_ids
        seqs = [prefix + c for c in chunk]
        length = max(len(q) for q in seqs)
        x = torch.full((len(seqs), length), self.pad, dtype=torch.long, device=self.device)
        mask = torch.zeros((len(seqs), length - 1), dtype=torch.bool, device=self.device)
        for i, q in enumerate(seqs):
            x[i, :len(q)] = torch.tensor(q, dtype=torch.long)
            mask[i, len(prefix) - 1:len(q) - 1] = True
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = self.model(x[:, :-1])
        nll = torch.nn.functional.cross_entropy((logits.float() / self.a.temperature).transpose(1, 2), x[:, 1:],
                                                reduction="none")
        return -(nll * mask).sum(1), int(mask.sum())

    def sequence_log_probs(self, completions, autocast=True, micro=2):
        out = []
        with torch.no_grad():
            for s in range(0, len(completions), micro):
                out += self._chunk_log_probs(completions[s:s + micro], autocast)[0].tolist()
        return out

    def reinforce(self, completions, reward, micro=2):
        """One score-function step on the joint log-likelihood of `completions` (token lists)."""
        advantage = reward - self.baseline
        self.opt.zero_grad(set_to_none=True)
        joint, tokens = 0.0, 0
        for s in range(0, len(completions), micro):
            lp, n = self._chunk_log_probs(completions[s:s + micro], autocast=True)
            lp = lp.sum()
            (-(advantage / LOSS_TOKENS) * lp).backward()
            joint += float(lp)
            tokens += n
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return dict(reward=reward, baseline=self.baseline, advantage=advantage, joint_log_likelihood=joint,
                    completions=len(completions), tokens=tokens, gradient_norm_before_clip=norm)

    # state --------------------------------------------------------------------------------
    def save(self, path):
        state = dict(round=self.round, baseline=self.baseline, action=self.action, rng=self.rng.get_state(),
                     checkpoint_sha256=self.checkpoint_sha256, train=self.a.train)
        if self.a.train:
            state.update(model=self.model.state_dict(), optimizer=self.opt.state_dict())
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        return sha256(path)

    def restore(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["checkpoint_sha256"] != self.checkpoint_sha256 or state["train"] != self.a.train:
            raise ValueError("generator state belongs to another model or arm")
        self.round, self.baseline, self.action = state["round"], state["baseline"], state["action"]
        self.rng.set_state(state["rng"])
        if self.a.train:
            self.model.load_state_dict(state["model"])
            self.opt.load_state_dict(state["optimizer"])


def write_jsonl(path, rows, drop=()):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k not in drop}) + "\n")


def verdicts(items):
    return dict(Counter(it["verdict"] for it in items))


def panel(a):
    g = Generator(a, PANEL_SEED)
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    items = []
    for b in range(PANEL_SAMPLES // a.batch):
        batch = g.sample_batch("panel", b * a.batch)
        g.screen(batch, exclude_panel=False)
        items += batch
    admitted = [it for it in items if it["verdict"] == "admitted"]
    g.probe(admitted)
    human = {"id": "human", "standard": to_standard(g.human_text), "json": g.human_json, "verdict": "human"}
    ext = check_texts([{"id": "human", "text": human["standard"]}], a.engine_dir, workers=1, script="ps_extract.js")
    c = from_engine(ext[0])
    human.update(mechanics_key=mechanics_key(c), n_rules=len(c.rules))
    ok, why = E.admission(g.human_json)
    if not ok:
        raise RuntimeError(f"the human original is not admissible: {why}")
    g.probe([human])
    reps, seen = [], set()
    for it in admitted:
        if not it["probe"].get("ok"):
            continue
        cls = tuple(it["probe"]["probes"])
        if cls not in seen:
            seen.add(cls)
            reps.append(it)
    if not human["probe"].get("ok"):
        raise RuntimeError("the human original failed its behaviour probe")
    human_class = tuple(human["probe"]["probes"])
    rows = [dict(gid=gid, id=it["id"], source="human" if it is human else "generated",
                 mechanics_key=it["mechanics_key"], n_rules=it["n_rules"], standard=it["standard"], json=it["json"],
                 probes=it["probe"]["probes"], same_class_as_human=tuple(it["probe"]["probes"]) == human_class)
            for gid, it in enumerate(reps + [human])]
    write_jsonl(a.out / "panel.jsonl", rows)
    write_jsonl(a.out / "samples.jsonl", items, drop=("json",))
    summary = dict(seed=PANEL_SEED, samples=len(items), verdicts=verdicts(items), admitted=len(admitted),
                   behaviour_classes=len(seen), panel_games=len(rows),
                   human_class_also_generated=human_class in seen,
                   sampler=a.sampler, temperature=a.temperature, level=a.level,
                   model_checkpoint_sha256=g.checkpoint_sha256, seconds=time.time() - t0,
                   panel_sha256=sha256(a.out / "panel.jsonl"))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(json.dumps(summary))


def serve(a):
    g = Generator(a, a.seed)
    for line in open(a.panel / "panel.jsonl"):
        key = json.loads(line)["mechanics_key"]
        if key is not None:
            g.panel_keys.add(key)
    a.state.mkdir(parents=True, exist_ok=True)
    if a.resume:
        g.restore(a.resume)
        log(f"resumed generator at round {g.round}")
    out_stream = sys.stdout
    sys.stdout = sys.stderr  # stray prints from libraries must not reach the driver
    for line in sys.stdin:
        req = json.loads(line)
        r, out = int(req["round"]), Path(req["out"])
        if r != g.round + 1:
            raise RuntimeError(f"round {r} requested after round {g.round}")
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        update = None
        reward = req.get("reward")
        if reward is not None:
            if a.train:
                update = g.reinforce(g.action, float(reward))
            g.baseline = 0.9 * g.baseline + 0.1 * float(reward)
        update_seconds = time.time() - t0
        drawn, action = g.draw(f"r{r:03d}", a.pool, exclude_panel=True)
        pool = [it for it in action if it["verdict"] == "admitted"]
        g.probe(pool)
        for k, it in enumerate(pool):
            it["gid"] = int(req["gid_base"]) + k
        write_jsonl(out / "samples.jsonl", drawn, drop=("json",))
        rows = [dict(gid=it["gid"], id=it["id"], json=it["json"], standard=it["standard"],
                     mechanics_key=it["mechanics_key"], n_rules=it["n_rules"], train_copy=it["train_copy"],
                     rollout=it["rollout"], probe=it.get("probe")) for it in pool]
        write_jsonl(out / "pool.jsonl", rows)
        write_jsonl(out / "pool-texts.jsonl", rows, drop=("json",))
        g.action, g.round = [it["tokens"] for it in action], r
        state_sha = g.save(a.state / f"r{r:03d}.pt")
        for old in sorted(a.state.glob("r*.pt"))[:-2]:
            old.unlink()
        new_tokens = sum(it["n_new_tokens"] for it in action)
        classes = Counter(tuple(it["probe"]["probes"]) for it in pool if it.get("probe", {}).get("ok"))
        summary = dict(round=r, reward_received=reward, update=update, baseline_after=g.baseline,
                       drawn=len(drawn), action_samples=len(action), action_tokens=new_tokens,
                       admission_rate_in_action=len(pool) / len(action), verdicts_in_action=verdicts(action),
                       rejections_per_token=(sum(it["constrain"]["rejections"] for it in action if "constrain" in it)
                                             / max(1, new_tokens)),
                       dead_ends=sum(bool(it.get("constrain", {}).get("dead_end")) for it in action),
                       mean_rules=float(np.mean([it["n_rules"] for it in pool if it["n_rules"] is not None]))
                       if any(it["n_rules"] is not None for it in pool) else None,
                       distinct_mechanics=len({it["mechanics_key"] for it in pool}),
                       behaviour_classes=len(classes), train_copies=sum(it["train_copy"] for it in pool),
                       update_seconds=update_seconds, seconds=time.time() - t0, state_sha256=state_sha)
        (out / "generator.json").write_text(json.dumps(summary, indent=2) + "\n")
        out_stream.write(json.dumps(dict(ok=True, round=r, pool=str(out / "pool.jsonl"), state_sha256=state_sha)) + "\n")
        out_stream.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["panel", "serve"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--cpp", type=Path, required=True, help="script-doctor's compiled _puzzlescript_cpp extension")
    ap.add_argument("--level", required=True)
    ap.add_argument("--sampler", choices=["full", "none"], default="full")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, help="panel: output directory")
    ap.add_argument("--state", type=Path, help="serve: state directory")
    ap.add_argument("--panel", type=Path, help="serve: the panel directory (its games are excluded)")
    ap.add_argument("--seed", type=int, help="serve: sampler seed")
    ap.add_argument("--train", action="store_true", help="serve: REINFORCE on the driver's rewards")
    ap.add_argument("--resume", type=Path, help="serve: a saved state (STATE/rNNN.pt)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if a.mode == "panel":
        panel(a)
    else:
        if a.state is None or a.panel is None or a.seed is None:
            ap.error("serve needs --state, --panel and --seed")
        serve(a)


if __name__ == "__main__":
    main()

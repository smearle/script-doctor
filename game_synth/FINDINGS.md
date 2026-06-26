# Game synthesis + GP↔WM co-evolution (`game_synth/`)

Generate a large pool of *distinct* PuzzleScript environments and co-evolve them
against a learning world model (WM): breed toward games the WM cannot yet model.
This is the environment-generation analog of the active *data* collection in
`nca_wm/active_learning/` (see that subproject's `REPORT.md`).

## Dataset (`dataset/`, 1,000 environments)

1,000 novel games via two tracks, then deduplicated:
- **GP + search** (~700): `gp_generate.py` mutates a rule grammar (`rule_gp.py`)
  and assembles games (`rule_game.py`); search plays them to keep the playable ones.
- **LLM ELM** (~300): `llm_generate.py` mutates existing generated games. Local
  vLLM serving fails (engine-core init); HF `transformers` (Qwen3-8B) works
  locally, big models (Qwen3-VL-235B) serve on the torch HPC via SLURM
  (`gp_evolve_235b.sbatch`, vLLM TP=4).

**Dedup (`dedup.py`).** `token_key(code)` parses with the lark parser, strips,
builds the PS tree, **drops no-op rules** (`_drop_noop_tree_rules`: a rule whose
left kernels == right kernels and has no command — e.g. `[Player|ObjA]->[Player|ObjA]`),
then tokenizes (`tokenize_game`, `encode_sprites=False`) and hashes. No-op
stripping is done on the *parsed tree*, not by regex (regex is only the fallback
for unparseable games). The no-op step caught 13 further duplicates (998→987
distinct). Earlier dedup tiers kept in `dataset_v1_crude_dedup/`,
`dataset_v2_tokendedup/` for provenance.

## Engines compared (`engine_train.py`, `code_cond_train.py`)

All games are fixed to 8×8, single-level, objects `OBJ6 =
[background,wall,player,objA,objB,objC]`.

**The honest baseline is no-rule movement** (`norule_next`): default PuzzleScript
movement + collision (player steps into an empty cell or is blocked; valid here
because all objects share one collision layer). It costs ~2.0 NLL. A WM only
"earns its keep" by modeling rule *effects* beyond movement, so fitness uses
**excess loss = model_NLL − norule_NLL** (high = the WM has *not* captured the
rule effects = the game is hard).

Held-out NLL (lower = better; no-rule ≈ 2.0):

| engine | held-out NLL |
|---|---:|
| in-context NCA-belief (`engine_train.py`) | **0.36** |
| code-conditioned, rule_attn encoder | 1.14 |
| code-conditioned, mean-pooled encoder | 1.87 |

The **in-context** engine generalizes best: it conditions on observed
transitions, so it infers dynamics rather than having to learn a code→dynamics
map. The code-conditioned models tokenize the game (`tokenize_game`) and FiLM the
NCA; `rule_attn` (a `RuleSlotEncoder` of K=16 learned-query rule slots with
per-cell cross-attention, ported from the repo's JAX rule_attn) beats mean
pooling but still lags in-context. The recurrence + code-conditioning **hybrid**
is owned by a separate agent — do not rebuild it here.

## GP↔WM co-evolution (`gp_evolve.py`, `coevolve.py`)

A proper GP with non-stationary WM-loss fitness: each generation trains the WM
`updates_per_gen` steps on the population, **re-evaluates all fitness** (because
the WM is moving), tournament-selects high-excess-loss games, produces offspring
(GP mutate/crossover and/or an LLM operator — `--operator {gp,llm,mixed}`,
pluggable via `_llm_child`), and culls the lowest. New games are token-deduped
before entering the pool.

**How online transitions are collected.** There is no replay buffer or fixed
dataset. Each WM update, `train_batch` samples a random population game and rolls
a **fresh random-action trajectory** through the C++ engine (`sample_traj` →
`process_input`), taking a random `(state, action, next_state)` from it, with the
per-step `haoo'` resample (`o'` via `backup_level`/`restore_level` + `seed_rng`).
Fitness eval samples the same way. So training data is a continuously regenerated
stream of **random-policy** transitions. Consequence: rules that only fire in rare
configurations are seldom triggered by random play, so neither the WM nor the
fitness sees them (and the no-rule baseline also matches when nothing fires →
excess ≈ 0) — a navigation/coverage collection policy would be needed to surface
sparse-trigger mechanics.

## Key finding: the in-sample-loss frontier collapses

Running the GP with in-sample WM-loss as fitness, **max fitness goes negative
after ~gen 4** (transient spikes, e.g. +3.8 at gen 10), and a GP-only 30-gen run
ended at max fitness −0.76 over 549 games. Two compounding causes:

1. **Train-on-eval.** The WM trains on the very games it scores, so it fits
   whatever is in the pool — in-sample "hardness" decays to noise. The honest fix
   is **held-out fitness**: score each game with a WM not trained on it.
2. **GP does not scale complexity.** Mean rules/game stayed flat at ~1.9 the whole
   run. `rule_gp.py` only emits single-bracket rules with basic
   prefixes/modifiers/commands — it never produces the constructs that are
   genuinely hard to model. Simple games are trivially learnable, so no quantity
   of them sustains a frontier.

Figure: `nca_wm/figures/gp_evolve_frontier.{png,pdf}`.

## Levers to sustain the frontier (open work)

In rough priority:

1. **Richer GP grammar.** `rule_gp.py` currently has directional / `late` /
   `random` prefixes, `> < ^ v no random randomDir` modifiers, and
   `again`/`win`/… commands; it *structurally* supports multi-part rules and
   multi-object cells but never generates them. Missing generators, ranked by how
   hard the result is for the WM (the parser/tokenizer/`_drop_noop_tree_rules`
   already accept all of these):
   - **High (WM blind spots):** `...` ellipsis (long-range), `[A][B]` multi-bracket
     (non-local), rule **groups** (`startloop`/`endloop`, `+` ordered groups —
     cascades/loops). These force the global / iterative computation the NCA WM is
     weak at (cf. the global-context `axis_pool`/`global_pool` flags in
     `nca_wm/models.py`), so they produce games that resist fitting.
   - **Medium:** aggregate/property objects (`Obstacle = Wall or Crate`),
     multi-object cells, `stationary`/`moving`/`action` modifiers.
   - **Lower:** `rigid`/`perpendicular`/`parallel`, richer win conditions.
2. **Held-out fitness** — score games with a WM that did not train on them, so
   "hard" means "hard to learn," not "not yet memorized."
3. **Evolution : SGD ratio** — at `updates_per_gen=250` the WM masters the
   (simple) population within a generation; fewer updates/gen or more
   offspring/gen lets GP stay ahead. A tuning knob, secondary to (1)–(2): more
   rounds of *simple* games still get mastered.

## File map

`rule_gp.py` rule-AST grammar + mutators · `rule_game.py` game assembly + random
levels · `gp_generate.py` GP+search generation · `llm_generate.py` LLM mutation ·
`dedup.py` token + no-op-rule dedup · `engine_train.py` in-context NCA-belief
engine (+ `norule_next`, `sample_traj`) · `code_cond_train.py` code-conditioned
WM (pool + rule_attn encoders) · `gp_evolve.py` proper GP with WM-loss fitness +
pluggable LLM operator · `coevolve.py` / `coevolve_report.py` co-evolution loop +
pool/mechanics report · `render_games.py` render generated games ·
`human_transfer.py` cross-domain transfer eval · `autumn/`, `autumn_port/` Autumn
ELM + PuzzleScript ports (separate tracks). Generated pools/checkpoints under
`game_synth/*/` are gitignored; `dataset*/` corpora are tracked.

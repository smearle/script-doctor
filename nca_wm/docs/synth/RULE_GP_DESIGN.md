# Rule-Grammar GP Curriculum — Design Sketch

Status: draft, not implemented. Discuss before building.

## The question we're trying to answer

Can the NCA world model **distill the PuzzleScript engine's rule grammar**,
or does it just memorize the dynamics of whichever games happen to be in the
training set? "Rule distillation" here means: train on rule-grammar
configurations A, B, C; test on held-out configuration D drawn from the
*same grammar surface* but never seen during training. If the WM
generalizes to D, it has internalized the rule semantics, not the
sokoban-specific transition table.

LLM-driven game generation (`game_curriculum.py`) is the wrong tool for
this question. LLMs produce *human-plausible* games clustered around their
training corpus; they don't systematically span the rule-grammar space, and
the smoketest in `GAME_CURRICULUM_REPORT.md` confirms the failure mode
(`init` was 0/2 compile, the gen-3 winner was a `mutate` of a seed rather
than a structurally novel rule). What we want is **systematic coverage of
rule structure**, which is what AST-level GP gives us for free.

`varislide.txt` is already the existence proof of the right object: a
single-rule synthetic game, parametric across levels, designed to probe one
specific dynamic (`again`-loop iteration count). What's missing is the
**generator** that produces a varislide-shaped game for every cell in the
rule-grammar space.

## The rule-grammar surface (from `puzzlescript_jax/syntax.lark`)

A `rule_data_actual` node is the unit of variation:

```
rule_data_actual: RULE_PREFIX* (rule_part RULE_PREFIX?)+ THEN (command | rule_part)* NEWLINES
rule_part:        "[" (rule_content? cell_border*)+ "]"
rule_content:     rule_object_with_modifier | rule_object | line_detector
RULE_PREFIX:      late | horizontal | vertical | left | right | up | down | + | random | ...
OBJECT_MODIFIER:  > | < | ^ | v | random | randomDir | no | parallel | perpendicular | ...
line_detector:    "..." | "]...["
command:          again | cancel | checkpoint | restart | win
```

That gives us a finite, enumerable set of axes to mutate:

| axis | concrete options | what it probes in the WM |
| --- | --- | --- |
| rule prefix | none, `late`, `random`, `horizontal/vertical`, directional (`left/right/up/down`) | global rule scheduling vs local pattern-matching |
| LHS clause count | 1, 2, 3+ patterns (`[A][B]`, `[A][B][C]`) | multi-clause conjunction |
| cell count per pattern | 1, 2, 3+ cells (`[A]`, `[A\|B]`, `[A\|B\|C]`) | spatial neighborhood reasoning |
| line connector | none, `...`, `]...[` | unbounded-distance / ellipsis semantics |
| object modifier | none, `>`, `<`, `^`, `v`, `no`, `parallel`, `perpendicular`, `random` | direction propagation, negation, relational direction |
| RHS rewrite | identity, swap, delete (`[A] -> []`), spawn (`[A] -> [A B]`), composition with modifier | arity-changing rewrites |
| trailing command | none, `again`, `cancel`, `win` | iteration / termination semantics |
| inside `startloop`/`endloop` | yes / no | rule-block fixpoint vs single-pass |

A rule-set is a tuple of rules; the second-order axes are **interaction**
between rules (rule order, late vs eager, shared object vocabulary).

## AST mutator: architecture

Two layers. Keep them separate so we can fuzz at either level.

### 1. Rule-tree mutator (`nca_wm/rule_gp.py`, new)

Operates on the lark tree of a single rule. Atomic operations:

- `add_prefix(prefix)` / `drop_prefix()`
- `add_clause(rule_part)` / `drop_clause(i)`
- `add_cell(part_i, slot_i, content)` / `drop_cell(part_i, slot_i)`
- `set_modifier(part_i, slot_i, mod)` / `clear_modifier`
- `insert_ellipsis(part_i, after_slot)` / `drop_ellipsis`
- `set_rhs_rewrite(rhs_tree)`
- `add_command(cmd)` / `drop_command`

Each operation returns a new tree; serialization back to PuzzleScript text
is a small unparser (lark trees → string preserving cell-border layout).
**Not** an LLM — a deterministic AST-to-text printer. ~200 LOC.

### 2. Game assembler (`nca_wm/rule_game.py`, new)

Stitches a `(rule_set, object_vocab, collision_layout)` triple into a
complete PuzzleScript file. The non-rule sections are *boilerplate
templated* off varislide's structure:

- `OBJECTS`: Background + Wall + Player + N typed mover/blocker objects,
  fixed sprite palette (single-pixel solid colors so we don't pollute the
  rule signal with sprite-decoding cost).
- `LEGEND`: 1-char alias per object.
- `SOUNDS`: empty.
- `COLLISIONLAYERS`: derived deterministically from object roles
  (`Background; Targets; Player, Movers, Wall`).
- `RULES`: the GP-mutated rule set.
- `WINCONDITIONS`: parametric (`no X`, `all X on Y`, `some X`) — a small
  enum, also subject to mutation since win-condition shape interacts with
  level design.
- `LEVELS`: filled in by the level inner loop (see below).

Object vocab is fixed-size (e.g. {Player, A, B, C, Wall}) per game so the
WM's input channels are stable across the whole curriculum — that's
deliberate, since channel reorganization would confound the
generalization signal.

## Co-evolution of levels (the bit you flagged)

Mutating a rule changes which levels are non-trivial. A `right [Player |
no Wall] -> [ | Player]` rule needs at least one open cell to the right of
the player; replacing `no Wall` with `Wall` makes every existing level a
no-op. So **rule mutations invalidate level libraries**.

Hierarchical loop:

```
for ruleset in rule_GA(parent_rulesets):
    levels = level_GA(ruleset, target_difficulty)   # uses synthetic_levels infra
    if levels is None:
        # ruleset is degenerate (no level achieves target difficulty);
        # discard or repair
        continue
    score = wm_proxy(ruleset, levels)               # or actual_wm_loss(...)
    record(ruleset, levels, score)
```

The inner level loop is **already built** as
`synthetic_levels._evolve_levels` + `_eval_candidate_fitness`. Adapting it
needs:

- Generalize `LevelGenerator` to take an *arbitrary just-compiled JSON
  spec* instead of an authored game — the empirical tile-pattern sampler
  needs at least a few seed levels to bootstrap from. For a fresh GP
  ruleset that has no authored levels, use a uniform/handcrafted prior
  for the first iteration of level GA, then bootstrap from any solvable
  levels found.
- Plumb the BFS validity gate (already exists,
  `synthetic_levels.search_validate`) — drop a candidate ruleset if no
  level of any reasonable size hits min_states / max_iters bounds.

A pragmatic shortcut for the first prototype: skip level GA entirely and
**pair every ruleset with one shared template** — a small open room
(say 8×8) with a few scattered objects of each typed kind, walls on the
border. One template is genuinely too narrow to exercise the full rule
grammar (vertical/perpendicular rules, multi-clause spatial separation,
ellipsis-across-distance all stay silent on it), but it's enough to get
the pipeline end-to-end, and we promote the level to a per-axis probe
library only once we observe the no-op rejection rate climbing.

To keep no-op mutations from polluting the curriculum, every (ruleset,
level) pair runs through a **transition-diff filter**:

1. Compile and BFS the level with the candidate ruleset → transition
   table T_with.
2. Compile and BFS the level with the candidate rule *removed* (or
   neutralized to identity) → T_without.
3. If `T_with == T_without` on the explored states, the rule is a no-op
   on this level — reject the pair.

This gives us no-op detection for free without designing the probe
library yet, and the rejection rate itself becomes the trigger for
expanding template coverage. The `puzzlescript_cpp` backend already
exposes `collect_transitions_bfs` (used by `synthetic_levels`), so this
is one extra BFS pass per candidate, not a new infrastructure piece.

## What the WM-side experiment looks like

Not in this doc's scope to design end-to-end, but the punchline test is:

1. Enumerate rule-grammar configurations across the axes table above (call
   it ~50–500 distinct rulesets, each with its own pinned level set).
2. Train one shared, conditional WM on a held-in subset (e.g. 80%).
3. Evaluate per-axis: held-out *prefix*, held-out *modifier*, held-out
   *clause-count*, held-out *ellipsis*, held-out *command*. Per-axis
   transfer is the actual rule-distillation metric, not aggregate held-out
   loss.

This is the "rule-conditioned generalization" plot the
ARCHITECTURE_REPORT keeps gesturing at without ever having a clean
test bed for. The varislide canary is a single point on the
"line_detector ellipsis × `again`" axis; rule-GP would let us draw the
whole surface.

## Reusing what already exists

| existing | reuse for |
| --- | --- |
| `puzzlescript_jax/syntax.lark` + `init_ps_lark_parser` | parsing + AST manipulation source |
| `puzzlescript_jax.preprocessing` | section detection, validity gate |
| `puzzlescript_cpp.CppPuzzleScriptBackend` | compile + serialize (already wired in `game_curriculum.py`) |
| `nca_wm.tokenize_game.{get_game_tree_from_js, tokenize_game}` | rule-tokenization for the WM (already conditional on rule grammar) |
| `nca_wm.synthetic_levels.LevelGenerator` + `_evolve_levels` | inner level loop |
| BFS / A* via cpp engine | validity + difficulty estimation |
| `nca_wm.train --conditional` | the WM that actually consumes the curriculum |
| `nca_wm/figures/varislide_*` plotting precedent | per-axis transfer plots |

The new code is ~3 modules: `rule_gp.py` (mutator + unparser),
`rule_game.py` (assembler + boilerplate templates), and a thin
`rule_curriculum.py` that wires the hierarchical loop. No LLM dependency,
no external API. CPU-only generation.

## Open design questions

1. **Object vocabulary: fixed or growing?** Fixed is cleaner for the
   transfer experiment but limits the rule space. Suggest fixed at the
   start (player + 3 typed objects + wall), revisit if we hit a coverage
   ceiling.
2. **Where does the `LEGEND` come from for assembled games?** Trivial
   1:1 char-to-object mapping is fine; only matters for human readability
   of `custom_games/` artifacts.
3. **How do we score a ruleset's curriculum value without training a WM?**
   The current proxy (search hardness + token diversity) doesn't measure
   what we care about. Two options:
   - cheap: per-axis *coverage* — pick rulesets that fill grammar cells
     not yet covered by the held-in set.
   - expensive: pair-eval — for each candidate ruleset, train a small WM
     and measure transfer to a fixed eval set.
   Suggest start with coverage, layer pair-eval when we have headline
   transfer numbers worth optimizing.
4. **Composability: do we ever combine rule-GP with LLM theming?** Could
   imagine GP-generated rule cores re-skinned with LLM-generated object
   names / sprites for human-facing demos. Not load-bearing for the
   distillation question.
5. **Rule equivalence classes**: structurally distinct rules can be
   semantically identical (e.g. `[> Player | A] -> [Player | >A]` vs.
   `right [Player | A] -> [Player | >A]` differ in surface but coincide
   on directional inputs). A canonicalizer that detects equivalence
   prevents the curriculum from over-counting variety. Probably not
   needed for a first pass — observed behavior collisions can be a
   downstream filter.

## Suggested next concrete step

Build the *minimum viable* path end-to-end before fleshing anything out:

1. Implement `rule_gp.py` with **just two mutators**: `set_modifier` and
   `add_command(again)`. (~80 LOC.)
2. Implement `rule_game.py` with the varislide-shaped boilerplate and a
   single 8×8 open-room template seeded with a few scattered typed
   objects. (~150 LOC.)
3. Write a script that enumerates ~16 rulesets across the cross-product
   of `{none, >, no} × {none, again}` × `{1,2}-cell LHS`, materializes
   them into `custom_games/`, runs BFS to filter for non-trivial
   solvability, applies the transition-diff no-op filter, and reports
   both the survival rate and the no-op rejection rate per axis.

If the no-op rate climbs above ~50% on any axis, that's the trigger to
start building the per-axis probe library — the survival rate alone
hides this because BFS-solvability and rule-firing are different
properties (a rule can be a no-op on a level the player can still beat
by ignoring it).

If that produces a working pipeline of 8–12 distinct, valid synthetic
games in an afternoon, the architecture is sound and we expand the
mutator catalog. If most rulesets compile but BFS rejects them, the level
template is wrong and we need to start the level inner loop sooner than
planned.

## Smoketest results (2026-05-03)

Built and ran the minimum-viable pipeline:
- `nca_wm/rule_gp.py` — Rule dataclass, unparser, four mutators
  (`set_object_modifier`, `truncate_cells`, `set_command`, `set_prefix`),
  `enumerate_smoketest_rulesets()`.
- `nca_wm/rule_game.py` — varislide-shaped boilerplate assembler with the
  8×8 open-room template.
- `nca_wm/scripts/rule_gp_smoke.py` — materialize + compile + BFS + diff
  against an empty-rules baseline.

Run command: `.venv/bin/python -m nca_wm.scripts.rule_gp_smoke`. Wall: ~3 s.

| metric | result |
| --- | --- |
| compile rate | **12/12** |
| no-op rate (of compiled) | **8/12 (67%)** |
| survivors | **4/12** |

Per-axis no-op rate:

| axis | value | no-ops / total |
| --- | --- | --- |
| `lhs_cells` | 1 | 6/6 |
| `lhs_cells` | 2 | 2/6 |
| `modifier`  | none | 2/4 |
| `modifier`  | `>` | 4/4 |
| `modifier`  | `no` | 2/4 |
| `command`   | none | 4/6 |
| `command`   | `again` | 4/6 |

The 4 survivors are the cells=2 rulesets without the `>` modifier. The
3 high-no-op axes confirm exactly the limitations of the single-template
shortcut:

- `lhs_cells=1` (6/6 no-op): truncating to one cell makes every rule
  `[Player] -> [Player]` (identity) — a trivially detectable no-op,
  filter working as intended.
- `modifier='>'` (4/4 no-op): `> ObjA` requires ObjA to *be moving*, but
  the level template has no rule that moves ObjA, so the LHS never
  matches. Probe library would need a level with already-moving objects
  (or a chained rule that creates motion).
- `command='again'` not differentiating: the rewrite `[A] -> [B]` doesn't
  change the LHS for the next pass, so `again` never re-fires here.
  Distinguishing `again` from no-`again` needs a rule whose RHS keeps
  the LHS true (e.g. a move-and-leave-behind rule).

**Engine quirk discovered during build:** the JS-side
`serializeCompiledState` raises `TypeError: Cannot read properties of
undefined (reading 'length')` when there are ≥3 typed objects with
single-character names that legend-alias to themselves. Multi-char object
names (`ObjA`/`ObjB`/`ObjC`) plus distinct single-char level pixels
(`A`/`B`/`C` → `ObjA`/etc.) sidesteps it. Encoded in `_OBJECT_PIXEL` in
`rule_game.py`. This is a JS engine bug worth tracking down separately;
the test passes the lark parser cleanly.

### Verdict

Architecture is sound — the no-op filter is doing exactly what it's
designed to do, and the per-axis numbers point unambiguously at which
template/probe extensions are needed next. The 67% no-op rate is above
the 50% promotion trigger from above; the next mutator-catalog expansion
should come *with* the per-axis probe library (motion-induction probe for
`>`, RHS-preserving rule for `again`), rather than just adding more rule
mutations to the same template.

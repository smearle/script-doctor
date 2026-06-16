"""Build a canonical mechanic graph from a PuzzleScript game tree.

The mechanic graph captures the *structure* of a game's rules, objects, collision
layers, and win conditions in a way that is invariant to object naming. Two games
with shuffled object/rule names but identical underlying mechanics will produce
the same canonical hash.

Key idea: objects are identified not by name but by their *structural role*:
  - Which collision layer they belong to
  - How they participate in rules (as what modifier, on which side, in which cell position)
  - Whether they appear in win conditions (and with what quantifier)

We build a node-and-edge-attributed multigraph, then compute a Weisfeiler-Lehman
hash for a canonical fingerprint.

Usage:
    from puzzlejax.mechanic_graph import build_mechanic_graph, canonical_hash
    G = build_mechanic_graph(tree)
    h = canonical_hash(G)
"""
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Optional

import networkx as nx

from puzzlescript_jax.ps_game import PSGameTree, Rule, RuleBlock, WinCondition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_rules_flat(rules) -> list[Rule]:
    out = []
    for r in rules:
        if isinstance(r, RuleBlock):
            out.extend(_iter_rules_flat(r.rules))
        elif isinstance(r, Rule):
            out.append(r)
    return out


def _is_in_loop(rules, target_rule: Rule) -> bool:
    """Check whether a rule lives inside a startloop/endloop block."""
    for r in rules:
        if isinstance(r, RuleBlock) and r.looping:
            flat = _iter_rules_flat(r.rules)
            if target_rule in flat:
                return True
    return False


def _iter_rule_tokens(node):
    if node is None:
        return
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if isinstance(cur, str):
            for tok in cur.split():
                yield tok.lower()
        elif isinstance(cur, Iterable):
            stack.extend(reversed(list(cur)))


MODIFIERS = {
    ">", "v", "<", "^", "no", "random", "randomdir",
    "horizontal", "vertical", "perpendicular", "parallel",
}
COMMANDS = {"again", "cancel", "checkpoint", "restart", "win"}
NON_OBJECT_TOKENS = MODIFIERS | COMMANDS | {"...", ""}


def _parse_kernel_cells(kernel):
    """Parse a kernel (list of rule_parts, each a list of cells, each a list of
    token strings) into a list of (cell_index, object_name, modifier_or_None) tuples.

    Kernel structure: [rule_part[cell[token_string]]]
    A rule_part is one [...] bracket group. Each cell within is separated by |.
    """
    entries = []
    if not kernel:
        return entries
    for part in kernel:
        for cell_i, cell in enumerate(part):
            if isinstance(cell, str):
                tokens = cell.split()
            elif isinstance(cell, list):
                tokens = []
                for t in cell:
                    tokens.extend(t.split() if isinstance(t, str) else [str(t)])
            else:
                continue

            # Parse modifier-object pairs from token list
            modifier = None
            for tok in tokens:
                tok_l = tok.lower()
                if tok_l in NON_OBJECT_TOKENS:
                    if tok_l in MODIFIERS:
                        modifier = tok_l
                    continue
                if tok_l == "...":
                    continue
                # This is an object name
                entries.append((cell_i, tok_l, modifier))
                modifier = None
    return entries


def _parse_kernel_commands(kernel):
    """Extract command tokens from a kernel."""
    cmds = []
    for tok in _iter_rule_tokens(kernel):
        if tok in COMMANDS:
            cmds.append(tok)
    return cmds


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_mechanic_graph(tree: PSGameTree) -> nx.MultiDiGraph:
    """Build a mechanic interaction graph from a parsed PuzzleScript game.

    Nodes: one per object (keyed by lowercase name).
    Node attributes:
        - layer_index: which collision layer (int)
        - layer_size: how many objects share that layer
        - is_player: whether name contains 'player' (heuristic, not used for hashing)
        - wc_quantifier: win condition quantifier if object is a wc source (or None)
        - wc_role: 'src', 'trg', or None

    Edges: one per rule interaction.
    Edge attributes:
        - rule_index: index of the rule in flattened rule list
        - side: 'lhs', 'rhs', or 'both' (if object appears on both sides)
        - cell_index: which cell within the rule pattern
        - modifier: directional/other modifier applied to this object
        - relation: 'same_cell', 'adjacent_cell', or 'cross_side'
        - in_loop: whether the rule is inside a loop block
        - prefixes: frozenset of rule prefixes

    There is also a special "GAME" node that holds global attributes.
    """
    G = nx.MultiDiGraph()

    # --- Global node for prelude flags ---
    prelude = tree.prelude
    G.add_node("__GAME__", type="game",
               noaction=prelude.noaction,
               require_player_movement=prelude.require_player_movement,
               run_rules_on_level_start=prelude.run_rules_on_level_start,
               has_flickscreen=prelude.flickscreen is not None,
               has_zoomscreen=prelude.zoomscreen is not None)

    # --- Object nodes ---
    # Build layer membership map
    obj_to_layer = {}
    for layer_i, layer in enumerate(tree.collision_layers):
        for obj_name in layer:
            obj_to_layer[obj_name.lower()] = layer_i

    for obj_name in tree.objects:
        name_l = obj_name.lower()
        layer_i = obj_to_layer.get(name_l, -1)
        layer_size = len(tree.collision_layers[layer_i]) if layer_i >= 0 else 0
        G.add_node(name_l, type="object",
                   layer_index=layer_i,
                   layer_size=layer_size,
                   wc_quantifier=None,
                   wc_role=None,
                   wc_has_target=False)

    # --- Win condition annotations ---
    for wc in tree.win_conditions:
        src = wc.src_obj.lower() if wc.src_obj else None
        trg = wc.trg_obj.lower() if wc.trg_obj else None
        # Ensure win condition objects exist as nodes (they may be legend aliases)
        for obj in (src, trg):
            if obj and obj not in G:
                G.add_node(obj, type="object", layer_index=-1, layer_size=0,
                           wc_quantifier=None, wc_role=None, wc_has_target=False)
        if src and src in G:
            G.nodes[src]["wc_quantifier"] = wc.quantifier
            G.nodes[src]["wc_role"] = "src"
            G.nodes[src]["wc_has_target"] = trg is not None
        if trg and trg in G:
            if G.nodes[trg]["wc_role"] is None:
                G.nodes[trg]["wc_role"] = "trg"
            # Add a wc edge
            if src:
                G.add_edge(src, trg, type="wincond", quantifier=wc.quantifier)

    # --- Rule edges ---
    flat_rules = _iter_rules_flat(tree.rules)
    for rule_i, rule in enumerate(flat_rules):
        prefixes = frozenset(p.lower() for p in rule.prefixes)
        in_loop = _is_in_loop(tree.rules, rule)

        lhs_entries = _parse_kernel_cells(rule.left_kernels)
        rhs_entries = _parse_kernel_cells(rule.right_kernels)
        lhs_cmds = _parse_kernel_commands(rule.right_kernels)
        if rule.command:
            lhs_cmds.extend(rule.command.lower().split())

        # Collect objects per cell per side
        lhs_by_cell = defaultdict(list)
        rhs_by_cell = defaultdict(list)
        for cell_i, obj, mod in lhs_entries:
            lhs_by_cell[cell_i].append((obj, mod))
        for cell_i, obj, mod in rhs_entries:
            rhs_by_cell[cell_i].append((obj, mod))

        all_lhs_objs = {obj for _, obj, _ in lhs_entries}
        all_rhs_objs = {obj for _, obj, _ in rhs_entries}
        all_objs = all_lhs_objs | all_rhs_objs

        # Ensure all referenced objects exist as nodes
        for obj in all_objs:
            if obj not in G:
                G.add_node(obj, type="object", layer_index=-1, layer_size=0,
                           wc_quantifier=None, wc_role=None, wc_has_target=False)

        # Same-cell interactions (within LHS)
        for cell_i, objs in lhs_by_cell.items():
            for i, (obj_a, mod_a) in enumerate(objs):
                for obj_b, mod_b in objs[i + 1:]:
                    G.add_edge(obj_a, obj_b, type="rule_same_cell",
                               rule_i=rule_i, side="lhs", cell_i=cell_i,
                               mod_a=mod_a, mod_b=mod_b,
                               prefixes=prefixes, in_loop=in_loop)

        # Same-cell interactions (within RHS)
        for cell_i, objs in rhs_by_cell.items():
            for i, (obj_a, mod_a) in enumerate(objs):
                for obj_b, mod_b in objs[i + 1:]:
                    G.add_edge(obj_a, obj_b, type="rule_same_cell",
                               rule_i=rule_i, side="rhs", cell_i=cell_i,
                               mod_a=mod_a, mod_b=mod_b,
                               prefixes=prefixes, in_loop=in_loop)

        # Adjacent-cell interactions (LHS cells with different indices)
        all_cells = sorted(lhs_by_cell.keys())
        for ci in range(len(all_cells)):
            for cj in range(ci + 1, len(all_cells)):
                for obj_a, mod_a in lhs_by_cell[all_cells[ci]]:
                    for obj_b, mod_b in lhs_by_cell[all_cells[cj]]:
                        G.add_edge(obj_a, obj_b, type="rule_adjacent_cell",
                                   rule_i=rule_i, side="lhs",
                                   cell_delta=all_cells[cj] - all_cells[ci],
                                   mod_a=mod_a, mod_b=mod_b,
                                   prefixes=prefixes, in_loop=in_loop)

        # Cross-side interactions: object changes between LHS and RHS
        for cell_i in set(lhs_by_cell) & set(rhs_by_cell):
            lhs_objs_in_cell = {o for o, _ in lhs_by_cell[cell_i]}
            rhs_objs_in_cell = {o for o, _ in rhs_by_cell[cell_i]}
            # Objects that appear/disappear
            created = rhs_objs_in_cell - lhs_objs_in_cell
            destroyed = lhs_objs_in_cell - rhs_objs_in_cell
            for obj in created:
                # Find what was in that cell on LHS (the "cause")
                for cause_obj, cause_mod in lhs_by_cell[cell_i]:
                    G.add_edge(cause_obj, obj, type="rule_creates",
                               rule_i=rule_i, cell_i=cell_i,
                               cause_mod=cause_mod,
                               prefixes=prefixes, in_loop=in_loop)
            for obj in destroyed:
                for effect_obj, effect_mod in rhs_by_cell[cell_i]:
                    G.add_edge(obj, effect_obj, type="rule_destroys",
                               rule_i=rule_i, cell_i=cell_i,
                               prefixes=prefixes, in_loop=in_loop)

            # Objects whose modifier changes (e.g., stationary -> moving)
            for obj in lhs_objs_in_cell & rhs_objs_in_cell:
                lhs_mod = next((m for o, m in lhs_by_cell[cell_i] if o == obj), None)
                rhs_mod = next((m for o, m in rhs_by_cell[cell_i] if o == obj), None)
                if lhs_mod != rhs_mod:
                    G.add_edge(obj, obj, type="rule_mod_change",
                               rule_i=rule_i, cell_i=cell_i,
                               mod_from=lhs_mod, mod_to=rhs_mod,
                               prefixes=prefixes, in_loop=in_loop)

        # Commands triggered by this rule
        for cmd in set(lhs_cmds):
            cmd_node = f"__CMD_{cmd.upper()}__"
            if cmd_node not in G:
                G.add_node(cmd_node, type="command", command=cmd)
            for obj in all_lhs_objs:
                G.add_edge(obj, cmd_node, type="rule_triggers_command",
                           rule_i=rule_i, command=cmd,
                           prefixes=prefixes, in_loop=in_loop)

    return G


# ---------------------------------------------------------------------------
# Canonical hash
# ---------------------------------------------------------------------------

def _node_label(G: nx.MultiDiGraph, node) -> str:
    """Compute a structural label for a node (ignoring its name)."""
    attrs = dict(G.nodes[node])
    ntype = attrs.get("type", "unknown")

    if ntype == "object":
        # Label by structural role, not name
        parts = [
            f"L{attrs.get('layer_index', -1)}",
            f"LS{attrs.get('layer_size', 0)}",
        ]
        wc_q = attrs.get("wc_quantifier")
        wc_r = attrs.get("wc_role")
        if wc_q:
            parts.append(f"WQ:{wc_q}")
        if wc_r:
            parts.append(f"WR:{wc_r}")
        if attrs.get("wc_has_target"):
            parts.append("WT")
        return f"OBJ({'|'.join(parts)})"

    elif ntype == "command":
        return f"CMD({attrs.get('command', '?')})"

    elif ntype == "game":
        flags = []
        for flag in sorted(["noaction", "require_player_movement",
                            "run_rules_on_level_start",
                            "has_flickscreen", "has_zoomscreen"]):
            if attrs.get(flag):
                flags.append(flag)
        return f"GAME({','.join(flags)})"

    return f"UNKNOWN({ntype})"


def _edge_label(G: nx.MultiDiGraph, u, v, key) -> str:
    """Compute a structural label for an edge."""
    attrs = dict(G.edges[u, v, key])
    etype = attrs.get("type", "unknown")

    if etype == "wincond":
        return f"WC({attrs.get('quantifier', '?')})"

    elif etype == "rule_same_cell":
        return (f"SC(s={attrs.get('side')},ma={attrs.get('mod_a')},mb={attrs.get('mod_b')},"
                f"p={_prefix_str(attrs)},lp={attrs.get('in_loop', False)})")

    elif etype == "rule_adjacent_cell":
        return (f"AC(s={attrs.get('side')},d={attrs.get('cell_delta')},"
                f"ma={attrs.get('mod_a')},mb={attrs.get('mod_b')},"
                f"p={_prefix_str(attrs)},lp={attrs.get('in_loop', False)})")

    elif etype == "rule_creates":
        return (f"CR(cm={attrs.get('cause_mod')},"
                f"p={_prefix_str(attrs)},lp={attrs.get('in_loop', False)})")

    elif etype == "rule_destroys":
        return (f"DE(p={_prefix_str(attrs)},lp={attrs.get('in_loop', False)})")

    elif etype == "rule_mod_change":
        return (f"MC(f={attrs.get('mod_from')},t={attrs.get('mod_to')},"
                f"p={_prefix_str(attrs)},lp={attrs.get('in_loop', False)})")

    elif etype == "rule_triggers_command":
        return (f"TC(c={attrs.get('command')},"
                f"p={_prefix_str(attrs)},lp={attrs.get('in_loop', False)})")

    return f"E({etype})"


def _prefix_str(attrs) -> str:
    pfx = attrs.get("prefixes", frozenset())
    return "+".join(sorted(pfx)) if pfx else "_"


def canonical_hash(G: nx.MultiDiGraph, iterations: int = 5) -> str:
    """Compute a Weisfeiler-Lehman-style hash of the mechanic graph.

    This is invariant to node naming — two isomorphic graphs produce the
    same hash. Uses iterative neighborhood aggregation on structural labels.

    Args:
        G: mechanic graph from build_mechanic_graph
        iterations: WL refinement iterations (more = finer discrimination)

    Returns:
        hex digest string
    """
    # Initialize node labels from structural attributes
    labels = {}
    for node in G.nodes():
        labels[node] = _node_label(G, node)

    for _ in range(iterations):
        new_labels = {}
        for node in G.nodes():
            # Gather sorted neighbor labels with edge labels
            neighbor_parts = []
            for _, nbr, key in G.out_edges(node, keys=True):
                el = _edge_label(G, node, nbr, key)
                neighbor_parts.append(f"{el}->{labels[nbr]}")
            for pred, _, key in G.in_edges(node, keys=True):
                el = _edge_label(G, pred, node, key)
                neighbor_parts.append(f"{labels[pred]}->{el}")

            neighbor_parts.sort()
            combined = f"{labels[node]}|{'|'.join(neighbor_parts)}"
            new_labels[node] = hashlib.sha256(combined.encode()).hexdigest()[:16]
        labels = new_labels

    # Final hash: sorted multiset of all node labels
    all_labels = sorted(labels.values())
    final = hashlib.sha256("|".join(all_labels).encode()).hexdigest()
    return final


def canonical_label_vector(G: nx.MultiDiGraph, iterations: int = 5) -> Counter:
    """Like canonical_hash but returns the full multiset of WL node labels.

    This is useful for computing graph similarity (e.g., via multiset intersection)
    rather than exact equality.
    """
    labels = {}
    for node in G.nodes():
        labels[node] = _node_label(G, node)

    for _ in range(iterations):
        new_labels = {}
        for node in G.nodes():
            neighbor_parts = []
            for _, nbr, key in G.out_edges(node, keys=True):
                el = _edge_label(G, node, nbr, key)
                neighbor_parts.append(f"{el}->{labels[nbr]}")
            for pred, _, key in G.in_edges(node, keys=True):
                el = _edge_label(G, pred, node, key)
                neighbor_parts.append(f"{labels[pred]}->{el}")

            neighbor_parts.sort()
            combined = f"{labels[node]}|{'|'.join(neighbor_parts)}"
            new_labels[node] = hashlib.sha256(combined.encode()).hexdigest()[:16]
        labels = new_labels

    return Counter(labels.values())


# ---------------------------------------------------------------------------
# Mechanic similarity
# ---------------------------------------------------------------------------

def mechanic_similarity(G1: nx.MultiDiGraph, G2: nx.MultiDiGraph,
                        iterations: int = 5) -> float:
    """Compute similarity between two mechanic graphs as normalized multiset
    intersection of their WL label vectors.

    Returns a value in [0, 1] where 1 means identical mechanics.
    """
    v1 = canonical_label_vector(G1, iterations)
    v2 = canonical_label_vector(G2, iterations)
    intersection = sum((v1 & v2).values())
    union = sum((v1 | v2).values())
    return intersection / union if union > 0 else 1.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import pickle
    import sys
    import traceback

    from puzzlescript_jax.gen_tree import GenPSTree
    from puzzlescript_jax.globals import TREES_DIR
    from puzzlescript_jax.utils import get_list_of_games_for_testing

    ap = argparse.ArgumentParser(description="Build mechanic graphs and find duplicates")
    ap.add_argument("--dataset", default="pedro")
    ap.add_argument("--find-duplicates", action="store_true",
                    help="Find games with identical canonical hashes")
    ap.add_argument("--show-graph", type=str, default=None,
                    help="Print graph details for a specific game")
    args = ap.parse_args()

    games = get_list_of_games_for_testing(dataset=args.dataset, include_random=True)

    if args.show_graph:
        games = [args.show_graph]

    hash_to_games = defaultdict(list)
    failures = []

    for game in games:
        try:
            pkl_path = os.path.join(TREES_DIR, game + ".pkl")
            if not os.path.exists(pkl_path):
                failures.append((game, "no pkl"))
                continue
            with open(pkl_path, "rb") as f:
                min_tree = pickle.load(f)
            tree = GenPSTree().transform(min_tree)
            G = build_mechanic_graph(tree)
            h = canonical_hash(G)
            hash_to_games[h].append(game)

            if args.show_graph:
                print(f"\n=== {game} ===")
                print(f"Canonical hash: {h}")
                print(f"Nodes ({G.number_of_nodes()}):")
                for node in sorted(G.nodes()):
                    print(f"  {node}: {_node_label(G, node)}")
                print(f"Edges ({G.number_of_edges()}):")
                for u, v, k in sorted(G.edges(keys=True)):
                    print(f"  {u} -> {v}: {_edge_label(G, u, v, k)}")

        except Exception as e:
            failures.append((game, str(e)))

    if args.find_duplicates:
        print(f"\nProcessed {len(games) - len(failures)} games, {len(failures)} failures")
        dupes = {h: gs for h, gs in hash_to_games.items() if len(gs) > 1}
        if dupes:
            print(f"\nFound {len(dupes)} groups of mechanically identical games:")
            for h, gs in sorted(dupes.items(), key=lambda x: -len(x[1])):
                print(f"  [{len(gs)} games] hash={h[:12]}... : {', '.join(gs)}")
        else:
            print("No exact mechanical duplicates found.")

    if failures and not args.show_graph:
        print(f"\n{len(failures)} failures (first 10):")
        for g, reason in failures[:10]:
            print(f"  {g}: {reason}")

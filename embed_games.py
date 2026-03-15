"""Extract structural feature vectors from PuzzleScript games parsed via Lark.

Usage:
    python embed_games.py [--dataset pedro] [--out embeddings.npz]

Produces a .npz file with:
    names:      (N,) array of game names
    features:   (N, D) float32 feature matrix
    feat_names: (D,) array of feature names
"""
import argparse
import json
import os
import pickle
import sys
import traceback
from collections import Counter
from collections.abc import Iterable
from typing import List

import numpy as np

from puzzlescript_jax.gen_tree import GenPSTree
from puzzlescript_jax.globals import TREES_DIR
from puzzlescript_jax.ps_game import PSGameTree, Rule, RuleBlock, WinCondition
from puzzlescript_jax.utils import (
    get_list_of_games_for_testing,
    init_ps_lark_parser,
)
from puzzlescript_jax.preprocessing import get_tree_from_txt


# ---------------------------------------------------------------------------
# Helpers for walking the rule tree
# ---------------------------------------------------------------------------

def _iter_rules_flat(rules) -> list[Rule]:
    """Flatten RuleBlock nesting into a flat list of Rule objects."""
    out = []
    for r in rules:
        if isinstance(r, RuleBlock):
            out.extend(_iter_rules_flat(r.rules))
        elif isinstance(r, Rule):
            out.append(r)
    return out


def _count_rules_in_loops(rules) -> int:
    """Count rules that are inside a startloop/endloop block."""
    count = 0
    for r in rules:
        if isinstance(r, RuleBlock) and r.looping:
            count += len(_iter_rules_flat(r.rules))
    return count


def _iter_rule_tokens(node):
    """Yield all string tokens from a nested rule kernel structure."""
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


def _rule_objects(rule: Rule) -> set[str]:
    """Return set of object names referenced in a rule (both sides)."""
    objs = set()
    for tok in _iter_rule_tokens(rule.left_kernels):
        if tok not in NON_OBJECT_TOKENS:
            objs.add(tok)
    for tok in _iter_rule_tokens(rule.right_kernels):
        if tok not in NON_OBJECT_TOKENS:
            objs.add(tok)
    return objs


def _rule_object_pairs(rule: Rule) -> set[tuple[str, str]]:
    """Return pairs of objects that co-occur in the same rule."""
    objs = sorted(_rule_objects(rule))
    pairs = set()
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            pairs.add((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Prefix / modifier / command vocabularies
# ---------------------------------------------------------------------------

RULE_PREFIXES = [
    "late", "horizontal", "vertical", "left", "right", "up", "down",
    "random", "perpendicular", "parallel", "rigid",
]

OBJECT_MODIFIERS = {
    ">", "v", "<", "^",  # directional arrows
    "no", "random", "randomdir",
    "horizontal", "vertical",
    "perpendicular", "parallel",
}


COMMANDS = ["again", "cancel", "checkpoint", "restart", "win"]

WIN_QUANTIFIERS = ["all", "some", "no", "any"]

# Tokens that are not object names (modifiers, commands, ellipsis)
NON_OBJECT_TOKENS = OBJECT_MODIFIERS | set(COMMANDS) | {"...", ""}

PRELUDE_FLAGS = [
    "noaction", "noundo", "require_player_movement",
    "run_rules_on_level_start",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(tree: PSGameTree) -> dict[str, float]:
    """Extract a dict of named features from a parsed game tree."""
    feats = {}

    # --- Scale features ---
    n_objects = len(tree.objects)
    n_layers = len(tree.collision_layers)
    flat_rules = _iter_rules_flat(tree.rules)
    n_rules = len(flat_rules)
    n_rules_in_loops = _count_rules_in_loops(tree.rules)
    n_winconds = len(tree.win_conditions)
    n_levels = len(tree.levels)

    feats["n_objects"] = n_objects
    feats["n_layers"] = n_layers
    feats["n_rules"] = n_rules
    feats["n_rules_in_loops"] = n_rules_in_loops
    feats["n_winconds"] = n_winconds
    feats["n_levels"] = n_levels

    # --- Layer statistics ---
    layer_sizes = [len(layer) for layer in tree.collision_layers]
    feats["max_layer_size"] = max(layer_sizes) if layer_sizes else 0
    feats["mean_layer_size"] = np.mean(layer_sizes) if layer_sizes else 0

    # --- Rule prefix distribution ---
    prefix_counts = Counter()
    for rule in flat_rules:
        for p in rule.prefixes:
            prefix_counts[p.lower()] += 1

    for p in RULE_PREFIXES:
        feats[f"prefix_{p}"] = prefix_counts.get(p, 0)

    # --- Object modifier distribution (from rule kernels) ---
    modifier_counts = Counter()
    for rule in flat_rules:
        for tok in _iter_rule_tokens(rule.left_kernels):
            if tok in OBJECT_MODIFIERS:
                modifier_counts[tok] += 1
        for tok in _iter_rule_tokens(rule.right_kernels):
            if tok in OBJECT_MODIFIERS:
                modifier_counts[tok] += 1

    for m in OBJECT_MODIFIERS:
        feats[f"modifier_{m}"] = modifier_counts.get(m, 0)

    # --- Command usage ---
    cmd_counts = Counter()
    for rule in flat_rules:
        if rule.command:
            for tok in rule.command.lower().split():
                cmd_counts[tok] += 1
        # Also check right kernels for command tokens
        for tok in _iter_rule_tokens(rule.right_kernels):
            if tok in COMMANDS:
                cmd_counts[tok] += 1

    for c in COMMANDS:
        feats[f"cmd_{c}"] = cmd_counts.get(c, 0)

    # --- Win condition type distribution ---
    for q in WIN_QUANTIFIERS:
        feats[f"wc_{q}"] = sum(1 for wc in tree.win_conditions if wc.quantifier == q)

    feats["wc_has_on"] = sum(1 for wc in tree.win_conditions if wc.trg_obj is not None)
    feats["wc_no_on"] = sum(1 for wc in tree.win_conditions if wc.trg_obj is None)

    # --- Rule interaction density ---
    all_rule_objects = set()
    all_pairs = set()
    for rule in flat_rules:
        objs = _rule_objects(rule)
        all_rule_objects.update(objs)
        all_pairs.update(_rule_object_pairs(rule))

    n_rule_objs = len(all_rule_objects)
    max_pairs = n_rule_objs * (n_rule_objs - 1) / 2 if n_rule_objs > 1 else 1
    feats["rule_interaction_density"] = len(all_pairs) / max_pairs

    # --- Rule structural features ---
    n_cells_lhs = []
    n_cells_rhs = []
    for rule in flat_rules:
        if rule.left_kernels:
            n_cells_lhs.append(len(rule.left_kernels))
        if rule.right_kernels:
            n_cells_rhs.append(len(rule.right_kernels))

    feats["mean_cells_lhs"] = np.mean(n_cells_lhs) if n_cells_lhs else 0
    feats["max_cells_lhs"] = max(n_cells_lhs) if n_cells_lhs else 0
    feats["mean_cells_rhs"] = np.mean(n_cells_rhs) if n_cells_rhs else 0

    # Rules with commands but no RHS pattern (command-only rules)
    feats["n_command_only_rules"] = sum(
        1 for r in flat_rules if r.command and not r.right_kernels
    )

    # --- Prelude flags ---
    prelude = tree.prelude
    feats["has_noaction"] = float(prelude.noaction)
    feats["has_require_player_movement"] = float(prelude.require_player_movement)
    feats["has_run_rules_on_level_start"] = float(prelude.run_rules_on_level_start)
    feats["has_flickscreen"] = float(prelude.flickscreen is not None)
    feats["has_zoomscreen"] = float(prelude.zoomscreen is not None)

    # --- Legend complexity ---
    # tree.legend can be a dict (key -> LegendEntry) or list of LegendEntry
    legend_entries = tree.legend.values() if isinstance(tree.legend, dict) else tree.legend
    n_or_legends = 0
    n_and_legends = 0
    for le in legend_entries:
        if hasattr(le, "operator") and le.operator:
            op = le.operator.lower()
            if op == "or":
                n_or_legends += 1
            elif op == "and":
                n_and_legends += 1
    feats["n_or_legends"] = n_or_legends
    feats["n_and_legends"] = n_and_legends

    # --- Level size statistics ---
    if tree.levels:
        level_areas = []
        for lev in tree.levels:
            try:
                area = len(lev) * (len(lev[0]) if len(lev) > 0 else 0)
            except (TypeError, IndexError):
                area = 0
            level_areas.append(area)
        feats["mean_level_area"] = np.mean(level_areas)
        feats["max_level_area"] = max(level_areas)
    else:
        feats["mean_level_area"] = 0
        feats["max_level_area"] = 0

    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser_arg = argparse.ArgumentParser(description="Extract PuzzleScript game embeddings")
    parser_arg.add_argument("--dataset", default="pedro", choices=["priority", "gallery", "pedro", "increpare"])
    parser_arg.add_argument("--out", default="embeddings.npz")
    parser_arg.add_argument("--reparse", action="store_true", help="Re-parse from .txt instead of using cached .pkl trees")
    args = parser_arg.parse_args()

    games = get_list_of_games_for_testing(dataset=args.dataset, include_random=True)
    print(f"Processing {len(games)} games from dataset '{args.dataset}'")

    lark_parser = None
    if args.reparse:
        lark_parser = init_ps_lark_parser()

    results = {}
    failures = []

    for i, game in enumerate(games):
        try:
            tree = None

            # Try loading cached tree first
            if not args.reparse:
                pkl_path = os.path.join(TREES_DIR, game + ".pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as f:
                        min_tree = pickle.load(f)
                    tree = GenPSTree().transform(min_tree)

            # Fall back to parsing from txt
            if tree is None:
                if lark_parser is None:
                    lark_parser = init_ps_lark_parser()
                tree, status, err = get_tree_from_txt(lark_parser, game, test_env_init=False)
                if tree is None:
                    failures.append((game, str(status)))
                    continue

            feats = extract_features(tree)
            results[game] = feats

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(games)}] extracted features for {game}")

        except Exception as e:
            failures.append((game, str(e)))
            traceback.print_exc()
            continue

    if not results:
        print("No games successfully processed!")
        sys.exit(1)

    # Build arrays
    names = sorted(results.keys())
    feat_names = sorted(results[names[0]].keys())
    features = np.zeros((len(names), len(feat_names)), dtype=np.float32)

    for i, name in enumerate(names):
        for j, fn in enumerate(feat_names):
            features[i, j] = results[name].get(fn, 0.0)

    np.savez(args.out, names=names, features=features, feat_names=feat_names)
    print(f"\nSaved {len(names)} game embeddings ({len(feat_names)} features) to {args.out}")

    if failures:
        print(f"\n{len(failures)} games failed:")
        for game, reason in failures[:20]:
            print(f"  {game}: {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()

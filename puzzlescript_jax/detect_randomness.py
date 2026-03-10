from collections.abc import Iterable

from puzzlescript_jax.ps_game import PSGameTree, Rule, RuleBlock


RANDOM_RULE_PREFIXES = frozenset({"random"})
RANDOM_RULE_TOKENS = frozenset({"random", "randomdir"})


def _iter_rule_tokens(node):
    if node is None:
        return

    stack = [node]
    while stack:
        cur = stack.pop()
        if cur is None:
            continue
        if isinstance(cur, str):
            for token in cur.split():
                yield token.lower()
            continue
        if isinstance(cur, Iterable):
            stack.extend(reversed(list(cur)))


def rule_has_randomness(rule: Rule) -> bool:
    prefixes = {prefix.lower() for prefix in rule.prefixes}
    if prefixes & RANDOM_RULE_PREFIXES:
        return True

    return any(token in RANDOM_RULE_TOKENS for token in _iter_rule_tokens(rule.right_kernels))


def rule_block_has_randomness(rule_block: RuleBlock) -> bool:
    for rule in rule_block.rules:
        if hasattr(rule, "rules"):
            if rule_block_has_randomness(rule):
                return True
        elif rule_has_randomness(rule):
            return True
    return False


def tree_has_randomness(tree: PSGameTree) -> bool:
    for rule in tree.rules:
        if hasattr(rule, "rules"):
            if rule_block_has_randomness(rule):
                return True
        elif rule_has_randomness(rule):
            return True
    return False
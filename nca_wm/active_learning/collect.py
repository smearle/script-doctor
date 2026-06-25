"""Data-collection policies and fixed-dataset builders for online-vs-offline.

A *collection policy* chooses actions while rolling out haoo' sequences. The
value of a collection scheme is how well a world model trained on its data
predicts held-out informative transitions, at a matched budget.

Policies (shared signature ``policy(masks, rng, *, history_ids, engine)``):
  - random_policy      : uniform over actions (offline / passive baseline)
  - navigate_policy    : walk toward the nearest seed, then bump it (informative
                         upper bound; needs only observable seed positions)
  - make_ig_policy(...) : greedy / shallow-expectimax IG planner (the learned
                          active learner)
"""
from __future__ import annotations

import random

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.data import sample_training_tokens


def _positions(masks: list[int], bit: int) -> list[tuple[int, int]]:
    out = []
    for i, m in enumerate(masks):
        if m & (1 << bit):
            out.append((i % V.GRID_W, i // V.GRID_W))  # (x, y)
    return out


def random_policy(masks, r, **kw) -> str:
    return r.choice(V.ACTIONS)


# Direction action -> (dx, dy) in grid coords (y down). Engine input ids:
# 0=up 1=left 2=down 3=right (see vocab.ACTION_TO_INPUT).
_DIR_DELTA = {V.UP: (0, -1), V.DOWN: (0, 1), V.LEFT: (-1, 0), V.RIGHT: (1, 0)}


def navigate_policy(masks, r, **kw) -> str:
    """Greedy walk toward the nearest seed; bump it once adjacent."""
    pbit, sbit = V.NAME_TO_BIT["player"], V.NAME_TO_BIT["seed"]
    players = _positions(masks, pbit)
    seeds = _positions(masks, sbit)
    if not players or not seeds:
        return r.choice(V.ACTIONS)
    px, py = players[0]
    sx, sy = min(seeds, key=lambda s: abs(s[0] - px) + abs(s[1] - py))
    dist = abs(sx - px) + abs(sy - py)
    if dist <= 1:
        # Adjacent: bump into the seed (its direction) to keep triggering.
        for a, (dx, dy) in _DIR_DELTA.items():
            if (px + dx, py + dy) == (sx, sy):
                return a
        return r.choice(V.ACTIONS)
    # Move along the axis with the larger gap, reducing Manhattan distance.
    cand = []
    if sx != px:
        cand.append(V.RIGHT if sx > px else V.LEFT)
    if sy != py:
        cand.append(V.DOWN if sy > py else V.UP)
    return r.choice(cand)


def make_ig_policy(model, device, depth: int = 3, n_chance: int = 4,
                   greedy: bool = False, epsilon: float = 0.1):
    """Collection policy that picks actions by predicted information gain.

    greedy=True -> depth-1 (cheap, myopic); else shallow expectimax (can plan
    multi-step navigation toward informative regions, like the prototype walking
    to the TV before pressing space). epsilon-random keeps coverage / breaks ties.
    """
    from nca_wm.active_learning.inference import estimate_information_gain
    from nca_wm.active_learning.planner import PlannerConfig, plan_action

    cfg = PlannerConfig(depth=1 if greedy else depth, n_chance=n_chance)

    def policy(masks, r, *, history_ids, engine=None, **kw) -> str:
        if r.random() < epsilon:
            return r.choice(V.ACTIONS)
        if greedy:
            igs = {a: estimate_information_gain(model, history_ids, a, device,
                                                n_samples=n_chance)
                   for a in V.ACTIONS}
            return max(igs, key=igs.get)
        best, _ = plan_action(model, history_ids, cfg, device)
        return best

    return policy


def make_mixed_policy(p_navigate: float = 0.5):
    """Per-step mix of navigate (reaches+pushes the box) and random (coverage)."""
    def policy(masks, r, **kw):
        return (navigate_policy if r.random() < p_navigate else random_policy)(masks, r, **kw)
    return policy


def is_push_action(masks, action: str) -> bool:
    """True if `action` moves the player into an adjacent box (a push attempt)."""
    if action not in _DIR_DELTA:
        return False
    pbit, sbit = V.NAME_TO_BIT["player"], V.NAME_TO_BIT["seed"]
    players, boxes = _positions(masks, pbit), _positions(masks, sbit)
    if not players or not boxes:
        return False
    px, py = players[0]
    dx, dy = _DIR_DELTA[action]
    return (px + dx, py + dy) in set(boxes)


def build_dataset(family: W.WorldFamily, policy, n_seqs: int,
                  min_prefix_steps: int = 4, max_prefix_steps: int = 12,
                  max_seq_len: int = 400, seed: int = 0) -> list[list[int]]:
    """Generate `n_seqs` haoo' token-id sequences under `policy`."""
    r = random.Random(seed)
    out: list[list[int]] = []
    while len(out) < n_seqs:
        toks = sample_training_tokens(family, min_prefix_steps, max_prefix_steps,
                                      rng=r, policy=policy)
        ids = V.encode(toks)
        if len(ids) <= max_seq_len:
            out.append(ids)
    return out

"""Depth-limited expectimax over the learned model, IG as per-step reward.

Per-step chance reward r(h,a,o) = log q1(o|h,a,o) - log q0(o|h,a). Ported from
the prototype; vocab/import swapped only.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from nca_wm.active_learning.inference import (logprob_of_sequence,
                                              sample_obs_block)
from nca_wm.active_learning.vocab import ACT, ACTIONS, RESAMPLE_OBS, STOI


@dataclass
class PlannerConfig:
    depth: int = 2
    n_chance: int = 4
    discount: float = 1.0
    temperature: float = 1.0


def _step_reward_and_obs(model, history_ids, action_token, cfg, device):
    act_id, a_id, rs_id = STOI[ACT], STOI[action_token], STOI[RESAMPLE_OBS]
    prefix_q0 = history_ids + [act_id, a_id]
    out = []
    for _ in range(cfg.n_chance):
        o_block = sample_obs_block(model, prefix_q0, device, temperature=cfg.temperature)
        log_q0 = logprob_of_sequence(model, prefix_q0, o_block, device)
        prefix_q1 = prefix_q0 + o_block + [rs_id]
        log_q1 = logprob_of_sequence(model, prefix_q1, o_block, device)
        out.append((o_block, log_q1 - log_q0))
    return out


@torch.no_grad()
def plan_action(model, history_ids, cfg, device, actions=None):
    actions = actions if actions is not None else ACTIONS
    values: dict[str, float] = {}

    def expectimax(h_ids, depth):
        if depth == 0:
            return 0.0
        best = -float("inf")
        for a in actions:
            children = _step_reward_and_obs(model, h_ids, a, cfg, device)
            tot = 0.0
            for o_block, r in children:
                next_h = h_ids + [STOI[ACT], STOI[a]] + o_block
                v_next = expectimax(next_h, depth - 1) if depth > 1 else 0.0
                tot += r + cfg.discount * v_next
            value = tot / max(len(children), 1)
            best = max(best, value)
            if depth == cfg.depth:
                values[a] = value
        return best

    expectimax(history_ids, cfg.depth)
    best_action = max(values, key=values.get)
    return best_action, values

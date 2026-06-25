"""IG estimator + obs-block sampling/scoring (ported from the prototype).

    IG(h, a) = E_{o ~ q0(o|h,a)} [ log q1(o | h, a, o) - log q0(o | h, a) ]

q0 is the next-obs distribution right after `<h> ACT <a>`; q1 is the distribution
right after `<h> ACT <a> OBS <o> END_OBS RESAMPLE_OBS`. The logic is token-
agnostic; only the vocab import differs from the prototype.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning.model import TinyTransformerLM
from nca_wm.active_learning.vocab import (ACT, END_OBS, OBS, PAD_ID,
                                          RESAMPLE_OBS, STOI)


@torch.no_grad()
def _next_logits(model, ids):
    return model(ids)[0, -1, :]


@torch.no_grad()
def sample_obs_block(model, prefix_ids, device, max_obs_len=None,
                     temperature=1.0):
    # An obs block is exactly OBS + N_CELLS cell tokens + END_OBS; read the
    # current geometry at call time so different families work without reimport.
    if max_obs_len is None:
        max_obs_len = V.N_CELLS + 2
    end_obs_id, obs_id = STOI[END_OBS], STOI[OBS]
    out = [obs_id]
    ids = torch.tensor(prefix_ids + out, dtype=torch.long, device=device)[None, :]
    for _ in range(max_obs_len):
        logits = _next_logits(model, ids)
        if temperature != 1.0:
            logits = logits / temperature
        logits[PAD_ID] = -1e9
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1).item())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == end_obs_id:
            break
    else:
        out.append(end_obs_id)
    return out


@torch.no_grad()
def logprob_of_sequence(model, prefix_ids, block_ids, device):
    if len(block_ids) == 0:
        return 0.0
    full = prefix_ids + block_ids
    ids = torch.tensor(full, dtype=torch.long, device=device)[None, :]
    logits = model(ids)
    start = len(prefix_ids) - 1
    end = start + len(block_ids)
    log_probs = F.log_softmax(logits[0, start:end, :], dim=-1)
    targets = torch.tensor(block_ids, dtype=torch.long, device=device)
    return log_probs.gather(-1, targets[:, None]).squeeze(-1).sum().item()


@torch.no_grad()
def estimate_information_gain(model, history_ids, action_token, device,
                              n_samples=16, temperature=1.0,
                              return_samples=False):
    act_id, a_id, rs_id = STOI[ACT], STOI[action_token], STOI[RESAMPLE_OBS]
    prefix_q0 = history_ids + [act_id, a_id]
    total = 0.0
    diagnostics = []
    for _ in range(n_samples):
        o_block = sample_obs_block(model, prefix_q0, device, temperature=temperature)
        log_q0 = logprob_of_sequence(model, prefix_q0, o_block, device)
        prefix_q1 = prefix_q0 + o_block + [rs_id]
        log_q1 = logprob_of_sequence(model, prefix_q1, o_block, device)
        total += (log_q1 - log_q0)
        if return_samples:
            diagnostics.append({"o_block": o_block, "log_q0": log_q0,
                                "log_q1": log_q1, "delta": log_q1 - log_q0})
    ig = total / max(n_samples, 1)
    return (ig, diagnostics) if return_samples else ig

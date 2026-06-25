"""haoo' training-sequence generation, engine-backed.

One sequence:

    BOS  OBS <obs0> END_OBS
         ( ACT <a> OBS <obs> END_OBS )* T times
         RESAMPLE_OBS OBS <obs'> END_OBS

The final RESAMPLE block is an i.i.d. redraw of the last observation from the
same pre-action engine snapshot under the same hidden world, obtained via
backup/restore + a different RNG seed. The rollout policy depends ONLY on the
observable history, so a_t is independent of theta given h_t.
"""
from __future__ import annotations

import random
from typing import Optional

import torch
from torch.utils.data import IterableDataset

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W


def _policy(masks: list[int], r: random.Random, **kw) -> str:
    """Default observable-history-only exploration policy.

    Mild bias toward ACTION when any seed (bit) is still present, else uniform.
    Extra kwargs (history_ids, engine) are accepted+ignored so all policies share
    one signature ``policy(masks, rng, *, history_ids, engine) -> action``.
    """
    seed_bit = V.NAME_TO_BIT["seed"]
    has_seed = any(m & (1 << seed_bit) for m in masks)
    if has_seed and r.random() < 0.4:
        return V.ACTION
    return r.choice(V.ACTIONS)


def sample_training_tokens(family: W.WorldFamily,
                           min_prefix_steps: int = 1,
                           max_prefix_steps: int = 8,
                           rng: Optional[random.Random] = None,
                           policy=_policy) -> list[str]:
    r = rng if rng is not None else random
    mech = r.choice(family.mechanisms)
    li = r.randrange(family.n_layouts)
    eng = W._new_engine(family.jsons[mech], li)
    id_to_bit = W._engine_id_to_canon_bit(eng)

    obs = W.read_obs(eng, id_to_bit)
    tokens: list[str] = [V.BOS]
    tokens.extend(V.serialize_obs(obs))

    T = r.randint(max(1, min_prefix_steps), max_prefix_steps)
    last_action = None
    bak = None
    for t in range(T):
        a = policy(obs, r, history_ids=V.encode(tokens), engine=eng)
        if t == T - 1:
            bak = eng.backup_level()  # snapshot pre-final-action
            last_action = a
        W.step_engine(eng, a, seed=str(r.getrandbits(48)))
        obs = W.read_obs(eng, id_to_bit)
        tokens.extend(V.serialize_action(a))
        tokens.extend(V.serialize_obs(obs))

    # i.i.d. redraw of the last obs from the same pre-action snapshot.
    eng.restore_level(bak)
    W.step_engine(eng, last_action, seed=str(r.getrandbits(48)))
    obs_resampled = W.read_obs(eng, id_to_bit)
    tokens.append(V.RESAMPLE_OBS)
    tokens.extend(V.serialize_obs(obs_resampled))
    return tokens


def pad_batch(seqs, max_len):
    B = len(seqs)
    out = torch.full((B, max_len), V.PAD_ID, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = min(len(s), max_len)
        out[i, :L] = torch.tensor(s[:L], dtype=torch.long)
        mask[i, :L] = True
    return out, mask


class HaooDataset(IterableDataset):
    def __init__(self, family: W.WorldFamily, min_prefix_steps: int = 1,
                 max_prefix_steps: int = 8, max_seq_len: int = 320, seed: int = 0,
                 policy=_policy):
        super().__init__()
        self.family = family
        self.min_prefix_steps = min_prefix_steps
        self.max_prefix_steps = max_prefix_steps
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.policy = policy

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        rng = random.Random(self.seed + 10_000 * wid + 1)
        while True:
            toks = sample_training_tokens(
                self.family, self.min_prefix_steps, self.max_prefix_steps, rng,
                policy=self.policy)
            ids = V.encode(toks)
            if len(ids) > self.max_seq_len:
                continue
            yield ids


def collate(batch):
    return pad_batch(batch, max(len(s) for s in batch))

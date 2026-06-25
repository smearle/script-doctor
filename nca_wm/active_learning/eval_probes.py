"""Fixed probe scenarios with known IG sign, for charting during training.

We build conditioning histories by rolling REAL worlds of a known mechanism for
k steps (recording only observable tokens), then estimate IG of a candidate
action. Expected pattern once the model has learned:

  fresh_unknown : HIGH  (mechanism unresolved -> acting reveals which world)
  rand_known    : ~0    (model knows it's stochastic; residual noise is
                         irreducible, carries no info about theta)
  det_known     : ~0    (deterministic and already known)
  noop_known    : ~0    (deterministic and already known)
"""
from __future__ import annotations

import random

from nca_wm.active_learning import vocab as V
from nca_wm.active_learning import worlds as W
from nca_wm.active_learning.data import _policy
from nca_wm.active_learning.inference import estimate_information_gain

# A module-global family so probes are consistent across calls within a run.
_FAMILY: W.WorldFamily | None = None


def set_family(family: W.WorldFamily) -> None:
    global _FAMILY
    _FAMILY = family


def _roll_history(mech: str, n_steps: int, rng: random.Random) -> list[int]:
    """Roll `mech` for `n_steps` real steps; return token-id history ending END_OBS."""
    fam = _FAMILY
    li = rng.randrange(fam.n_layouts)
    eng = W._new_engine(fam.jsons[mech], li)
    id2b = W._engine_id_to_canon_bit(eng)
    obs = W.read_obs(eng, id2b)
    toks = [V.BOS] + V.serialize_obs(obs)
    for _ in range(n_steps):
        a = _policy(obs, rng)
        W.step_engine(eng, a, seed=str(rng.getrandbits(48)))
        obs = W.read_obs(eng, id2b)
        toks += V.serialize_action(a) + V.serialize_obs(obs)
    return V.encode(toks)


# (name, mechanism, prefix_steps)
_PROBES = [
    ("fresh_unknown", "rand", 0),   # obs0 identical across mechanisms
    ("rand_known", "rand", 3),
    ("det_known", "det", 2),
    ("noop_known", "noop", 2),
]


def evaluate_probes(model, device, n_samples: int = 32, n_worlds: int = 8,
                    seed: int = 4321) -> dict[str, float]:
    assert _FAMILY is not None, "call set_family() first"
    rng = random.Random(seed)
    out: dict[str, float] = {}
    for name, mech, steps in _PROBES:
        vals = []
        for _ in range(n_worlds):
            h = _roll_history(mech, steps, rng)
            vals.append(estimate_information_gain(
                model, h, V.ACTION, device, n_samples=n_samples))
        out[name] = sum(vals) / len(vals)
    return out

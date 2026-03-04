"""Named RL sweep presets.

This file is the canonical place to track sweeps:
- add currently running sweeps under ACTIVE PRESETS
- keep completed sweeps under ARCHIVED PRESETS
"""

from dataclasses import dataclass, field
from typing import Dict, Type


@dataclass
class SweepConfig:
    """Base sweep preset payload.

    `sweep_axes` keys must match fields on TrainConfig/EnjoyConfig.
    """

    sweep_axes: dict = field(default_factory=lambda: {
        "seed": (0, 1, 2, 3, 4),
        "lr": (1.0e-4,),
    })


@dataclass
class SweepBasePreset(SweepConfig):
    """No-op preset: keep defaults unless overridden via Hydra CLI."""


@dataclass
class MaxEpisodeStepsSweep(SweepConfig):
    sweep_axes: dict = field(default_factory=lambda: {
        "seed": (0, 1, 2, 3, 4),
        "max_episode_steps": (
            # 100,
            200,
        ),
    })

@dataclass
class LearningRateSweep(SweepConfig):
    sweep_axes: dict = field(default_factory=lambda: {
        "seed": (0, 1, 2, 3, 4),
        "max_episode_steps": (
            # 100,
            200,
        ),
        "lr": (
            # 3.0e-5, 
            1.0e-4, 
            # 3.0e-4, 
            # 1.0e-3
        ),
    })


_NAMED_SWEEPS: Dict[str, Type[SweepConfig]] = {
    "sweep": SweepBasePreset,
    "learning_rate": LearningRateSweep,
    "max_episode_steps": MaxEpisodeStepsSweep,
}

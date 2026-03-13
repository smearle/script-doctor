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
            100,
            200,
        ),
    })

@dataclass
class LearningRateSweep(SweepConfig):
    sweep_axes: dict = field(default_factory=lambda: {
        "seed": (
            0,
            # 1, 2, 3, 4
        ),
        "max_episode_steps": (
            100,
            200,
        ),
        "lr": (
            # 3.0e-5, 
            1.0e-4, 
            # 3.0e-4, 
            # 1.0e-3
        ),
    })


@dataclass
class SmokeTestSweep(SweepConfig):
    """Single-seed smoke test — run with all_games=False total_timesteps=5000.

    JAX:     python sweep_rl.py sweep_name=smoke_test backend=jax all_games=False total_timesteps=5000 slurm=False
    PyTorch: python sweep_rl.py sweep_name=smoke_test backend=cpp all_games=False total_timesteps=5000 slurm=False

    Backend defaults applied by `sweep_rl.py`:
    - `backend=cpp`: `n_envs=128`, `cpp_num_threads=32`, Slurm requests `cpus_per_task=32`, `gpu:1`
    - `backend=nodejs`: `n_envs=16`, Slurm requests `cpus_per_task=32`
    """
    sweep_axes: dict = field(default_factory=lambda: {
        "seed": (100,),
    })


_NAMED_SWEEPS: Dict[str, Type[SweepConfig]] = {
    "sweep": SweepBasePreset,
    "learning_rate": LearningRateSweep,
    "max_episode_steps": MaxEpisodeStepsSweep,
    "smoke_test": SmokeTestSweep,
}

from dataclasses import dataclass, field
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore
import numpy as np


@dataclass
class PSConfig:
    game: str = "sokoban_basic"
    level: int = 0
    max_episode_steps: int = np.iinfo(np.int32).max
    overwrite: bool = False  # Whether to overwrite existing results
    vmap: bool = True


cs = ConfigStore.instance()
cs.store(name="ps_config", node=PSConfig)

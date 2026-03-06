import math
import os
from typing import Any


EXIT_TRAINING_CONFIG_FIELDS = (
    "game",
    "level_i",
    "n_iterations",
    "max_nodes",
    "batch_size",
    "cost_weight",
    "train_steps_per_iter",
    "train_batch_size",
    "lr",
    "blend_alpha",
    "replay_max_size",
    "initial_dim",
    "hidden_dim",
    "res_n",
)

EXIT_TRAINING_SUBDIR_FIELDS = (
    ("n_iterations", "it"),
    ("max_nodes", "mn"),
    ("batch_size", "bs"),
    ("cost_weight", "cw"),
    ("train_steps_per_iter", "ts"),
    ("train_batch_size", "tb"),
    ("lr", "lr"),
    ("blend_alpha", "ba"),
    ("replay_max_size", "rb"),
    ("initial_dim", "id"),
    ("hidden_dim", "hd"),
    ("res_n", "rn"),
)
EXIT_TRAINING_LABEL_FIELDS = tuple(field for field, _ in EXIT_TRAINING_SUBDIR_FIELDS)
EXIT_TRAINING_LABEL_NAMES = {
    "n_iterations": "Iterations",
    "max_nodes": "Max nodes",
    "batch_size": "Batch size",
    "cost_weight": "Cost weight",
    "train_steps_per_iter": "Train steps per iter",
    "train_batch_size": "Train batch size",
    "lr": "Learning rate",
    "blend_alpha": "Blend alpha",
    "replay_max_size": "Replay max size",
    "initial_dim": "Initial dim",
    "hidden_dim": "Hidden dim",
    "res_n": "Residual blocks",
}

EXIT_TRAINING_RELATIVE_DIR = os.path.join("data", "exit_training")


def format_run_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.12g}".replace("-", "m").replace(".", "p").replace("+", "")
    return str(value).replace(os.sep, "_").replace(" ", "_")


def build_run_config(
    game: str,
    level_i: int,
    n_iterations: int,
    max_nodes: int,
    batch_size: int,
    cost_weight: float,
    train_steps_per_iter: int,
    train_batch_size: int,
    lr: float,
    blend_alpha: float,
    replay_max_size: int,
    initial_dim: int,
    hidden_dim: int,
    res_n: int,
) -> dict[str, Any]:
    return {
        "game": game,
        "level_i": level_i,
        "n_iterations": n_iterations,
        "max_nodes": max_nodes,
        "batch_size": batch_size,
        "cost_weight": cost_weight,
        "train_steps_per_iter": train_steps_per_iter,
        "train_batch_size": train_batch_size,
        "lr": lr,
        "blend_alpha": blend_alpha,
        "replay_max_size": replay_max_size,
        "initial_dim": initial_dim,
        "hidden_dim": hidden_dim,
        "res_n": res_n,
    }


def run_subdir_name(run_config: dict[str, Any]) -> str:
    return "_".join(
        f"{alias}{format_run_value(run_config[key])}"
        for key, alias in EXIT_TRAINING_SUBDIR_FIELDS
    )


def format_run_label(run_config: dict[str, Any] | None, fallback_name: str) -> str:
    if not isinstance(run_config, dict):
        return fallback_name
    return "_".join(
        f"{alias}{format_run_value(run_config[key])}"
        for key, alias in EXIT_TRAINING_SUBDIR_FIELDS
        if key in run_config
    ) or fallback_name


def format_display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return f"{int(value):,}"
        return f"{value:.12g}"
    return str(value)


def varying_run_fields(run_configs: list[dict[str, Any] | None]) -> list[str]:
    valid_configs = [cfg for cfg in run_configs if isinstance(cfg, dict)]
    if not valid_configs:
        return []

    varying_fields = []
    for field in EXIT_TRAINING_LABEL_FIELDS:
        values = {cfg.get(field) for cfg in valid_configs if field in cfg}
        if len(values) > 1:
            varying_fields.append(field)
    return varying_fields


def format_variable_run_label(
    run_config: dict[str, Any] | None,
    fallback_name: str,
    varying_fields: list[str] | tuple[str, ...],
) -> str:
    if not isinstance(run_config, dict):
        return fallback_name

    parts = []
    for field in varying_fields:
        if field not in run_config:
            continue
        label = EXIT_TRAINING_LABEL_NAMES.get(field, field.replace("_", " ").title())
        parts.append(f"{label}={format_display_value(run_config[field])}")

    return ", ".join(parts) if parts else "Default config"

"""Config schema placeholders for experiments and clusters."""

from __future__ import annotations


def validate_experiment_config(config: dict) -> None:
    """Minimal validation stub."""
    if "experiment_name" not in config:
        raise ValueError("Missing 'experiment_name' in experiment config.")


def validate_cluster_config(config: dict) -> None:
    """Minimal validation stub."""
    if "cluster_name" not in config:
        raise ValueError("Missing 'cluster_name' in cluster config.")

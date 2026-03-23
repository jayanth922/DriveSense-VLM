"""YAML configuration loading, environment variable expansion, and deep merge.

Provides the canonical config loading interface used by all DriveSense modules.
Supports ``${VAR:default}`` syntax for environment variable interpolation so that
HPC paths (set via export HPC_DATA_ROOT=...) override local defaults without
modifying the YAML files.

Usage:
    from drivesense.utils.config import load_config, merge_configs

    config = load_config("configs/training.yaml")
    full = merge_configs(model_cfg, data_cfg, training_cfg)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load and validate a YAML config file, expanding environment variables.

    Processes ``${VAR:default}`` placeholders in string values:
    - If ``VAR`` is set in the environment, its value replaces the placeholder.
    - Otherwise, ``default`` is used (supports ``~`` for home directory expansion).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict containing the fully resolved configuration.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    return _expand_env_vars(raw)


def merge_configs(*configs: dict) -> dict:
    """Deep merge multiple config dicts; later configs override earlier ones.

    Recursively merges nested dicts. Non-dict values at the same key are
    overridden by the later config (last-writer-wins).

    Args:
        *configs: Two or more config dicts to merge in order.

    Returns:
        New merged dict. Input dicts are not modified.

    Example:
        >>> base = {"model": {"rank": 16, "alpha": 32}}
        >>> override = {"model": {"rank": 32}}
        >>> merge_configs(base, override)
        {"model": {"rank": 32, "alpha": 32}}
    """
    result: dict = {}
    for config in configs:
        result = _deep_merge(result, config)
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.

    Args:
        base: Base configuration dict.
        override: Override configuration dict.

    Returns:
        Merged dict with override values taking precedence.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env_vars(obj: object) -> object:
    """Recursively expand ``${VAR:default}`` placeholders in config values.

    Args:
        obj: Config value (dict, list, str, or scalar).

    Returns:
        Config value with environment variable placeholders resolved.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    if isinstance(obj, str):
        return _replace_env_placeholders(obj)
    return obj


def _replace_env_placeholders(value: str) -> str:
    """Replace ``${VAR:default}`` patterns in a single string value.

    Args:
        value: String potentially containing ``${VAR:default}`` placeholders.

    Returns:
        String with placeholders replaced by environment values or defaults.
    """
    pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) or ""
        resolved = os.environ.get(var_name, default)
        return str(Path(resolved).expanduser()) if resolved else default

    return pattern.sub(replacer, value)

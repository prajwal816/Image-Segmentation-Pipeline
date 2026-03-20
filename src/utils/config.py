"""YAML configuration loader with defaults merging."""

import copy
import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def merge_configs(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge override config into base config.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config_with_base(
    config_path: str, base_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load config with optional base config merging.

    If the config references a base_config key, or if base_config_path
    is provided, loads and merges the base config first.

    Args:
        config_path: Path to the config file.
        base_config_path: Optional path to base config.

    Returns:
        Merged configuration dictionary.
    """
    config = load_config(config_path)

    if base_config_path is None:
        base_config_path = config.get("ablation", {}).get("base_config")

    if base_config_path and os.path.exists(base_config_path):
        base_config = load_config(base_config_path)
        return merge_configs(base_config, config)

    return config


def get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot-separated key path.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., "model.num_classes").
        default: Default value if key not found.

    Returns:
        Value at the key path, or default.
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

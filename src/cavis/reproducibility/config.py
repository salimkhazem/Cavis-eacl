from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping using the non-executable safe loader."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(value).__name__}")
    return value


def deep_merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings from left to right without mutating inputs."""
    result: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    return result


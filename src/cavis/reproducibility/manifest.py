from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import load_yaml

IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Validate that every remote dependency is pinned to a full commit."""
    manifest = load_yaml(path)
    errors: list[str] = []
    sources = manifest.get("sources")
    models = manifest.get("models")
    if not isinstance(sources, dict) or not isinstance(models, dict):
        raise ValueError("manifest must contain source and model mappings")

    for section_name, section in (("sources", sources), ("models", models)):
        for key, entry in section.items():
            if not isinstance(entry, dict):
                errors.append(f"{section_name}.{key} must be a mapping")
                continue
            revision = entry.get("revision")
            if not isinstance(revision, str) or not IMMUTABLE_REVISION.fullmatch(revision):
                errors.append(f"{section_name}.{key}.revision is not a full SHA-1")
    if manifest.get("policy", {}).get("allow_pickle") is not False:
        errors.append("policy.allow_pickle must be false")
    if errors:
        raise ValueError("Invalid data manifest:\n- " + "\n- ".join(errors))
    return manifest


def validate_model_configs(
    manifest: dict[str, Any],
    config_dir: str | Path,
) -> list[str]:
    """Return human-readable consistency errors for model YAML files."""
    errors: list[str] = []
    manifest_models = manifest["models"]
    seen: set[str] = set()
    for path in sorted(Path(config_dir).glob("*.yaml")):
        payload = load_yaml(path)
        model = payload.get("model", {})
        key = model.get("key")
        if key not in manifest_models:
            errors.append(f"{path}: unknown model key {key!r}")
            continue
        seen.add(key)
        expected = manifest_models[key]
        for field in ("repo_id", "revision"):
            if model.get(field) != expected.get(field):
                errors.append(
                    f"{path}: {field}={model.get(field)!r}, expected {expected.get(field)!r}"
                )
    missing = set(manifest_models) - seen
    errors.extend(f"missing config for model {key}" for key in sorted(missing))
    return errors


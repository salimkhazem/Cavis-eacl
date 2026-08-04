from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import capture_environment
from .io import atomic_write_json, stable_json_dumps


@dataclass(frozen=True)
class RunDirectory:
    path: Path
    config_hash: str

    @classmethod
    def create(
        cls,
        root: str | Path,
        config: dict[str, Any],
        *,
        name: str,
        run_id: str | None = None,
    ) -> RunDirectory:
        digest = hashlib.sha256(stable_json_dumps(config).encode("utf-8")).hexdigest()
        if run_id is None:
            timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{timestamp}-{digest[:10]}"
        path = Path(root) / name / run_id
        path.mkdir(parents=True, exist_ok=False)
        atomic_write_json(path / "config.resolved.json", config)
        atomic_write_json(path / "environment.json", capture_environment())
        return cls(path=path, config_hash=digest)


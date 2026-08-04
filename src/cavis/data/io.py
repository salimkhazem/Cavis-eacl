"""Safe serialization helpers.

Only transparent, data-only formats are accepted.  In particular this module
never deserializes pickle, torch checkpoints, NumPy object arrays, or arbitrary
Python objects.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

SAFE_INPUT_SUFFIXES = frozenset({".json", ".jsonl", ".parquet"})
UNSAFE_PICKLE_SUFFIXES = frozenset(
    {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".npy", ".npz"}
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_safe_suffix(path: Path, expected: set[str] | frozenset[str]) -> None:
    suffix = path.suffix.lower()
    if suffix in UNSAFE_PICKLE_SUFFIXES:
        raise ValueError(f"Refusing unsafe serialized input: {path}")
    if suffix not in expected:
        allowed = ", ".join(sorted(expected))
        raise ValueError(f"Expected one of {allowed}, got: {path}")


def read_json(path: str | Path) -> Any:
    file_path = Path(path)
    _require_safe_suffix(file_path, {".json"})
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path) -> Iterator[Mapping[str, Any]]:
    file_path = Path(path)
    _require_safe_suffix(file_path, {".jsonl"})
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"Expected an object at {file_path}:{line_number}, "
                    f"got {type(value).__name__}"
                )
            yield value


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a parquet file without importing an optional backend at import time."""

    file_path = Path(path)
    _require_safe_suffix(file_path, {".parquet"})
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Reading parquet requires pyarrow; install the project data extra."
        ) from exc
    return pq.read_table(file_path).to_pylist()


def canonical_json_hash(value: Any) -> str:
    """Hash a JSON-compatible value with stable key and whitespace handling."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256_text(payload)

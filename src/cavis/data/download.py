"""Content-addressed downloads for the locked CAVIS data manifest."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import sha256_file

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DISALLOWED_REVISIONS = frozenset({"", "main", "master", "latest", "head"})


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    name: str
    kind: str
    destination: str
    revision: str | None = None
    url: str | None = None
    repo_id: str | None = None
    sha256: str | None = None
    include: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def assert_pinned(self) -> None:
        if self.kind in {"git", "huggingface_dataset"}:
            revision = (self.revision or "").lower()
            if revision in DISALLOWED_REVISIONS:
                raise ValueError(
                    f"{self.name}: {self.kind} revision must be an immutable hash"
                )
            if not COMMIT_RE.fullmatch(revision):
                raise ValueError(
                    f"{self.name}: expected a full 40-character revision hash"
                )
        elif self.kind == "http":
            if not self.sha256 or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
                raise ValueError(f"{self.name}: HTTP artifacts require SHA-256")
        else:
            raise ValueError(f"{self.name}: unsupported artifact kind {self.kind!r}")


def _parse_text_manifest(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - project installs PyYAML
            raise RuntimeError(
                "YAML manifest requires PyYAML; JSON manifests work without it"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, Mapping):
        raise ValueError("data_manifest.lock must contain a mapping")
    return value


def _iter_raw_entries(manifest: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    """Support a compact ``artifacts`` list and named category mappings."""

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, list):
        for raw in artifacts:
            if not isinstance(raw, Mapping) or "name" not in raw:
                raise ValueError("Every artifact list entry needs a name")
            yield str(raw["name"]), raw
        return
    if isinstance(artifacts, Mapping):
        yield from ((str(name), value) for name, value in artifacts.items())
        return
    for category in ("sources", "repositories", "datasets", "files"):
        entries = manifest.get(category, {})
        if isinstance(entries, Mapping):
            yield from ((str(name), value) for name, value in entries.items())


def load_manifest(path: str | Path) -> list[ArtifactSpec]:
    manifest_path = Path(path)
    raw_manifest = _parse_text_manifest(manifest_path)
    specs: list[ArtifactSpec] = []
    seen: set[str] = set()
    for name, raw in _iter_raw_entries(raw_manifest):
        if name in seen:
            raise ValueError(f"Duplicate manifest artifact: {name}")
        seen.add(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"Manifest artifact {name} must be a mapping")
        kind = str(raw.get("kind") or raw.get("type") or "")
        destination = str(
            raw.get("destination")
            or raw.get("path")
            or f"data/external/{name}"
        )
        include_value = raw.get("include") or raw.get("allow_patterns") or ()
        if isinstance(include_value, str):
            include = (include_value,)
        elif isinstance(include_value, Iterable):
            include = tuple(str(value) for value in include_value)
        else:
            raise ValueError(f"Manifest artifact {name} has invalid include list")
        known = {
            "name",
            "kind",
            "type",
            "destination",
            "path",
            "revision",
            "url",
            "repo_id",
            "sha256",
            "include",
            "allow_patterns",
        }
        spec = ArtifactSpec(
            name=name,
            kind=kind,
            destination=destination,
            revision=(
                str(raw["revision"]).lower() if raw.get("revision") is not None else None
            ),
            url=str(raw["url"]) if raw.get("url") is not None else None,
            repo_id=(
                str(raw["repo_id"]) if raw.get("repo_id") is not None else None
            ),
            sha256=(
                str(raw["sha256"]).lower() if raw.get("sha256") is not None else None
            ),
            include=include,
            metadata={key: value for key, value in raw.items() if key not in known},
        )
        spec.assert_pinned()
        specs.append(spec)
    if not specs:
        raise ValueError("No artifacts found in data_manifest.lock")
    return specs


def _resolve_destination(root: Path, relative: str) -> Path:
    destination = (root / relative).resolve()
    resolved_root = root.resolve()
    if destination == resolved_root or resolved_root not in destination.parents:
        raise ValueError(f"Artifact destination escapes data root: {relative}")
    return destination


def _atomic_install(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists: {destination}; verify it or remove it explicitly"
        )
    staging.replace(destination)


def _git_revision(path: Path) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip().lower() if process.returncode == 0 else None


def artifact_is_current(spec: ArtifactSpec, root: Path) -> bool:
    destination = _resolve_destination(root, spec.destination)
    if not destination.exists():
        return False
    if spec.kind == "git":
        return (
            destination.is_dir()
            and _git_revision(destination) == spec.revision
            and _declared_files_match(spec, destination)
        )
    if spec.kind == "http":
        return destination.is_file() and sha256_file(destination) == spec.sha256
    if spec.kind == "huggingface_dataset":
        marker = destination / ".cavis_revision"
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == (
            spec.revision or ""
        )
    return False


def _declared_files_match(spec: ArtifactSpec, artifact_root: Path) -> bool:
    """Validate optional ``*_path``/``*_sha256`` pairs from the lock."""

    checked = False
    for key, relative in spec.metadata.items():
        if not key.endswith("_path") or not isinstance(relative, str):
            continue
        digest_key = f"{key[:-5]}_sha256"
        expected = spec.metadata.get(digest_key)
        if not isinstance(expected, str):
            continue
        candidate = (artifact_root / relative).resolve()
        if artifact_root.resolve() not in candidate.parents:
            return False
        if not candidate.is_file() or sha256_file(candidate) != expected.lower():
            return False
        checked = True
    # No declared file hashes is valid; the immutable repository revision is
    # itself the content lock in that case.
    return True if not checked else True


def _download_git(spec: ArtifactSpec, staging: Path) -> None:
    if not spec.url:
        raise ValueError(f"{spec.name}: git artifact requires url")
    process = subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            spec.url,
            str(staging),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(f"git clone failed for {spec.name}: {process.stderr}")
    process = subprocess.run(
        ["git", "-C", str(staging), "checkout", "--detach", spec.revision or ""],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(f"git checkout failed for {spec.name}: {process.stderr}")
    actual = _git_revision(staging)
    if actual != spec.revision:
        raise RuntimeError(
            f"{spec.name}: resolved git revision {actual}, expected {spec.revision}"
        )
    if not _declared_files_match(spec, staging):
        raise RuntimeError(f"{spec.name}: a locked file hash does not match")


def _download_http(spec: ArtifactSpec, staging: Path) -> None:
    if not spec.url:
        raise ValueError(f"{spec.name}: HTTP artifact requires url")
    request = urllib.request.Request(
        spec.url, headers={"User-Agent": "cavis-reproducibility/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with staging.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    actual = sha256_file(staging)
    if actual != spec.sha256:
        raise RuntimeError(
            f"{spec.name}: SHA-256 mismatch: expected {spec.sha256}, got {actual}"
        )


def _download_huggingface(spec: ArtifactSpec, staging: Path) -> None:
    if not spec.repo_id:
        raise ValueError(f"{spec.name}: Hugging Face artifact requires repo_id")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Hugging Face downloads require huggingface_hub"
        ) from exc
    staging.mkdir(parents=True, exist_ok=False)
    snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        revision=spec.revision,
        local_dir=staging,
        allow_patterns=list(spec.include) or None,
    )
    (staging / ".cavis_revision").write_text(
        f"{spec.revision}\n", encoding="utf-8"
    )


def download_artifact(spec: ArtifactSpec, *, data_root: str | Path) -> Path:
    """Download one locked artifact without overwriting existing paths."""

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = _resolve_destination(root, spec.destination)
    if artifact_is_current(spec, root):
        return destination
    if destination.exists():
        raise RuntimeError(
            f"{destination} exists but does not match the lock; "
            "move it aside explicitly before downloading"
        )

    temp_parent = destination.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    if spec.kind == "http":
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=temp_parent
        )
        os.close(file_descriptor)
        staging = Path(temp_name)
    else:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=temp_parent)
        )
        # git clone and snapshot_download require a nonexistent output path.
        staging.rmdir()
    try:
        if spec.kind == "git":
            _download_git(spec, staging)
        elif spec.kind == "http":
            _download_http(spec, staging)
        elif spec.kind == "huggingface_dataset":
            _download_huggingface(spec, staging)
        else:  # guarded by assert_pinned
            raise AssertionError(spec.kind)
        _atomic_install(staging, destination)
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
        elif staging.exists():
            staging.unlink()
        raise
    return destination

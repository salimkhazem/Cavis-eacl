"""Deterministic implementation fingerprints for immutable research artifacts."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cavis.data.io import canonical_json_hash

from .io import sha256_file

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "cavis.implementation_fingerprint.v1"
GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ANONYMOUS_ARTIFACT_MANIFEST = "ARTIFACT_MANIFEST.json"


def _git_output(*arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()
        raise RuntimeError(
            "Implementation provenance requires an intact Git checkout"
            f" ({detail or 'git command failed'})"
        )
    return process.stdout.strip()


def _anonymous_artifact_git_state(
    *,
    normalized_paths: Sequence[str],
    source_hashes: Mapping[str, str],
    git_error: RuntimeError,
) -> dict[str, Any]:
    """Use a content-bound pseudo revision only in a packaged review artifact.

    The normal repository path remains fail-closed on every Git failure.  A
    tarball deliberately contains no ``.git`` directory, so it may substitute
    its externally checksummed allowlist manifest after verifying that every
    requested implementation file is present with the exact recorded digest.
    """

    manifest_path = ROOT / ANONYMOUS_ARTIFACT_MANIFEST
    if not manifest_path.is_file():
        raise git_error
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Anonymous artifact provenance manifest is unreadable") from error
    if manifest.get("anonymized_for_double_blind_review") is not True:
        raise RuntimeError("Anonymous artifact provenance manifest is not review-bound")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("Anonymous artifact provenance manifest has no file allowlist")

    recorded: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("Anonymous artifact provenance contains an invalid file entry")
        path = entry.get("path")
        digest = entry.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path in recorded
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise RuntimeError("Anonymous artifact provenance contains an invalid file binding")
        recorded[path] = digest

    mismatches = [
        path
        for path in normalized_paths
        if recorded.get(path) != source_hashes.get(path)
    ]
    if mismatches:
        raise RuntimeError(
            "Anonymous artifact implementation files differ from the release manifest: "
            + ", ".join(mismatches)
        )
    artifact_revision = canonical_json_hash(
        {
            "artifact_name": manifest.get("artifact_name"),
            "files": recorded,
        }
    )[:40]
    return {
        "commit": artifact_revision,
        "dirty": False,
        "tracked_dirty": False,
        "untracked_implementation_files": [],
    }


def implementation_fingerprint(
    relative_paths: Sequence[str],
) -> dict[str, Any]:
    """Fingerprint one implementation stage, including an honest Git state."""

    normalized = sorted(set(relative_paths))
    if not normalized or len(normalized) != len(relative_paths):
        raise ValueError(
            "Implementation fingerprint paths must be non-empty and unique"
        )
    files: dict[str, str] = {}
    for relative in normalized:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Implementation path must be repository-relative: {relative}"
            )
        source = ROOT / path
        if not source.is_file():
            raise FileNotFoundError(
                f"Implementation fingerprint source is missing: {source}"
            )
        files[path.as_posix()] = sha256_file(source)

    try:
        commit = _git_output("rev-parse", "HEAD").lower()
    except RuntimeError as git_error:
        git_state = _anonymous_artifact_git_state(
            normalized_paths=normalized,
            source_hashes=files,
            git_error=git_error,
        )
    else:
        if GIT_REVISION_PATTERN.fullmatch(commit) is None:
            raise RuntimeError(f"Git HEAD is not a full commit revision: {commit!r}")
        tracked_files = set(_git_output("ls-files", "--", *normalized).splitlines())
        untracked_implementation_files = sorted(set(normalized) - tracked_files)
        tracked_dirty = bool(
            _git_output(
                "status",
                "--porcelain",
                "--untracked-files=no",
            )
        )
        git_state = {
            "commit": commit,
            # Ignore unrelated untracked result caches, while remaining honest
            # about source files that are not yet represented by HEAD.
            "dirty": tracked_dirty or bool(untracked_implementation_files),
            "tracked_dirty": tracked_dirty,
            "untracked_implementation_files": untracked_implementation_files,
        }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git": git_state,
        "files": files,
    }
    payload["fingerprint_sha256"] = canonical_json_hash(payload)
    return payload


def verify_implementation_fingerprint(
    recorded: Any,
    *,
    relative_paths: Sequence[str],
    context: str,
) -> dict[str, Any]:
    """Fail closed unless a recorded fingerprint equals the current sources."""

    if not isinstance(recorded, Mapping):
        raise ValueError(f"{context}: implementation fingerprint is missing")
    schema = recorded.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"{context}: implementation fingerprint schema must be "
            f"{SCHEMA_VERSION}"
        )
    git = recorded.get("git")
    if not isinstance(git, Mapping):
        raise ValueError(f"{context}: implementation Git state is missing")
    commit = git.get("commit")
    dirty = git.get("dirty")
    tracked_dirty = git.get("tracked_dirty")
    untracked_implementation_files = git.get(
        "untracked_implementation_files"
    )
    if (
        not isinstance(commit, str)
        or GIT_REVISION_PATTERN.fullmatch(commit) is None
        or not isinstance(dirty, bool)
        or not isinstance(tracked_dirty, bool)
        or not isinstance(untracked_implementation_files, list)
        or any(
            not isinstance(path, str) or not path
            for path in untracked_implementation_files
        )
        or untracked_implementation_files
        != sorted(set(untracked_implementation_files))
        or dirty
        != (tracked_dirty or bool(untracked_implementation_files))
    ):
        raise ValueError(f"{context}: invalid implementation Git state")
    files = recorded.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{context}: implementation file hashes are missing")
    if any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or SHA256_PATTERN.fullmatch(digest) is None
        for path, digest in files.items()
    ):
        raise ValueError(f"{context}: invalid implementation path/SHA-256 mapping")
    expected = implementation_fingerprint(relative_paths)
    if dict(recorded) != expected:
        raise ValueError(
            f"{context}: implementation fingerprint does not match the "
            "current Git state and source hashes"
        )
    return expected

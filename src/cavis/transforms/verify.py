"""External Lean verification with explicit, narrowly scoped evidence."""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from cavis.data.io import sha256_text

from .base import TransformResult

EvidenceState = Literal[
    "unverified",
    "lean_compiles",
    "lean_rejects",
    "unexpected_outcome",
    "verification_error",
]


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """Evidence about compiler behavior, never a theorem-equivalence assertion."""

    transform_name: str
    family: str
    source_sha256: str
    target_sha256: str
    command: tuple[str, ...]
    lean_version: str | None
    source_returncode: int | None
    target_returncode: int | None
    source_stdout: str
    source_stderr: str
    target_stdout: str
    target_stderr: str
    state: EvidenceState
    expected_source_outcome: str
    expected_target_outcome: str
    human_validation: str | None = None
    human_validator: str | None = None
    human_notes: str | None = None
    semantic_status: str = "not_established"

    def __post_init__(self) -> None:
        if self.semantic_status != "not_established":
            raise ValueError(
                "Compiler evidence alone cannot establish semantic equivalence "
                "or theorem invalidity"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _materialize_command(
    command: Sequence[str], source_path: Path, target_path: Path, which: str
) -> list[str]:
    selected = source_path if which == "source" else target_path
    replacements = {
        "{source}": str(source_path),
        "{target}": str(target_path),
        "{file}": str(selected),
    }
    output: list[str] = []
    found_placeholder = False
    for argument in command:
        value = argument
        for placeholder, replacement in replacements.items():
            if placeholder in value:
                found_placeholder = True
                value = value.replace(placeholder, replacement)
        output.append(value)
    if not found_placeholder:
        output.append(str(selected))
    return output


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None,
    timeout_seconds: float,
    max_output_chars: int,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    # Bound evidence logs so a pathological compiler failure cannot make a
    # result artifact enormous.
    process.stdout = process.stdout[-max_output_chars:]
    process.stderr = process.stderr[-max_output_chars:]
    return process


def _version(command: Sequence[str], cwd: Path | None) -> str | None:
    executable_prefix: list[str]
    if len(command) >= 3 and command[:2] == ["lake", "env"]:
        executable_prefix = ["lake", "env", command[2]]
    else:
        executable_prefix = [command[0]]
    try:
        process = subprocess.run(
            [*executable_prefix, "--version"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (process.stdout or process.stderr).strip()
    return value[:1000] or None


def verify_with_command(
    result: TransformResult,
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    verify_source: bool = True,
    expected_source_outcome: Literal["compile", "reject"] = "compile",
    expected_target_outcome: Literal["compile", "reject"] | None = None,
    human_validation: Literal["approved", "rejected"] | None = None,
    human_validator: str | None = None,
    human_notes: str | None = None,
    timeout_seconds: float = 60.0,
    max_output_chars: int = 20_000,
) -> VerificationEvidence:
    """Compile temporary source/target files using a non-shell command.

    The command may use ``{source}``, ``{target}``, or ``{file}`` placeholders.
    If no placeholder occurs, the selected path is appended.  A successful G0
    target receives ``lean_compiles``; a rejected G1 target receives
    ``lean_rejects``.  Neither state is called semantic preservation/invalidity.
    """

    if not command or any(not isinstance(argument, str) for argument in command):
        raise ValueError("command must be a non-empty sequence of strings")
    working_directory = Path(cwd) if cwd is not None else None
    shown_command = tuple(command)
    target_expectation = expected_target_outcome or result.spec.expected_lean_outcome
    lean_version = _version(command, working_directory)
    source_process: subprocess.CompletedProcess[str] | None = None
    target_process: subprocess.CompletedProcess[str] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="cavis-lean-") as temp_directory:
            temp = Path(temp_directory)
            source_path = temp / "source.lean"
            target_path = temp / "target.lean"
            source_path.write_text(result.source, encoding="utf-8")
            target_path.write_text(result.target, encoding="utf-8")
            if verify_source:
                source_process = _run(
                    _materialize_command(command, source_path, target_path, "source"),
                    cwd=working_directory,
                    timeout_seconds=timeout_seconds,
                    max_output_chars=max_output_chars,
                )
            target_process = _run(
                _materialize_command(command, source_path, target_path, "target"),
                cwd=working_directory,
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return VerificationEvidence(
            transform_name=result.spec.name,
            family=result.spec.family,
            source_sha256=result.source_sha256,
            target_sha256=result.target_sha256,
            command=shown_command,
            lean_version=lean_version,
            source_returncode=(
                source_process.returncode if source_process is not None else None
            ),
            target_returncode=None,
            source_stdout=source_process.stdout if source_process else "",
            source_stderr=source_process.stderr if source_process else "",
            target_stdout="",
            target_stderr=f"{type(exc).__name__}: {exc}",
            state="verification_error",
            expected_source_outcome=expected_source_outcome,
            expected_target_outcome=target_expectation,
            human_validation=human_validation,
            human_validator=human_validator,
            human_notes=human_notes,
        )

    assert target_process is not None
    source_ok = not verify_source or (
        source_process is not None
        and (
            (expected_source_outcome == "compile" and source_process.returncode == 0)
            or (
                expected_source_outcome == "reject"
                and source_process.returncode != 0
            )
        )
    )
    target_compiles = target_process.returncode == 0
    if not source_ok:
        state: EvidenceState = "verification_error"
    elif target_expectation == "compile" and target_compiles:
        state = "lean_compiles"
    elif target_expectation == "reject" and not target_compiles:
        state = "lean_rejects"
    else:
        state = "unexpected_outcome"
    return VerificationEvidence(
        transform_name=result.spec.name,
        family=result.spec.family,
        source_sha256=result.source_sha256,
        target_sha256=result.target_sha256,
        command=shown_command,
        lean_version=lean_version,
        source_returncode=source_process.returncode if source_process else None,
        target_returncode=target_process.returncode,
        source_stdout=source_process.stdout if source_process else "",
        source_stderr=source_process.stderr if source_process else "",
        target_stdout=target_process.stdout,
        target_stderr=target_process.stderr,
        state=state,
        expected_source_outcome=expected_source_outcome,
        expected_target_outcome=target_expectation,
        human_validation=human_validation,
        human_validator=human_validator,
        human_notes=human_notes,
    )


def command_for_log(command: Sequence[str]) -> str:
    """Render a command for logs only; execution never passes through a shell."""

    return shlex.join(command)


def to_core_transform_evidence(
    evidence: VerificationEvidence, *, transformation_id: str
):
    """Convert compiler evidence to the repository-wide typed schema.

    A nonzero G1 compiler result is retained as evidence but is *not* marked
    mechanically verified: rejection of one proof script does not prove that
    the mutated theorem is false.  This conservative mapping is intentional.
    """

    from cavis.schemas import TransformEvidence

    if not evidence.lean_version:
        raise ValueError("A core TransformEvidence requires a recorded Lean version")
    return TransformEvidence(
        transformation_id=transformation_id,
        command=evidence.command,
        lean_version=evidence.lean_version,
        return_code=evidence.target_returncode if evidence.target_returncode is not None else -1,
        source_hash=evidence.source_sha256,
        target_hash=evidence.target_sha256,
        mechanically_verified=evidence.state == "lean_compiles",
        human_validation=evidence.human_validation,
        stdout_hash=sha256_text(evidence.target_stdout),
        stderr_hash=sha256_text(evidence.target_stderr),
    )

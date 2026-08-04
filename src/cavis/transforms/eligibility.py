"""Compiler evidence and conservative CAVIS eligibility joins."""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cavis.data.io import sha256_text

from .contracts import check_g0_contract, declaration_statement_changed


@dataclass(frozen=True, slots=True)
class LeanCompileEvidence:
    schema_version: str
    target_hash: str
    command: tuple[str, ...]
    lean_version: str
    environment_revision: str
    mathlib_revision: str
    validator_worktrees_clean: bool
    returncode: int | None
    state: str
    stdout_hash: str
    stderr_hash: str
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command)
        return value


def _version_command(command: Sequence[str]) -> list[str]:
    for index, argument in enumerate(command):
        if Path(argument).name == "lean":
            return [*command[: index + 1], "--version"]
    return [command[0], "--version"]


def inspect_pinned_environment(
    *,
    command: Sequence[str],
    cwd: str | Path,
    expected_lean_version: str,
    expected_environment_revision: str,
    expected_mathlib_revision: str,
) -> tuple[str, str, str]:
    """Fail closed unless Lean and both source worktrees match their pins."""

    working_directory = Path(cwd)
    version = subprocess.run(
        _version_command(command),
        cwd=working_directory,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual_version = (version.stdout or version.stderr).strip()
    if version.returncode or expected_lean_version not in actual_version:
        raise RuntimeError(
            f"Lean version mismatch: expected substring {expected_lean_version!r}, "
            f"got {actual_version!r}"
        )
    revision = subprocess.run(
        ["git", "-C", str(working_directory), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual_revision = revision.stdout.strip().lower()
    if revision.returncode or actual_revision != expected_environment_revision.lower():
        raise RuntimeError(
            f"Lean environment revision mismatch: expected "
            f"{expected_environment_revision}, got {actual_revision or '<unavailable>'}"
        )
    environment_status = subprocess.run(
        [
            "git",
            "-C",
            str(working_directory),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if environment_status.returncode or environment_status.stdout.strip():
        raise RuntimeError(
            "Lean environment worktree contains modifications or untracked files"
        )

    leanpkg_path = working_directory / "leanpkg.toml"
    try:
        leanpkg_text = leanpkg_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"Cannot read pinned Lean package file: {leanpkg_path}") from error
    if expected_mathlib_revision.lower() not in leanpkg_text.lower():
        raise RuntimeError(
            "Lean package metadata does not contain the expected mathlib revision"
        )

    mathlib_directory = working_directory / "_target" / "deps" / "mathlib"
    mathlib_revision = subprocess.run(
        ["git", "-C", str(mathlib_directory), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    actual_mathlib_revision = mathlib_revision.stdout.strip().lower()
    if (
        mathlib_revision.returncode
        or actual_mathlib_revision != expected_mathlib_revision.lower()
    ):
        raise RuntimeError(
            "Mathlib revision mismatch: expected "
            f"{expected_mathlib_revision}, got "
            f"{actual_mathlib_revision or '<unavailable>'}"
        )
    mathlib_status = subprocess.run(
        [
            "git",
            "-C",
            str(mathlib_directory),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if mathlib_status.returncode or mathlib_status.stdout.strip():
        raise RuntimeError("Mathlib worktree contains modifications or untracked files")
    return actual_version, actual_revision, actual_mathlib_revision


def _materialize_command(command: Sequence[str], target_path: Path) -> list[str]:
    found = False
    output: list[str] = []
    for argument in command:
        if "{file}" in argument:
            found = True
            output.append(argument.replace("{file}", str(target_path)))
        else:
            output.append(argument)
    if not found:
        output.append(str(target_path))
    return output


def compile_lean_text(
    text: str,
    *,
    command: Sequence[str],
    cwd: str | Path,
    lean_version: str,
    environment_revision: str,
    mathlib_revision: str,
    timeout_seconds: float,
    max_output_chars: int = 20_000,
) -> LeanCompileEvidence:
    target_hash = sha256_text(text)
    process: subprocess.CompletedProcess[str] | None = None
    error = ""
    try:
        with tempfile.TemporaryDirectory(prefix="cavis-lean-compile-") as temp:
            target = Path(temp) / f"{target_hash[:16]}.lean"
            target.write_text(text, encoding="utf-8")
            process = subprocess.run(
                _materialize_command(command, target),
                cwd=Path(cwd),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    stdout = process.stdout if process is not None else ""
    stderr = process.stderr if process is not None else error
    returncode = process.returncode if process is not None else None
    if returncode is not None and returncode < 0:
        signal_error = (
            "ProcessSignalError: Lean process terminated by signal "
            f"{-returncode}"
        )
        stderr = f"{stderr.rstrip()}\n{signal_error}".lstrip()
        returncode = None
    state = (
        "compile"
        if returncode == 0
        else ("reject" if returncode is not None else "error")
    )
    return LeanCompileEvidence(
        schema_version="cavis.lean_evidence.v2",
        target_hash=target_hash,
        command=tuple(command),
        lean_version=lean_version,
        environment_revision=environment_revision,
        mathlib_revision=mathlib_revision,
        validator_worktrees_clean=True,
        returncode=returncode,
        state=state,
        stdout_hash=sha256_text(stdout),
        stderr_hash=sha256_text(stderr),
        stdout_tail=stdout[-max_output_chars:],
        stderr_tail=stderr[-max_output_chars:],
    )


def compile_unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    command: Sequence[str],
    cwd: str | Path,
    lean_version: str,
    environment_revision: str,
    mathlib_revision: str,
    jobs: int,
    timeout_seconds: float,
    existing: Mapping[str, LeanCompileEvidence] | None = None,
    checkpoint: (
        Callable[[Mapping[str, LeanCompileEvidence]], None] | None
    ) = None,
) -> dict[str, LeanCompileEvidence]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    texts = {sha256_text(str(row["reasoning"])): str(row["reasoning"]) for row in rows}
    evidence = {
        target_hash: value
        for target_hash, value in (existing or {}).items()
        if target_hash in texts
        and value.command == tuple(command)
        and value.schema_version == "cavis.lean_evidence.v2"
        and value.lean_version == lean_version
        and value.environment_revision == environment_revision
        and value.mathlib_revision == mathlib_revision
        and value.validator_worktrees_clean is True
        and value.returncode is not None
        and value.state in {"compile", "reject"}
    }
    missing = {
        target_hash: text
        for target_hash, text in texts.items()
        if target_hash not in evidence
    }
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                compile_lean_text,
                text,
                command=command,
                cwd=cwd,
                lean_version=lean_version,
                environment_revision=environment_revision,
                mathlib_revision=mathlib_revision,
                timeout_seconds=timeout_seconds,
            ): target_hash
            for target_hash, text in missing.items()
        }
        for future in as_completed(futures):
            result = future.result()
            evidence[result.target_hash] = result
            if checkpoint is not None:
                checkpoint(evidence)
    return {target_hash: evidence[target_hash] for target_hash in sorted(texts)}


def load_human_validations(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    validations: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Human validation line {line_number} is not an object")
            pair_id = str(value.get("pair_id", ""))
            if not pair_id or pair_id in validations:
                raise ValueError(
                    f"Missing or duplicate human validation pair_id at line {line_number}"
                )
            validations[pair_id] = value
    return validations


def _human_pair_approved(
    validation: Mapping[str, Any] | None,
    *,
    pair_id: str,
    positive_hash: str,
    negative_hash: str,
) -> tuple[bool, str]:
    if validation is None:
        return False, "human_validation_missing"
    required = {
        "pair_id": pair_id,
        "decision": "approved",
        "policy": "paired_process_validity_v1",
        "positive_hash": positive_hash,
        "negative_hash": negative_hash,
        "reviewed_statement_change": True,
    }
    for key, expected in required.items():
        if validation.get(key) != expected:
            return False, f"human_validation_mismatch:{key}"
    if not str(validation.get("validator", "")).strip():
        return False, "human_validator_missing"
    if not str(validation.get("notes", "")).strip():
        return False, "human_notes_missing"
    return True, "paired_human_validation_approved"


def join_cavis_eligibility(
    rows: Sequence[Mapping[str, Any]],
    *,
    compiler_evidence: Mapping[str, LeanCompileEvidence],
    human_validations: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join evidence without promoting raw compiler rejection to validity."""

    output = [dict(row) for row in rows]
    by_item = {str(row["item_id"]): row for row in output}
    if len(by_item) != len(output):
        raise ValueError("Prepared LeanTwin item_id values must be unique")
    validations = human_validations or {}
    g1_pair_ids = [
        str(row.get("pair_id") or "")
        for row in output
        if row.get("transform_kind") == "g1"
    ]
    if any(not pair_id for pair_id in g1_pair_ids):
        raise ValueError("Every prepared G1 root must have a non-empty pair_id")
    if len(g1_pair_ids) != len(set(g1_pair_ids)):
        raise ValueError("Prepared LeanTwin G1 pair_id values must be unique")
    unknown_validations = sorted(set(validations) - set(g1_pair_ids))
    if unknown_validations:
        preview = ", ".join(unknown_validations[:3])
        raise ValueError(
            "Human validation ledger contains pair IDs absent from the "
            f"prepared LeanTwin input: {preview}"
        )

    def evidence_for(row: Mapping[str, Any]) -> LeanCompileEvidence:
        target_hash = str(row["target_hash"])
        try:
            return compiler_evidence[target_hash]
        except KeyError as exc:
            raise ValueError(f"Missing compiler evidence for {target_hash}") from exc

    # Evaluate G1 roots first so their approvals can gate negative G0 children.
    pair_approved: dict[str, bool] = {}
    for row in output:
        if row.get("transform_kind") != "g1":
            continue
        parent = by_item[str(row["parent_variant_id"])]
        source_evidence = evidence_for(parent)
        target_evidence = evidence_for(row)
        statement = declaration_statement_changed(
            str(parent["reasoning"]), str(row["reasoning"])
        )
        pair_id = str(row["pair_id"])
        human_ok, human_reason = _human_pair_approved(
            validations.get(pair_id),
            pair_id=pair_id,
            positive_hash=str(row["source_hash"]),
            negative_hash=str(row["target_hash"]),
        )
        compiler_pattern = (
            source_evidence.returncode == 0
            and target_evidence.returncode is not None
            and target_evidence.returncode != 0
        )
        approved = compiler_pattern and statement.verified and human_ok
        pair_approved[pair_id] = approved
        row.update(
            {
                "source_returncode": source_evidence.returncode,
                "target_returncode": target_evidence.returncode,
                "statement_changed": statement.verified,
                "syntactic_contract_verified": False,
                "compiler_outcomes_match": compiler_pattern,
                "paired_validity_approved": approved,
                "mechanically_verified": False,
                "cavis_eligible": approved,
                "evidence_state": (
                    "paired_validity_approved"
                    if approved
                    else "paired_validity_ineligible"
                ),
                "semantic_status": (
                    "paired_process_invalidity_approved"
                    if approved
                    else "not_established"
                ),
                "eligibility_reason": (
                    "source_compiles_target_rejects_statement_changed_and_human_approved"
                    if approved
                    else ";".join(
                        reason
                        for condition, reason in (
                            (compiler_pattern, "compiler_pattern_failed"),
                            (statement.verified, statement.reason),
                            (human_ok, human_reason),
                        )
                        if not condition
                    )
                ),
            }
        )

    for row in output:
        kind = str(row.get("transform_kind", ""))
        target_evidence = evidence_for(row)
        if kind == "g1":
            continue
        if kind == "base":
            is_valid = row.get("label") == 1
            eligible = bool(is_valid and target_evidence.returncode == 0)
            row.update(
                {
                    "source_returncode": target_evidence.returncode,
                    "target_returncode": target_evidence.returncode,
                    "statement_changed": False,
                    "syntactic_contract_verified": False,
                    "compiler_outcomes_match": True,
                    "paired_validity_approved": False,
                    "mechanically_verified": eligible,
                    "cavis_eligible": eligible,
                    "evidence_state": (
                        "base_compiles" if eligible else "base_ineligible"
                    ),
                    "semantic_status": (
                        "upstream_validity_lean_checked"
                        if eligible
                        else "not_established"
                    ),
                    "eligibility_reason": (
                        "upstream_valid_label_and_lean_compile"
                        if eligible
                        else "base_not_positive_compiling"
                    ),
                }
            )
            continue
        if kind != "g0":
            row.update(
                {
                    "cavis_eligible": False,
                    "eligibility_reason": f"unknown_transform_kind:{kind}",
                }
            )
            continue
        parent = by_item[str(row["parent_variant_id"])]
        source_evidence = evidence_for(parent)
        try:
            parameters = json.loads(str(row["transform_parameters_json"]))
        except json.JSONDecodeError:
            parameters = {}
        contract = check_g0_contract(
            str(row["transform_name"]),
            str(parent["reasoning"]),
            str(row["reasoning"]),
            parameters,
        )
        both_compile = (
            source_evidence.returncode == 0 and target_evidence.returncode == 0
        )
        both_reject = (
            source_evidence.returncode is not None
            and target_evidence.returncode is not None
            and source_evidence.returncode != 0
            and target_evidence.returncode != 0
        )
        outcomes_match = both_compile or both_reject
        pair_side = row.get("pair_side")
        if pair_side == "positive":
            parent_approved = both_compile
        elif pair_side == "negative":
            pair_id = str(row.get("pair_id") or "")
            parent_approved = both_reject and pair_approved.get(pair_id, False)
        else:
            parent_approved = False
        eligible = contract.verified and outcomes_match and parent_approved
        row.update(
            {
                "source_returncode": source_evidence.returncode,
                "target_returncode": target_evidence.returncode,
                "statement_changed": False,
                "syntactic_contract_verified": contract.verified,
                "compiler_outcomes_match": outcomes_match,
                "paired_validity_approved": (
                    bool(pair_approved.get(str(row.get("pair_id")), False))
                    if pair_side == "negative"
                    else False
                ),
                # Core TransformEvidence reserves this field for successful
                # compile/compile verification.  Negative reject/reject G0 uses
                # cavis_eligible instead.
                "mechanically_verified": eligible and both_compile,
                "cavis_eligible": eligible,
                "evidence_state": (
                    "g0_cavis_eligible" if eligible else "g0_ineligible"
                ),
                "semantic_status": (
                    "g0_rendering_contract_verified"
                    if eligible
                    else "not_established"
                ),
                "eligibility_reason": (
                    (
                        "exact_g0_contract_and_compile_compile"
                        if both_compile
                        else "exact_g0_contract_and_reject_reject_after_pair_approval"
                    )
                    if eligible
                    else ";".join(
                        reason
                        for condition, reason in (
                            (contract.verified, contract.reason),
                            (outcomes_match, "compiler_outcomes_differ"),
                            (parent_approved, "parent_validity_evidence_missing"),
                        )
                        if not condition
                    )
                ),
            }
        )

    counts = Counter(
        f"{row.get('transform_kind')}:{'eligible' if row.get('cavis_eligible') else 'ineligible'}"
        for row in output
    )
    pair_structure: dict[str, dict[str, int]] = {}
    for pair_id in sorted(pair_approved):
        negative_rows = [
            row
            for row in output
            if row.get("pair_id") == pair_id and row.get("pair_side") == "negative"
        ]
        positive_id = str(negative_rows[0]["positive_semantic_variant_id"])
        positive_rows = [
            row
            for row in output
            if row.get("semantic_variant_id") == positive_id
            and row.get("pair_side") == "positive"
        ]
        pair_structure[pair_id] = {
            "positive_g0": sum(
                row.get("transform_kind") == "g0"
                and bool(row.get("cavis_eligible"))
                for row in positive_rows
            ),
            "negative_g0": sum(
                row.get("transform_kind") == "g0"
                and bool(row.get("cavis_eligible"))
                for row in negative_rows
            ),
            "pair_approved": int(pair_approved[pair_id]),
        }
    report = {
        "counts": dict(counts),
        "pairs": pair_structure,
        "eligible_pairs_with_both_radii": sum(
            value["pair_approved"]
            and value["positive_g0"] > 0
            and value["negative_g0"] > 0
            for value in pair_structure.values()
        ),
        "scope_boundary": (
            "Compiler rejection alone never establishes theorem invalidity; "
            "G1 eligibility requires hash-bound human paired-validity approval."
        ),
    }
    return output, report


def parse_command(value: str) -> tuple[str, ...]:
    command = tuple(shlex.split(value))
    if not command:
        raise ValueError("Lean command cannot be empty")
    return command

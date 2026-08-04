"""Fail-closed exclusion of canonical Lean dependencies after fixed-budget timeouts."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cavis.data.dependence import enrich_prepared_dependence_ids
from cavis.data.io import canonical_json_hash, sha256_file, sha256_text
from cavis.reproducibility.implementation import (
    verify_implementation_fingerprint,
)

from .eligibility import LeanCompileEvidence

RESOURCE_EXCLUSION_SCHEMA = "cavis.lean_resource_exclusion.v1"
RESOURCE_EXCLUSION_POLICY = (
    "fixed_budget_timeout_canonical_dependence_exclusion"
)
RESOURCE_EXCLUSION_IMPLEMENTATION_PATHS = (
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "scripts/verify_leantwin.py",
    "src/cavis/data/dependence.py",
    "src/cavis/data/io.py",
    "src/cavis/reproducibility/implementation.py",
    "src/cavis/transforms/eligibility.py",
    "src/cavis/transforms/resource_exclusion.py",
)
_TIMEOUT_PATTERN = re.compile(
    r"timed out after\s+(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s+seconds",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LeanResourceExclusionPlan:
    """Canonical closure and retained evidence for one fixed timeout budget."""

    rows: tuple[dict[str, Any], ...]
    retained_rows: tuple[dict[str, Any], ...]
    excluded_rows: tuple[dict[str, Any], ...]
    retained_evidence: Mapping[str, LeanCompileEvidence]
    trigger_evidence: tuple[LeanCompileEvidence, ...]
    excluded_dependence_ids: tuple[str, ...]
    excluded_group_ids: tuple[str, ...]
    excluded_target_hashes: tuple[str, ...]
    missing_excluded_target_hashes: tuple[str, ...]


def _nonnegative_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}: expected a non-negative integer")
    return value


def _bound_file_metadata(
    value: Any,
    *,
    context: str,
) -> tuple[Path, str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: file metadata must be an object")
    raw_path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{context}: path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f"{context}: path must be absolute and resolved")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{context}: invalid SHA-256")
    records = _nonnegative_integer(
        value.get("records"),
        context=f"{context}.records",
    )
    if not path.is_file():
        raise FileNotFoundError(f"{context}: bound file is missing: {path}")
    if sha256_file(path) != digest:
        raise ValueError(f"{context}: bound file hash changed: {path}")
    return path, digest, records


def _sorted_unique_strings(value: Any, *, context: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(
            f"{context}: expected a sorted list of unique non-empty strings"
        )
    return value


def verify_resource_exclusion_report(
    *,
    report_path: Path,
    retained_source_path: Path,
    retained_evidence_path: Path,
) -> tuple[dict[str, Any], str]:
    """Verify the fixed-budget report and its complete immutable file chain."""

    report_path = report_path.resolve()
    retained_source_path = retained_source_path.resolve()
    retained_evidence_path = retained_evidence_path.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Lean resource exclusion report is missing: {report_path}"
        )
    report_hash = sha256_file(report_path)
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"{report_path}: invalid Lean resource exclusion report: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"{report_path}: Lean resource exclusion report must be an object"
        )
    if value.get("schema_version") != RESOURCE_EXCLUSION_SCHEMA:
        raise ValueError(
            f"{report_path}: unsupported Lean resource exclusion schema"
        )
    if value.get("policy") != RESOURCE_EXCLUSION_POLICY:
        raise ValueError(
            f"{report_path}: Lean resource exclusion policy mismatch"
        )
    verify_implementation_fingerprint(
        value.get("implementation_fingerprint"),
        relative_paths=RESOURCE_EXCLUSION_IMPLEMENTATION_PATHS,
        context=f"{report_path}: Lean resource exclusion producer",
    )

    fixed_budget = value.get("fixed_budget")
    if not isinstance(fixed_budget, Mapping):
        raise ValueError(f"{report_path}: fixed_budget is missing")
    timeout_seconds = fixed_budget.get("timeout_seconds")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
        or fixed_budget.get("formal_policy_attempts_per_unique_target") != 1
        or fixed_budget.get("timeout_state") != "error"
        or fixed_budget.get("timeout_returncode") is not None
        or fixed_budget.get("historical_attempt_count_encoded_in_v2") is not False
        or fixed_budget.get("prior_exploratory_attempts_may_exist") is not True
        or fixed_budget.get("rerun_policy")
        != "reuse_timeout_and_never_retry_after_policy_freeze"
    ):
        raise ValueError(f"{report_path}: invalid fixed timeout budget")

    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError(f"{report_path}: execution metadata is missing")
    command = execution.get("command")
    if (
        _nonnegative_integer(
            execution.get("jobs"),
            context=f"{report_path}: execution.jobs",
        )
        < 1
        or not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
        or execution.get("validator_worktrees_clean") is not True
    ):
        raise ValueError(f"{report_path}: invalid execution metadata")
    for field in ("lean_version", "environment_revision", "mathlib_revision"):
        field_value = execution.get(field)
        if not isinstance(field_value, str) or not field_value:
            raise ValueError(
                f"{report_path}: execution.{field} must be a non-empty string"
            )

    inputs = value.get("inputs")
    outputs = value.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError(f"{report_path}: input/output bindings are missing")
    _, _, source_records = _bound_file_metadata(
        inputs.get("source_parquet"),
        context=f"{report_path}: inputs.source_parquet",
    )
    _, _, raw_evidence_records = _bound_file_metadata(
        inputs.get("raw_compiler_evidence"),
        context=f"{report_path}: inputs.raw_compiler_evidence",
    )
    human_metadata = inputs.get("human_validations")
    if human_metadata is not None:
        _bound_file_metadata(
            human_metadata,
            context=f"{report_path}: inputs.human_validations",
        )
    bound_source, _, retained_source_records = _bound_file_metadata(
        outputs.get("retained_source_parquet"),
        context=f"{report_path}: outputs.retained_source_parquet",
    )
    bound_evidence, _, retained_evidence_records = _bound_file_metadata(
        outputs.get("retained_compiler_evidence"),
        context=f"{report_path}: outputs.retained_compiler_evidence",
    )
    if bound_source != retained_source_path:
        raise ValueError(
            f"{report_path}: retained source path does not match the formal input"
        )
    if bound_evidence != retained_evidence_path:
        raise ValueError(
            f"{report_path}: retained evidence path does not match the formal input"
        )

    counts = value.get("counts")
    exclusion = value.get("exclusion")
    triggers = value.get("triggers")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(exclusion, Mapping)
        or not isinstance(triggers, list)
    ):
        raise ValueError(
            f"{report_path}: counts, exclusion, and triggers are required"
        )
    count_fields = (
        "input_rows",
        "input_unique_targets",
        "timeout_triggers",
        "excluded_dependencies",
        "excluded_groups",
        "excluded_rows",
        "excluded_unique_targets",
        "missing_excluded_targets",
        "retained_rows",
        "retained_unique_targets",
    )
    parsed_counts = {
        field: _nonnegative_integer(
            counts.get(field),
            context=f"{report_path}: counts.{field}",
        )
        for field in count_fields
    }
    dependence_ids = _sorted_unique_strings(
        exclusion.get("dependence_ids"),
        context=f"{report_path}: exclusion.dependence_ids",
    )
    group_ids = _sorted_unique_strings(
        exclusion.get("group_ids"),
        context=f"{report_path}: exclusion.group_ids",
    )
    target_hashes = _sorted_unique_strings(
        exclusion.get("target_hashes"),
        context=f"{report_path}: exclusion.target_hashes",
    )
    missing_target_hashes = _sorted_unique_strings(
        exclusion.get("missing_target_hashes"),
        context=f"{report_path}: exclusion.missing_target_hashes",
    )
    if not set(missing_target_hashes).issubset(target_hashes):
        raise ValueError(
            f"{report_path}: missing targets lie outside the excluded closure"
        )
    expected_hashes = {
        "dependence_ids_sha256": canonical_json_hash(dependence_ids),
        "group_ids_sha256": canonical_json_hash(group_ids),
        "target_hashes_sha256": canonical_json_hash(target_hashes),
    }
    if any(
        exclusion.get(field) != expected
        for field, expected in expected_hashes.items()
    ):
        raise ValueError(f"{report_path}: exclusion list hash mismatch")
    if (
        exclusion.get("dependence_unit")
        != "canonical_lean_statement_dependence_id"
    ):
        raise ValueError(f"{report_path}: exclusion dependence unit mismatch")

    expected_count_values = {
        "input_rows": source_records,
        "timeout_triggers": len(triggers),
        "excluded_dependencies": len(dependence_ids),
        "excluded_groups": len(group_ids),
        "excluded_unique_targets": len(target_hashes),
        "missing_excluded_targets": len(missing_target_hashes),
        "retained_rows": retained_source_records,
        "retained_unique_targets": retained_evidence_records,
    }
    if any(
        parsed_counts[field] != expected
        for field, expected in expected_count_values.items()
    ):
        raise ValueError(f"{report_path}: resource exclusion count mismatch")
    if (
        parsed_counts["input_rows"]
        != parsed_counts["retained_rows"] + parsed_counts["excluded_rows"]
        or parsed_counts["input_unique_targets"]
        != parsed_counts["retained_unique_targets"]
        + parsed_counts["excluded_unique_targets"]
        or raw_evidence_records
        != parsed_counts["input_unique_targets"]
        - parsed_counts["missing_excluded_targets"]
    ):
        raise ValueError(f"{report_path}: resource attrition arithmetic mismatch")

    trigger_hashes: set[str] = set()
    triggered_dependencies: set[str] = set()
    for index, trigger in enumerate(triggers, start=1):
        if not isinstance(trigger, Mapping):
            raise ValueError(f"{report_path}: trigger {index} must be an object")
        target_hash = trigger.get("target_hash")
        observed_timeout = trigger.get("observed_timeout_seconds")
        if (
            not isinstance(target_hash, str)
            or target_hash not in target_hashes
            or target_hash in trigger_hashes
            or trigger.get("state") != "error"
            or trigger.get("returncode") is not None
            or trigger.get("error_kind") != "timeout"
            or isinstance(observed_timeout, bool)
            or not isinstance(observed_timeout, (int, float))
            or not math.isclose(
                float(observed_timeout),
                float(timeout_seconds),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(f"{report_path}: invalid timeout trigger {index}")
        trigger_hashes.add(target_hash)
        trigger_dependencies = _sorted_unique_strings(
            trigger.get("dependence_ids"),
            context=f"{report_path}: trigger {index}.dependence_ids",
        )
        trigger_groups = _sorted_unique_strings(
            trigger.get("group_ids"),
            context=f"{report_path}: trigger {index}.group_ids",
        )
        if (
            not trigger_dependencies
            or not set(trigger_dependencies).issubset(dependence_ids)
            or not set(trigger_groups).issubset(group_ids)
        ):
            raise ValueError(
                f"{report_path}: timeout trigger {index} escapes its closure"
            )
        triggered_dependencies.update(trigger_dependencies)

    has_exclusion = bool(dependence_ids)
    if (
        triggered_dependencies != set(dependence_ids)
        or bool(trigger_hashes) != has_exclusion
        or bool(group_ids) != has_exclusion
        or bool(target_hashes) != has_exclusion
        or (parsed_counts["excluded_rows"] > 0) != has_exclusion
    ):
        raise ValueError(
            f"{report_path}: timeout triggers do not justify the complete "
            "canonical exclusion closure"
        )

    if sha256_file(report_path) != report_hash:
        raise ValueError(
            f"{report_path}: Lean resource exclusion changed while being read"
        )
    return value, report_hash


def timeout_seconds_from_evidence(
    evidence: LeanCompileEvidence,
) -> float | None:
    """Return the recorded TimeoutExpired budget, never a rejection code."""

    if evidence.returncode is not None or evidence.state != "error":
        return None
    match = _TIMEOUT_PATTERN.search(evidence.stderr_tail)
    if match is None or "TimeoutExpired" not in evidence.stderr_tail:
        return None
    return float(match.group("seconds"))


def timeout_trigger_hashes(
    *,
    evidence: Mapping[str, LeanCompileEvidence],
    expected_target_hashes: set[str],
    timeout_seconds: float,
) -> set[str]:
    """Classify in-scope execution errors under one objective timeout budget."""

    triggers: set[str] = set()
    for target_hash, record in evidence.items():
        if target_hash not in expected_target_hashes:
            continue
        if record.returncode is not None:
            expected_state = "compile" if record.returncode == 0 else "reject"
            if record.state != expected_state:
                raise ValueError(
                    f"Lean evidence {target_hash}: state/returncode mismatch"
                )
            continue
        observed_timeout = timeout_seconds_from_evidence(record)
        if observed_timeout is None:
            raise RuntimeError(
                "Lean execution error is not a fixed-budget TimeoutExpired "
                f"and cannot be excluded: {target_hash}"
            )
        if not math.isclose(
            observed_timeout,
            timeout_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                f"Lean timeout budget mismatch for {target_hash}: evidence "
                f"records {observed_timeout:g}s, invocation requires "
                f"{timeout_seconds:g}s"
            )
        triggers.add(target_hash)
    return triggers


def precompile_resource_filter(
    rows: Sequence[Mapping[str, Any]],
    *,
    existing_evidence: Mapping[str, LeanCompileEvidence],
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Remove previously timed-out dependencies before scheduling any work."""

    enriched, _ = enrich_prepared_dependence_ids(rows)
    expected_hashes = {
        sha256_text(str(row["reasoning"])) for row in enriched
    }
    trigger_hashes = timeout_trigger_hashes(
        evidence=existing_evidence,
        expected_target_hashes=expected_hashes,
        timeout_seconds=timeout_seconds,
    )
    excluded_dependencies = {
        str(row["dependence_id"])
        for row in enriched
        if str(row["target_hash"]) in trigger_hashes
    }
    retained = [
        row
        for row in enriched
        if str(row["dependence_id"]) not in excluded_dependencies
    ]
    return retained, excluded_dependencies, trigger_hashes


def build_resource_exclusion_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    compiler_evidence: Mapping[str, LeanCompileEvidence],
    timeout_seconds: float,
) -> LeanResourceExclusionPlan:
    """Exclude every row sharing a canonical dependence with a timeout.

    Missing evidence is permitted only inside a dependence already closed by
    an observed fixed-budget timeout. This lets an interrupted parallel batch
    terminate without treating never-launched siblings as compiler rejection.
    """

    enriched, _ = enrich_prepared_dependence_ids(rows)
    hash_to_dependencies: dict[str, set[str]] = defaultdict(set)
    expected_hashes: set[str] = set()
    for index, row in enumerate(enriched, start=1):
        reasoning = row.get("reasoning")
        if not isinstance(reasoning, str):
            raise ValueError(f"Lean input row {index}: reasoning must be a string")
        target_hash = sha256_text(reasoning)
        if row.get("target_hash") not in (None, target_hash):
            raise ValueError(
                f"Lean input row {index}: target_hash does not match reasoning"
            )
        expected_hashes.add(target_hash)
        hash_to_dependencies[target_hash].add(str(row["dependence_id"]))

    extra_evidence = sorted(set(compiler_evidence) - expected_hashes)
    if extra_evidence:
        raise ValueError(
            "Compiler evidence contains target hashes outside the Lean input: "
            + ", ".join(extra_evidence[:3])
        )
    trigger_hashes = timeout_trigger_hashes(
        evidence=compiler_evidence,
        expected_target_hashes=expected_hashes,
        timeout_seconds=timeout_seconds,
    )
    excluded_dependencies = {
        dependence_id
        for target_hash in trigger_hashes
        for dependence_id in hash_to_dependencies[target_hash]
    }
    retained_rows = [
        row
        for row in enriched
        if str(row["dependence_id"]) not in excluded_dependencies
    ]
    excluded_rows = [
        row
        for row in enriched
        if str(row["dependence_id"]) in excluded_dependencies
    ]
    retained_hashes = {
        sha256_text(str(row["reasoning"])) for row in retained_rows
    }
    excluded_hashes = {
        sha256_text(str(row["reasoning"])) for row in excluded_rows
    }
    missing_retained = sorted(retained_hashes - set(compiler_evidence))
    if missing_retained:
        raise RuntimeError(
            "Compiler evidence is incomplete outside resource-excluded "
            f"dependencies: {', '.join(missing_retained[:3])}"
        )
    retained_evidence: dict[str, LeanCompileEvidence] = {}
    for target_hash in sorted(retained_hashes):
        record = compiler_evidence[target_hash]
        expected_state = "compile" if record.returncode == 0 else "reject"
        if (
            record.returncode is None
            or record.state not in {"compile", "reject"}
            or record.state != expected_state
        ):
            raise RuntimeError(
                "Resource timeout closure failed to remove non-terminal Lean "
                f"evidence: {target_hash}"
            )
        retained_evidence[target_hash] = record

    return LeanResourceExclusionPlan(
        rows=tuple(enriched),
        retained_rows=tuple(retained_rows),
        excluded_rows=tuple(excluded_rows),
        retained_evidence=retained_evidence,
        trigger_evidence=tuple(
            compiler_evidence[target_hash]
            for target_hash in sorted(trigger_hashes)
        ),
        excluded_dependence_ids=tuple(sorted(excluded_dependencies)),
        excluded_group_ids=tuple(
            sorted({str(row["group_id"]) for row in excluded_rows})
        ),
        excluded_target_hashes=tuple(sorted(excluded_hashes)),
        missing_excluded_target_hashes=tuple(
            sorted(excluded_hashes - set(compiler_evidence))
        ),
    )


def resource_exclusion_core(
    plan: LeanResourceExclusionPlan,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Return deterministic scientific contents for the hash-bound report."""

    triggers = [
        {
            "target_hash": record.target_hash,
            "dependence_ids": sorted(
                {
                    str(row["dependence_id"])
                    for row in plan.excluded_rows
                    if str(row["target_hash"]) == record.target_hash
                }
            ),
            "group_ids": sorted(
                {
                    str(row["group_id"])
                    for row in plan.excluded_rows
                    if str(row["target_hash"]) == record.target_hash
                }
            ),
            "state": record.state,
            "returncode": record.returncode,
            "error_kind": "timeout",
            "observed_timeout_seconds": timeout_seconds_from_evidence(record),
            "stderr_hash": record.stderr_hash,
            "evidence_record_sha256": canonical_json_hash(record.to_dict()),
        }
        for record in plan.trigger_evidence
    ]
    return {
        "schema_version": RESOURCE_EXCLUSION_SCHEMA,
        "policy": RESOURCE_EXCLUSION_POLICY,
        "fixed_budget": {
            "formal_policy_attempts_per_unique_target": 1,
            "timeout_seconds": timeout_seconds,
            "timeout_state": "error",
            "timeout_returncode": None,
            "historical_attempt_count_encoded_in_v2": False,
            "prior_exploratory_attempts_may_exist": True,
            "rerun_policy": (
                "reuse_timeout_and_never_retry_after_policy_freeze"
            ),
        },
        "triggers": triggers,
        "exclusion": {
            "dependence_unit": "canonical_lean_statement_dependence_id",
            "dependence_ids": list(plan.excluded_dependence_ids),
            "group_ids": list(plan.excluded_group_ids),
            "target_hashes": list(plan.excluded_target_hashes),
            "missing_target_hashes": list(
                plan.missing_excluded_target_hashes
            ),
            "dependence_ids_sha256": canonical_json_hash(
                list(plan.excluded_dependence_ids)
            ),
            "group_ids_sha256": canonical_json_hash(
                list(plan.excluded_group_ids)
            ),
            "target_hashes_sha256": canonical_json_hash(
                list(plan.excluded_target_hashes)
            ),
        },
        "counts": {
            "input_rows": len(plan.rows),
            "input_unique_targets": len(
                {str(row["target_hash"]) for row in plan.rows}
            ),
            "timeout_triggers": len(plan.trigger_evidence),
            "excluded_dependencies": len(plan.excluded_dependence_ids),
            "excluded_groups": len(plan.excluded_group_ids),
            "excluded_rows": len(plan.excluded_rows),
            "excluded_unique_targets": len(plan.excluded_target_hashes),
            "missing_excluded_targets": len(
                plan.missing_excluded_target_hashes
            ),
            "retained_rows": len(plan.retained_rows),
            "retained_unique_targets": len(plan.retained_evidence),
        },
        "scope_boundary": (
            "Timeout is an execution/resource outcome, never evidence that a "
            "Lean theorem is false. The complete canonical dependence is "
            "removed from review, fitting, calibration, and evaluation."
        ),
    }


__all__ = [
    "LeanResourceExclusionPlan",
    "RESOURCE_EXCLUSION_IMPLEMENTATION_PATHS",
    "RESOURCE_EXCLUSION_POLICY",
    "RESOURCE_EXCLUSION_SCHEMA",
    "build_resource_exclusion_plan",
    "precompile_resource_filter",
    "resource_exclusion_core",
    "timeout_seconds_from_evidence",
    "timeout_trigger_hashes",
    "verify_resource_exclusion_report",
]

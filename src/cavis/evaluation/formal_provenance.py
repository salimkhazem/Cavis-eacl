"""Fail-closed provenance checks for confirmatory LeanTwin evaluation.

The extraction launcher already binds formal GPU jobs to the post-review
design and eligibility reports.  This module independently verifies the same
chain when score caches are consumed, so a copied or partially written
``scores.jsonl`` cannot silently enter a confirmatory table.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cavis.data.io import canonical_json_hash
from cavis.reproducibility.config import load_yaml
from cavis.reproducibility.io import read_jsonl, sha256_file
from cavis.transforms.resource_exclusion import (
    verify_resource_exclusion_report,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class FormalEvaluationBinding:
    """Exact immutable inputs authorized for confirmatory evaluation."""

    design_report_path: Path
    design_report_sha256: str
    eligibility_report_path: Path
    eligibility_report_sha256: str
    merge_report_path: Path
    merge_report_sha256: str
    formal_source_path: Path
    formal_source_sha256: str
    compiler_evidence_path: Path
    compiler_evidence_sha256: str
    resource_exclusion_path: Path
    resource_exclusion_sha256: str
    combined_validations_path: Path
    combined_validations_sha256: str
    formal_parquet_path: Path
    formal_parquet_sha256: str
    exclusion_path: Path
    exclusion_sha256: str
    excluded_dependence_ids: tuple[str, ...]

    @property
    def execution_preflight(self) -> dict[Path, str]:
        """Artifacts that every formal extraction signature must bind."""

        return {
            self.design_report_path: self.design_report_sha256,
            self.eligibility_report_path: self.eligibility_report_sha256,
            self.merge_report_path: self.merge_report_sha256,
            self.formal_source_path: self.formal_source_sha256,
            self.compiler_evidence_path: self.compiler_evidence_sha256,
            self.resource_exclusion_path: self.resource_exclusion_sha256,
            self.combined_validations_path: self.combined_validations_sha256,
            self.formal_parquet_path: self.formal_parquet_sha256,
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "design_report": _metadata(
                self.design_report_path,
                self.design_report_sha256,
            ),
            "eligibility_report": _metadata(
                self.eligibility_report_path,
                self.eligibility_report_sha256,
            ),
            "merge_report": _metadata(
                self.merge_report_path,
                self.merge_report_sha256,
            ),
            "formal_source": _metadata(
                self.formal_source_path,
                self.formal_source_sha256,
            ),
            "compiler_evidence": _metadata(
                self.compiler_evidence_path,
                self.compiler_evidence_sha256,
            ),
            "resource_exclusion": _metadata(
                self.resource_exclusion_path,
                self.resource_exclusion_sha256,
            ),
            "combined_validations": _metadata(
                self.combined_validations_path,
                self.combined_validations_sha256,
            ),
            "formal_parquet": _metadata(
                self.formal_parquet_path,
                self.formal_parquet_sha256,
            ),
            "dependence_exclusion": {
                **_metadata(self.exclusion_path, self.exclusion_sha256),
                "selected_dependence_ids": list(self.excluded_dependence_ids),
                "selected_dependence_count": len(self.excluded_dependence_ids),
            },
        }


def _metadata(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path), "sha256": digest}


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    digest = sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{path}: invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {label} must be a JSON object")
    if sha256_file(path) != digest:
        raise ValueError(f"{path}: {label} changed while being read")
    return value, digest


def _clean_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context}: sha256 must be 64 lowercase hexadecimal digits")
    return value


def _clean_path(value: Any, *, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: path must be a non-empty string")
    return Path(value).resolve()


def _bound_file(metadata: Any, *, context: str) -> tuple[Path, str]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context}: file metadata must be an object")
    path = _clean_path(metadata.get("path"), context=context)
    digest = _clean_sha256(metadata.get("sha256"), context=context)
    if not path.is_file():
        raise FileNotFoundError(f"{context}: bound file is missing: {path}")
    if sha256_file(path) != digest:
        raise ValueError(f"{context}: bound file hash changed: {path}")
    return path, digest


def _require_same_binding(
    metadata: Any,
    *,
    expected_path: Path,
    expected_sha256: str,
    context: str,
) -> None:
    path, digest = _bound_file(metadata, context=context)
    if path != expected_path or digest != expected_sha256:
        raise ValueError(f"{context}: path/hash does not match the frozen chain")


def _require_provenance_path_hash(
    provenance: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    expected_path: Path,
    expected_sha256: str,
    context: str,
) -> None:
    actual_path = _clean_path(provenance.get(path_field), context=context)
    actual_hash = _clean_sha256(provenance.get(hash_field), context=context)
    if actual_path != expected_path or actual_hash != expected_sha256:
        raise ValueError(f"{context}: path/hash does not match the frozen chain")
    if not actual_path.is_file() or sha256_file(actual_path) != actual_hash:
        raise ValueError(f"{context}: current file does not match its frozen hash")


def validate_formal_evaluation_binding(
    *,
    design_report_path: Path,
    eligibility_report_path: Path,
    merge_report_path: Path,
    dependence_input_path: Path,
    exclude_selection_paths: Sequence[Path],
) -> FormalEvaluationBinding:
    """Validate design→merge→eligibility and return exact evaluation inputs."""

    design_report_path = design_report_path.resolve()
    eligibility_report_path = eligibility_report_path.resolve()
    merge_report_path = merge_report_path.resolve()
    dependence_input_path = dependence_input_path.resolve()
    if len(exclude_selection_paths) != 1:
        raise ValueError(
            "Confirmatory LeanTwin evaluation requires exactly one formal "
            "dependence exclusion file"
        )
    supplied_exclusion_path = exclude_selection_paths[0].resolve()

    design, design_hash = _read_json_object(
        design_report_path,
        label="formal design report",
    )
    if (
        design.get("schema_version")
        != "cavis.formal_post_adjudication_design.v1"
        or design.get("passes") is not True
        or design.get("ready_for_merge") is not True
    ):
        raise ValueError(f"{design_report_path}: formal design gate did not pass")
    design_inputs = design.get("inputs")
    if not isinstance(design_inputs, Mapping):
        raise ValueError(f"{design_report_path}: formal design inputs are missing")
    source_path, source_hash = _bound_file(
        design_inputs.get("eligible_parquet"),
        context=f"{design_report_path}: eligible_parquet",
    )
    evidence_path, evidence_hash = _bound_file(
        design_inputs.get("compiler_evidence"),
        context=f"{design_report_path}: compiler_evidence",
    )
    resource_path, resource_hash = _bound_file(
        design_inputs.get("resource_exclusion"),
        context=f"{design_report_path}: resource_exclusion",
    )
    _, reread_resource_hash = verify_resource_exclusion_report(
        report_path=resource_path,
        retained_source_path=source_path,
        retained_evidence_path=evidence_path,
    )
    if reread_resource_hash != resource_hash:
        raise ValueError("Lean resource exclusion changed during validation")
    exclusion_path, exclusion_hash = _bound_file(
        design_inputs.get("dependence_exclusion"),
        context=f"{design_report_path}: dependence_exclusion",
    )
    if exclusion_path != supplied_exclusion_path:
        raise ValueError(
            "Confirmatory exclusion path is not the exact exclusion bound by "
            "the formal design report"
        )
    exclusion, reread_exclusion_hash = _read_json_object(
        exclusion_path,
        label="formal dependence exclusion",
    )
    if reread_exclusion_hash != exclusion_hash:
        raise ValueError("Formal dependence exclusion changed during validation")
    if (
        exclusion.get("schema_version")
        != "cavis.formal_dependence_exclusion.v1"
    ):
        raise ValueError(f"{exclusion_path}: unsupported exclusion schema")
    dependence_ids = exclusion.get("selected_dependence_ids")
    if (
        not isinstance(dependence_ids, list)
        or not dependence_ids
        or any(not isinstance(value, str) or not value for value in dependence_ids)
        or len(dependence_ids) != len(set(dependence_ids))
    ):
        raise ValueError(
            f"{exclusion_path}: selected_dependence_ids must be non-empty and unique"
        )
    recorded_exclusion_count = design_inputs["dependence_exclusion"].get(
        "excluded_dependencies"
    )
    if (
        recorded_exclusion_count is not None
        and recorded_exclusion_count != len(dependence_ids)
    ):
        raise ValueError(
            f"{design_report_path}: dependence exclusion count does not match"
        )

    merge, merge_hash = _read_json_object(
        merge_report_path,
        label="formal human-validation merge report",
    )
    if (
        merge.get("schema_version") != "cavis.human_validation_merge.v1"
        or merge.get("policy") != "paired_process_validity_v1"
    ):
        raise ValueError(f"{merge_report_path}: unsupported merge report schema")
    merge_inputs = merge.get("inputs")
    merge_output = merge.get("output")
    dependence_isolation = merge.get("dependence_isolation")
    if not isinstance(merge_inputs, Mapping) or not isinstance(merge_output, Mapping):
        raise ValueError(f"{merge_report_path}: merge provenance is incomplete")
    if (
        not isinstance(dependence_isolation, Mapping)
        or dependence_isolation.get("overlap_count") != 0
    ):
        raise ValueError(
            f"{merge_report_path}: pilot/formal dependence isolation did not pass"
        )
    _require_same_binding(
        merge_inputs.get("formal_design_report"),
        expected_path=design_report_path,
        expected_sha256=design_hash,
        context=f"{merge_report_path}: formal_design_report",
    )
    _require_same_binding(
        merge_inputs.get("formal_eligible_parquet"),
        expected_path=source_path,
        expected_sha256=source_hash,
        context=f"{merge_report_path}: formal_eligible_parquet",
    )
    _require_same_binding(
        merge_inputs.get("formal_compiler_evidence"),
        expected_path=evidence_path,
        expected_sha256=evidence_hash,
        context=f"{merge_report_path}: formal_compiler_evidence",
    )
    _require_same_binding(
        merge_inputs.get("formal_resource_exclusion"),
        expected_path=resource_path,
        expected_sha256=resource_hash,
        context=f"{merge_report_path}: formal_resource_exclusion",
    )
    _require_same_binding(
        merge_inputs.get("formal_dependence_exclusion"),
        expected_path=exclusion_path,
        expected_sha256=exclusion_hash,
        context=f"{merge_report_path}: formal_dependence_exclusion",
    )
    validations_path, validations_hash = _bound_file(
        merge_output,
        context=f"{merge_report_path}: output",
    )

    eligibility, eligibility_hash = _read_json_object(
        eligibility_report_path,
        label="formal eligibility report",
    )
    provenance = eligibility.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{eligibility_report_path}: provenance is missing")
    if provenance.get("evidence_mode") != "reused_read_only":
        raise ValueError(
            f"{eligibility_report_path}: compiler evidence was not reused read-only"
        )
    if (
        provenance.get("validator_worktrees_clean") is not True
        or provenance.get("environment_validated_before_and_after") is not True
    ):
        raise ValueError(
            f"{eligibility_report_path}: validator cleanliness checks are missing"
        )
    _require_provenance_path_hash(
        provenance,
        path_field="evidence_binding_report_path",
        hash_field="evidence_binding_report_sha256",
        expected_path=design_report_path,
        expected_sha256=design_hash,
        context=f"{eligibility_report_path}: design binding",
    )
    _require_provenance_path_hash(
        provenance,
        path_field="input_path",
        hash_field="input_sha256",
        expected_path=source_path,
        expected_sha256=source_hash,
        context=f"{eligibility_report_path}: formal source",
    )
    if provenance.get("bound_input_sha256") != source_hash:
        raise ValueError(f"{eligibility_report_path}: bound input hash mismatch")
    _require_provenance_path_hash(
        provenance,
        path_field="evidence_path",
        hash_field="evidence_sha256",
        expected_path=evidence_path,
        expected_sha256=evidence_hash,
        context=f"{eligibility_report_path}: compiler evidence",
    )
    if provenance.get("bound_evidence_sha256") != evidence_hash:
        raise ValueError(f"{eligibility_report_path}: bound evidence hash mismatch")
    _require_provenance_path_hash(
        provenance,
        path_field="formal_resource_exclusion_report_path",
        hash_field="formal_resource_exclusion_report_sha256",
        expected_path=resource_path,
        expected_sha256=resource_hash,
        context=f"{eligibility_report_path}: resource exclusion",
    )
    _require_provenance_path_hash(
        provenance,
        path_field="human_validations_path",
        hash_field="human_validations_sha256",
        expected_path=validations_path,
        expected_sha256=validations_hash,
        context=f"{eligibility_report_path}: combined validations",
    )
    _require_provenance_path_hash(
        provenance,
        path_field="human_validation_merge_report_path",
        hash_field="human_validation_merge_report_sha256",
        expected_path=merge_report_path,
        expected_sha256=merge_hash,
        context=f"{eligibility_report_path}: validation merge",
    )
    if (
        provenance.get("bound_human_validations_sha256") != validations_hash
        or provenance.get("human_validation_records")
        != merge_output.get("records")
    ):
        raise ValueError(
            f"{eligibility_report_path}: combined-validation binding mismatch"
        )
    formal_path = _clean_path(
        provenance.get("output_path"),
        context=f"{eligibility_report_path}: formal output",
    )
    formal_hash = _clean_sha256(
        provenance.get("output_sha256"),
        context=f"{eligibility_report_path}: formal output",
    )
    if formal_path != dependence_input_path:
        raise ValueError(
            "Dependence input is not the exact formal eligible Parquet bound by "
            "the eligibility report"
        )
    if not formal_path.is_file() or sha256_file(formal_path) != formal_hash:
        raise ValueError("Formal eligible Parquet is missing or hash-mismatched")

    frozen = {
        design_report_path: design_hash,
        eligibility_report_path: eligibility_hash,
        merge_report_path: merge_hash,
        source_path: source_hash,
        evidence_path: evidence_hash,
        resource_path: resource_hash,
        validations_path: validations_hash,
        formal_path: formal_hash,
        exclusion_path: exclusion_hash,
    }
    for path, expected_hash in frozen.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Formal evaluation input changed during validation: {path}")
    return FormalEvaluationBinding(
        design_report_path=design_report_path,
        design_report_sha256=design_hash,
        eligibility_report_path=eligibility_report_path,
        eligibility_report_sha256=eligibility_hash,
        merge_report_path=merge_report_path,
        merge_report_sha256=merge_hash,
        formal_source_path=source_path,
        formal_source_sha256=source_hash,
        compiler_evidence_path=evidence_path,
        compiler_evidence_sha256=evidence_hash,
        resource_exclusion_path=resource_path,
        resource_exclusion_sha256=resource_hash,
        combined_validations_path=validations_path,
        combined_validations_sha256=validations_hash,
        formal_parquet_path=formal_path,
        formal_parquet_sha256=formal_hash,
        exclusion_path=exclusion_path,
        exclusion_sha256=exclusion_hash,
        excluded_dependence_ids=tuple(sorted(dependence_ids)),
    )


def _resolved_hash_mapping(value: Any, *, context: str) -> dict[Path, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{context}: expected a non-empty path→sha256 mapping")
    output: dict[Path, str] = {}
    for raw_path, raw_hash in value.items():
        path = _clean_path(raw_path, context=context)
        digest = _clean_sha256(raw_hash, context=f"{context}: {raw_path}")
        if path in output:
            raise ValueError(f"{context}: duplicate resolved path {path}")
        output[path] = digest
    return output


def load_and_validate_score_cache(
    score_path: Path,
    *,
    binding: FormalEvaluationBinding,
    expected_experiment: str,
    expected_dataset: str,
    expected_model: str,
    expected_precision: str | None = None,
    expected_experiment_config_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one cache only after signature, manifest and formal bindings pass."""

    if (
        expected_experiment == "formal_audit"
        and expected_experiment_config_sha256 is None
    ):
        raise ValueError(
            "formal_audit score validation requires the frozen current "
            "experiment-config sha256"
        )
    if expected_experiment_config_sha256 is not None:
        expected_experiment_config_sha256 = _clean_sha256(
            expected_experiment_config_sha256,
            context="expected experiment config",
        )
    score_path = score_path.resolve()
    signature_path = score_path.parent / "job_signature.json"
    run_manifest_path = score_path.parent / "run_manifest.json"
    signature, signature_file_hash = _read_json_object(
        signature_path,
        label="job signature",
    )
    manifest, run_manifest_hash = _read_json_object(
        run_manifest_path,
        label="run manifest",
    )
    score_hash = sha256_file(score_path)
    records = read_jsonl(score_path)
    if sha256_file(score_path) != score_hash:
        raise ValueError(f"{score_path}: score cache changed while being read")

    if signature.get("schema_version") != 1:
        raise ValueError(f"{signature_path}: unsupported signature schema")
    recorded_signature = _clean_sha256(
        signature.get("signature"),
        context=f"{signature_path}: signature",
    )
    unsigned = {key: value for key, value in signature.items() if key != "signature"}
    if canonical_json_hash(unsigned) != recorded_signature:
        raise ValueError(f"{signature_path}: self-hash mismatch")
    expected_fields = {
        "experiment_key": expected_experiment,
        "dataset_key": expected_dataset,
        "model_key": expected_model,
    }
    for field, expected in expected_fields.items():
        if signature.get(field) != expected:
            raise ValueError(
                f"{signature_path}: {field} must be {expected!r}, "
                f"got {signature.get(field)!r}"
            )
    if (
        expected_experiment_config_sha256 is not None
        and signature.get("experiment_config_sha256")
        != expected_experiment_config_sha256
    ):
        raise ValueError(
            f"{signature_path}: experiment_config_sha256 does not match the "
            "frozen current experiment config"
        )
    if (
        expected_precision is not None
        and signature.get("precision_tag") != expected_precision
    ):
        raise ValueError(
            f"{signature_path}: precision_tag does not match its cache directory"
        )

    signature_input_path = _clean_path(
        signature.get("input_path"),
        context=f"{signature_path}: input",
    )
    signature_input_hash = _clean_sha256(
        signature.get("input_sha256"),
        context=f"{signature_path}: input",
    )
    if not signature_input_path.is_file():
        raise FileNotFoundError(
            f"{signature_path}: extraction input is missing: {signature_input_path}"
        )
    if sha256_file(signature_input_path) != signature_input_hash:
        raise ValueError(f"{signature_path}: extraction input hash mismatch")
    if expected_dataset == "geometry_leantwin":
        if signature.get("input_kind") != "verified":
            raise ValueError(f"{signature_path}: LeanTwin input_kind must be verified")
        if (
            signature_input_path != binding.formal_parquet_path
            or signature_input_hash != binding.formal_parquet_sha256
        ):
            raise ValueError(
                f"{signature_path}: LeanTwin cache does not use the exact formal Parquet"
            )

    recorded_preflight = _resolved_hash_mapping(
        signature.get("execution_preflight"),
        context=f"{signature_path}: execution_preflight",
    )
    expected_preflight = binding.execution_preflight
    if recorded_preflight != expected_preflight:
        raise ValueError(
            f"{signature_path}: execution_preflight does not exactly match "
            "the formal evaluation chain"
        )
    for path, digest in recorded_preflight.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(
                f"{signature_path}: preflight artifact is missing or changed: {path}"
            )

    model_config_path = _clean_path(
        signature.get("model_config_path"),
        context=f"{signature_path}: model config",
    )
    model_config_hash = _clean_sha256(
        signature.get("model_config_sha256"),
        context=f"{signature_path}: model config",
    )
    if (
        not model_config_path.is_file()
        or sha256_file(model_config_path) != model_config_hash
    ):
        raise ValueError(f"{signature_path}: model config hash mismatch")
    model_payload = load_yaml(model_config_path)
    model = model_payload.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"{model_config_path}: model config is incomplete")
    model_revision = model.get("revision")
    model_id = model.get("repo_id")
    if not isinstance(model_revision, str) or not model_revision:
        raise ValueError(f"{model_config_path}: model revision is missing")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"{model_config_path}: model repo_id is missing")

    implementation = _resolved_hash_mapping(
        signature.get("implementation_sha256"),
        context=f"{signature_path}: implementation_sha256",
    )
    for path, digest in implementation.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(
                f"{signature_path}: extraction implementation changed: {path}"
            )

    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
    ):
        raise ValueError(f"{run_manifest_path}: extraction is not completed")
    manifest_scores_path = manifest.get("scores_path")
    if not isinstance(manifest_scores_path, str) or not manifest_scores_path:
        raise ValueError(f"{run_manifest_path}: scores_path is missing")
    resolved_manifest_scores = (run_manifest_path.parent / manifest_scores_path).resolve()
    if resolved_manifest_scores != score_path:
        raise ValueError(f"{run_manifest_path}: scores_path does not name this cache")
    if manifest.get("scores_sha256") != score_hash:
        raise ValueError(f"{run_manifest_path}: scores hash mismatch")
    manifest_input_path = _clean_path(
        manifest.get("input_path"),
        context=f"{run_manifest_path}: input",
    )
    if (
        manifest_input_path != signature_input_path
        or manifest.get("input_sha256") != signature_input_hash
    ):
        raise ValueError(f"{run_manifest_path}: input binding mismatches job signature")
    if (
        manifest.get("model_revision") != model_revision
        or manifest.get("model_id") != model_id
    ):
        raise ValueError(f"{run_manifest_path}: model identity mismatches model config")
    if manifest.get("records") != len(records):
        raise ValueError(f"{run_manifest_path}: record count mismatches score cache")
    for index, record in enumerate(records):
        if (
            record.get("model_revision") != model_revision
            or record.get("model_id") != model_id
        ):
            raise ValueError(
                f"{score_path} record {index}: model identity mismatches run manifest"
            )
    sidecar_hash_fields = {
        "config.resolved.json": "config_sha256",
        "environment.json": "environment_sha256",
        "seed.json": "seed_sha256",
        "selection.json": "selection_sha256",
        "runtime.json": "runtime_sha256",
    }
    for filename, field in sidecar_hash_fields.items():
        path = score_path.parent / filename
        expected_hash = _clean_sha256(
            manifest.get(field),
            context=f"{run_manifest_path}: {field}",
        )
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"{run_manifest_path}: {filename} hash mismatch")

    immutable = {
        signature_path: signature_file_hash,
        run_manifest_path: run_manifest_hash,
        score_path: score_hash,
        **binding.execution_preflight,
    }
    for path, digest in immutable.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Formal score-cache input changed during validation: {path}")
    return records, {
        "scores": _metadata(score_path, score_hash),
        "job_signature": _metadata(signature_path, signature_file_hash),
        "run_manifest": _metadata(run_manifest_path, run_manifest_hash),
        "input": _metadata(signature_input_path, signature_input_hash),
        "model_config": _metadata(model_config_path, model_config_hash),
        "model_id": model_id,
        "model_revision": model_revision,
        "experiment_config_sha256": signature.get(
            "experiment_config_sha256"
        ),
        "records": len(records),
        "execution_preflight": {
            str(path): digest
            for path, digest in sorted(
                recorded_preflight.items(),
                key=lambda item: str(item[0]),
            )
        },
    }

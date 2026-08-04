#!/usr/bin/env python3
"""Evaluate completed extraction caches with frozen, leakage-safe protocols."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cavis.data.dependence import enrich_prepared_dependence_ids
from cavis.data.io import read_parquet_rows
from cavis.evaluation.formal_provenance import (
    FormalEvaluationBinding,
    load_and_validate_score_cache,
    validate_formal_evaluation_binding,
)
from cavis.evaluation.sweep import (
    discover_score_families,
    evaluate_observational_family,
    select_cavis_family,
)
from cavis.reproducibility.config import load_yaml
from cavis.reproducibility.io import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from cavis.scores.probe import (
    PROBE_FIT_SCOPE,
    defensible_probe_fit_mask,
    fit_nested_linear_probe,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL_EXPERIMENT_CONFIG = ROOT / "configs/experiment/formal_audit.yaml"
EVALUATION_IMPLEMENTATION_PATHS = (
    ROOT / "scripts/evaluate_sweep.py",
    ROOT / "src/cavis/certificates/conformal.py",
    ROOT / "src/cavis/data/dependence.py",
    ROOT / "src/cavis/data/splits.py",
    ROOT / "src/cavis/evaluation/calibration.py",
    ROOT / "src/cavis/evaluation/formal_provenance.py",
    ROOT / "src/cavis/evaluation/protocol.py",
    ROOT / "src/cavis/evaluation/sweep.py",
    ROOT / "src/cavis/evaluation/metrics.py",
    ROOT / "src/cavis/schemas.py",
    ROOT / "src/cavis/scores/probe.py",
)

DEFAULT_FAMILIES = (
    "length",
    "perplexity",
    "mean_log_likelihood",
    "mean_token_entropy",
    "max_token_entropy",
    "hfer",
    "fiedler",
    "smoothness",
    "spectral_entropy",
    "linear_probe",
)


def _frozen_formal_protocol() -> dict[str, Any]:
    """Load and validate the canonical primary formal-audit protocol."""

    config_hash = sha256_file(FORMAL_EXPERIMENT_CONFIG)
    payload = load_yaml(FORMAL_EXPERIMENT_CONFIG)
    if sha256_file(FORMAL_EXPERIMENT_CONFIG) != config_hash:
        raise ValueError(
            f"{FORMAL_EXPERIMENT_CONFIG}: config changed while being read"
        )
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("key") != "formal_audit":
        raise ValueError(
            f"{FORMAL_EXPERIMENT_CONFIG}: expected experiment.key=formal_audit"
        )
    raw_seeds = experiment.get("seeds")
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
        or len(raw_seeds) != len(set(raw_seeds))
    ):
        raise ValueError(
            f"{FORMAL_EXPERIMENT_CONFIG}: seeds must be non-empty unique integers"
        )
    primary_alpha = experiment.get("primary_alpha")
    if (
        isinstance(primary_alpha, bool)
        or not isinstance(primary_alpha, int | float)
        or not 0.0 < float(primary_alpha) < 1.0
    ):
        raise ValueError(
            f"{FORMAL_EXPERIMENT_CONFIG}: primary_alpha must lie in (0, 1)"
        )
    primary_calibration_size = experiment.get("primary_calibration_size")
    if (
        isinstance(primary_calibration_size, bool)
        or not isinstance(primary_calibration_size, int)
        or primary_calibration_size <= 0
    ):
        raise ValueError(
            f"{FORMAL_EXPERIMENT_CONFIG}: primary_calibration_size must be "
            "a positive integer"
        )
    calibration_sizes = experiment.get("calibration_sizes")
    if (
        not isinstance(calibration_sizes, list)
        or primary_calibration_size not in calibration_sizes
    ):
        raise ValueError(
            f"{FORMAL_EXPERIMENT_CONFIG}: primary_calibration_size must appear "
            "in calibration_sizes"
        )
    configured_alphas = experiment.get("alpha")
    if (
        not isinstance(configured_alphas, list)
        or not any(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isclose(
                float(value),
                float(primary_alpha),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for value in configured_alphas
        )
    ):
        raise ValueError(
            f"{FORMAL_EXPERIMENT_CONFIG}: primary_alpha must appear in alpha"
        )
    return {
        "path": FORMAL_EXPERIMENT_CONFIG.resolve(),
        "sha256": config_hash,
        "split_seeds": tuple(int(seed) for seed in raw_seeds),
        "primary_alpha": float(primary_alpha),
        "primary_calibration_size": primary_calibration_size,
    }


def _require_frozen_formal_protocol(
    *,
    split_seeds: tuple[int, ...],
    alpha: float,
    calibration_size: int | None,
) -> dict[str, Any]:
    protocol = _frozen_formal_protocol()
    if split_seeds != protocol["split_seeds"]:
        raise ValueError(
            "formal_audit split seeds are frozen by "
            f"{FORMAL_EXPERIMENT_CONFIG}: expected "
            f"{protocol['split_seeds']}, got {split_seeds}"
        )
    if not math.isclose(
        alpha,
        protocol["primary_alpha"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "formal_audit alpha is frozen by "
            f"{FORMAL_EXPERIMENT_CONFIG}: expected "
            f"{protocol['primary_alpha']}, got {alpha}"
        )
    if calibration_size != protocol["primary_calibration_size"]:
        raise ValueError(
            "formal_audit calibration size is frozen by "
            f"{FORMAL_EXPERIMENT_CONFIG}: expected "
            f"{protocol['primary_calibration_size']}, got {calibration_size}"
        )
    return protocol


def evaluation_implementation_sha256() -> dict[str, str]:
    """Hash every implementation file that can change frozen evaluation."""

    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in EVALUATION_IMPLEMENTATION_PATHS
    }


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = sorted({key for row in rows for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metric_rows(
    metrics: dict[str, Any],
    *,
    experiment: str,
    dataset: str,
    model: str,
    family: str,
    selected_score: str,
    split_seed: int,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "cavis":
        test = metrics["test"]
        sections = {
            "binary": test["binary"],
            "invariance": test["invariance"],
            "certificates": test["certificates"],
            "pairs": test["pairs"],
        }
    else:
        sections = {"binary": metrics["test"]["binary"]}
    rows = []
    for section, values in sections.items():
        for metric, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append(
                    {
                        "experiment": experiment,
                        "dataset": dataset,
                        "model": model,
                        "score_family": family,
                        "selected_score": selected_score,
                        "split_seed": split_seed,
                        "mode": mode,
                        "section": section,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows


def _records_with_probe(
    records: list[dict[str, Any]],
    *,
    score_path: Path,
    split_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features = []
    labels = []
    groups = []
    for row in records:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("probe requires a metadata mapping on every record")
        relative = metadata.get("pooled_hidden_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError(
                f"{score_path}: pooled hidden artifacts are required for linear_probe"
            )
        artifact_path = Path(relative)
        if not artifact_path.is_absolute():
            artifact_path = score_path.parent / artifact_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing pooled hidden artifact: {artifact_path}")
        expected_hash = row.get("artifact_hash")
        actual_hash = sha256_file(artifact_path)
        if expected_hash != actual_hash:
            raise ValueError(f"Artifact hash mismatch: {artifact_path}")
        features.append(np.load(artifact_path, allow_pickle=False))
        labels.append(1 if int(row["label"]) == 1 else 0)
        dependence_id = metadata.get("dependence_id")
        if not isinstance(dependence_id, str) or not dependence_id:
            raise ValueError(
                f"{score_path}: metadata.dependence_id is required for probe CV"
            )
        groups.append(dependence_id)
    fit_mask = defensible_probe_fit_mask(records)
    logits, parameters, assignments = fit_nested_linear_probe(
        np.stack(features),
        labels,
        groups,
        split_seed=split_seed,
        fit_mask=fit_mask,
    )
    enriched = []
    for row, logit, fit_eligible in zip(
        records, logits, fit_mask, strict=True
    ):
        probe_split = assignments[str(row["metadata"]["dependence_id"])]
        enriched.append(
            {
                **row,
                "scores": {
                    **row["scores"],
                    "linear_probe_logit": float(logit),
                },
                "metadata": {
                    **row["metadata"],
                    "probe_split": probe_split,
                    "probe_fit_scope": PROBE_FIT_SCOPE,
                    "probe_fit_eligible": bool(fit_eligible),
                    "probe_used_for_fit": bool(
                        fit_eligible and probe_split == "train"
                    ),
                },
            }
        )
    return enriched, parameters.to_dict()


def _validate_dependence_metadata(
    records: list[dict[str, Any]],
    *,
    score_path: Path,
) -> None:
    """Fail closed unless source provenance maps to one canonical unit."""

    dependence_by_group: dict[str, str] = {}
    for row in records:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{score_path}: every record needs metadata")
        group_id = metadata.get("group_id")
        dependence_id = metadata.get("dependence_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(
                f"{score_path}: every record needs metadata.group_id provenance"
            )
        if not isinstance(dependence_id, str) or not dependence_id:
            raise ValueError(
                f"{score_path}: every record needs metadata.dependence_id"
            )
        previous = dependence_by_group.setdefault(group_id, dependence_id)
        if previous != dependence_id:
            raise ValueError(
                f"{score_path}: one metadata.group_id maps to multiple "
                "metadata.dependence_id values"
            )


def _enrich_score_dependence_metadata(
    records: list[dict[str, Any]],
    *,
    score_path: Path,
    prepared_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join immutable score caches to canonical prepared-row identities.

    The score JSONL is never rewritten.  Every joined row must agree on exact
    item ID, source group provenance, and target text hash.
    """

    canonical_rows, diagnostics = enrich_prepared_dependence_ids(prepared_rows)
    prepared_by_item: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("dependence input contains an invalid item_id")
        if item_id in prepared_by_item:
            raise ValueError(
                f"dependence input contains duplicate item_id {item_id!r}"
            )
        prepared_by_item[item_id] = row

    enriched: list[dict[str, Any]] = []
    injected = 0
    already_present = 0
    for score_row in records:
        item_id = score_row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{score_path}: every score record needs item_id")
        try:
            prepared = prepared_by_item[item_id]
        except KeyError as exc:
            raise ValueError(
                f"{score_path}: item_id {item_id!r} is absent from dependence input"
            ) from exc
        metadata = score_row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{score_path}: every score record needs metadata")
        score_group = metadata.get("group_id")
        prepared_group = prepared.get("group_id")
        if (
            not isinstance(score_group, str)
            or not score_group
            or score_group != prepared_group
        ):
            raise ValueError(
                f"{score_path}: group_id mismatch for item_id {item_id!r}"
            )
        score_target_hash = metadata.get("target_hash")
        prepared_target_hash = prepared.get("target_hash")
        if (
            not isinstance(score_target_hash, str)
            or not score_target_hash
            or not isinstance(prepared_target_hash, str)
            or not prepared_target_hash
            or score_target_hash != prepared_target_hash
        ):
            raise ValueError(
                f"{score_path}: target_hash mismatch for item_id {item_id!r}"
            )
        canonical_id = prepared["dependence_id"]
        existing_id = metadata.get("dependence_id")
        if existing_id is None:
            injected += 1
        elif existing_id != canonical_id:
            raise ValueError(
                f"{score_path}: dependence_id mismatch for item_id {item_id!r}"
            )
        else:
            already_present += 1
        enriched.append(
            {
                **score_row,
                "metadata": {
                    **metadata,
                    "dependence_id": canonical_id,
                    "theorem_name": prepared["theorem_name"],
                    "statement_sha256": prepared["statement_sha256"],
                    "dependence_join": "prepared_item_group_target_hash_exact",
                },
            }
        )
    _validate_dependence_metadata(enriched, score_path=score_path)
    return (
        enriched,
        {
            "records": len(records),
            "injected_records": injected,
            "already_present_records": already_present,
            "prepared": asdict(diagnostics),
        },
    )


def _aggregate(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = (
        "experiment",
        "dataset",
        "model",
        "score_family",
        "mode",
        "section",
        "metric",
    )
    for row in raw_rows:
        buckets[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for bucket_key, rows in sorted(buckets.items()):
        finite = np.asarray(
            [row["value"] for row in rows if math.isfinite(float(row["value"]))],
            dtype=np.float64,
        )
        if finite.size == 0:
            mean = std = value_min = value_max = math.nan
        else:
            mean = float(finite.mean())
            std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            value_min = float(finite.min())
            value_max = float(finite.max())
        values = dict(zip(keys, bucket_key, strict=True))
        values.update(
            {
                "table_id": (
                    "table3_certificates"
                    if values["mode"] == "cavis"
                    else "table4_transfer"
                ),
                "status": "frozen_result_from_local_inference",
                "evidence_scope": (
                    "confirmatory_cavis"
                    if values["mode"] == "cavis"
                    else "observational_transfer_only"
                ),
                "selected_scores": "|".join(
                    sorted({str(row["selected_score"]) for row in rows})
                ),
                "value": mean,
                "std": std,
                # Split seeds reuse the same examples and are not independent
                # replications. Never mislabel a t interval across them as an
                # inferential 95% confidence interval.
                "ci95_lower": math.nan,
                "ci95_upper": math.nan,
                "descriptive_min": value_min,
                "descriptive_max": value_max,
                "n": int(finite.size),
                "seed_type": "grouped_split_seed",
                "uncertainty_type": (
                    "mean_std_and_range_across_repeated_grouped_splits;"
                    "not_an_inferential_confidence_interval"
                ),
            }
        )
        output.append(values)
    return output


def evaluate_experiment(
    *,
    experiment: str,
    input_root: Path,
    output_root: Path,
    split_seeds: tuple[int, ...],
    alpha: float,
    requested_families: tuple[str, ...],
    exclude_selection_paths: tuple[Path, ...] = (),
    dependence_input: Path | None = None,
    calibration_size: int | None = None,
    formal_design_report: Path | None = None,
    formal_eligibility_report: Path | None = None,
    formal_merge_report: Path | None = None,
) -> dict[str, Any]:
    if calibration_size is not None and calibration_size <= 0:
        raise ValueError("calibration_size must be a positive dependence-group count")
    score_paths = sorted((input_root / experiment).glob("*/*/scores.jsonl"))
    if not score_paths:
        raise FileNotFoundError(
            f"No completed score caches below {input_root / experiment}"
        )
    raw_metric_rows: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    formal_binding: FormalEvaluationBinding | None = None
    formal_paths = (
        formal_design_report,
        formal_eligibility_report,
        formal_merge_report,
    )
    formal_requested = any(path is not None for path in formal_paths)
    formal_required = experiment == "formal_audit"
    formal_protocol: dict[str, Any] | None = None
    if formal_required or formal_requested:
        if any(path is None for path in formal_paths):
            raise ValueError(
                "Confirmatory formal_audit evaluation requires "
                "--formal-design-report, --formal-eligibility-report, and "
                "--formal-merge-report together"
            )
        if dependence_input is None:
            raise ValueError(
                "Confirmatory formal_audit evaluation requires --dependence-input"
            )
        formal_binding = validate_formal_evaluation_binding(
            design_report_path=formal_design_report,
            eligibility_report_path=formal_eligibility_report,
            merge_report_path=formal_merge_report,
            dependence_input_path=dependence_input,
            exclude_selection_paths=exclude_selection_paths,
        )
        if formal_required:
            formal_protocol = _require_frozen_formal_protocol(
                split_seeds=split_seeds,
                alpha=alpha,
                calibration_size=calibration_size,
            )
    prepared_dependence_rows: list[dict[str, Any]] | None = None
    dependence_input_hash: str | None = None
    dependence_joins: dict[str, dict[str, Any]] = {}
    if dependence_input is not None:
        if not dependence_input.is_file():
            raise FileNotFoundError(
                f"Dependence input is missing: {dependence_input}"
            )
        prepared_dependence_rows = read_parquet_rows(dependence_input)
        dependence_input_hash = sha256_file(dependence_input)
        inputs[str(dependence_input)] = dependence_input_hash
    excluded_dependencies: set[str] = set()
    exclusion_inputs: dict[str, str] = {}
    for selection_path in exclude_selection_paths:
        if not selection_path.is_file():
            raise FileNotFoundError(f"Exclusion selection is missing: {selection_path}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        dependence_ids = selection.get("selected_dependence_ids")
        if not isinstance(dependence_ids, list) or any(
            not isinstance(dependence_id, str) or not dependence_id
            for dependence_id in dependence_ids
        ):
            raise ValueError(
                f"{selection_path}: selected_dependence_ids must be nonempty strings"
            )
        excluded_dependencies.update(dependence_ids)
        exclusion_inputs[str(selection_path)] = sha256_file(selection_path)
    excluded_records_by_input: dict[str, int] = {}
    score_cache_validations: dict[str, dict[str, Any]] = {}
    run_count = 0
    for score_path in score_paths:
        dataset = score_path.parent.parent.name
        model = score_path.parent.name
        if formal_binding is not None:
            records, cache_validation = load_and_validate_score_cache(
                score_path,
                binding=formal_binding,
                expected_experiment=experiment,
                expected_dataset=dataset,
                expected_model=model,
                expected_experiment_config_sha256=(
                    formal_protocol["sha256"]
                    if formal_protocol is not None
                    else None
                ),
            )
            score_cache_validations[str(score_path)] = cache_validation
        else:
            records = read_jsonl(score_path)
        inputs[str(score_path)] = sha256_file(score_path)
        missing_dependence = any(
            not isinstance(record.get("metadata"), dict)
            or not isinstance(
                record["metadata"].get("dependence_id"),
                str,
            )
            or not record["metadata"]["dependence_id"]
            for record in records
        )
        if prepared_dependence_rows is not None:
            records, join_diagnostics = _enrich_score_dependence_metadata(
                records,
                score_path=score_path,
                prepared_rows=prepared_dependence_rows,
            )
            dependence_joins[str(score_path)] = join_diagnostics
        elif missing_dependence:
            raise ValueError(
                f"{score_path}: metadata.dependence_id is missing and no "
                "--dependence-input was provided"
            )
        else:
            _validate_dependence_metadata(records, score_path=score_path)
        if excluded_dependencies:
            kept = []
            for record in records:
                metadata = record.get("metadata")
                if not isinstance(metadata, dict) or not isinstance(
                    metadata.get("dependence_id"), str
                ):
                    raise ValueError(
                        f"{score_path}: every record needs metadata.dependence_id "
                        "when dependence exclusions are active"
                    )
                if metadata["dependence_id"] not in excluded_dependencies:
                    kept.append(record)
            excluded_records_by_input[str(score_path)] = len(records) - len(kept)
            records = kept
            if not records:
                raise ValueError(
                    f"{score_path}: dependence exclusions removed every record"
                )
        available = discover_score_families(records)
        missing = sorted(
            set(requested_families) - set(available) - {"linear_probe"}
        )
        if missing:
            raise ValueError(f"{score_path}: missing requested families {missing}")
        mode = "cavis" if dataset == "geometry_leantwin" else "observational"
        for family in requested_families:
            for split_seed in split_seeds:
                output_dir = (
                    output_root
                    / experiment
                    / dataset
                    / model
                    / family
                    / f"split_seed_{split_seed}"
                )
                family_records = records
                candidate_names = available.get(family, ())
                if family == "linear_probe":
                    family_records, probe_parameters = _records_with_probe(
                        records,
                        score_path=score_path,
                        split_seed=split_seed,
                    )
                    candidate_names = ("linear_probe_logit",)
                    atomic_write_json(
                        output_dir / "probe_parameters.json",
                        probe_parameters,
                    )
                if mode == "cavis":
                    selected = select_cavis_family(
                        family_records,
                        family=family,
                        candidate_names=candidate_names,
                        split_seed=split_seed,
                        alpha=alpha,
                        calibration_size=calibration_size,
                    )
                    metrics = {
                        **dict(selected.result.metrics),
                        "selection": {
                            "family": family,
                            "selected_score_name": selected.selected_score_name,
                            "candidate_train_aurocs": selected.candidate_train_aurocs,
                            "selection_bound_delta_0_05": (
                                selected.selection_bound_delta_0_05
                            ),
                        },
                    }
                    per_item = [dict(row) for row in selected.result.per_item]
                    selected_score = selected.selected_score_name
                else:
                    selected_observational = evaluate_observational_family(
                        family_records,
                        family=family,
                        candidate_names=candidate_names,
                        split_seed=split_seed,
                    )
                    metrics = selected_observational.metrics
                    per_item = list(selected_observational.per_item)
                    selected_score = selected_observational.selected_score_name
                atomic_write_json(output_dir / "metrics.json", _safe(metrics))
                write_jsonl(
                    output_dir / "per_item.jsonl",
                    (_safe(row) for row in per_item),
                )
                raw_metric_rows.extend(
                    _metric_rows(
                        metrics,
                        experiment=experiment,
                        dataset=dataset,
                        model=model,
                        family=family,
                        selected_score=selected_score,
                        split_seed=split_seed,
                        mode=mode,
                    )
                )
                run_count += 1

    experiment_output = output_root / experiment
    if (
        formal_protocol is not None
        and sha256_file(formal_protocol["path"]) != formal_protocol["sha256"]
    ):
        raise ValueError(
            f"{formal_protocol['path']}: formal config changed during evaluation"
        )
    aggregated = _aggregate(raw_metric_rows)
    summary_path = experiment_output / "summary.csv"
    _atomic_csv(summary_path, [_safe(row) for row in aggregated])
    write_jsonl(
        experiment_output / "per_seed_metrics.jsonl",
        (_safe(row) for row in raw_metric_rows),
    )
    manifest = {
        "schema_version": 1,
        "experiment": experiment,
        "split_seeds": list(split_seeds),
        "seed_type": "canonical_dependence_split_seed",
        "dependence_unit": "metadata.dependence_id",
        "alpha": alpha,
        "calibration_size_per_side_dependence_groups": calibration_size,
        "families": list(requested_families),
        "implementation_sha256": evaluation_implementation_sha256(),
        "inputs": inputs,
        "formal_provenance": (
            formal_binding.to_manifest()
            if formal_binding is not None
            else None
        ),
        "formal_primary_protocol": (
            {
                "config_path": str(formal_protocol["path"]),
                "config_sha256": formal_protocol["sha256"],
                "split_seeds": list(formal_protocol["split_seeds"]),
                "alpha": formal_protocol["primary_alpha"],
                "calibration_size_per_side_dependence_groups": (
                    formal_protocol["primary_calibration_size"]
                ),
                "override_policy": "refuse",
            }
            if formal_protocol is not None
            else None
        ),
        "score_cache_validations": score_cache_validations,
        "dependence_enrichment": {
            "input_path": (
                str(dependence_input)
                if dependence_input is not None
                else None
            ),
            "input_sha256": dependence_input_hash,
            "joins_by_score_input": dependence_joins,
            "score_caches_mutated": False,
        },
        "dependence_exclusions": {
            "dependence_unit": "metadata.dependence_id",
            "selection_inputs": exclusion_inputs,
            "excluded_dependence_ids": sorted(excluded_dependencies),
            "excluded_dependence_count": len(excluded_dependencies),
            "excluded_records_by_input": excluded_records_by_input,
        },
        "runs": run_count,
        "summary_rows": len(aggregated),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "scope_boundary": (
            "CAVIS rows require eligible Lean evidence. External rows are "
            "observational transfer and carry no invariance or causal assertion."
        ),
    }
    atomic_write_json(experiment_output / "evaluation_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/runs/extractions"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/runs/evaluations"),
    )
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[17, 42, 97])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--calibration-size",
        type=int,
        help=(
            "Optional number of canonical dependence/theorem groups sampled "
            "separately for each certificate side. By default all eligible "
            "calibration groups are used."
        ),
    )
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument(
        "--dependence-input",
        type=Path,
        help=(
            "Prepared Geometry/LeanTwin parquet used to inject canonical "
            "metadata.dependence_id into legacy score caches in memory."
        ),
    )
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        action="append",
        default=[],
        help=(
            "Selection JSON whose selected_dependence_ids are removed before "
            "fitting and evaluation; repeatable."
        ),
    )
    parser.add_argument(
        "--formal-design-report",
        type=Path,
        help=(
            "Passing post-adjudication formal-design report. Required, with "
            "the other formal provenance arguments, for formal_audit."
        ),
    )
    parser.add_argument(
        "--formal-eligibility-report",
        type=Path,
        help=(
            "Hash-bound formal eligibility report. Required, with the other "
            "formal provenance arguments, for formal_audit."
        ),
    )
    parser.add_argument(
        "--formal-merge-report",
        type=Path,
        help=(
            "Pilot+formal human-validation merge report. Required, with the "
            "other formal provenance arguments, for formal_audit."
        ),
    )
    args = parser.parse_args()
    manifest = evaluate_experiment(
        experiment=args.experiment,
        input_root=args.input_root,
        output_root=args.output_root,
        split_seeds=tuple(args.split_seeds),
        alpha=args.alpha,
        requested_families=tuple(args.families),
        exclude_selection_paths=tuple(args.exclude_selection),
        dependence_input=args.dependence_input,
        calibration_size=args.calibration_size,
        formal_design_report=args.formal_design_report,
        formal_eligibility_report=args.formal_eligibility_report,
        formal_merge_report=args.formal_merge_report,
    )
    print(
        f"Evaluated {manifest['runs']} family/split runs; "
        f"summary rows={manifest['summary_rows']}"
    )


if __name__ == "__main__":
    main()

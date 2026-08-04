#!/usr/bin/env python3
"""Run post-extraction CAVIS ablations from BF16 and 4-bit score caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from cavis.data.io import read_parquet_rows
from cavis.evaluation.ablations import (
    filter_transform_records,
    residualize_score_records,
)
from cavis.evaluation.formal_provenance import (
    FormalEvaluationBinding,
    load_and_validate_score_cache,
    validate_formal_evaluation_binding,
)
from cavis.evaluation.metrics import roc_auc
from cavis.evaluation.protocol import (
    _dependence_label_mean_scores,
    _select_threshold,
    parse_protocol_rows,
)
from cavis.evaluation.sweep import (
    discover_score_families,
    evaluate_observational_family,
    select_cavis_family,
)
from cavis.reproducibility.io import (
    atomic_write_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)
from scripts.evaluate_sweep import (
    _enrich_score_dependence_metadata,
    evaluation_implementation_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def ablation_implementation_sha256() -> dict[str, str]:
    hashes = evaluation_implementation_sha256()
    for path in (
        ROOT / "scripts/run_ablations.py",
        ROOT / "src/cavis/evaluation/ablations.py",
    ):
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = sorted({key for row in rows for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_dependence_exclusions(
    paths: tuple[Path, ...],
) -> tuple[set[str], dict[str, str]]:
    excluded: set[str] = set()
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Ablation exclusion selection is missing: {path}")
        original_hash = sha256_file(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{path}: invalid exclusion JSON: {error}") from error
        dependence_ids = payload.get("selected_dependence_ids")
        if (
            not isinstance(dependence_ids, list)
            or not dependence_ids
            or any(
                not isinstance(dependence_id, str) or not dependence_id
                for dependence_id in dependence_ids
            )
            or len(dependence_ids) != len(set(dependence_ids))
        ):
            raise ValueError(
                f"{path}: selected_dependence_ids must be a non-empty unique "
                "list of non-empty strings"
            )
        if sha256_file(path) != original_hash:
            raise ValueError(f"{path}: exclusion selection changed while being read")
        excluded.update(dependence_ids)
        hashes[str(path)] = original_hash
    return excluded, hashes


def _exclude_geometry_dependencies(
    records: list[dict[str, Any]],
    *,
    excluded_dependence_ids: set[str],
    score_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    if not excluded_dependence_ids:
        return records, 0
    retained: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        metadata = record.get("metadata")
        dependence_id = (
            metadata.get("dependence_id")
            if isinstance(metadata, dict)
            else None
        )
        if not isinstance(dependence_id, str) or not dependence_id:
            raise ValueError(
                f"{score_path} record {index}: metadata.dependence_id is "
                "required before pilot exclusion"
            )
        if dependence_id not in excluded_dependence_ids:
            retained.append(record)
    excluded_count = len(records) - len(retained)
    if not retained:
        raise ValueError(
            f"{score_path}: pilot dependence exclusions removed every record"
        )
    return retained, excluded_count


def _defensible_geometry_records(
    records: list[dict[str, Any]],
    *,
    score_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Retain upstream roots and only evidence-eligible generated variants."""

    retained: list[dict[str, Any]] = []
    excluded = 0
    for index, record in enumerate(records):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{score_path} record {index}: metadata must be a mapping")
        canonical_kind = metadata.get("variant_kind")
        alias_kind = metadata.get("transform_kind")
        if (
            canonical_kind is not None
            and alias_kind is not None
            and canonical_kind != alias_kind
        ):
            raise ValueError(
                f"{score_path} record {index}: conflicting "
                "variant_kind/transform_kind"
            )
        variant_kind = (
            canonical_kind if canonical_kind is not None else alias_kind
        )
        source_kind = metadata.get("source_transform_kind", variant_kind)
        if variant_kind not in {"base", "g1", "g0"} or source_kind not in {
            "base",
            "g1",
            "g0",
        }:
            raise ValueError(
                f"{score_path} record {index}: unsupported transform provenance"
            )
        is_generated_transform = variant_kind == "g0" or source_kind == "g1"
        if not is_generated_transform:
            retained.append(record)
            continue
        eligible = metadata.get("cavis_eligible")
        if type(eligible) is not bool:
            raise ValueError(
                f"{score_path} record {index}: generated transformations require "
                "literal-boolean metadata.cavis_eligible"
            )
        if eligible:
            retained.append(record)
        else:
            excluded += 1
    if not retained:
        raise ValueError(
            f"{score_path}: evidence filtering removed every geometry record"
        )
    return retained, excluded


def _core_metrics(result: Any) -> dict[str, float]:
    test = result.result.metrics["test"]
    return {
        "auroc": float(test["binary"]["auroc"]),
        "balanced_accuracy": float(test["binary"]["balanced_accuracy"]),
        "inv_flip": float(test["invariance"]["inv_flip"]),
        "sensitivity": float(test["pairs"]["sensitivity"]),
        "certified_pair_rate": float(test["pairs"]["certified_pair_rate"]),
        "abstention_rate": float(test["certificates"]["abstention_rate"]),
    }


def _append_cavis(
    rows: list[dict[str, Any]],
    *,
    records: list[dict[str, Any]],
    candidates: tuple[str, ...],
    model: str,
    precision: str,
    condition: str,
    setting: str,
    split_seed: int,
    alpha: float = 0.1,
    calibration_size: int | None = None,
    allow_not_applicable: bool = False,
) -> None:
    try:
        selected = select_cavis_family(
            records,
            family="hfer",
            candidate_names=candidates,
            split_seed=split_seed,
            alpha=alpha,
            calibration_size=calibration_size,
        )
    except ValueError as error:
        if not allow_not_applicable:
            raise
        rows.append(
            {
                "table_id": "table5_ablations",
                "status": "not_applicable",
                "model": model,
                "precision": precision,
                "condition": condition,
                "setting": setting,
                "split_seed": split_seed,
                "metric": "not_applicable",
                "value": "",
                "reason": str(error),
            }
        )
        return
    for metric, value in _core_metrics(selected).items():
        rows.append(
            {
                "table_id": "table5_ablations",
                "status": "frozen_result_from_local_inference",
                "model": model,
                "precision": precision,
                "condition": condition,
                "setting": setting,
                "split_seed": split_seed,
                "metric": metric,
                "value": value,
                "selected_score": selected.selected_score_name,
                "alpha": alpha,
                "calibration_size": selected.result.state.calibration_size,
                "calibration_pool_size": (
                    selected.result.state.calibration_pool_size
                ),
            }
        )


def _residualized_candidates(
    records: list[dict[str, Any]],
    candidates: tuple[str, ...],
    *,
    nuisances: tuple[str, ...],
    split_seed: int,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    output = deepcopy(records)
    names = []
    for candidate in candidates:
        output, name = residualize_score_records(
            output,
            target_score=candidate,
            nuisance_names=nuisances,
            split_seed=split_seed,
        )
        names.append(name)
    return output, tuple(names)


def _oracle_rows(
    records: list[dict[str, Any]],
    candidates: tuple[str, ...],
    *,
    model: str,
    precision: str,
) -> list[dict[str, Any]]:
    values = []
    for name in candidates:
        parsed = parse_protocol_rows(records, score_name=name)
        roots = [
            row
            for row in parsed
            if row.is_root and (not row.is_g1_root or row.cavis_eligible)
        ]
        _, labels, raw = _dependence_label_mean_scores(roots)
        positive = roc_auc(labels, raw)
        negative = roc_auc(labels, -raw)
        orientation = -1 if negative > positive else 1
        oriented = orientation * raw
        _, balanced = _select_threshold(labels, oriented)
        values.append((max(positive, negative), balanced, name))
    auroc, balanced, selected = max(values, key=lambda row: (row[0], row[1], row[2]))
    return [
        {
            "table_id": "table5_ablations",
            "status": "diagnostic_oracle_full_data_do_not_compare",
            "model": model,
            "precision": precision,
            "condition": "threshold_selection",
            "setting": "oracle_full_data",
            "split_seed": "full_data",
            "metric": metric,
            "value": value,
            "selected_score": selected,
        }
        for metric, value in (
            ("auroc", auroc),
            ("balanced_accuracy", balanced),
        )
    ]


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = [
        row
        for row in rows
        if row["status"] == "frozen_result_from_local_inference"
    ]
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("model", "precision", "condition", "setting", "metric")
    for row in numeric:
        buckets[tuple(str(row[key]) for key in keys)].append(row)
    output = [
        row
        for row in rows
        if row["status"] != "frozen_result_from_local_inference"
    ]
    for bucket_key, bucket in sorted(buckets.items()):
        values = np.asarray(
            [
                float(row["value"])
                for row in bucket
                if math.isfinite(float(row["value"]))
            ],
            dtype=np.float64,
        )
        mean = float(values.mean()) if values.size else math.nan
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        value_min = float(values.min()) if values.size else math.nan
        value_max = float(values.max()) if values.size else math.nan
        aggregate = dict(zip(keys, bucket_key, strict=True))
        aggregate.update(
            {
                "table_id": "table5_ablations",
                "status": "frozen_result_from_local_inference",
                "value": mean,
                "std": std,
                "ci95_lower": math.nan,
                "ci95_upper": math.nan,
                "descriptive_min": value_min,
                "descriptive_max": value_max,
                "n": int(values.size),
                "seed_type": "grouped_split_seed",
                "uncertainty_type": (
                    "mean_std_and_range_across_repeated_grouped_splits;"
                    "not_an_inferential_confidence_interval"
                ),
                "selected_scores": "|".join(
                    sorted(
                        {
                            str(row.get("selected_score", ""))
                            for row in bucket
                        }
                    )
                ),
            }
        )
        output.append(aggregate)
    return output


def run_ablations(
    *,
    input_root: Path,
    output_dir: Path,
    split_seeds: tuple[int, ...],
    exclude_selection_paths: tuple[Path, ...] = (),
    dependence_input: Path | None = None,
    formal_design_report: Path | None = None,
    formal_eligibility_report: Path | None = None,
    formal_merge_report: Path | None = None,
    require_formal_provenance: bool = False,
) -> dict[str, Any]:
    score_paths = sorted((input_root / "ablations").glob("*/*/*/scores.jsonl"))
    if not score_paths:
        raise FileNotFoundError(
            f"No ablation score caches below {input_root / 'ablations'}"
        )
    rows: list[dict[str, Any]] = []
    input_hashes = {}
    formal_binding: FormalEvaluationBinding | None = None
    formal_paths = (
        formal_design_report,
        formal_eligibility_report,
        formal_merge_report,
    )
    formal_requested = any(path is not None for path in formal_paths)
    if require_formal_provenance or formal_requested:
        if any(path is None for path in formal_paths):
            raise ValueError(
                "Confirmatory LeanTwin ablations require "
                "--formal-design-report, --formal-eligibility-report, and "
                "--formal-merge-report together"
            )
        if dependence_input is None:
            raise ValueError(
                "Confirmatory LeanTwin ablations require --dependence-input"
            )
        formal_binding = validate_formal_evaluation_binding(
            design_report_path=formal_design_report,
            eligibility_report_path=formal_eligibility_report,
            merge_report_path=formal_merge_report,
            dependence_input_path=dependence_input,
            exclude_selection_paths=exclude_selection_paths,
        )
    (
        excluded_dependence_ids,
        exclusion_selection_hashes,
    ) = _load_dependence_exclusions(exclude_selection_paths)
    excluded_records_by_input: dict[str, int] = {}
    ineligible_transform_records_by_input: dict[str, int] = {}
    prepared_dependence_rows: list[dict[str, Any]] | None = None
    dependence_input_hash: str | None = None
    dependence_joins: dict[str, dict[str, Any]] = {}
    score_cache_validations: dict[str, dict[str, Any]] = {}
    if dependence_input is not None:
        if not dependence_input.is_file():
            raise FileNotFoundError(
                f"Ablation dependence input is missing: {dependence_input}"
            )
        dependence_input_hash = sha256_file(dependence_input)
        prepared_dependence_rows = read_parquet_rows(dependence_input)
        if sha256_file(dependence_input) != dependence_input_hash:
            raise ValueError(
                f"{dependence_input}: dependence input changed while being read"
            )
        input_hashes[str(dependence_input)] = dependence_input_hash
    for score_path in score_paths:
        dataset = score_path.parents[2].name
        model = score_path.parents[1].name
        precision = score_path.parent.name
        if formal_binding is not None:
            records, cache_validation = load_and_validate_score_cache(
                score_path,
                binding=formal_binding,
                expected_experiment="ablations",
                expected_dataset=dataset,
                expected_model=model,
                expected_precision=precision,
            )
            score_cache_validations[str(score_path)] = cache_validation
        else:
            records = read_jsonl(score_path)
        input_hashes[str(score_path)] = sha256_file(score_path)
        if dataset == "geometry_leantwin":
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
                    "dependence_input was provided"
                )
            records, excluded_count = _exclude_geometry_dependencies(
                records,
                excluded_dependence_ids=excluded_dependence_ids,
                score_path=score_path,
            )
            excluded_records_by_input[str(score_path)] = excluded_count
            records, ineligible_count = _defensible_geometry_records(
                records,
                score_path=score_path,
            )
            ineligible_transform_records_by_input[str(score_path)] = (
                ineligible_count
            )
        families = discover_score_families(records)
        candidates = families["hfer"]
        if dataset != "geometry_leantwin":
            for split_seed in split_seeds:
                result = evaluate_observational_family(
                    records,
                    family="hfer",
                    candidate_names=candidates,
                    split_seed=split_seed,
                )
                rows.append(
                    {
                        "table_id": "table5_ablations",
                        "status": "frozen_result_from_local_inference",
                        "model": model,
                        "precision": precision,
                        "condition": f"precision_transfer:{dataset}",
                        "setting": precision,
                        "split_seed": split_seed,
                        "metric": "auroc",
                        "value": result.metrics["test"]["binary"]["auroc"],
                        "selected_score": result.selected_score_name,
                    }
                )
            continue

        g0_names = sorted(
            {
                str(record["metadata"].get("transform_name"))
                for record in records
                if record["metadata"].get("variant_kind") == "g0"
            }
        )
        g1_names = sorted(
            {
                str(record["metadata"].get("transform_name"))
                for record in records
                if record["metadata"].get("source_transform_kind") == "g1"
            }
        )
        rows.extend(
            _oracle_rows(
                records,
                candidates,
                model=model,
                precision=precision,
            )
        )
        for split_seed in split_seeds:
            _append_cavis(
                rows,
                records=records,
                candidates=candidates,
                model=model,
                precision=precision,
                condition="main",
                setting="raw_alpha_0.10",
                split_seed=split_seed,
            )
            for alpha in (0.05, 0.2):
                _append_cavis(
                    rows,
                    records=records,
                    candidates=candidates,
                    model=model,
                    precision=precision,
                    condition="alpha",
                    setting=str(alpha),
                    split_seed=split_seed,
                    alpha=alpha,
                    allow_not_applicable=True,
                )
            for size in (5, 10, 15, 20):
                _append_cavis(
                    rows,
                    records=records,
                    candidates=candidates,
                    model=model,
                    precision=precision,
                    condition="calibration_size",
                    setting=str(size),
                    split_seed=split_seed,
                    calibration_size=size,
                    allow_not_applicable=True,
                )
            for count in (1, 2, 4):
                filtered = filter_transform_records(
                    records,
                    seed=split_seed,
                    max_g0_per_root=count,
                )
                _append_cavis(
                    rows,
                    records=filtered,
                    candidates=candidates,
                    model=model,
                    precision=precision,
                    condition="g0_count",
                    setting=str(count),
                    split_seed=split_seed,
                    allow_not_applicable=True,
                )
                filtered = filter_transform_records(
                    records,
                    seed=split_seed,
                    max_g1_per_positive=count,
                )
                _append_cavis(
                    rows,
                    records=filtered,
                    candidates=candidates,
                    model=model,
                    precision=precision,
                    condition="g1_count",
                    setting=str(count),
                    split_seed=split_seed,
                    allow_not_applicable=True,
                )
            for name in g0_names:
                filtered = filter_transform_records(
                    records,
                    seed=split_seed,
                    drop_g0_names=frozenset({name}),
                )
                _append_cavis(
                    rows,
                    records=filtered,
                    candidates=candidates,
                    model=model,
                    precision=precision,
                    condition="leave_one_g0_out",
                    setting=name,
                    split_seed=split_seed,
                    allow_not_applicable=True,
                )
            for name in g1_names:
                filtered = filter_transform_records(
                    records,
                    seed=split_seed,
                    drop_g1_names=frozenset({name}),
                )
                _append_cavis(
                    rows,
                    records=filtered,
                    candidates=candidates,
                    model=model,
                    precision=precision,
                    condition="leave_one_g1_out",
                    setting=name,
                    split_seed=split_seed,
                    allow_not_applicable=True,
                )
            for nuisance_names, setting in (
                (("token_length",), "length"),
                (("token_length", "perplexity"), "length_and_perplexity"),
            ):
                residualized, residual_names = _residualized_candidates(
                    records,
                    candidates,
                    nuisances=nuisance_names,
                    split_seed=split_seed,
                )
                _append_cavis(
                    rows,
                    records=residualized,
                    candidates=residual_names,
                    model=model,
                    precision=precision,
                    condition="residualization",
                    setting=setting,
                    split_seed=split_seed,
                    allow_not_applicable=True,
                )

    write_jsonl(
        output_dir / "per_seed_metrics.jsonl",
        (
            {key: value for key, value in row.items() if key != "table_id"}
            for row in rows
        ),
    )
    summary = _aggregate(rows)
    _atomic_csv(output_dir / "summary.csv", summary)
    manifest = {
        "schema_version": 1,
        "inputs": input_hashes,
        "implementation_sha256": ablation_implementation_sha256(),
        "formal_provenance": (
            formal_binding.to_manifest()
            if formal_binding is not None
            else None
        ),
        "score_cache_validations": score_cache_validations,
        "split_seeds": list(split_seeds),
        "seed_type": "grouped_split_seed",
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
        "eligibility_filter": {
            "upstream_base_roots_retained": True,
            "generated_transform_requirement": (
                "metadata.cavis_eligible == true"
            ),
            "excluded_records_by_input": (
                ineligible_transform_records_by_input
            ),
        },
        "dependence_exclusions": {
            "applied_before_all_geometry_fits": True,
            "dependence_unit": "metadata.dependence_id",
            "selection_inputs": exclusion_selection_hashes,
            "excluded_dependence_ids": sorted(excluded_dependence_ids),
            "excluded_dependence_count": len(excluded_dependence_ids),
            "excluded_records_by_input": excluded_records_by_input,
        },
        "summary_rows": len(summary),
        "scope_boundary": (
            "Oracle rows are diagnostic only. Precision rows require paired "
            "BF16/4-bit caches; missing calibration sizes are explicit."
        ),
    }
    atomic_write_json(output_dir / "ablation_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/runs/extractions"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/runs/evaluations/ablations"),
    )
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[17, 42, 97])
    parser.add_argument(
        "--dependence-input",
        type=Path,
        default=Path("data/cache/prepared/geometry_leantwin.parquet"),
        help=(
            "Prepared LeanTwin parquet used to inject canonical dependence "
            "metadata into legacy score caches in memory."
        ),
    )
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        action="append",
        default=[],
        help=(
            "Selection JSON whose selected_dependence_ids are removed from "
            "LeanTwin before every oracle, fit, and ablation; repeatable."
        ),
    )
    parser.add_argument(
        "--formal-design-report",
        type=Path,
        help="Passing post-adjudication formal-design report (required).",
    )
    parser.add_argument(
        "--formal-eligibility-report",
        type=Path,
        help="Hash-bound formal eligibility report (required).",
    )
    parser.add_argument(
        "--formal-merge-report",
        type=Path,
        help="Pilot+formal human-validation merge report (required).",
    )
    args = parser.parse_args()
    manifest = run_ablations(
        input_root=args.input_root,
        output_dir=args.output_dir,
        split_seeds=tuple(args.split_seeds),
        exclude_selection_paths=tuple(args.exclude_selection),
        dependence_input=args.dependence_input,
        formal_design_report=args.formal_design_report,
        formal_eligibility_report=args.formal_eligibility_report,
        formal_merge_report=args.formal_merge_report,
        require_formal_provenance=True,
    )
    print(f"Wrote {manifest['summary_rows']} ablation summary rows")


if __name__ == "__main__":
    main()

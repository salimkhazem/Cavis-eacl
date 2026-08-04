from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cavis.data.splits import stable_hash
from cavis.reproducibility import (
    atomic_write_json,
    capture_environment,
    load_yaml,
    sha256_file,
)
from cavis.reproducibility.seed import seed_everything
from cavis.scores.extractor import ExtractionConfig, HFScoreExtractor


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
    elif suffix == ".csv":
        rows = pd.read_csv(path).to_dict(orient="records")
    else:
        raise ValueError("Input must be .jsonl, .json, .parquet, or .csv")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Input must contain a sequence of record objects")
    return rows


def _rank(value: str, *, seed: int, namespace: str) -> tuple[int, str]:
    return stable_hash(value, seed=seed, namespace=namespace), value


def _deterministic_subset(
    rows: list[dict[str, Any]],
    *,
    max_base_items: int | None,
    max_transforms_per_family: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select complete semantic groups and bounded LeanTwin transformations."""

    if max_base_items is not None and max_base_items <= 0:
        raise ValueError("max_base_items must be positive")
    if max_transforms_per_family is not None and max_transforms_per_family <= 0:
        raise ValueError("max_transforms_per_family must be positive")
    input_groups = sorted({str(row["group_id"]) for row in rows})
    ranked_groups = sorted(
        input_groups,
        key=lambda value: _rank(value, seed=seed, namespace="extraction-group"),
    )
    base_counts = {
        group: sum(
            str(row["group_id"]) == group
            and str(row.get("transform_kind", "base")) == "base"
            for row in rows
        )
        for group in input_groups
    }
    if max_base_items is None:
        selected_groups = ranked_groups
    else:
        selected_groups = []
        selected_base_count = 0
        for group in ranked_groups:
            count = base_counts[group]
            if count <= 0 or selected_base_count + count > max_base_items:
                continue
            selected_groups.append(group)
            selected_base_count += count
            if selected_base_count == max_base_items:
                break
        if not selected_groups:
            raise ValueError("no complete group fits within max_base_items")
    selected_group_set = set(selected_groups)
    candidates = [row for row in rows if str(row["group_id"]) in selected_group_set]

    selected_pairs: set[str] | None = None
    if max_transforms_per_family is not None:
        selected_pairs = set()
        g1_by_positive: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            if str(row.get("transform_kind", "base")) != "g1":
                continue
            positive_id = str(row.get("positive_semantic_variant_id") or "")
            g1_by_positive.setdefault(positive_id, []).append(row)
        for positive_id, pair_rows in g1_by_positive.items():
            ranked = sorted(
                pair_rows,
                key=lambda row: _rank(
                    str(row["item_id"]),
                    seed=seed,
                    namespace=f"extraction-g1:{positive_id}",
                ),
            )
            selected_pairs.update(
                str(row["pair_id"])
                for row in ranked[:max_transforms_per_family]
            )

        keep_ids: set[str] = set()
        g0_by_parent: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            kind = str(row.get("transform_kind", "base"))
            if kind == "base":
                keep_ids.add(str(row["item_id"]))
            elif kind == "g1" and str(row.get("pair_id")) in selected_pairs:
                keep_ids.add(str(row["item_id"]))
            elif kind == "g0":
                pair_id = row.get("pair_id")
                if pair_id is None or str(pair_id) in selected_pairs:
                    parent_id = str(row.get("parent_variant_id") or "")
                    g0_by_parent.setdefault(parent_id, []).append(row)
        for parent_id, transform_rows in g0_by_parent.items():
            ranked = sorted(
                transform_rows,
                key=lambda row: _rank(
                    str(row["item_id"]),
                    seed=seed,
                    namespace=f"extraction-g0:{parent_id}",
                ),
            )
            keep_ids.update(
                str(row["item_id"])
                for row in ranked[:max_transforms_per_family]
            )
        candidates = [row for row in candidates if str(row["item_id"]) in keep_ids]

    selected = sorted(
        candidates,
        key=lambda row: (
            str(row["dataset"]),
            str(row["group_id"]),
            str(row["item_id"]),
            str(row.get("transformation_id", "base")),
        ),
    )
    report = {
        "selection_seed": seed,
        "max_base_items": max_base_items,
        "max_transforms_per_family": max_transforms_per_family,
        "input_rows": len(rows),
        "selected_rows": len(selected),
        "input_groups": len(input_groups),
        "selected_groups": len(selected_groups),
        "selected_base_items": sum(base_counts[group] for group in selected_groups),
        "selected_group_ids": selected_groups,
        "selected_pair_ids": sorted(selected_pairs or set()),
    }
    return selected, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen-model CAVIS scores.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device")
    parser.add_argument("--quantization", choices=["none", "4bit"])
    parser.add_argument("--max-base-items", type=int)
    parser.add_argument("--max-transforms-per-family", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_yaml(args.model_config)
    model = dict(payload["model"])
    extraction = dict(payload.get("extraction", {}))
    config_values = {
        "model_id": model["repo_id"],
        "revision": model["revision"],
        **extraction,
    }
    if args.device is not None:
        config_values["device"] = args.device
    if args.quantization is not None:
        config_values["quantization"] = (
            None if args.quantization == "none" else args.quantization
        )
    if "layers" in config_values:
        config_values["layers"] = tuple(config_values["layers"])
    config = ExtractionConfig(**config_values)
    seed_report = seed_everything(args.seed)
    rows = _read_rows(args.input)
    rows, selection_report = _deterministic_subset(
        rows,
        max_base_items=args.max_base_items,
        max_transforms_per_family=args.max_transforms_per_family,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "config.resolved.json", as_serializable(config))
    atomic_write_json(args.output / "environment.json", capture_environment())
    atomic_write_json(args.output / "seed.json", seed_report)
    atomic_write_json(args.output / "selection.json", selection_report)
    extractor = HFScoreExtractor(config)
    started_at = datetime.now(timezone.utc)
    total_start = time.perf_counter()
    load_start = time.perf_counter()
    try:
        extractor.load()
        load_seconds = time.perf_counter() - load_start
        import torch

        cuda_metrics = bool(
            config.device.startswith("cuda") and torch.cuda.is_available()
        )
        if cuda_metrics:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        extraction_start = time.perf_counter()
        records = extractor.extract_to_jsonl(
            rows,
            args.output / "scores.jsonl",
            seed=args.seed,
            artifact_dir=args.output / "artifacts",
        )
        if cuda_metrics:
            torch.cuda.synchronize()
        extraction_seconds = time.perf_counter() - extraction_start
        runtime = {
            "status": "completed",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_load_seconds": load_seconds,
            "extraction_seconds": extraction_seconds,
            "wall_seconds": time.perf_counter() - total_start,
            "records": len(records),
            "reasoning_tokens": sum(record.token_length for record in records),
            "records_per_second": len(records) / max(extraction_seconds, 1e-12),
            "tokens_per_second": (
                sum(record.token_length for record in records)
                / max(extraction_seconds, 1e-12)
            ),
            "cuda_metrics_available": cuda_metrics,
            "peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if cuda_metrics else None
            ),
            "peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if cuda_metrics else None
            ),
            "energy_kwh": None,
            "energy_note": (
                "Not measured by the extractor; attach a separately logged power "
                "meter trace when available."
            ),
        }
        atomic_write_json(args.output / "runtime.json", runtime)
    finally:
        extractor.close()
    score_path = args.output / "scores.jsonl"
    run_manifest = {
        "schema_version": 1,
        "status": "completed",
        "model_id": config.model_id,
        "model_revision": config.revision,
        "input_path": str(args.input),
        "input_sha256": sha256_file(args.input),
        "scores_path": "scores.jsonl",
        "scores_sha256": sha256_file(score_path),
        "records": len(records),
        "artifact_records": sum(record.artifact_hash is not None for record in records),
        "config_sha256": sha256_file(args.output / "config.resolved.json"),
        "environment_sha256": sha256_file(args.output / "environment.json"),
        "seed_sha256": sha256_file(args.output / "seed.json"),
        "selection_sha256": sha256_file(args.output / "selection.json"),
        "runtime_sha256": sha256_file(args.output / "runtime.json"),
        "scope_boundary": (
            "A completed extraction cache is model evidence only. CAVIS "
            "transformation-specific conclusions additionally require eligible audited "
            "edits and frozen evaluation."
        ),
    }
    atomic_write_json(args.output / "run_manifest.json", run_manifest)
    print(f"Extracted {len(records)} records to {score_path}")


def as_serializable(config: ExtractionConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(config)


if __name__ == "__main__":
    main()

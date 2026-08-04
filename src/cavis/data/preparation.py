"""Materialize canonical, hashed Parquet inputs for GPU extraction."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cavis.transforms import make_transform

from .adapters import (
    load_geometry,
    load_math_shepherd,
    load_prm800k,
    load_processbench,
)
from .download import ArtifactSpec, load_manifest
from .io import canonical_json_hash, sha256_file, sha256_text
from .records import ReasoningExample
from .splits import assign_grouped_splits, proportional_stratified_sample


@dataclass(frozen=True, slots=True)
class PreparationSummary:
    dataset_key: str
    output_path: str
    output_sha256: str
    record_count: int
    label_counts: Mapping[str, int]
    split_counts: Mapping[str, Mapping[str, int]]
    source_files: Mapping[str, str]
    fingerprint: str
    cached: bool
    candidate_transform_counts: Mapping[str, int]
    not_applicable_counts: Mapping[str, int]
    semantic_status: str


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - core dependency
        raise RuntimeError("Dataset preparation requires PyYAML") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _base_row(
    example: ReasoningExample,
    *,
    split_seeds: Sequence[int],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reasoning_hash = sha256_text(example.reasoning_text)
    assignments = {
        seed: assign_grouped_splits([example.dependence_id], seed=seed)[
            example.dependence_id
        ]
        for seed in split_seeds
    }
    merged_metadata = dict(example.metadata)
    if metadata:
        merged_metadata.update(metadata)
    merged_metadata.update(
        {
            "source_group_id": example.group_id,
            "dependence_id": example.dependence_id,
        }
    )
    # Spectral scores are already handled by the read-only public audit.  They
    # are intentionally not copied into GPU input Parquet files.
    merged_metadata.pop("spectral", None)
    merged_metadata.pop("token_baselines", None)
    # Absolute cache locations are machine-local provenance, not semantic data.
    # Source hashes and relative paths are recorded in the preparation sidecar.
    merged_metadata.pop("proof_path", None)
    return {
        "item_id": example.item_id,
        "group_id": example.group_id,
        "source_group_id": example.group_id,
        "dependence_id": example.dependence_id,
        "theorem_name": example.metadata.get("theorem_name"),
        "statement_sha256": example.metadata.get("statement_sha256"),
        "dataset": example.dataset,
        "prompt": example.problem,
        "reasoning": example.reasoning_text,
        "label": int(example.valid) if example.valid is not None else None,
        "transformation_id": "base",
        "transform_family": "base",
        "transform_name": "base",
        "transform_kind": "base",
        "transform_parameters_json": "{}",
        "parent_item_id": None,
        "parent_variant_id": None,
        "semantic_variant_id": example.item_id,
        "pair_id": None,
        "pair_ids_json": "[]",
        "pair_side": None,
        "positive_item_id": None,
        "negative_item_id": None,
        "positive_semantic_variant_id": None,
        "negative_semantic_variant_id": None,
        # Transform evidence is keyed by the exact Lean/reasoning text. The
        # adapter's upstream row hash is retained separately.
        "source_hash": reasoning_hash,
        "target_hash": reasoning_hash,
        "upstream_source_hash": example.source_hash,
        "semantic_status": "observed_label",
        "evidence_state": "upstream_annotation",
        "mechanically_verified": False,
        "syntactic_contract_verified": False,
        "compiler_outcomes_match": False,
        "paired_validity_approved": False,
        "cavis_eligible": False,
        "step_labels_json": _json(list(example.step_labels)),
        "metadata_json": _json(merged_metadata),
        **{f"split_s{seed}": split for seed, split in assignments.items()},
    }


def materialize_leantwin_candidates(
    examples: Sequence[ReasoningExample],
    *,
    split_seeds: Sequence[int],
    transform_seed: int,
    transform_names: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Create explicitly unverified LeanTwin candidate rows.

    Model extraction may be run on these candidates because it is label-blind,
    but confirmatory evaluation must filter to separately recorded Lean
    evidence.  The function never upgrades ``semantic_status`` or
    ``mechanically_verified``.
    """

    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    g0_names = [
        name for name in transform_names if make_transform(name).spec.family == "g0"
    ]
    g1_names = [
        name for name in transform_names if make_transform(name).spec.family == "g1"
    ]
    for example in examples:
        base_row = _base_row(
            example,
            split_seeds=split_seeds,
            metadata={"leantwin_candidate": False},
        )
        base_row.update(
            {
                "semantic_status": "not_established",
                "evidence_state": "unverified",
                "pair_side": "positive" if example.valid else "negative",
            }
        )
        rows.append(base_row)
        if not example.reasoning_text.strip():
            skipped["missing_proof_text"] += 1
            continue
        positive_rows = [base_row]
        for name in g0_names:
            transform = make_transform(name)
            try:
                result = transform.apply(
                    example.reasoning_text,
                    item_id=example.item_id,
                    seed=transform_seed,
                )
            except ValueError:
                skipped[name] += 1
                continue
            variant_id = (
                f"{example.item_id}__{result.spec.name}__"
                f"s{transform_seed}__{result.target_sha256[:12]}"
            )
            assignments = {
                seed: assign_grouped_splits(
                    [example.dependence_id], seed=seed
                )[
                    example.dependence_id
                ]
                for seed in split_seeds
            }
            row = {
                "item_id": variant_id,
                "group_id": example.group_id,
                "source_group_id": example.group_id,
                "dependence_id": example.dependence_id,
                "theorem_name": example.metadata.get("theorem_name"),
                "statement_sha256": example.metadata.get(
                    "statement_sha256"
                ),
                "dataset": example.dataset,
                "prompt": example.problem,
                "reasoning": result.target,
                "label": int(example.valid) if example.valid is not None else None,
                "transformation_id": result.transformation_id,
                "transform_family": "g0",
                "transform_name": result.spec.name,
                "transform_kind": "g0",
                "transform_parameters_json": _json(dict(result.parameters)),
                "parent_item_id": example.item_id,
                "parent_variant_id": example.item_id,
                "semantic_variant_id": example.item_id,
                "pair_id": None,
                "pair_ids_json": "[]",
                "pair_side": "positive" if example.valid else "negative",
                "positive_item_id": None,
                "negative_item_id": None,
                "positive_semantic_variant_id": (
                    example.item_id if example.valid else None
                ),
                "negative_semantic_variant_id": (
                    example.item_id if not example.valid else None
                ),
                "source_hash": result.source_sha256,
                "target_hash": result.target_sha256,
                "upstream_source_hash": example.source_hash,
                "semantic_status": "not_established",
                "evidence_state": "unverified",
                "mechanically_verified": False,
                "syntactic_contract_verified": False,
                "compiler_outcomes_match": False,
                "paired_validity_approved": False,
                "cavis_eligible": False,
                "step_labels_json": _json(
                    [
                        int(example.valid)
                        if example.valid is not None
                        else None
                    ]
                ),
                "metadata_json": _json(
                    {
                        "leantwin_candidate": True,
                        "expected_lean_outcome": result.spec.expected_lean_outcome,
                        "construction_seed": transform_seed,
                        "warning": (
                            "Design intent is not semantic evidence; join "
                            "external Lean evidence before evaluation."
                        ),
                    }
                ),
                **{
                    f"split_s{seed}": split
                    for seed, split in assignments.items()
                },
            }
            rows.append(row)
            positive_rows.append(row)
            counts[result.spec.name] += 1

        if example.valid is not True:
            continue
        pair_ids: list[str] = []
        for name in g1_names:
            transform = make_transform(name)
            try:
                negative_result = transform.apply(
                    example.reasoning_text,
                    item_id=example.item_id,
                    seed=transform_seed,
                )
            except ValueError:
                skipped[name] += 1
                continue
            negative_id = (
                f"{example.item_id}__{negative_result.spec.name}__"
                f"s{transform_seed}__{negative_result.target_sha256[:12]}"
            )
            pair_id = (
                f"{example.item_id}::{negative_result.spec.name}::s{transform_seed}"
            )
            pair_ids.append(pair_id)
            assignments = {
                seed: assign_grouped_splits(
                    [example.dependence_id], seed=seed
                )[
                    example.dependence_id
                ]
                for seed in split_seeds
            }
            negative_root = {
                "item_id": negative_id,
                "group_id": example.group_id,
                "source_group_id": example.group_id,
                "dependence_id": example.dependence_id,
                "theorem_name": example.metadata.get("theorem_name"),
                "statement_sha256": example.metadata.get(
                    "statement_sha256"
                ),
                "dataset": example.dataset,
                "prompt": example.problem,
                "reasoning": negative_result.target,
                # Intended process-validity label only; default evaluation
                # eligibility remains false until independent paired evidence.
                "label": 0,
                "transformation_id": negative_result.transformation_id,
                "transform_family": "g1",
                "transform_name": negative_result.spec.name,
                "transform_kind": "g1",
                "transform_parameters_json": _json(
                    dict(negative_result.parameters)
                ),
                "parent_item_id": example.item_id,
                "parent_variant_id": example.item_id,
                "semantic_variant_id": negative_id,
                "pair_id": pair_id,
                "pair_ids_json": _json([pair_id]),
                "pair_side": "negative",
                "positive_item_id": example.item_id,
                "negative_item_id": negative_id,
                "positive_semantic_variant_id": example.item_id,
                "negative_semantic_variant_id": negative_id,
                "source_hash": negative_result.source_sha256,
                "target_hash": negative_result.target_sha256,
                "upstream_source_hash": example.source_hash,
                "semantic_status": "not_established",
                "evidence_state": "unverified",
                "mechanically_verified": False,
                "syntactic_contract_verified": False,
                "compiler_outcomes_match": False,
                "paired_validity_approved": False,
                "cavis_eligible": False,
                "step_labels_json": "[0]",
                "metadata_json": _json(
                    {
                        "leantwin_candidate": True,
                        "expected_lean_outcome": "reject",
                        "construction_seed": transform_seed,
                        "warning": (
                            "Compiler rejection alone is not theorem invalidity; "
                            "paired human validity evidence is required."
                        ),
                    }
                ),
                **{
                    f"split_s{seed}": split
                    for seed, split in assignments.items()
                },
            }
            rows.append(negative_root)
            counts[negative_result.spec.name] += 1

            # Negative-side G0 descendants make r_i^- observable.  They remain
            # ineligible until the G1 parent has independent validity approval
            # and the exact G0 rewrite contract/compiler outcome is checked.
            for g0_name in g0_names:
                g0_transform = make_transform(g0_name)
                try:
                    descendant = g0_transform.apply(
                        negative_result.target,
                        item_id=negative_id,
                        seed=transform_seed,
                    )
                except ValueError:
                    skipped[f"negative/{g0_name}"] += 1
                    continue
                descendant_id = (
                    f"{negative_id}__{descendant.spec.name}__"
                    f"s{transform_seed}__{descendant.target_sha256[:12]}"
                )
                rows.append(
                    {
                        **negative_root,
                        "item_id": descendant_id,
                        "reasoning": descendant.target,
                        "transformation_id": (
                            f"{negative_result.transformation_id}>"
                            f"{descendant.transformation_id}"
                        ),
                        "transform_family": "g0",
                        "transform_name": descendant.spec.name,
                        "transform_kind": "g0",
                        "transform_parameters_json": _json(
                            dict(descendant.parameters)
                        ),
                        "parent_item_id": negative_id,
                        "parent_variant_id": negative_id,
                        "source_hash": descendant.source_sha256,
                        "target_hash": descendant.target_sha256,
                        "metadata_json": _json(
                            {
                                "leantwin_candidate": True,
                                "composed_after_g1": (
                                    negative_result.spec.name
                                ),
                                "expected_source_outcome": "reject",
                                "expected_target_outcome": "reject",
                                "construction_seed": transform_seed,
                            }
                        ),
                    }
                )
                counts[f"negative_g0/{descendant.spec.name}"] += 1

        if pair_ids:
            for row in positive_rows:
                row["pair_ids_json"] = _json(pair_ids)
                row["positive_semantic_variant_id"] = example.item_id
    return rows, dict(counts), dict(skipped)


def _canonical_rows(
    examples: Iterable[ReasoningExample], *, split_seeds: Sequence[int]
) -> list[dict[str, Any]]:
    return [
        _base_row(example, split_seeds=split_seeds)
        for example in examples
        if example.valid is not None and example.reasoning_text.strip()
    ]


def _write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty dataset: {path}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - core dependency
        raise RuntimeError("Dataset preparation requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        table = pa.Table.from_pylist(rows)
        pq.write_table(
            table,
            temp_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(_json(value) + "\n", encoding="utf-8")
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _artifact_by_name(specs: Sequence[ArtifactSpec]) -> dict[str, ArtifactSpec]:
    return {spec.name: spec for spec in specs}


def _source_path(project_root: Path, spec: ArtifactSpec) -> Path:
    path = (project_root / spec.destination).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Locked source {spec.name} is missing at {path}; run download_data.py first"
        )
    return path


def _dataset_source_files(
    key: str,
    config: Mapping[str, Any],
    *,
    project_root: Path,
    artifacts: Mapping[str, ArtifactSpec],
) -> tuple[list[ReasoningExample], dict[str, str], dict[str, Any]]:
    source_name = str(config["source"])
    spec = artifacts[source_name]
    source_root = _source_path(project_root, spec)
    extra: dict[str, Any] = {}
    if source_name == "geometry_of_reason":
        extraction_relative = str(
            spec.metadata.get(
                "extraction_path",
                "data/results/rebuttal/llama8b_full_extraction.json",
            )
        )
        extraction = source_root / extraction_relative
        proofs = source_root / "data" / "experiment_ready"
        examples = load_geometry(
            extraction, proof_root=proofs, label_field="label_corrected"
        )
        source_files = {
            str(extraction.relative_to(project_root)): sha256_file(extraction)
        }
        for proof in sorted(proofs.glob("*/*.lean")):
            source_files[str(proof.relative_to(project_root))] = sha256_file(proof)
        extra["proof_root"] = str(proofs.relative_to(project_root))
        return examples, source_files, extra
    if source_name == "processbench":
        subset = str(config["subset"])
        source_file = source_root / f"{subset}.json"
        examples = load_processbench(source_file, subset=subset)
        return (
            examples,
            {str(source_file.relative_to(project_root)): sha256_file(source_file)},
            extra,
        )
    if source_name == "prm800k":
        source_file = source_root / "phase2_test.jsonl"
        examples = load_prm800k(source_file)
        extra["upstream_split"] = "phase2_test"
        return (
            examples,
            {str(source_file.relative_to(project_root)): sha256_file(source_file)},
            extra,
        )
    if source_name == "math_shepherd":
        candidates = sorted((source_root / "data").glob("test-*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"Math-Shepherd test parquet not found below {source_root / 'data'}"
            )
        source_file = candidates[0]
        examples = load_math_shepherd(source_file)
        extra["upstream_split"] = "test"
        return (
            examples,
            {str(source_file.relative_to(project_root)): sha256_file(source_file)},
            extra,
        )
    raise ValueError(f"Unsupported source {source_name!r} for dataset {key}")


def prepare_dataset(
    config_path: str | Path,
    *,
    manifest_path: str | Path,
    project_root: str | Path,
    build_leantwin: bool,
    sample_seed: int = 17,
    transform_seed: int = 17,
    force: bool = False,
) -> PreparationSummary:
    """Prepare one configured dataset and its content-addressed sidecar."""

    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (root / config_file).resolve()
    locked_manifest = Path(manifest_path)
    if not locked_manifest.is_absolute():
        locked_manifest = (root / locked_manifest).resolve()
    payload = _load_yaml(config_file)
    config = payload.get("dataset")
    if not isinstance(config, Mapping):
        raise ValueError(f"{config_file} lacks a dataset mapping")
    key = str(config["key"])
    specs = load_manifest(locked_manifest)
    artifacts = _artifact_by_name(specs)
    if str(config["source"]) not in artifacts:
        raise KeyError(f"No locked source for {config['source']}")
    examples, source_files, source_extra = _dataset_source_files(
        key, config, project_root=root, artifacts=artifacts
    )
    split_config = config.get("split", {})
    split_seeds = tuple(
        int(seed)
        for seed in (
            split_config.get("seeds")
            if isinstance(split_config, Mapping) and split_config.get("seeds")
            else config.get("seeds", [17, 42, 97])
        )
    )
    unlabeled_count = sum(example.valid is None for example in examples)
    labeled = [example for example in examples if example.valid is not None]
    requested_sample = config.get("sample_size")
    if requested_sample is not None:
        requested = int(requested_sample)
        if len(labeled) < requested:
            raise ValueError(
                f"{key}: requested {requested} labeled rows, only {len(labeled)} available"
            )
        labeled = proportional_stratified_sample(
            labeled,
            requested,
            seed=sample_seed,
            key=lambda example: example.item_id,
            strata=lambda example: example.valid,
        )

    transform_counts: dict[str, int] = {}
    not_applicable: dict[str, int] = {}
    if key == "geometry_leantwin" and build_leantwin:
        transform_config = config.get("transformations", {})
        if not isinstance(transform_config, Mapping):
            raise ValueError("Geometry transformations config must be a mapping")
        names = [
            *[str(name) for name in transform_config.get("g0", [])],
            *[str(name) for name in transform_config.get("g1", [])],
        ]
        rows, transform_counts, not_applicable = materialize_leantwin_candidates(
            labeled,
            split_seeds=split_seeds,
            transform_seed=transform_seed,
            transform_names=names,
        )
        semantic_status = "contains_unverified_candidates"
    else:
        rows = _canonical_rows(labeled, split_seeds=split_seeds)
        semantic_status = (
            "base_only_build_leantwin_not_requested"
            if key == "geometry_leantwin"
            else "upstream_annotations"
        )

    for row in rows:
        row["dataset"] = key
    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["dependence_id"]),
            str(row["group_id"]),
            str(row["item_id"]),
            str(row["transformation_id"]),
        )
    )
    output = (root / str(config["prepared_path"])).resolve()
    if root not in output.parents:
        raise ValueError(f"Prepared output escapes project root: {output}")
    fingerprint_payload = {
        "config": payload,
        "manifest_sha256": sha256_file(locked_manifest),
        "source_files": source_files,
        "implementation_files": {
            str(path.relative_to(Path(__file__).parents[1])): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("adapters.py").resolve(),
                Path(__file__).with_name("dependence.py").resolve(),
                Path(__file__).with_name("splits.py").resolve(),
                (Path(__file__).parents[1] / "transforms" / "g0.py").resolve(),
                (Path(__file__).parents[1] / "transforms" / "g1.py").resolve(),
                (Path(__file__).parents[1] / "transforms" / "lexical.py").resolve(),
            )
        },
        "build_leantwin": build_leantwin and key == "geometry_leantwin",
        "sample_seed": sample_seed,
        "transform_seed": transform_seed,
        "split_seeds": split_seeds,
    }
    fingerprint = canonical_json_hash(fingerprint_payload)
    sidecar = output.with_suffix(output.suffix + ".meta.json")
    if not force and output.is_file() and sidecar.is_file():
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            cached.get("fingerprint") == fingerprint
            and cached.get("output_sha256") == sha256_file(output)
        ):
            return PreparationSummary(**{**cached["summary"], "cached": True})

    _write_parquet_atomic(output, rows)
    output_hash = sha256_file(output)
    label_counts = Counter(
        "unknown"
        if row["label"] is None
        else ("valid" if int(row["label"]) == 1 else "invalid")
        for row in rows
    )
    split_counts = {
        f"split_s{seed}": dict(
            Counter(str(row[f"split_s{seed}"]) for row in rows)
        )
        for seed in split_seeds
    }
    summary = PreparationSummary(
        dataset_key=key,
        output_path=str(output.relative_to(root)),
        output_sha256=output_hash,
        record_count=len(rows),
        label_counts=dict(label_counts),
        split_counts=split_counts,
        source_files=source_files,
        fingerprint=fingerprint,
        cached=False,
        candidate_transform_counts=transform_counts,
        not_applicable_counts={
            **not_applicable,
            "unlabeled_source_rows": unlabeled_count,
        },
        semantic_status=semantic_status,
    )
    sidecar_payload = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "output_sha256": output_hash,
        "summary": {**asdict(summary), "cached": False},
        "source_revision": artifacts[str(config["source"])].revision,
        "source_extra": source_extra,
    }
    _write_json_atomic(sidecar, sidecar_payload)
    return summary


def prepare_from_configs(
    *,
    config_dir: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    selected: Sequence[str] = (),
    build_leantwin: bool = False,
    sample_seed: int = 17,
    transform_seed: int = 17,
    force: bool = False,
    output_manifest: str | Path = "data/cache/prepared/preparation_manifest.json",
) -> list[PreparationSummary]:
    root = Path(project_root).resolve()
    configs = Path(config_dir)
    if not configs.is_absolute():
        configs = (root / configs).resolve()
    config_paths = sorted(configs.glob("*.yaml"))
    indexed = {
        str(_load_yaml(path)["dataset"]["key"]): path for path in config_paths
    }
    unknown = set(selected).difference(indexed)
    if unknown:
        raise ValueError(f"Unknown dataset config(s): {', '.join(sorted(unknown))}")
    keys = list(selected) if selected else sorted(indexed)
    summaries = [
        prepare_dataset(
            indexed[key],
            manifest_path=manifest_path,
            project_root=root,
            build_leantwin=build_leantwin,
            sample_seed=sample_seed,
            transform_seed=transform_seed,
            force=force,
        )
        for key in keys
    ]
    try:
        import pyarrow

        pyarrow_version = pyarrow.__version__
    except ImportError:  # pragma: no cover
        pyarrow_version = None
    manifest_output = (root / output_manifest).resolve()
    if root not in manifest_output.parents:
        raise ValueError("Preparation manifest escapes project root")
    locked_manifest = Path(manifest_path)
    if not locked_manifest.is_absolute():
        locked_manifest = root / locked_manifest
    _write_json_atomic(
        manifest_output,
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha256_file(locked_manifest),
            "sample_seed": sample_seed,
            "transform_seed": transform_seed,
            "build_leantwin": build_leantwin,
            "python": platform.python_version(),
            "pyarrow": pyarrow_version,
            "datasets": [asdict(summary) for summary in summaries],
            "scope_boundary": (
                "Rows with semantic_status=not_established or "
                "evidence_state=unverified are score-cache candidates only."
            ),
        },
    )
    return summaries

"""Deterministic post-extraction ablations over immutable ScoreRecords."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from cavis.data.splits import stable_hash
from cavis.evaluation.protocol import exact_grouped_split


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("every score record must contain metadata")
    return metadata


def _dependence_id(row: dict[str, Any]) -> str:
    value = _metadata(row).get("dependence_id")
    if not isinstance(value, str) or not value:
        raise ValueError(
            "every confirmatory score record needs metadata.dependence_id"
        )
    return value


def filter_transform_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    drop_g0_names: frozenset[str] = frozenset(),
    drop_g1_names: frozenset[str] = frozenset(),
    max_g0_per_root: int | None = None,
    max_g1_per_positive: int | None = None,
) -> list[dict[str, Any]]:
    """Filter complete transform subtrees without changing root scores."""

    if max_g0_per_root is not None and max_g0_per_root <= 0:
        raise ValueError("max_g0_per_root must be positive")
    if max_g1_per_positive is not None and max_g1_per_positive <= 0:
        raise ValueError("max_g1_per_positive must be positive")

    g1_by_positive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        metadata = _metadata(row)
        if metadata.get("source_transform_kind") == "g1":
            positive_id = str(metadata.get("g1_positive_id") or "")
            g1_by_positive[positive_id].append(row)

    all_g1_ids = {
        str(row["item_id"])
        for roots in g1_by_positive.values()
        for row in roots
    }
    kept_g1_ids: set[str] = set()
    for positive_id, roots in g1_by_positive.items():
        eligible = [
            row
            for row in roots
            if str(_metadata(row).get("transform_name", "")) not in drop_g1_names
        ]
        eligible.sort(
            key=lambda row: (
                stable_hash(
                    str(row["item_id"]),
                    seed=seed,
                    namespace=f"ablation-g1:{positive_id}",
                ),
                str(row["item_id"]),
            )
        )
        if max_g1_per_positive is not None:
            eligible = eligible[:max_g1_per_positive]
        kept_g1_ids.update(str(row["item_id"]) for row in eligible)

    provisional = []
    for row in records:
        metadata = _metadata(row)
        kind = metadata.get("variant_kind")
        if metadata.get("source_transform_kind") == "g1":
            if str(row["item_id"]) not in kept_g1_ids:
                continue
        elif kind == "g0":
            parent = str(metadata.get("g0_parent_id") or "")
            if parent in all_g1_ids and parent not in kept_g1_ids:
                continue
            if str(metadata.get("transform_name", "")) in drop_g0_names:
                continue
        provisional.append(row)

    if max_g0_per_root is None:
        return sorted(provisional, key=lambda row: str(row["item_id"]))
    g0_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots = []
    for row in provisional:
        metadata = _metadata(row)
        if metadata.get("variant_kind") == "g0":
            g0_by_parent[str(metadata["g0_parent_id"])].append(row)
        else:
            roots.append(row)
    selected = list(roots)
    for parent_id, children in g0_by_parent.items():
        children.sort(
            key=lambda row: (
                stable_hash(
                    str(row["item_id"]),
                    seed=seed,
                    namespace=f"ablation-g0:{parent_id}",
                ),
                str(row["item_id"]),
            )
        )
        selected.extend(children[:max_g0_per_root])
    return sorted(selected, key=lambda row: str(row["item_id"]))


def residualize_score_records(
    records: list[dict[str, Any]],
    *,
    target_score: str,
    nuisance_names: tuple[str, ...],
    split_seed: int,
) -> tuple[list[dict[str, Any]], str]:
    """Fit a nonlinear nuisance model on train roots and residualize all rows."""

    if not nuisance_names:
        raise ValueError("nuisance_names must not be empty")
    assignments = exact_grouped_split(
        (_dependence_id(row) for row in records),
        seed=split_seed,
    )

    def nuisance_vector(row: dict[str, Any]) -> list[float]:
        values = []
        for name in nuisance_names:
            value = (
                row["token_length"]
                if name == "token_length"
                else row["scores"][name]
            )
            values.append(float(value))
        return values

    train = [
        row
        for row in records
        if assignments[_dependence_id(row)] == "train"
        and _metadata(row).get("variant_kind") != "g0"
    ]
    if len(train) < 4:
        raise ValueError("at least four train roots are required for residualization")
    train_x = np.asarray([nuisance_vector(row) for row in train], dtype=np.float64)
    train_y = np.asarray(
        [float(row["scores"][target_score]) for row in train],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(train_x)) or not np.all(np.isfinite(train_y)):
        raise ValueError("residualization inputs must be finite")
    model = Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=min(5, max(2, len(train) // 4)),
                    degree=3,
                    include_bias=False,
                ),
            ),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train_x, train_y)
    all_x = np.asarray([nuisance_vector(row) for row in records], dtype=np.float64)
    residuals = np.asarray(
        [float(row["scores"][target_score]) for row in records],
        dtype=np.float64,
    ) - model.predict(all_x)
    suffix = "_and_".join(name.replace(".", "_") for name in nuisance_names)
    output_name = f"residualized.{target_score}.on_{suffix}"
    output = deepcopy(records)
    for row, residual in zip(output, residuals, strict=True):
        row["scores"][output_name] = float(residual)
        row["metadata"]["residualization"] = {
            "target_score": target_score,
            "nuisance_names": list(nuisance_names),
            "fit_scope": "train_roots_only",
            "split_seed": split_seed,
        }
    return output, output_name


__all__ = ["filter_transform_records", "residualize_score_records"]

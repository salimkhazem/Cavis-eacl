"""Leakage-safe layer selection and observational transfer evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from cavis.evaluation.calibration import fit_platt_scaler
from cavis.evaluation.metrics import binary_classification_metrics, roc_auc
from cavis.evaluation.protocol import (
    ProtocolResult,
    _select_threshold,
    evaluate_frozen_protocol,
    exact_grouped_split,
    fit_frozen_protocol,
    parse_protocol_rows,
)

SPECTRAL_METRICS = ("hfer", "fiedler", "smoothness", "spectral_entropy", "energy")
TOKEN_FAMILIES = {
    "length": ("n_tokens",),
    "perplexity": ("perplexity",),
    "mean_log_likelihood": ("mean_log_likelihood",),
    "mean_token_entropy": ("mean_entropy",),
    "max_token_entropy": ("max_entropy",),
}


def discover_score_families(records: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    """Discover a fixed family map from homogeneous ScoreRecord dictionaries."""

    if not records:
        raise ValueError("records must not be empty")
    key_sets = []
    for record in records:
        scores = record.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("each record must contain a scores mapping")
        key_sets.append(set(str(key) for key in scores))
    common = set.intersection(*key_sets)
    families = {
        family: names for family, names in TOKEN_FAMILIES.items() if set(names) <= common
    }
    for metric in SPECTRAL_METRICS:
        pattern = re.compile(rf"^layer_\d+\.{re.escape(metric)}$")
        candidates = tuple(sorted(name for name in common if pattern.fullmatch(name)))
        if candidates:
            families[metric] = candidates
        mean_name = f"layers_mean.{metric}"
        if mean_name in common:
            families[f"mean_layers_{metric}"] = (mean_name,)
    if not families:
        raise ValueError("no supported score families are common to all records")
    return dict(sorted(families.items()))


def selection_optimism_bound(*, n: int, m: int, delta: float = 0.05) -> float:
    if n <= 0 or m <= 0 or not 0.0 < delta < 1.0:
        raise ValueError("n/m must be positive and delta must lie in (0,1)")
    return math.sqrt(math.log(2.0 * m / delta) / (2.0 * n))


@dataclass(frozen=True, slots=True)
class SelectedCavisFamily:
    family: str
    selected_score_name: str
    candidate_train_aurocs: dict[str, float]
    selection_bound_delta_0_05: float
    result: ProtocolResult


def select_cavis_family(
    records: list[dict[str, Any]],
    *,
    family: str,
    candidate_names: tuple[str, ...],
    split_seed: int,
    alpha: float,
    calibration_size: int | None = None,
) -> SelectedCavisFamily:
    """Select a layer on train only, then evaluate its frozen CAVIS state."""

    if not candidate_names:
        raise ValueError("candidate_names must not be empty")
    fitted = {}
    parsed = {}
    for score_name in candidate_names:
        rows = parse_protocol_rows(records, score_name=score_name)
        parsed[score_name] = rows
        fitted[score_name] = fit_frozen_protocol(
            rows,
            score_name=score_name,
            split_seed=split_seed,
            alpha=alpha,
            calibration_size=calibration_size,
        )
    selected_name = min(
        candidate_names,
        key=lambda name: (-fitted[name].train_auroc, name),
    )
    state = fitted[selected_name]
    result = evaluate_frozen_protocol(parsed[selected_name], state)
    train_groups = sum(
        split == "train" for _, split in state.dependence_assignments
    )
    return SelectedCavisFamily(
        family=family,
        selected_score_name=selected_name,
        candidate_train_aurocs={
            name: fitted[name].train_auroc for name in sorted(candidate_names)
        },
        selection_bound_delta_0_05=selection_optimism_bound(
            n=train_groups,
            m=len(candidate_names),
        ),
        result=result,
    )


@dataclass(frozen=True, slots=True)
class ObservationalFamilyResult:
    family: str
    selected_score_name: str
    candidate_train_aurocs: dict[str, float]
    selection_bound_delta_0_05: float
    metrics: dict[str, Any]
    per_item: tuple[dict[str, Any], ...]


def evaluate_observational_family(
    records: list[dict[str, Any]],
    *,
    family: str,
    candidate_names: tuple[str, ...],
    split_seed: int,
) -> ObservationalFamilyResult:
    """Grouped train/calibration/test transfer protocol without G0 conclusions."""

    if not candidate_names:
        raise ValueError("candidate_names must not be empty")
    rows_by_name = {
        name: parse_protocol_rows(records, score_name=name) for name in candidate_names
    }
    reference = rows_by_name[candidate_names[0]]
    assignments = exact_grouped_split(
        (row.dependence_id for row in reference),
        seed=split_seed,
    )

    fitted: dict[str, tuple[int, float, float]] = {}
    for name, rows in rows_by_name.items():
        train = [
            row
            for row in rows
            if assignments[row.dependence_id] == "train"
            and row.is_root
            and not row.is_g1_root
        ]
        labels = np.asarray([row.label for row in train], dtype=np.int64)
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("observational train split must contain both classes")
        raw = np.asarray([row.score for row in train], dtype=np.float64)
        positive_auc = roc_auc(labels, raw)
        negative_auc = roc_auc(labels, -raw)
        orientation = -1 if negative_auc > positive_auc else 1
        oriented = orientation * raw
        threshold, _ = _select_threshold(labels, oriented)
        fitted[name] = (orientation, threshold, float(max(positive_auc, negative_auc)))

    selected_name = min(
        candidate_names,
        key=lambda name: (-fitted[name][2], name),
    )
    orientation, threshold, train_auroc = fitted[selected_name]
    selected_rows = rows_by_name[selected_name]
    calibration = [
        row
        for row in selected_rows
        if assignments[row.dependence_id] == "calibration"
        and row.is_root
        and not row.is_g1_root
    ]
    calibration_labels = np.asarray([row.label for row in calibration], dtype=np.int64)
    calibration_scores = orientation * np.asarray(
        [row.score for row in calibration],
        dtype=np.float64,
    )
    platt = fit_platt_scaler(
        calibration_labels,
        calibration_scores,
        seed=split_seed,
    )

    test = [
        row
        for row in selected_rows
        if assignments[row.dependence_id] == "test"
        and row.is_root
        and not row.is_g1_root
    ]
    labels = np.asarray([row.label for row in test], dtype=np.int64)
    scores = orientation * np.asarray([row.score for row in test], dtype=np.float64)
    probabilities = platt.predict(scores) if platt is not None else None
    binary = binary_classification_metrics(
        labels,
        scores,
        threshold=threshold,
        probabilities=probabilities,
    )
    train_groups = sum(split == "train" for split in assignments.values())
    metrics = {
        "protocol_version": "cavis-observational-v1",
        "family": family,
        "selected_score_name": selected_name,
        "candidate_train_aurocs": {
            name: fitted[name][2] for name in sorted(candidate_names)
        },
        "selection_bound_delta_0_05": selection_optimism_bound(
            n=train_groups,
            m=len(candidate_names),
        ),
        "split_seed": split_seed,
        "orientation": orientation,
        "threshold": threshold,
        "train_auroc": train_auroc,
        "probability_calibration_size": (
            platt.calibration_size if platt is not None else 0
        ),
        "test": {"binary": binary, "n_items": len(test)},
        "scope_boundary": (
            "Observational transfer only: no G0 transformations, invariance "
            "certificate, causal sensitivity, or CertifiedPairRate."
        ),
        "dependence_unit": "metadata.dependence_id",
        "dependence_assignments": dict(sorted(assignments.items())),
        # Deprecated alias: keys are canonical dependence IDs, not group_id.
        "group_assignments": dict(sorted(assignments.items())),
    }
    per_item = tuple(
        {
            "item_id": row.item_id,
            "dependence_id": row.dependence_id,
            "group_id": row.group_id,
            "label": row.label,
            "raw_score": row.score,
            "oriented_score": float(score),
            "probability_valid": (
                float(probabilities[index]) if probabilities is not None else None
            ),
            "prediction": int(score > threshold),
        }
        for index, (row, score) in enumerate(zip(test, scores, strict=True))
    )
    return ObservationalFamilyResult(
        family=family,
        selected_score_name=selected_name,
        candidate_train_aurocs=metrics["candidate_train_aurocs"],
        selection_bound_delta_0_05=metrics["selection_bound_delta_0_05"],
        metrics=metrics,
        per_item=per_item,
    )


__all__ = [
    "ObservationalFamilyResult",
    "SelectedCavisFamily",
    "discover_score_families",
    "evaluate_observational_family",
    "select_cavis_family",
    "selection_optimism_bound",
]

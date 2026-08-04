"""Frozen train/calibration/test protocol for CAVIS.

This module is deliberately stricter than a generic metric runner.  It consumes
``ScoreRecord``-style dictionaries and requires the following keys inside each
record's ``metadata`` mapping:

``semantic_variant_id``
    Globally unique identifier for the exact scored text/proof variant.
``dependence_id``
    Dataset-scoped canonical dependence unit.  For LeanTwin this is the
    canonical Lean theorem-statement fingerprint, not a theorem name or
    source-row alias.  It is the only identifier used for splitting,
    calibration, aggregation, and gates.
``group_id``
    Source provenance identifier.  It is reported for traceability but never
    used as the confirmatory dependence unit.
``variant_kind``
    ``"base"``, ``"g1"`` (a validity-changing negative root), or ``"g0"``.
    ``transform_kind`` is accepted as the extractor-facing alias.
``g0_parent_id``
    For a ``g0`` row, the ``semantic_variant_id`` of its base row; otherwise
    ``null``.  ``parent_variant_id`` is accepted as the extractor-facing alias.
``g1_pair_id``, ``g1_side``, and ``g1_positive_id``
    Optional validity-changing pair identifier and side
    (``"positive"``/``"negative"``).  In LeanTwin's one-to-many form, only a
    G1 negative root carries the pair fields and ``g1_positive_id`` points to a
    shared positive base.  The older explicit-two-sides form remains accepted.
    G0 descendants may inherit pair provenance, but never count as pair sides.
``cavis_eligible``
    A literal boolean controlling all confirmatory G0 and G1 calculations.
``mechanically_verified``
    A literal boolean retained only as evidence provenance.  It never controls
    CAVIS inclusion: reject/reject negative-side G0 variants can legitimately
    be CAVIS-eligible without being mechanically compile-verified.

The top-level record follows :class:`cavis.schemas.ScoreRecord`: in particular
it contains ``item_id``, dataset/model identifiers, ``label``, ``scores``, and
``metadata``.  The evaluator operates on exactly one dataset/model/revision/
extraction-seed slice and one named scalar score at a time.

Protocol choices are made in three disjoint stages:

1. group assignments and score orientation/threshold use train data only,
   with one mean-score observation per dependence ID and validity label;
2. side-specific conformal radii use one maximum eligible G0 radius per
   calibration theorem group for positive/base and G1-negative roots;
3. all reported performance, invariance, coverage, abstention, and pair
   metrics use test data only.

For group counts not divisible by five, "exact 60/20/20" means the deterministic
largest-remainder integer allocation.  Assignment is by a stable seeded hash,
so row order cannot affect a split.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from cavis.certificates import (
    certificate_decision,
    certificate_interval,
    finite_sample_quantile,
)
from cavis.data.splits import stable_hash
from cavis.evaluation.calibration import fit_platt_scaler
from cavis.evaluation.metrics import (
    binary_classification_metrics,
    invariance_flip_rate,
    roc_auc,
)
from cavis.schemas import ValidityDecision

PROTOCOL_VERSION = "cavis-frozen-v4-dependence-label-aggregation"
SPLIT_NAMES = ("train", "calibration", "test")
SPLIT_RATIOS = (0.6, 0.2, 0.2)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _aliased_metadata(
    metadata: Mapping[str, Any],
    canonical: str,
    alias: str,
) -> Any:
    """Read a canonical metadata field while checking a legacy/extractor alias."""

    has_canonical = canonical in metadata
    has_alias = alias in metadata
    if has_canonical and has_alias and metadata[canonical] != metadata[alias]:
        raise ValueError(f"metadata.{canonical} conflicts with its alias metadata.{alias}")
    if has_canonical:
        return metadata[canonical]
    return metadata.get(alias)


def _binary_label(value: Any) -> int:
    if type(value) is not int:  # reject bool, 1.0, and arbitrary truthy values
        raise ValueError("label must be an integer in {-1, 0, 1}")
    integer = int(value)
    if integer in {0, -1}:
        return 0
    if integer == 1:
        return 1
    raise ValueError("label must use {-1, +1} or {0, 1} encoding")


@dataclass(frozen=True, slots=True)
class ProtocolRow:
    """Validated long-form row used internally by the frozen protocol."""

    item_id: str
    semantic_variant_id: str
    dependence_id: str
    group_id: str
    dataset: str
    model_id: str
    model_revision: str
    extraction_seed: int
    transformation_id: str
    label: int
    score: float
    variant_kind: Literal["base", "g1", "g0"]
    source_transform_kind: Literal["base", "g1", "g0"]
    g0_parent_id: str | None
    g1_pair_id: str | None
    g1_side: Literal["positive", "negative"] | None
    g1_positive_id: str | None
    cavis_eligible: bool
    mechanically_verified: bool

    @property
    def is_root(self) -> bool:
        """Whether this row is a scored base/G1 root rather than a G0 child."""

        return self.variant_kind != "g0"

    @property
    def is_g1_root(self) -> bool:
        """Whether provenance identifies this root as a G1 corruption."""

        return self.is_root and (self.variant_kind == "g1" or self.source_transform_kind == "g1")

    @classmethod
    def from_score_record(
        cls,
        payload: Mapping[str, Any],
        *,
        score_name: str,
    ) -> ProtocolRow:
        """Parse one strict ``ScoreRecord``-style mapping."""

        if not isinstance(payload, Mapping):
            raise ValueError("each score record must be a mapping")
        metadata = payload.get("metadata")
        scores = payload.get("scores")
        if not isinstance(metadata, Mapping):
            raise ValueError("record metadata must be a mapping")
        if not isinstance(scores, Mapping):
            raise ValueError("record scores must be a mapping")
        if score_name not in scores:
            raise ValueError(f"score {score_name!r} is missing from a record")

        score = float(scores[score_name])
        if not math.isfinite(score):
            raise ValueError(f"score {score_name!r} must be finite")
        variant_kind = _aliased_metadata(metadata, "variant_kind", "transform_kind")
        if variant_kind not in {"base", "g1", "g0"}:
            raise ValueError("metadata.variant_kind must be 'base', 'g1', or 'g0'")
        source_transform_kind = metadata.get("source_transform_kind", variant_kind)
        if source_transform_kind not in {"base", "g1", "g0"}:
            raise ValueError("metadata.source_transform_kind must be 'base', 'g1', or 'g0'")
        if (variant_kind == "g0") != (source_transform_kind == "g0"):
            raise ValueError("G0 variant and source transform kinds must agree")
        parent_id = _optional_string(
            _aliased_metadata(metadata, "g0_parent_id", "parent_variant_id"),
            "g0_parent_id",
        )
        if variant_kind == "g0" and parent_id is None:
            raise ValueError("a G0 row must name metadata.g0_parent_id")
        if variant_kind != "g0" and parent_id is not None:
            raise ValueError("a root row cannot name metadata.g0_parent_id")

        pair_id = _optional_string(
            _aliased_metadata(metadata, "g1_pair_id", "pair_id"), "g1_pair_id"
        )
        pair_side = _aliased_metadata(metadata, "g1_side", "pair_side")
        if pair_side is not None and pair_side not in {"positive", "negative"}:
            raise ValueError("metadata.g1_side must be 'positive', 'negative', or null")
        if (pair_id is None) != (pair_side is None):
            raise ValueError("metadata.g1_pair_id and metadata.g1_side must occur together")
        positive_id = _optional_string(
            _aliased_metadata(
                metadata,
                "g1_positive_id",
                "positive_semantic_variant_id",
            ),
            "g1_positive_id",
        )
        if positive_id is not None and pair_side not in {None, "negative"}:
            raise ValueError("metadata.g1_positive_id belongs to a negative pair side")

        eligible = metadata.get("cavis_eligible")
        if type(eligible) is not bool:
            raise ValueError("metadata.cavis_eligible must be a literal boolean")

        verified = metadata.get("mechanically_verified")
        if type(verified) is not bool:
            raise ValueError("metadata.mechanically_verified must be a literal boolean")
        seed = payload.get("seed")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        item_id = _nonempty_string(payload.get("item_id"), "item_id")
        semantic_variant_id = _nonempty_string(
            metadata.get("semantic_variant_id"), "semantic_variant_id"
        )
        if semantic_variant_id != item_id:
            raise ValueError(
                "metadata.semantic_variant_id must equal the exact ScoreRecord item_id"
            )
        return cls(
            item_id=item_id,
            semantic_variant_id=semantic_variant_id,
            dependence_id=_nonempty_string(
                metadata.get("dependence_id"), "metadata.dependence_id"
            ),
            group_id=_nonempty_string(metadata.get("group_id"), "group_id"),
            dataset=_nonempty_string(payload.get("dataset"), "dataset"),
            model_id=_nonempty_string(payload.get("model_id"), "model_id"),
            model_revision=_nonempty_string(payload.get("model_revision"), "model_revision"),
            extraction_seed=seed,
            transformation_id=_nonempty_string(
                payload.get("transformation_id"), "transformation_id"
            ),
            label=_binary_label(payload.get("label")),
            score=score,
            variant_kind=variant_kind,
            source_transform_kind=source_transform_kind,
            g0_parent_id=parent_id,
            g1_pair_id=pair_id,
            g1_side=pair_side,
            g1_positive_id=positive_id,
            cavis_eligible=eligible,
            mechanically_verified=verified,
        )


@dataclass(frozen=True, slots=True)
class FrozenProtocolState:
    """All choices frozen before the test partition is inspected."""

    protocol_version: str
    score_name: str
    split_seed: int
    alpha: float
    orientation: int
    threshold: float
    train_auroc: float
    train_balanced_accuracy: float
    q_alpha_positive: float
    q_alpha_negative: float
    # Conservative backward-compatible alias: max of both side quantiles.
    q_alpha: float
    calibration_group_size: int
    calibration_group_pool_size: int
    calibration_root_pool_size: int
    calibration_positive_group_size: int
    calibration_positive_group_pool_size: int
    calibration_positive_root_pool_size: int
    calibration_negative_group_size: int
    calibration_negative_group_pool_size: int
    calibration_negative_root_pool_size: int
    # Deprecated generic aliases count the smaller side-specific group pool.
    calibration_size: int
    calibration_pool_size: int
    platt_coefficient: float | None
    platt_intercept: float | None
    probability_calibration_size: int
    dependence_assignments: tuple[tuple[str, str], ...]

    def assignments(self) -> dict[str, str]:
        """Return a mutable copy of the frozen dependence assignment."""

        return dict(self.dependence_assignments)

    @property
    def group_assignments(self) -> tuple[tuple[str, str], ...]:
        """Deprecated alias for readers of pre-v3 result objects."""

        return self.dependence_assignments


@dataclass(frozen=True, slots=True)
class ProtocolResult:
    """Complete test result and deterministic per-base-item records."""

    state: FrozenProtocolState
    metrics: Mapping[str, Any]
    per_item: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _ResolvedPair:
    pair_id: str
    positive: ProtocolRow | None
    negative: ProtocolRow | None


def exact_grouped_split(
    dependence_ids: Iterable[str],
    *,
    seed: int,
) -> dict[str, str]:
    """Assign dependence units to an exact deterministic 60/20/20 partition.

    Integer split sizes use the largest-remainder rule.  Tied remainders are
    resolved in ``train``, ``calibration``, ``test`` order.
    """

    units = {
        _nonempty_string(dependence_id, "dependence_id")
        for dependence_id in dependence_ids
    }
    if not units:
        raise ValueError("at least one dependence unit is required")
    n_groups = len(units)
    quotas = [ratio * n_groups for ratio in SPLIT_RATIOS]
    counts = [math.floor(quota) for quota in quotas]
    remainder = n_groups - sum(counts)
    priority = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in priority[:remainder]:
        counts[index] += 1

    ranked = sorted(
        units,
        key=lambda dependence_id: (
            stable_hash(
                dependence_id,
                seed=seed,
                namespace="cavis-protocol-split",
            ),
            dependence_id,
        ),
    )
    assignments: dict[str, str] = {}
    cursor = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        for dependence_id in ranked[cursor : cursor + count]:
            assignments[dependence_id] = split_name
        cursor += count
    return assignments


def parse_protocol_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    score_name: str,
) -> list[ProtocolRow]:
    """Parse and validate one homogeneous experiment slice."""

    score_name = _nonempty_string(score_name, "score_name")
    rows = [ProtocolRow.from_score_record(record, score_name=score_name) for record in records]
    if not rows:
        raise ValueError("the score record collection is empty")

    slice_keys = {
        (row.dataset, row.model_id, row.model_revision, row.extraction_seed) for row in rows
    }
    if len(slice_keys) != 1:
        raise ValueError(
            "protocol input must contain exactly one dataset/model/revision/seed slice"
        )
    by_variant: dict[str, ProtocolRow] = {}
    dependence_by_provenance_group: dict[str, str] = {}
    for row in rows:
        if row.semantic_variant_id in by_variant:
            raise ValueError(f"duplicate semantic_variant_id: {row.semantic_variant_id!r}")
        by_variant[row.semantic_variant_id] = row
        previous_dependence = dependence_by_provenance_group.setdefault(
            row.group_id,
            row.dependence_id,
        )
        if previous_dependence != row.dependence_id:
            raise ValueError(
                "one provenance group_id cannot map to multiple dependence_id values"
            )

    for row in rows:
        if row.variant_kind != "g0":
            continue
        parent = by_variant.get(row.g0_parent_id or "")
        if parent is None or not parent.is_root:
            raise ValueError(f"G0 parent {row.g0_parent_id!r} is missing or is not a root row")
        if row.group_id != parent.group_id:
            raise ValueError("a G0 row and its parent must share group_id")
        if row.dependence_id != parent.dependence_id:
            raise ValueError("a G0 row and its parent must share dependence_id")
        if row.label != parent.label:
            raise ValueError("a G0 row and its parent must share the validity label")
        if row.cavis_eligible and not parent.cavis_eligible:
            raise ValueError("a CAVIS-eligible G0 row requires an eligible parent")

    _resolve_pair_roots(rows)
    return rows


def _resolve_pair_roots(rows: Sequence[ProtocolRow]) -> list[_ResolvedPair]:
    """Resolve modern pointer pairs and legacy explicit-two-side pairs.

    Only root rows participate.  G0 descendants may carry inherited pair
    provenance without becoming duplicate sides.
    """

    roots = {row.semantic_variant_id: row for row in rows if row.is_root}
    explicit: dict[str, dict[str, ProtocolRow]] = defaultdict(dict)
    for row in roots.values():
        if row.is_g1_root and (row.g1_pair_id is None or row.g1_side != "negative"):
            raise ValueError("a G1 root must be a named negative pair side")
        if row.g1_positive_id is not None and row.g1_side != "negative":
            raise ValueError("a root g1_positive_id requires g1_side='negative'")
        if row.g1_pair_id is None:
            continue
        side = row.g1_side or ""
        if side in explicit[row.g1_pair_id]:
            raise ValueError(f"duplicate root side {side!r} for G1 pair {row.g1_pair_id!r}")
        explicit[row.g1_pair_id][side] = row

    resolved: list[_ResolvedPair] = []
    for pair_id in sorted(explicit):
        sides = explicit[pair_id]
        negative = sides.get("negative")
        positive = sides.get("positive")
        if negative is not None and negative.g1_positive_id is not None:
            pointed_positive = roots.get(negative.g1_positive_id)
            if pointed_positive is None:
                raise ValueError(f"G1 positive root {negative.g1_positive_id!r} is missing")
            if positive is not None and positive != pointed_positive:
                raise ValueError("explicit positive side conflicts with g1_positive_id")
            positive = pointed_positive
        if positive is not None:
            if positive.label != 1:
                raise ValueError("a G1 positive root must have the positive label")
        if negative is not None:
            if negative.label != 0:
                raise ValueError("a G1 negative root must have the negative label")
        if positive is not None and negative is not None:
            if positive.dependence_id != negative.dependence_id:
                raise ValueError("both roots of a G1 pair must share dependence_id")
            if positive.semantic_variant_id == negative.semantic_variant_id:
                raise ValueError("G1 positive and negative roots must be distinct")
        resolved.append(
            _ResolvedPair(
                pair_id=pair_id,
                positive=positive,
                negative=negative,
            )
        )
    return resolved


def _split_rows(
    rows: Sequence[ProtocolRow],
    assignments: Mapping[str, str],
) -> dict[str, list[ProtocolRow]]:
    output = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        output[assignments[row.dependence_id]].append(row)
    for split_rows in output.values():
        split_rows.sort(key=lambda row: row.semantic_variant_id)
    return output


def _eligible_radii(rows: Sequence[ProtocolRow]) -> dict[str, float]:
    """Return max deviations using only explicitly CAVIS-eligible G0 rows."""

    bases = {row.semantic_variant_id: row for row in rows if row.is_root}
    deviations: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.variant_kind != "g0" or not row.cavis_eligible:
            continue
        parent = bases[row.g0_parent_id or ""]
        deviations[parent.semantic_variant_id].append(abs(row.score - parent.score))
    return {
        parent_id: max(parent_deviations) for parent_id, parent_deviations in deviations.items()
    }


def _group_max_radii(
    rows: Sequence[ProtocolRow],
    root_radii: Mapping[str, float],
) -> dict[str, float]:
    """Aggregate root radii to the theorem dependence unit."""

    roots = {row.semantic_variant_id: row for row in rows if row.is_root}
    grouped: dict[str, list[float]] = defaultdict(list)
    for root_id, radius in root_radii.items():
        root = roots.get(root_id)
        if root is None:
            raise ValueError(f"radius parent {root_id!r} is not a root row")
        grouped[root.dependence_id].append(float(radius))
    return {
        group_id: max(group_radii)
        for group_id, group_radii in grouped.items()
    }


def _certificate_side(row: ProtocolRow) -> Literal["positive", "negative"]:
    """Return the semantic validity side governing a root's radius."""

    return "positive" if row.label == 1 else "negative"


def _side_group_max_radii(
    rows: Sequence[ProtocolRow],
    root_radii: Mapping[str, float],
) -> tuple[
    dict[str, float],
    dict[str, float],
    int,
    int,
]:
    """Return positive/negative theorem maxima and their root counts."""

    roots = {row.semantic_variant_id: row for row in rows if row.is_root}
    grouped: dict[str, dict[str, list[float]]] = {
        "positive": defaultdict(list),
        "negative": defaultdict(list),
    }
    root_counts = Counter({"positive": 0, "negative": 0})
    for root_id, radius in root_radii.items():
        root = roots.get(root_id)
        if root is None:
            raise ValueError(f"radius parent {root_id!r} is not a root row")
        side = _certificate_side(root)
        grouped[side][root.dependence_id].append(float(radius))
        root_counts[side] += 1
    positive = {
        group_id: max(group_radii)
        for group_id, group_radii in grouped["positive"].items()
    }
    negative = {
        group_id: max(group_radii)
        for group_id, group_radii in grouped["negative"].items()
    }
    return (
        positive,
        negative,
        int(root_counts["positive"]),
        int(root_counts["negative"]),
    )


def _labelled_analysis_roots(rows: Sequence[ProtocolRow]) -> list[ProtocolRow]:
    """Return roots with a defensible label for discrimination/calibration.

    Upstream base labels remain available for the explicitly observational
    discrimination audit. A generated G1 label, however, is only established
    after the evidence join marks that root CAVIS-eligible.
    """

    return [
        row
        for row in rows
        if row.is_root and (not row.is_g1_root or row.cavis_eligible)
    ]


def _dependence_label_mean_scores(
    rows: Sequence[ProtocolRow],
) -> tuple[tuple[tuple[str, int], ...], np.ndarray, np.ndarray]:
    """Return one equally weighted mean-score cell per dependence and label.

    Multiple proof attempts or provenance aliases for the same proposition
    remain in one split, but they must not give that proposition extra weight
    when fitting or reporting the primary binary verifier metrics.  Distinct
    labels are retained because a dependence may contain both valid and
    invalid reasoning traces.
    """

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row.dependence_id, row.label)].append(float(row.score))
    if not grouped:
        raise ValueError("at least one labelled dependence-label cell is required")
    keys = tuple(sorted(grouped))
    labels = np.asarray([label for _, label in keys], dtype=np.int64)
    scores = np.asarray(
        [np.mean(grouped[key], dtype=np.float64) for key in keys],
        dtype=np.float64,
    )
    return keys, labels, scores


def _balanced_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    predictions = scores > threshold
    positives = labels == 1
    negatives = labels == 0
    if not np.any(positives) or not np.any(negatives):
        raise ValueError(
            "train dependence-label cells must contain both validity classes"
        )
    return 0.5 * (float(predictions[positives].mean()) + float((~predictions[negatives]).mean()))


def _select_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    unique_scores = np.unique(scores)
    lower_edge = np.nextafter(unique_scores[0], -math.inf)
    midpoints = (
        unique_scores[:-1] + 0.5 * (unique_scores[1:] - unique_scores[:-1])
    )
    candidates = np.r_[
        lower_edge if math.isfinite(float(lower_edge)) else unique_scores[0],
        midpoints,
        unique_scores[-1],
    ]
    values = np.asarray(
        [_balanced_accuracy(labels, scores, float(candidate)) for candidate in candidates]
    )
    best = float(np.max(values))
    # Stable, fully specified tie rule: smallest score threshold.
    best_index = int(np.flatnonzero(np.isclose(values, best, rtol=0.0, atol=1e-15))[0])
    return float(candidates[best_index]), best


def fit_frozen_protocol(
    rows: Sequence[ProtocolRow],
    *,
    score_name: str,
    split_seed: int,
    alpha: float,
    calibration_size: int | None = None,
) -> FrozenProtocolState:
    """Fit train choices and the calibration quantile without reading test data."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    assignments = exact_grouped_split(
        (row.dependence_id for row in rows),
        seed=split_seed,
    )
    partitions = _split_rows(rows, assignments)
    train_bases = _labelled_analysis_roots(partitions["train"])
    if not train_bases:
        raise ValueError("the train split contains no root rows")
    _, train_labels, raw_train_scores = _dependence_label_mean_scores(
        train_bases
    )
    if set(train_labels.tolist()) != {0, 1}:
        raise ValueError(
            "train dependence-label cells must contain both validity classes"
        )
    positive_auc = roc_auc(train_labels, raw_train_scores)
    negative_auc = roc_auc(train_labels, -raw_train_scores)
    orientation = -1 if negative_auc > positive_auc else 1
    oriented_train_scores = orientation * raw_train_scores
    threshold, train_balanced_accuracy = _select_threshold(train_labels, oriented_train_scores)

    calibration_root_radii = _eligible_radii(partitions["calibration"])
    if not calibration_root_radii:
        raise ValueError("the calibration split has no CAVIS-eligible G0 radius")
    (
        calibration_positive_group_radii,
        calibration_negative_group_radii,
        calibration_positive_root_pool_size,
        calibration_negative_root_pool_size,
    ) = _side_group_max_radii(
        partitions["calibration"],
        calibration_root_radii,
    )
    if not calibration_positive_group_radii:
        raise ValueError(
            "the calibration split has no eligible positive-side theorem group"
        )
    if not calibration_negative_group_radii:
        raise ValueError(
            "the calibration split has no eligible negative-side theorem group"
        )

    def rank_side(
        group_radii: Mapping[str, float],
        *,
        side: str,
    ) -> list[tuple[str, float]]:
        return sorted(
            group_radii.items(),
            key=lambda item: (
                stable_hash(
                    item[0],
                    seed=split_seed,
                    namespace=f"cavis-calibration-{side}-subsample",
                ),
                item[0],
            ),
        )

    ranked_positive = rank_side(
        calibration_positive_group_radii,
        side="positive",
    )
    ranked_negative = rank_side(
        calibration_negative_group_radii,
        side="negative",
    )
    positive_group_pool_size = len(ranked_positive)
    negative_group_pool_size = len(ranked_negative)
    if calibration_size is not None:
        if calibration_size <= 0:
            raise ValueError("calibration_size theorem-group count must be positive")
        if (
            calibration_size > positive_group_pool_size
            or calibration_size > negative_group_pool_size
        ):
            raise ValueError(
                "requested calibration_size exceeds a side-specific eligible "
                "theorem-group pool"
            )
        ranked_positive = ranked_positive[:calibration_size]
        ranked_negative = ranked_negative[:calibration_size]
    positive_radii = np.asarray(
        [radius for _, radius in ranked_positive],
        dtype=np.float64,
    )
    negative_radii = np.asarray(
        [radius for _, radius in ranked_negative],
        dtype=np.float64,
    )
    q_alpha_positive = finite_sample_quantile(positive_radii, alpha)
    q_alpha_negative = finite_sample_quantile(negative_radii, alpha)
    q_alpha = max(q_alpha_positive, q_alpha_negative)
    selected_dependence_ids = {
        dependence_id
        for dependence_id, _ in (*ranked_positive, *ranked_negative)
    }
    pooled_dependence_ids = (
        set(calibration_positive_group_radii)
        | set(calibration_negative_group_radii)
    )
    calibration_roots = _labelled_analysis_roots(partitions["calibration"])
    (
        _,
        calibration_labels,
        raw_calibration_scores,
    ) = _dependence_label_mean_scores(
        calibration_roots
    )
    calibration_scores = orientation * raw_calibration_scores
    platt = fit_platt_scaler(
        calibration_labels,
        calibration_scores,
        seed=split_seed,
    )
    return FrozenProtocolState(
        protocol_version=PROTOCOL_VERSION,
        score_name=score_name,
        split_seed=int(split_seed),
        alpha=float(alpha),
        orientation=orientation,
        threshold=threshold,
        train_auroc=float(max(positive_auc, negative_auc)),
        train_balanced_accuracy=train_balanced_accuracy,
        q_alpha_positive=q_alpha_positive,
        q_alpha_negative=q_alpha_negative,
        q_alpha=q_alpha,
        calibration_group_size=len(selected_dependence_ids),
        calibration_group_pool_size=len(pooled_dependence_ids),
        calibration_root_pool_size=len(calibration_root_radii),
        calibration_positive_group_size=int(positive_radii.size),
        calibration_positive_group_pool_size=positive_group_pool_size,
        calibration_positive_root_pool_size=(
            calibration_positive_root_pool_size
        ),
        calibration_negative_group_size=int(negative_radii.size),
        calibration_negative_group_pool_size=negative_group_pool_size,
        calibration_negative_root_pool_size=(
            calibration_negative_root_pool_size
        ),
        # Deprecated aliases expose the smaller side-specific effective pool.
        calibration_size=min(
            int(positive_radii.size),
            int(negative_radii.size),
        ),
        calibration_pool_size=min(
            positive_group_pool_size,
            negative_group_pool_size,
        ),
        platt_coefficient=(platt.coefficient if platt is not None else None),
        platt_intercept=(platt.intercept if platt is not None else None),
        probability_calibration_size=(
            platt.calibration_size if platt is not None else 0
        ),
        dependence_assignments=tuple(sorted(assignments.items())),
    )


def _standardized_mean(values: Sequence[float]) -> tuple[float, float]:
    """Return the arithmetic mean and finite paired-effect ``d_z``."""

    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    scale = float(array.std(ddof=1)) if array.size > 1 else abs(mean)
    # A constant nonzero effect has zero sampling dispersion. Use its RMS
    # magnitude as a finite diagnostic fallback rather than serializing inf.
    if scale <= np.finfo(np.float64).eps:
        scale = float(np.sqrt(np.mean(np.square(array))))
    effect = mean / scale if scale > np.finfo(np.float64).eps else 0.0
    return mean, effect


def _exact_two_sided_sign_test_pvalue(values: Sequence[float]) -> float:
    """Return the exact two-sided binomial sign-test p-value.

    Zero group means are discarded as ties. Under the null, every nonzero
    theorem-level mean gap has equal probability of either sign. The number of
    positive signs is therefore binomial. This test deliberately ignores gap
    magnitudes and is not mislabeled as a mean-statistic randomization test.
    """

    nonzero = [float(value) for value in values if float(value) != 0.0]
    if not nonzero:
        return math.nan
    positives = sum(value > 0.0 for value in nonzero)
    tail = min(positives, len(nonzero) - positives)
    probability = sum(
        math.comb(len(nonzero), index)
        for index in range(tail + 1)
    ) / (2 ** len(nonzero))
    return min(1.0, 2.0 * probability)


def _group_mean(flags: Mapping[str, Sequence[bool]]) -> float:
    if not flags:
        return math.nan
    return float(
        np.mean(
            [
                np.mean(np.asarray(group_flags, dtype=np.float64))
                for group_flags in flags.values()
            ]
        )
    )


def _pair_results(
    test_rows: Sequence[ProtocolRow],
    *,
    orientation: int,
    radii: Mapping[str, float],
    threshold: float,
    q_alpha_positive: float,
    q_alpha_negative: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    pairs = _resolve_pair_roots(test_rows)

    sensitivity_flags: list[bool] = []
    empirical_robust_flags: list[bool] = []
    certificate_flags: list[bool] = []
    score_gaps: list[float] = []
    decision_flip_flags: list[bool] = []
    directional_decision_flags: list[bool] = []
    group_score_gaps: dict[str, list[float]] = defaultdict(list)
    group_sensitivity_flags: dict[str, list[bool]] = defaultdict(list)
    group_decision_flip_flags: dict[str, list[bool]] = defaultdict(list)
    group_directional_decision_flags: dict[str, list[bool]] = defaultdict(list)
    group_empirical_robust_flags: dict[str, list[bool]] = defaultdict(list)
    group_certificate_flags: dict[str, list[bool]] = defaultdict(list)
    pair_details: dict[str, dict[str, Any]] = {}
    counts = Counter(
        {
            "n_pair_ids": len(pairs),
            "n_pairs_complete": 0,
            "n_pairs_incomplete": 0,
            "n_pairs_root_eligible": 0,
            "n_pairs_ineligible_roots": 0,
            "n_pairs_with_both_radii": 0,
            "n_pairs_missing_one_or_both_radii": 0,
            "n_pairs_eligible": 0,
            "n_empirically_robust_pairs": 0,
            "n_certified_pairs": 0,
        }
    )
    for pair in pairs:
        pair_id = pair.pair_id
        positive = pair.positive
        negative = pair.negative
        detail: dict[str, Any] = {
            "complete": positive is not None and negative is not None,
            "roots_cavis_eligible": False,
            "has_both_radii": False,
            "cavis_eligible": False,
            "observed_radius_separation": None,
            "conformal_certified_pair": None,
            # Backward-compatible alias for conformal_certified_pair.
            "certified": None,
        }
        if positive is None or negative is None:
            counts["n_pairs_incomplete"] += 1
            pair_details[pair_id] = detail
            continue
        counts["n_pairs_complete"] += 1
        if not (positive.cavis_eligible and negative.cavis_eligible):
            counts["n_pairs_ineligible_roots"] += 1
            pair_details[pair_id] = detail
            continue
        counts["n_pairs_root_eligible"] += 1
        detail["roots_cavis_eligible"] = True
        positive_score = orientation * positive.score
        negative_score = orientation * negative.score
        positive_radius = radii.get(positive.semantic_variant_id)
        negative_radius = radii.get(negative.semantic_variant_id)
        if positive_radius is None or negative_radius is None:
            counts["n_pairs_missing_one_or_both_radii"] += 1
            pair_details[pair_id] = detail
            continue
        counts["n_pairs_with_both_radii"] += 1
        counts["n_pairs_eligible"] += 1
        detail["has_both_radii"] = True
        detail["cavis_eligible"] = True
        detail["dependence_id"] = positive.dependence_id
        detail["positive_group_id"] = positive.group_id
        detail["negative_group_id"] = negative.group_id
        score_gap = positive_score - negative_score
        positive_decision = positive_score > threshold
        negative_decision = negative_score > threshold
        sensitive = score_gap > 0.0
        decision_flip = positive_decision != negative_decision
        directional_decision = positive_decision and not negative_decision
        empirical_robust = score_gap > (positive_radius + negative_radius)
        certified = score_gap > (q_alpha_positive + q_alpha_negative)
        sensitivity_flags.append(sensitive)
        score_gaps.append(score_gap)
        decision_flip_flags.append(decision_flip)
        directional_decision_flags.append(directional_decision)
        empirical_robust_flags.append(empirical_robust)
        certificate_flags.append(certified)
        group_score_gaps[positive.dependence_id].append(score_gap)
        group_sensitivity_flags[positive.dependence_id].append(sensitive)
        group_decision_flip_flags[positive.dependence_id].append(decision_flip)
        group_directional_decision_flags[positive.dependence_id].append(
            directional_decision
        )
        group_empirical_robust_flags[positive.dependence_id].append(empirical_robust)
        group_certificate_flags[positive.dependence_id].append(certified)
        detail["score_gap"] = score_gap
        detail["decision_flip"] = decision_flip
        detail["directional_decision"] = directional_decision
        detail["observed_radius_separation"] = empirical_robust
        detail["conformal_certified_pair"] = certified
        # Backward-compatible names retained for existing result readers.
        detail["empirical_pair_robust"] = empirical_robust
        detail["certified"] = certified
        counts["n_empirically_robust_pairs"] += int(empirical_robust)
        counts["n_certified_pairs"] += int(certified)
        pair_details[pair_id] = detail

    mean_score_gap, paired_effect_dz = _standardized_mean(score_gaps)
    group_mean_gaps = [
        float(np.mean(group_score_gaps[group_id]))
        for group_id in sorted(group_score_gaps)
    ]
    mean_group_score_gap, paired_effect_group_dz = _standardized_mean(
        group_mean_gaps
    )

    return (
        {
            **dict(counts),
            "aggregation_unit": (
                "metadata.dependence_id (canonical theorem/dependence group)"
            ),
            "n_pair_groups_eligible": len(group_mean_gaps),
            "n_pair_dependence_groups_eligible": len(group_mean_gaps),
            "n_pair_groups_nonzero_gap": sum(gap != 0.0 for gap in group_mean_gaps),
            "n_pair_dependence_groups_nonzero_gap": sum(
                gap != 0.0 for gap in group_mean_gaps
            ),
            "sensitivity": (float(np.mean(sensitivity_flags)) if sensitivity_flags else math.nan),
            "sensitivity_group_rate": _group_mean(group_sensitivity_flags),
            "sensitivity_dependence_group_rate": _group_mean(
                group_sensitivity_flags
            ),
            "mean_score_gap": mean_score_gap,
            "paired_effect_dz": paired_effect_dz,
            "mean_group_score_gap": mean_group_score_gap,
            "mean_dependence_group_score_gap": mean_group_score_gap,
            "paired_effect_group_dz": paired_effect_group_dz,
            "paired_effect_dependence_group_dz": paired_effect_group_dz,
            "paired_effect_group_sign_test_pvalue": (
                _exact_two_sided_sign_test_pvalue(group_mean_gaps)
            ),
            "paired_effect_dependence_group_sign_test_pvalue": (
                _exact_two_sided_sign_test_pvalue(group_mean_gaps)
            ),
            "decision_flip_rate": (
                float(np.mean(decision_flip_flags))
                if decision_flip_flags
                else math.nan
            ),
            "decision_flip_group_rate": _group_mean(group_decision_flip_flags),
            "decision_flip_dependence_group_rate": _group_mean(
                group_decision_flip_flags
            ),
            "directional_decision_rate": (
                float(np.mean(directional_decision_flags))
                if directional_decision_flags
                else math.nan
            ),
            "directional_decision_group_rate": _group_mean(
                group_directional_decision_flags
            ),
            "directional_decision_dependence_group_rate": _group_mean(
                group_directional_decision_flags
            ),
            "n_observed_radius_separations": counts[
                "n_empirically_robust_pairs"
            ],
            "observed_radius_separation_rate": (
                float(np.mean(empirical_robust_flags))
                if empirical_robust_flags
                else math.nan
            ),
            "observed_radius_separation_group_rate": _group_mean(
                group_empirical_robust_flags
            ),
            "observed_radius_separation_dependence_group_rate": _group_mean(
                group_empirical_robust_flags
            ),
            # Backward-compatible aliases for observed-radius separation.
            "empirical_pair_robust_rate": (
                float(np.mean(empirical_robust_flags))
                if empirical_robust_flags
                else math.nan
            ),
            "empirical_pair_robust_group_rate": _group_mean(
                group_empirical_robust_flags
            ),
            "empirical_pair_robust_dependence_group_rate": _group_mean(
                group_empirical_robust_flags
            ),
            "n_conformal_certified_pairs": counts["n_certified_pairs"],
            "conformal_certified_pair_rate": (
                float(np.mean(certificate_flags))
                if certificate_flags
                else math.nan
            ),
            "conformal_certified_pair_group_rate": _group_mean(
                group_certificate_flags
            ),
            "conformal_certified_pair_dependence_group_rate": _group_mean(
                group_certificate_flags
            ),
            # Backward-compatible aliases for conformal pair certification.
            "certified_pair_rate": (
                float(np.mean(certificate_flags)) if certificate_flags else math.nan
            ),
            "certified_pair_group_rate": _group_mean(group_certificate_flags),
            "certified_pair_dependence_group_rate": _group_mean(
                group_certificate_flags
            ),
        },
        pair_details,
    )


def evaluate_frozen_protocol(
    rows: Sequence[ProtocolRow],
    state: FrozenProtocolState,
) -> ProtocolResult:
    """Evaluate a previously frozen state on its test groups only."""

    partitions = _split_rows(rows, state.assignments())
    test_rows = partitions["test"]
    all_test_roots = [row for row in test_rows if row.is_root]
    test_bases = _labelled_analysis_roots(test_rows)
    if not test_bases:
        raise ValueError("the test split contains no root rows")

    (
        test_dependence_label_cells,
        binary_labels,
        raw_binary_scores,
    ) = _dependence_label_mean_scores(test_bases)
    binary_scores = state.orientation * raw_binary_scores
    root_labels = np.asarray(
        [row.label for row in test_bases],
        dtype=np.int64,
    )
    root_scores = state.orientation * np.asarray(
        [row.score for row in test_bases],
        dtype=np.float64,
    )
    binary_probabilities = None
    root_probabilities = None
    if state.platt_coefficient is not None and state.platt_intercept is not None:
        from scipy.special import expit

        binary_probabilities = expit(
            state.platt_coefficient * binary_scores + state.platt_intercept
        )
        root_probabilities = expit(
            state.platt_coefficient * root_scores + state.platt_intercept
        )
    binary = binary_classification_metrics(
        binary_labels,
        binary_scores,
        threshold=state.threshold,
        probabilities=binary_probabilities,
    )
    root_binary_diagnostic = binary_classification_metrics(
        root_labels,
        root_scores,
        threshold=state.threshold,
        probabilities=root_probabilities,
    )

    base_lookup = {row.semantic_variant_id: row for row in test_bases}
    eligible_g0 = [row for row in test_rows if row.variant_kind == "g0" and row.cavis_eligible]
    base_g0_scores = np.asarray(
        [state.orientation * base_lookup[row.g0_parent_id or ""].score for row in eligible_g0],
        dtype=np.float64,
    )
    transformed_scores = np.asarray(
        [state.orientation * row.score for row in eligible_g0],
        dtype=np.float64,
    )
    if eligible_g0:
        inv_flip = invariance_flip_rate(
            base_g0_scores,
            transformed_scores,
            state.threshold,
        )
        grouped_invariance_flags: dict[str, list[bool]] = defaultdict(list)
        for row, base_score, transformed_score in zip(
            eligible_g0,
            base_g0_scores,
            transformed_scores,
            strict=True,
        ):
            grouped_invariance_flags[row.dependence_id].append(
                bool(
                    (base_score > state.threshold)
                    != (transformed_score > state.threshold)
                )
            )
        inv_flip_group_rate = _group_mean(grouped_invariance_flags)
        transform_deviations = np.abs(
            transformed_scores - base_g0_scores
        )
        transform_sides = [
            _certificate_side(base_lookup[row.g0_parent_id or ""])
            for row in eligible_g0
        ]
        transform_coverage_flags = [
            bool(
                deviation
                <= (
                    state.q_alpha_negative
                    if side == "negative"
                    else state.q_alpha_positive
                )
            )
            for deviation, side in zip(
                transform_deviations,
                transform_sides,
                strict=True,
            )
        ]
        empirical_transform_coverage = float(
            np.mean(transform_coverage_flags)
        )
        positive_transform_flags = [
            covered
            for covered, side in zip(
                transform_coverage_flags,
                transform_sides,
                strict=True,
            )
            if side == "positive"
        ]
        negative_transform_flags = [
            covered
            for covered, side in zip(
                transform_coverage_flags,
                transform_sides,
                strict=True,
            )
            if side == "negative"
        ]
        empirical_positive_transform_coverage = (
            float(np.mean(positive_transform_flags))
            if positive_transform_flags
            else math.nan
        )
        empirical_negative_transform_coverage = (
            float(np.mean(negative_transform_flags))
            if negative_transform_flags
            else math.nan
        )
    else:
        inv_flip = math.nan
        inv_flip_group_rate = math.nan
        grouped_invariance_flags = {}
        empirical_transform_coverage = math.nan
        empirical_positive_transform_coverage = math.nan
        empirical_negative_transform_coverage = math.nan
    test_radii = _eligible_radii(test_rows)
    test_group_radii = _group_max_radii(test_rows, test_radii)
    (
        test_positive_group_radii,
        test_negative_group_radii,
        test_positive_root_count,
        test_negative_root_count,
    ) = _side_group_max_radii(test_rows, test_radii)
    test_root_lookup = {
        row.semantic_variant_id: row
        for row in test_rows
        if row.is_root
    }
    empirical_radius_coverage = (
        float(
            np.mean(
                [
                    radius
                    <= (
                        state.q_alpha_negative
                        if _certificate_side(test_root_lookup[root_id])
                        == "negative"
                        else state.q_alpha_positive
                    )
                    for root_id, radius in test_radii.items()
                ]
            )
        )
        if test_radii
        else math.nan
    )
    side_group_coverage_flags = [
        *(
            radius <= state.q_alpha_positive
            for radius in test_positive_group_radii.values()
        ),
        *(
            radius <= state.q_alpha_negative
            for radius in test_negative_group_radii.values()
        ),
    ]
    empirical_group_radius_coverage = (
        float(np.mean(side_group_coverage_flags))
        if side_group_coverage_flags
        else math.nan
    )
    empirical_positive_group_radius_coverage = (
        float(
            np.mean(
                np.asarray(
                    list(test_positive_group_radii.values()),
                    dtype=np.float64,
                )
                <= state.q_alpha_positive
            )
        )
        if test_positive_group_radii
        else math.nan
    )
    empirical_negative_group_radius_coverage = (
        float(
            np.mean(
                np.asarray(
                    list(test_negative_group_radii.values()),
                    dtype=np.float64,
                )
                <= state.q_alpha_negative
            )
        )
        if test_negative_group_radii
        else math.nan
    )

    decisions: list[ValidityDecision] = []
    certificate_population: list[bool] = []
    per_item: list[dict[str, Any]] = []
    for row, score in zip(test_bases, root_scores, strict=True):
        diagnostic_side = _certificate_side(row)
        diagnostic_side_q_alpha = (
            state.q_alpha_negative
            if diagnostic_side == "negative"
            else state.q_alpha_positive
        )
        # A deployed verifier does not know the true validity side.  The
        # conservative maximum is therefore the only admissible radius for a
        # standalone interval or decision.
        row_q_alpha = state.q_alpha
        lower, upper = certificate_interval(float(score), row_q_alpha)
        decision = certificate_decision(
            float(score),
            state.threshold,
            row_q_alpha,
        )
        decisions.append(decision)
        certificate_population.append(row.cavis_eligible)
        per_item.append(
            {
                "item_id": row.item_id,
                "semantic_variant_id": row.semantic_variant_id,
                "dependence_id": row.dependence_id,
                "group_id": row.group_id,
                "label": row.label,
                "source_transform_kind": row.source_transform_kind,
                "raw_score": row.score,
                "oriented_score": float(score),
                "probability_valid": (
                    float(root_probabilities[len(per_item)])
                    if root_probabilities is not None
                    else None
                ),
                "orientation": state.orientation,
                "threshold": state.threshold,
                "q_alpha": row_q_alpha,
                "q_alpha_policy": "label_free_max_of_side_quantiles",
                "diagnostic_validity_side": diagnostic_side,
                "diagnostic_side_q_alpha": diagnostic_side_q_alpha,
                "interval_lower": lower,
                "interval_upper": upper,
                "decision": decision.value,
                "certificate_population_eligible": row.cavis_eligible,
                "certified": (
                    row.cavis_eligible
                    and decision is not ValidityDecision.ABSTAIN
                ),
                "g0_radius": test_radii.get(row.semantic_variant_id),
                "g1_pair_id": row.g1_pair_id,
                "g1_side": row.g1_side,
                "g1_positive_id": row.g1_positive_id,
                "cavis_eligible": row.cavis_eligible,
                "mechanically_verified": row.mechanically_verified,
            }
        )

    non_abstaining_mask = np.asarray(
        [decision is not ValidityDecision.ABSTAIN for decision in decisions],
        dtype=np.bool_,
    )
    eligible_root_mask = np.asarray(certificate_population, dtype=np.bool_)
    certified_mask = eligible_root_mask & non_abstaining_mask
    threshold_predictions = root_scores > state.threshold
    if np.any(certified_mask):
        certificate_predictions = np.asarray(
            [decision is ValidityDecision.VALID for decision in decisions],
            dtype=np.bool_,
        )
        selective_accuracy = float(
            np.mean(
                certificate_predictions[certified_mask]
                == root_labels[certified_mask]
            )
        )
    else:
        selective_accuracy = math.nan

    resolved_test_pairs = _resolve_pair_roots(test_rows)
    pair_metrics, pair_details = _pair_results(
        test_rows,
        orientation=state.orientation,
        radii=test_radii,
        threshold=state.threshold,
        q_alpha_positive=state.q_alpha_positive,
        q_alpha_negative=state.q_alpha_negative,
    )
    pair_memberships: dict[str, list[str]] = defaultdict(list)
    for pair in resolved_test_pairs:
        if pair.positive is not None:
            pair_memberships[pair.positive.semantic_variant_id].append(pair.pair_id)
        if pair.negative is not None:
            pair_memberships[pair.negative.semantic_variant_id].append(pair.pair_id)
    for item in per_item:
        memberships = sorted(pair_memberships.get(item["semantic_variant_id"], []))
        item["g1_pair_ids"] = memberships
        item["pair_audits"] = {pair_id: pair_details[pair_id] for pair_id in memberships}
        if len(memberships) == 1:  # compatibility for simple explicit pairs
            item["pair_audit"] = pair_details[memberships[0]]

    split_group_counts = Counter(state.assignments().values())
    split_row_counts = {name: len(partitions[name]) for name in SPLIT_NAMES}
    ineligible_g0_by_split = {
        name: sum(row.variant_kind == "g0" and not row.cavis_eligible for row in partitions[name])
        for name in SPLIT_NAMES
    }
    all_pairs = _resolve_pair_roots(rows)
    paired_root_ids = {
        root.semantic_variant_id
        for pair in all_pairs
        for root in (pair.positive, pair.negative)
        if root is not None
    }
    ineligible_paired_roots_by_split = {
        name: sum(
            row.is_root and row.semantic_variant_id in paired_root_ids and not row.cavis_eligible
            for row in partitions[name]
        )
        for name in SPLIT_NAMES
    }
    mechanically_verified_by_split = {
        name: sum(row.mechanically_verified for row in partitions[name]) for name in SPLIT_NAMES
    }
    eligible_test_roots = int(eligible_root_mask.sum())
    eligible_decision_count = int(eligible_root_mask.sum())
    if eligible_decision_count:
        decision_coverage = float(
            non_abstaining_mask[eligible_root_mask].mean()
        )
        abstention_rate = 1.0 - decision_coverage
    else:
        decision_coverage = math.nan
        abstention_rate = math.nan
    excluded_ineligible_g1_test_roots = sum(
        row.is_g1_root and not row.cavis_eligible for row in all_test_roots
    )
    metrics: dict[str, Any] = {
        "protocol_version": state.protocol_version,
        "slice": {
            "dataset": rows[0].dataset,
            "model_id": rows[0].model_id,
            "model_revision": rows[0].model_revision,
            "extraction_seed": rows[0].extraction_seed,
            "score_name": state.score_name,
        },
        "frozen": {
            "split_seed": state.split_seed,
            "alpha": state.alpha,
            "orientation": state.orientation,
            "threshold": state.threshold,
            "train_auroc": state.train_auroc,
            "train_balanced_accuracy": state.train_balanced_accuracy,
            "binary_fit_aggregation_unit": (
                "dependence_id_x_label_mean_score"
            ),
            "probability_calibration_aggregation_unit": (
                "dependence_id_x_label_mean_score"
            ),
            "q_alpha_positive": state.q_alpha_positive,
            "q_alpha_negative": state.q_alpha_negative,
            # Label-free radius used by every standalone interval/decision.
            "q_alpha": state.q_alpha,
            "standalone_q_alpha_policy": "max_of_side_quantiles_label_free",
            "side_quantile_role": (
                "diagnostic coverage and paired certificates only"
            ),
            # Numeric 0/1 fields survive JSON sanitization and CSV aggregation
            # even when an unavailable finite-sample quantile is represented
            # mathematically as +inf (and serialized as null).
            "finite_q_alpha_positive_available": int(
                math.isfinite(state.q_alpha_positive)
            ),
            "finite_q_alpha_negative_available": int(
                math.isfinite(state.q_alpha_negative)
            ),
            "finite_q_alpha_both_sides_available": int(
                math.isfinite(state.q_alpha_positive)
                and math.isfinite(state.q_alpha_negative)
            ),
            "calibration_dependence_unit": "metadata.dependence_id",
            "calibration_dependence_unit_description": (
                "canonical theorem/dependence group"
            ),
            "calibration_group_size": state.calibration_group_size,
            "calibration_group_pool_size": state.calibration_group_pool_size,
            "calibration_root_pool_size": state.calibration_root_pool_size,
            "calibration_positive_group_size": (
                state.calibration_positive_group_size
            ),
            "calibration_positive_group_pool_size": (
                state.calibration_positive_group_pool_size
            ),
            "calibration_positive_root_pool_size": (
                state.calibration_positive_root_pool_size
            ),
            "calibration_negative_group_size": (
                state.calibration_negative_group_size
            ),
            "calibration_negative_group_pool_size": (
                state.calibration_negative_group_pool_size
            ),
            "calibration_negative_root_pool_size": (
                state.calibration_negative_root_pool_size
            ),
            # Deprecated aliases count the smaller side-specific group pool.
            "calibration_size": state.calibration_size,
            "calibration_pool_size": state.calibration_pool_size,
            "platt_coefficient": state.platt_coefficient,
            "platt_intercept": state.platt_intercept,
            "probability_calibration_size": state.probability_calibration_size,
        },
        "splits": {
            "dependence_unit": "metadata.dependence_id",
            "dependence_unit_description": (
                "canonical theorem/dependence group; group_id is provenance only"
            ),
            "dependence_counts": {
                name: int(split_group_counts.get(name, 0))
                for name in SPLIT_NAMES
            },
            "dependence_assignments": dict(state.dependence_assignments),
            # Deprecated aliases: values are canonical dependence-unit counts
            # and assignments, never source-provenance group_id values.
            "group_counts": {name: int(split_group_counts.get(name, 0)) for name in SPLIT_NAMES},
            "row_counts": split_row_counts,
            "group_assignments": dict(state.group_assignments),
        },
        "exclusions": {
            "ineligible_g0_by_split": ineligible_g0_by_split,
            "ineligible_g0_total": int(sum(ineligible_g0_by_split.values())),
            "ineligible_paired_roots_by_split": ineligible_paired_roots_by_split,
            "ineligible_paired_roots_total": int(sum(ineligible_paired_roots_by_split.values())),
            # Deprecated count aliases retained for old result parsers.  Their
            # values now follow cavis_eligible, never mechanical provenance.
            "unverified_g0_by_split": ineligible_g0_by_split,
            "unverified_g0_total": int(sum(ineligible_g0_by_split.values())),
        },
        "provenance": {
            "mechanically_verified_rows_by_split": mechanically_verified_by_split,
            "mechanically_verified_rows_total": int(sum(mechanically_verified_by_split.values())),
        },
        "test": {
            "binary": binary,
            "binary_root_level_diagnostic": root_binary_diagnostic,
            "binary_population": {
                "primary_aggregation_unit": "dependence_id_x_label_mean_score",
                "n_dependence_label_cells": len(
                    test_dependence_label_cells
                ),
                "n_primary_binary_observations": len(
                    test_dependence_label_cells
                ),
                "n_root_rows": len(test_bases),
                "n_all_root_rows_before_g1_label_filter": len(all_test_roots),
                "n_base_roots": sum(row.source_transform_kind == "base" for row in test_bases),
                "n_g1_roots": sum(row.is_g1_root for row in test_bases),
                "n_excluded_ineligible_g1_roots": excluded_ineligible_g1_test_roots,
                "n_cavis_eligible_roots": int(eligible_test_roots),
                "n_cavis_ineligible_roots": int(len(test_bases) - eligible_test_roots),
                "includes_upstream_labels": True,
                "includes_ineligible_g1_candidate_labels": False,
            },
            "invariance": {
                "aggregation_unit": (
                    "metadata.dependence_id "
                    "(canonical theorem/dependence group)"
                ),
                "inv_flip": inv_flip,
                "inv_flip_group_rate": inv_flip_group_rate,
                "n_invariance_groups": len(grouped_invariance_flags),
                # Root-level coverage is diagnostic; theorem-group maximum
                # coverage is aligned with the group-aware conformal unit.
                "empirical_radius_coverage": empirical_radius_coverage,
                "empirical_group_radius_coverage": (
                    empirical_group_radius_coverage
                ),
                "empirical_positive_group_radius_coverage": (
                    empirical_positive_group_radius_coverage
                ),
                "empirical_negative_group_radius_coverage": (
                    empirical_negative_group_radius_coverage
                ),
                # Reported separately to expose within-item transform behavior.
                "empirical_transform_coverage": empirical_transform_coverage,
                "empirical_positive_transform_coverage": (
                    empirical_positive_transform_coverage
                ),
                "empirical_negative_transform_coverage": (
                    empirical_negative_transform_coverage
                ),
                "n_eligible_g0_rows": len(eligible_g0),
                "n_root_items_with_radius": len(test_radii),
                "n_groups_with_radius": len(test_group_radii),
                "n_dependence_groups_with_radius": len(test_group_radii),
                "n_positive_root_items_with_radius": test_positive_root_count,
                "n_negative_root_items_with_radius": test_negative_root_count,
                "n_positive_groups_with_radius": len(
                    test_positive_group_radii
                ),
                "n_positive_dependence_groups_with_radius": len(
                    test_positive_group_radii
                ),
                "n_negative_groups_with_radius": len(
                    test_negative_group_radii
                ),
                "n_negative_dependence_groups_with_radius": len(
                    test_negative_group_radii
                ),
                # Deprecated aliases for pre-LeanTwin report readers.
                "n_verified_g0_rows": len(eligible_g0),
                "n_base_items_with_radius": len(test_radii),
            },
            "certificates": {
                "standalone_q_alpha": state.q_alpha,
                "standalone_q_alpha_policy": (
                    "max_of_side_quantiles_label_free"
                ),
                "side_quantile_role": (
                    "diagnostic coverage and paired certificates only"
                ),
                "finite_q_alpha_positive_available": int(
                    math.isfinite(state.q_alpha_positive)
                ),
                "finite_q_alpha_negative_available": int(
                    math.isfinite(state.q_alpha_negative)
                ),
                "finite_q_alpha_both_sides_available": int(
                    math.isfinite(state.q_alpha_positive)
                    and math.isfinite(state.q_alpha_negative)
                ),
                "abstention_rate": abstention_rate,
                "decision_coverage": decision_coverage,
                "selective_accuracy": selective_accuracy,
                "n_certified_decisions": int(certified_mask.sum()),
                "n_eligible_root_items": eligible_decision_count,
                "n_ineligible_root_items_excluded": int(
                    len(test_bases) - eligible_decision_count
                ),
                "n_test_root_items": len(test_bases),
                "n_test_base_items": len(test_bases),
            },
            "pairs": pair_metrics,
            "diagnostic_threshold_accuracy_root_level": float(
                np.mean(threshold_predictions == root_labels)
            ),
            # Backward-compatible alias. This is deliberately not the primary
            # dependence-label-cell binary accuracy.
            "diagnostic_threshold_accuracy": float(
                np.mean(threshold_predictions == root_labels)
            ),
        },
    }
    return ProtocolResult(
        state=state,
        metrics=metrics,
        per_item=tuple(per_item),
    )


def run_frozen_protocol(
    records: Iterable[Mapping[str, Any]],
    *,
    score_name: str,
    split_seed: int = 17,
    alpha: float = 0.1,
    calibration_size: int | None = None,
) -> ProtocolResult:
    """Parse records, freeze train/calibration choices, and evaluate test."""

    rows = parse_protocol_rows(records, score_name=score_name)
    state = fit_frozen_protocol(
        rows,
        score_name=score_name,
        split_seed=split_seed,
        alpha=alpha,
        calibration_size=calibration_size,
    )
    return evaluate_frozen_protocol(rows, state)

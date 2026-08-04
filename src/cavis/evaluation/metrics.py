"""Deterministic evaluation metrics for CAVIS experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cavis.schemas import ValidityDecision

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _finite_vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _binary_labels(labels: ArrayLike) -> IntArray:
    array = np.asarray(labels)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    unique = set(np.unique(array).tolist())
    if unique <= {0, 1}:
        return array.astype(np.int64)
    if unique <= {-1, 1}:
        return (array == 1).astype(np.int64)
    raise ValueError("labels must use either {0, 1} or {-1, +1} encoding")


def _average_ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # One-indexed average rank of this tie block.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def roc_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Compute AUROC via the tie-aware Mann--Whitney statistic."""

    labels = _binary_labels(y_true)
    values = _finite_vector(scores, "scores")
    if labels.size != values.size:
        raise ValueError("labels and scores must have the same length")
    n_positive = int(labels.sum())
    n_negative = labels.size - n_positive
    if n_positive == 0 or n_negative == 0:
        return math.nan
    positive_rank_sum = float(_average_ranks(values)[labels == 1].sum())
    statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return statistic / (n_positive * n_negative)


def average_precision(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Compute non-interpolated average precision with stable tie handling."""

    labels = _binary_labels(y_true)
    values = _finite_vector(scores, "scores")
    if labels.size != values.size:
        raise ValueError("labels and scores must have the same length")
    n_positive = int(labels.sum())
    if n_positive == 0:
        return math.nan

    # Group equal scores so the result does not depend on order within ties.
    order = np.argsort(-values, kind="mergesort")
    sorted_scores = values[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    end_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    indices = np.flatnonzero(end_of_tie)
    tp = cumulative_tp[indices].astype(np.float64)
    fp = (indices + 1).astype(np.float64) - tp
    recall = tp / n_positive
    precision = tp / (tp + fp)
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def expected_calibration_error(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    n_bins: int = 15,
) -> float:
    """Return equal-width expected calibration error for positive probability."""

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    labels = _binary_labels(y_true)
    probs = _finite_vector(probabilities, "probabilities")
    if labels.size != probs.size:
        raise ValueError("labels and probabilities must have the same length")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    # A probability of exactly one belongs to the final bin.
    bins = np.minimum((probs * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bins == bin_id
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probs[mask].mean())
            )
    return ece


def risk_coverage_auc(correct: ArrayLike, confidence: ArrayLike) -> float:
    """Area under the empirical selective risk--coverage curve.

    Examples are included from highest to lowest confidence.  The result is the
    mean cumulative error and lies in ``[0, 1]``.
    """

    correct_array = np.asarray(correct)
    if correct_array.ndim != 1 or correct_array.size == 0:
        raise ValueError("correct must be a non-empty one-dimensional array")
    if not set(np.unique(correct_array).tolist()) <= {False, True}:
        raise ValueError("correct must be binary")
    confidence_array = _finite_vector(confidence, "confidence")
    if confidence_array.size != correct_array.size:
        raise ValueError("correct and confidence must have the same length")
    order = np.argsort(-confidence_array, kind="mergesort")
    errors = 1.0 - correct_array[order].astype(np.float64)
    risks = np.cumsum(errors) / np.arange(1, errors.size + 1)
    return float(risks.mean())


def _f1(labels: IntArray, predictions: IntArray, positive_class: int) -> float:
    positive = labels == positive_class
    predicted_positive = predictions == positive_class
    tp = int(np.sum(positive & predicted_positive))
    fp = int(np.sum(~positive & predicted_positive))
    fn = int(np.sum(positive & ~predicted_positive))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def binary_classification_metrics(
    y_true: ArrayLike,
    scores: ArrayLike,
    *,
    threshold: float = 0.5,
    probabilities: ArrayLike | None = None,
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute the common frozen-score binary evaluation suite.

    Scores are oriented so larger values indicate validity.  Calibration
    metrics are returned only when ``probabilities`` is supplied, or when all
    scores already lie in ``[0, 1]``.  Otherwise ``brier`` and ``ece`` are NaN.
    """

    labels = _binary_labels(y_true)
    values = _finite_vector(scores, "scores")
    threshold = float(threshold)
    if labels.size != values.size:
        raise ValueError("labels and scores must have the same length")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    predictions = (values > threshold).astype(np.int64)
    positives = labels == 1
    negatives = ~positives
    sensitivity = (
        float(predictions[positives].mean()) if np.any(positives) else math.nan
    )
    specificity = (
        float((predictions[negatives] == 0).mean()) if np.any(negatives) else math.nan
    )
    balanced_accuracy = (
        0.5 * (sensitivity + specificity)
        if math.isfinite(sensitivity) and math.isfinite(specificity)
        else math.nan
    )
    processbench_f1 = (
        2.0 * sensitivity * specificity / (sensitivity + specificity)
        if math.isfinite(sensitivity)
        and math.isfinite(specificity)
        and sensitivity + specificity > 0.0
        else 0.0
    )
    macro_f1 = 0.5 * (_f1(labels, predictions, 0) + _f1(labels, predictions, 1))
    confidence = np.abs(values - threshold)

    probs: FloatArray | None
    if probabilities is not None:
        probs = _finite_vector(probabilities, "probabilities")
        if probs.size != labels.size:
            raise ValueError("labels and probabilities must have the same length")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")
    elif np.all((values >= 0.0) & (values <= 1.0)):
        probs = values
    else:
        probs = None

    result = {
        "auroc": roc_auc(labels, values),
        "auprc": average_precision(labels, values),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        # ProcessBench names this harmonic mean of correct-class and
        # error-class accuracies "F1". It differs from standard class F1.
        "processbench_f1": processbench_f1,
        "aurc": risk_coverage_auc(predictions == labels, confidence),
        "brier": math.nan,
        "ece": math.nan,
    }
    if probs is not None:
        result["brier"] = float(np.mean((probs - labels) ** 2))
        result["ece"] = expected_calibration_error(labels, probs, n_bins=n_bins)
    return result


def invariance_flip_rate(
    base_scores: ArrayLike,
    transformed_scores: ArrayLike,
    threshold: float,
) -> float:
    """Fraction of ``(item, transform)`` decisions that change."""

    base = _finite_vector(base_scores, "base_scores")
    transformed = np.asarray(transformed_scores, dtype=np.float64)
    if transformed.ndim == 1:
        transformed = transformed[:, None]
    if transformed.ndim != 2 or transformed.shape[0] != base.size:
        raise ValueError(
            "transformed_scores must have shape (n,) or (n, k) matching base_scores"
        )
    if transformed.shape[1] == 0 or not np.all(np.isfinite(transformed)):
        raise ValueError("transformed_scores must be finite and include a transform")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    base_predictions = base[:, None] > threshold
    transformed_predictions = transformed > threshold
    return float(np.mean(base_predictions != transformed_predictions))


def pair_sensitivity(positive_scores: ArrayLike, negative_scores: ArrayLike) -> float:
    """Fraction of pairs ordered in the validity-consistent direction."""

    positive = _finite_vector(positive_scores, "positive_scores")
    negative = _finite_vector(negative_scores, "negative_scores")
    if positive.shape != negative.shape:
        raise ValueError("positive_scores and negative_scores must have equal shape")
    return float(np.mean(positive > negative))


def certified_pair_rate(
    positive_scores: ArrayLike,
    negative_scores: ArrayLike,
    positive_radii: ArrayLike,
    negative_radii: ArrayLike,
) -> float:
    """Fraction of pairs with disjoint, correctly ordered robust intervals."""

    from cavis.certificates import certified_pair_mask

    return float(
        np.mean(
            certified_pair_mask(
                positive_scores,
                negative_scores,
                positive_radii,
                negative_radii,
            )
        )
    )


def abstention_rate(decisions: Sequence[ValidityDecision | str]) -> float:
    """Return the fraction of three-way predictions that abstain."""

    if not decisions:
        raise ValueError("decisions must not be empty")
    normalized = [ValidityDecision(value) for value in decisions]
    return float(
        np.mean([decision is ValidityDecision.ABSTAIN for decision in normalized])
    )


def empirical_invariance_coverage(
    base_scores: ArrayLike,
    transformed_scores: ArrayLike,
    quantile: float,
) -> float:
    """Fraction of observed transformations inside a symmetric certificate."""

    base = _finite_vector(base_scores, "base_scores")
    transformed = np.asarray(transformed_scores, dtype=np.float64)
    if transformed.ndim == 1:
        transformed = transformed[:, None]
    if transformed.ndim != 2 or transformed.shape[0] != base.size:
        raise ValueError(
            "transformed_scores must have shape (n,) or (n, k) matching base_scores"
        )
    if not np.all(np.isfinite(transformed)):
        raise ValueError("transformed_scores must be finite")
    quantile = float(quantile)
    if math.isnan(quantile) or quantile < 0.0:
        raise ValueError("quantile must be non-negative and not NaN")
    return float(np.mean(np.abs(transformed - base[:, None]) <= quantile))


def aggregate_metric_dicts(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate repeated runs as mean, sample std, and number of finite values."""

    if not rows:
        raise ValueError("rows must not be empty")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows[1:]):
        raise ValueError("all metric dictionaries must have identical keys")
    output: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[key] = {
            "mean": float(finite.mean()) if finite.size else math.nan,
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "n": float(finite.size),
        }
    return output

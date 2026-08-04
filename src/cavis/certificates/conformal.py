r"""Finite-sample invariance certificates used by CAVIS.

The calibration score for an item is its transformation radius

.. math::

   r_i = \max_{g \in \mathcal G_0} |s(gx_i) - s(x_i)|.

Given exchangeable calibration and test radii, the split-conformal order
statistic implemented here provides marginal coverage of at least
``1 - alpha``.  If the requested coverage is unattainable for the available
calibration size, the mathematically honest quantile is ``+inf`` and every
prediction abstains.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cavis.schemas import CertificateResult, ValidityDecision

FloatArray = NDArray[np.float64]


def _finite_vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def finite_sample_quantile(calibration_scores: ArrayLike, alpha: float) -> float:
    """Return the split-conformal finite-sample order statistic.

    The selected one-indexed rank is

    ``ceil((n + 1) * (1 - alpha))``.

    If that rank is ``n + 1``, no finite empirical radius can provide the
    requested distribution-free coverage.  Returning ``+inf`` preserves the
    guarantee and forces downstream predictions to abstain.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    scores = _finite_vector(calibration_scores, "calibration_scores")
    if np.any(scores < 0.0):
        raise ValueError("calibration scores must be non-negative radii")

    rank = math.ceil((scores.size + 1) * (1.0 - alpha))
    if rank > scores.size:
        return math.inf
    # np.partition avoids a full sort and is deterministic for scalar inputs.
    return float(np.partition(scores, rank - 1)[rank - 1])


def invariance_radii(base_scores: ArrayLike, transformed_scores: ArrayLike) -> FloatArray:
    """Compute each item's maximum absolute semantics-preserving deviation.

    ``transformed_scores`` can have shape ``(n,)`` for one transformation or
    ``(n, k)`` for ``k`` transformations per item.
    """

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
    return np.max(np.abs(transformed - base[:, None]), axis=1)


def certificate_interval(score: float, quantile: float) -> tuple[float, float]:
    """Return ``[score - quantile, score + quantile]``."""

    score = float(score)
    quantile = float(quantile)
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    if math.isnan(quantile) or quantile < 0.0:
        raise ValueError("quantile must be non-negative and not NaN")
    return score - quantile, score + quantile


def certificate_decision(
    score: float,
    threshold: float,
    quantile: float,
) -> ValidityDecision:
    """Return a strict three-way decision for an invariance interval.

    Touching the threshold is deliberately treated as abstention.  This agrees
    with the protocol's strict ``s(x) > threshold`` prediction rule.
    """

    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    lower, upper = certificate_interval(score, quantile)
    if lower > threshold:
        return ValidityDecision.VALID
    if upper < threshold:
        return ValidityDecision.INVALID
    return ValidityDecision.ABSTAIN


def certified_pair_mask(
    positive_scores: ArrayLike,
    negative_scores: ArrayLike,
    positive_radii: ArrayLike,
    negative_radii: ArrayLike,
) -> NDArray[np.bool_]:
    """Return pairs satisfying ``s+ - s- > r+ + r-``."""

    positive = _finite_vector(positive_scores, "positive_scores")
    negative = _finite_vector(negative_scores, "negative_scores")
    positive_radius = _finite_vector(positive_radii, "positive_radii")
    negative_radius = _finite_vector(negative_radii, "negative_radii")
    if not (
        positive.shape == negative.shape == positive_radius.shape == negative_radius.shape
    ):
        raise ValueError("all pair arrays must have the same shape")
    if np.any(positive_radius < 0.0) or np.any(negative_radius < 0.0):
        raise ValueError("radii must be non-negative")
    return (positive - negative) > (positive_radius + negative_radius)


@dataclass(slots=True)
class InvarianceCalibrator:
    """Stateful split-conformal calibrator for scalar verifier scores."""

    alpha: float = 0.1
    quantile_: float | None = None
    calibration_size_: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")

    @property
    def is_fitted(self) -> bool:
        """Whether calibration radii have been supplied."""

        return self.quantile_ is not None

    def fit(self, radii: ArrayLike) -> InvarianceCalibrator:
        """Fit the conformal radius and return ``self``."""

        array = _finite_vector(radii, "radii")
        self.quantile_ = finite_sample_quantile(array, self.alpha)
        self.calibration_size_ = int(array.size)
        return self

    def fit_from_transformations(
        self,
        base_scores: ArrayLike,
        transformed_scores: ArrayLike,
    ) -> InvarianceCalibrator:
        """Compute calibration radii from transformations and fit them."""

        return self.fit(invariance_radii(base_scores, transformed_scores))

    def predict_one(self, score: float, threshold: float) -> CertificateResult:
        """Certify one score against a fixed decision threshold."""

        if self.quantile_ is None:
            raise RuntimeError("the calibrator must be fitted before prediction")
        lower, upper = certificate_interval(score, self.quantile_)
        decision = certificate_decision(score, threshold, self.quantile_)
        return CertificateResult(
            score=float(score),
            alpha=self.alpha,
            quantile=self.quantile_,
            lower=lower,
            upper=upper,
            threshold=float(threshold),
            decision=decision,
            certified=decision is not ValidityDecision.ABSTAIN,
            calibration_size=self.calibration_size_,
        )

    def predict(
        self,
        scores: Iterable[float],
        threshold: float,
    ) -> list[CertificateResult]:
        """Certify a sequence while preserving input order."""

        return [self.predict_one(score, threshold) for score in scores]

"""Held-out probability calibration for frozen scalar scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True, slots=True)
class PlattScaler:
    coefficient: float
    intercept: float
    calibration_size: int

    def predict(self, scores: ArrayLike) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must be finite")
        return expit(self.coefficient * values + self.intercept)


def fit_platt_scaler(
    labels: ArrayLike,
    scores: ArrayLike,
    *,
    seed: int,
) -> PlattScaler | None:
    """Fit a one-dimensional sigmoid on calibration data only.

    A single-class calibration split cannot identify a sigmoid and returns
    ``None`` explicitly; callers then report calibration metrics as missing.
    """

    y = np.asarray(labels, dtype=np.int64)
    x = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or x.ndim != 1 or y.shape != x.shape or y.size == 0:
        raise ValueError("labels and scores must be aligned non-empty vectors")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError("labels must use 0/1 encoding")
    if not np.all(np.isfinite(x)):
        raise ValueError("scores must be finite")
    if np.unique(y).size < 2:
        return None
    model = LogisticRegression(
        C=1.0,
        max_iter=10_000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(x[:, None], y)
    return PlattScaler(
        coefficient=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
        calibration_size=int(y.size),
    )


__all__ = ["PlattScaler", "fit_platt_scaler"]

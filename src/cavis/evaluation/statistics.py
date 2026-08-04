"""Small-sample statistical utilities with explicit random seeds."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cavis.evaluation.metrics import roc_auc

FloatArray = NDArray[np.float64]


def _finite_vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Point estimate and percentile confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    n_groups: int


@dataclass(frozen=True, slots=True)
class PairedAUCConfidenceInterval:
    """Grouped percentile interval for a paired AUROC difference."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    requested_resamples: int
    valid_resamples: int
    one_class_resamples: int
    n_groups: int
    n_observations: int


@dataclass(frozen=True, slots=True)
class PermutationTestResult:
    """Result of a paired sign-flip randomization test."""

    estimate: float
    p_value: float
    alternative: str
    n_permutations: int
    exact: bool


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Exact two-sided McNemar test result."""

    discordant_a_only: int
    discordant_b_only: int
    p_value: float


def grouped_bootstrap_ci(
    values: ArrayLike,
    groups: ArrayLike,
    *,
    statistic: Callable[[FloatArray], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap that resamples whole groups with replacement.

    When a group is drawn multiple times, all of its rows are duplicated.  This
    preserves within-item dependence among transformed variants.
    """

    observations = _finite_vector(values, "values")
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.size != observations.size:
        raise ValueError("groups must be one-dimensional and match values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    if unique_groups.size == 0:
        raise ValueError("groups must not be empty")
    indices_by_group = [np.flatnonzero(inverse == index) for index in range(unique_groups.size)]
    estimate = float(statistic(observations))
    if not math.isfinite(estimate):
        raise ValueError("statistic must return a finite scalar")

    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_resamples, dtype=np.float64)
    for sample_index in range(n_resamples):
        selected = rng.integers(0, unique_groups.size, size=unique_groups.size)
        row_indices = np.concatenate([indices_by_group[index] for index in selected])
        bootstrap[sample_index] = float(statistic(observations[row_indices]))
    if not np.all(np.isfinite(bootstrap)):
        raise ValueError("statistic returned a non-finite bootstrap value")

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail])
    return ConfidenceInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_resamples=n_resamples,
        n_groups=int(unique_groups.size),
    )


def grouped_paired_auc_bootstrap_ci(
    labels: ArrayLike,
    first_scores: ArrayLike,
    second_scores: ArrayLike,
    groups: ArrayLike,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> PairedAUCConfidenceInterval:
    """Bootstrap ``AUROC(first) - AUROC(second)`` by dependence group.

    The two scores must be paired on exactly the same observations.  Sampling
    a group duplicates every label cell belonging to that dependence unit,
    which preserves both within-theorem dependence and valid/invalid pairs.
    Rare resamples containing only one class are skipped and reported rather
    than converted into zero-width or fabricated intervals.
    """

    first = _finite_vector(first_scores, "first_scores")
    second = _finite_vector(second_scores, "second_scores")
    label_array = np.asarray(labels)
    group_array = np.asarray(groups)
    if first.shape != second.shape:
        raise ValueError("paired score arrays must have identical shape")
    if label_array.ndim != 1 or label_array.size != first.size:
        raise ValueError("labels must be one-dimensional and match scores")
    if group_array.ndim != 1 or group_array.size != first.size:
        raise ValueError("groups must be one-dimensional and match scores")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    unique_labels = set(np.unique(label_array).tolist())
    if not (unique_labels <= {0, 1} or unique_labels <= {-1, 1}):
        raise ValueError("labels must use either {0, 1} or {-1, +1} encoding")
    if len(unique_labels) != 2:
        raise ValueError("paired AUROC requires both validity classes")

    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    if unique_groups.size == 0:
        raise ValueError("groups must not be empty")
    indices_by_group = [
        np.flatnonzero(inverse == index) for index in range(unique_groups.size)
    ]
    estimate = float(
        roc_auc(label_array, first) - roc_auc(label_array, second)
    )
    if not math.isfinite(estimate):
        raise ValueError("paired AUROC difference must be finite")

    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    one_class = 0
    for _ in range(n_resamples):
        selected = rng.integers(0, unique_groups.size, size=unique_groups.size)
        row_indices = np.concatenate([indices_by_group[index] for index in selected])
        first_auc = roc_auc(label_array[row_indices], first[row_indices])
        second_auc = roc_auc(label_array[row_indices], second[row_indices])
        if not math.isfinite(first_auc) or not math.isfinite(second_auc):
            one_class += 1
            continue
        bootstrap.append(float(first_auc - second_auc))
    if not bootstrap:
        raise ValueError("no bootstrap resample contained both classes")

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(
        np.asarray(bootstrap, dtype=np.float64),
        [tail, 1.0 - tail],
    )
    return PairedAUCConfidenceInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        requested_resamples=n_resamples,
        valid_resamples=len(bootstrap),
        one_class_resamples=one_class,
        n_groups=int(unique_groups.size),
        n_observations=int(first.size),
    )


def paired_permutation_test(
    first: ArrayLike,
    second: ArrayLike,
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    n_resamples: int = 10_000,
    seed: int = 0,
    exact_max_pairs: int = 18,
) -> PermutationTestResult:
    """Paired sign-flip test for a difference in means.

    All ``2**n`` assignments are enumerated for ``n <= exact_max_pairs``.
    Larger samples use a seeded Monte Carlo test with the standard plus-one
    correction, so the p-value can never be zero.
    """

    first_array = _finite_vector(first, "first")
    second_array = _finite_vector(second, "second")
    if first_array.shape != second_array.shape:
        raise ValueError("paired arrays must have identical shape")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    differences = first_array - second_array
    observed = float(differences.mean())

    def is_extreme(value: float) -> bool:
        tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 8.0
        if alternative == "greater":
            return value >= observed - tolerance
        if alternative == "less":
            return value <= observed + tolerance
        return abs(value) >= abs(observed) - tolerance

    if differences.size <= exact_max_pairs:
        extreme = 0
        count = 0
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
            permuted = float(np.mean(differences * np.asarray(signs)))
            extreme += int(is_extreme(permuted))
            count += 1
        return PermutationTestResult(
            estimate=observed,
            p_value=extreme / count,
            alternative=alternative,
            n_permutations=count,
            exact=True,
        )

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_resamples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=differences.size)
        extreme += int(is_extreme(float(np.mean(differences * signs))))
    return PermutationTestResult(
        estimate=observed,
        p_value=(extreme + 1.0) / (n_resamples + 1.0),
        alternative=alternative,
        n_permutations=n_resamples,
        exact=False,
    )


def benjamini_hochberg(p_values: ArrayLike) -> FloatArray:
    """Return Benjamini--Hochberg adjusted p-values in original order."""

    values = _finite_vector(p_values, "p_values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = ranked * values.size / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def mcnemar_exact(
    y_true: ArrayLike,
    predictions_a: ArrayLike,
    predictions_b: ArrayLike,
) -> McNemarResult:
    """Exact two-sided McNemar test using the conditional binomial law."""

    truth = np.asarray(y_true)
    first = np.asarray(predictions_a)
    second = np.asarray(predictions_b)
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array")
    if first.shape != truth.shape or second.shape != truth.shape:
        raise ValueError("predictions must match y_true")

    first_correct = first == truth
    second_correct = second == truth
    a_only = int(np.sum(first_correct & ~second_correct))
    b_only = int(np.sum(~first_correct & second_correct))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1)
        ) / (2.0**discordant)
        p_value = min(1.0, 2.0 * tail)
    return McNemarResult(
        discordant_a_only=a_only,
        discordant_b_only=b_only,
        p_value=p_value,
    )

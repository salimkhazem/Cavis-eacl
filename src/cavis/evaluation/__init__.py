"""Evaluation metrics and statistical tests for CAVIS."""

from .metrics import (
    abstention_rate,
    aggregate_metric_dicts,
    average_precision,
    binary_classification_metrics,
    certified_pair_rate,
    empirical_invariance_coverage,
    expected_calibration_error,
    invariance_flip_rate,
    pair_sensitivity,
    risk_coverage_auc,
    roc_auc,
)
from .statistics import (
    ConfidenceInterval,
    McNemarResult,
    PairedAUCConfidenceInterval,
    PermutationTestResult,
    benjamini_hochberg,
    grouped_bootstrap_ci,
    grouped_paired_auc_bootstrap_ci,
    mcnemar_exact,
    paired_permutation_test,
)

__all__ = [
    "ConfidenceInterval",
    "McNemarResult",
    "PairedAUCConfidenceInterval",
    "PermutationTestResult",
    "abstention_rate",
    "aggregate_metric_dicts",
    "average_precision",
    "benjamini_hochberg",
    "binary_classification_metrics",
    "certified_pair_rate",
    "empirical_invariance_coverage",
    "expected_calibration_error",
    "grouped_bootstrap_ci",
    "grouped_paired_auc_bootstrap_ci",
    "invariance_flip_rate",
    "mcnemar_exact",
    "pair_sensitivity",
    "paired_permutation_test",
    "risk_coverage_auc",
    "roc_auc",
]

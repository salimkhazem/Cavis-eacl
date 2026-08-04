from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cavis.evaluation.protocol import exact_grouped_split

PROBE_FIT_SCOPE = "outer_train_defensibly_labeled_roots_only"


@dataclass(frozen=True)
class LinearProbeParameters:
    c: float
    fit_scope: str
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    train_groups: tuple[str, ...]
    training_rows: int
    excluded_outer_train_rows: int
    selection_folds: int
    selection_auroc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pipeline(c: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def defensible_probe_fit_mask(
    records: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Select roots whose validity labels may supervise the diagnostic probe.

    Upstream base-root labels are treated as observational labels. Generated
    G1 roots enter only after CAVIS eligibility establishes their labels. G0
    descendants never enter because they duplicate a root label and would turn
    invariance transformations into supervised training augmentation.
    """

    mask: list[bool] = []
    for index, record in enumerate(records):
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"record {index} needs a metadata mapping")
        canonical_kind = metadata.get("variant_kind")
        alias_kind = metadata.get("transform_kind")
        if (
            canonical_kind is not None
            and alias_kind is not None
            and canonical_kind != alias_kind
        ):
            raise ValueError(
                f"record {index} has conflicting variant_kind/transform_kind"
            )
        variant_kind = (
            canonical_kind if canonical_kind is not None else alias_kind
        )
        if variant_kind not in {"base", "g1", "g0"}:
            raise ValueError(
                f"record {index} needs variant_kind in {{base, g1, g0}}"
            )
        source_kind = metadata.get("source_transform_kind", variant_kind)
        if source_kind not in {"base", "g1", "g0"}:
            raise ValueError(
                f"record {index} needs source_transform_kind in "
                "{base, g1, g0}"
            )
        eligible = metadata.get("cavis_eligible")
        if type(eligible) is not bool:
            raise ValueError(
                f"record {index} needs literal-boolean cavis_eligible"
            )
        is_root = variant_kind != "g0"
        is_g1_root = is_root and (
            variant_kind == "g1" or source_kind == "g1"
        )
        mask.append(is_root and (not is_g1_root or eligible))
    return np.asarray(mask, dtype=np.bool_)


def fit_nested_linear_probe(
    features: np.ndarray,
    labels: Sequence[int],
    group_ids: Sequence[str],
    *,
    split_seed: int,
    fit_mask: Sequence[bool] | None = None,
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    max_inner_folds: int = 5,
) -> tuple[np.ndarray, LinearProbeParameters, dict[str, str]]:
    """Fit a diagnostic linear probe on selected outer-train rows only.

    Returns decision logits for every input row, JSON-serializable parameters,
    and the exact outer group assignments. Calibration and test rows are never
    used for model or hyperparameter selection. ``fit_mask`` additionally
    excludes rows without defensible labels or with duplicated transformed
    labels while retaining predictions for them.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(group_ids)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("features must be a non-empty [rows,features] matrix")
    if y.shape != (x.shape[0],) or groups.shape != (x.shape[0],):
        raise ValueError("labels and group_ids must match the feature rows")
    unique_labels = set(np.unique(y).tolist())
    if not unique_labels or not unique_labels <= {0, 1}:
        raise ValueError("linear probe requires labels encoded as 0/1")
    if not np.all(np.isfinite(x)):
        raise ValueError("probe features must be finite")
    if not c_grid or any(c <= 0 for c in c_grid):
        raise ValueError("c_grid must contain positive values")
    if fit_mask is None:
        eligible_fit_mask = np.ones(x.shape[0], dtype=np.bool_)
    else:
        raw_fit_mask = np.asarray(fit_mask)
        if raw_fit_mask.shape != (x.shape[0],):
            raise ValueError("fit_mask must match the feature rows")
        if raw_fit_mask.dtype != np.bool_:
            raise ValueError("fit_mask must contain literal booleans")
        eligible_fit_mask = raw_fit_mask

    assignments = exact_grouped_split(groups.tolist(), seed=split_seed)
    outer_train_mask = np.asarray(
        [assignments[str(group)] == "train" for group in groups],
        dtype=np.bool_,
    )
    train_mask = outer_train_mask & eligible_fit_mask
    train_x, train_y, train_groups = x[train_mask], y[train_mask], groups[train_mask]
    if set(np.unique(train_y).tolist()) != {0, 1}:
        raise ValueError("outer train split must contain both classes")
    class_group_counts = [
        len(set(train_groups[train_y == label].tolist())) for label in (0, 1)
    ]
    n_folds = min(max_inner_folds, *class_group_counts)
    if n_folds < 2:
        raise ValueError("at least two training groups per class are needed for nested CV")

    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=split_seed,
    )
    candidate_scores: list[tuple[float, float]] = []
    for c in c_grid:
        fold_scores: list[float] = []
        for inner_train, inner_validation in splitter.split(
            train_x, train_y, groups=train_groups
        ):
            estimator = _pipeline(float(c), split_seed).fit(
                train_x[inner_train], train_y[inner_train]
            )
            probability = estimator.predict_proba(train_x[inner_validation])[:, 1]
            fold_scores.append(
                float(roc_auc_score(train_y[inner_validation], probability))
            )
        candidate_scores.append((float(np.mean(fold_scores)), float(c)))
    # Highest AUROC, then smallest C for a deterministic simplicity tie-break.
    best_auc, best_c = max(candidate_scores, key=lambda value: (value[0], -value[1]))
    estimator = _pipeline(best_c, split_seed).fit(train_x, train_y)
    logits = estimator.decision_function(x).astype(np.float64)
    scaler: StandardScaler = estimator.named_steps["scale"]
    classifier: LogisticRegression = estimator.named_steps["classifier"]
    parameters = LinearProbeParameters(
        c=best_c,
        fit_scope=PROBE_FIT_SCOPE,
        feature_mean=tuple(float(value) for value in scaler.mean_),
        feature_scale=tuple(float(value) for value in scaler.scale_),
        coefficients=tuple(float(value) for value in classifier.coef_[0]),
        intercept=float(classifier.intercept_[0]),
        train_groups=tuple(sorted(set(str(value) for value in train_groups))),
        training_rows=int(train_mask.sum()),
        excluded_outer_train_rows=int(
            np.sum(outer_train_mask & ~eligible_fit_mask)
        ),
        selection_folds=n_folds,
        selection_auroc=best_auc,
    )
    return logits, parameters, assignments

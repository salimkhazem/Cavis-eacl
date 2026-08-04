from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from cavis.reproducibility.environment import capture_environment
from cavis.reproducibility.io import atomic_write_json, sha256_file


@dataclass(frozen=True)
class GeometryAuditConfig:
    """Frozen choices for the read-only Geometry of Reason audit."""

    label_fields: tuple[str, ...] = ("label_corrected", "label_original")
    spectral_layer: str = "layer_30"
    seeds: tuple[int, ...] = (17, 42, 97)
    n_splits: int = 5
    n_knots: int = 5
    bootstrap_resamples: int = 2_000
    expected_sha256: str | None = None


def _as_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"valid", "true", "1", "+1"}:
        return 1
    if normalized in {"invalid", "false", "0", "-1"}:
        return 0
    raise ValueError(f"Unrecognized validity label: {value!r}")


def load_geometry_extraction(
    path: str | Path,
    *,
    spectral_layer: str = "layer_30",
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    """Validate and flatten the public JSON extraction.

    JSON is parsed as inert data. Pickle and executable deserializers are never
    used. The source hash is checked before analysis when supplied.
    """
    path = Path(path)
    actual_hash = sha256_file(path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_hash}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Geometry extraction must be a non-empty JSON list")

    rows: list[dict[str, Any]] = []
    required = {
        "file",
        "label_original",
        "label_corrected",
        "proof_text_length",
        "proof_token_count",
        "token_baselines",
        "spectral",
    }
    for index, record in enumerate(payload):
        if not isinstance(record, dict) or not required <= record.keys():
            missing = required - set(record if isinstance(record, dict) else {})
            raise ValueError(f"Malformed record {index}; missing fields: {sorted(missing)}")
        token = record["token_baselines"]
        spectral = record["spectral"]
        if spectral_layer not in spectral:
            raise ValueError(f"Record {index} lacks requested {spectral_layer}")
        layer = spectral[spectral_layer]
        rows.append(
            {
                "item_id": str(record["file"]),
                "label_original": _as_label(record["label_original"]),
                "label_corrected": _as_label(record["label_corrected"]),
                "proof_text_length": float(record["proof_text_length"]),
                "proof_token_count": float(record["proof_token_count"]),
                "mean_logprob": float(token["mean_logprob"]),
                "perplexity": float(token["perplexity"]),
                "mean_entropy": float(token["mean_entropy"]),
                "max_entropy": float(token["max_entropy"]),
                "hfer": float(layer["hfer"]),
                "fiedler": float(layer["fiedler"]),
                "smoothness": float(layer["smoothness"]),
                "spectral_entropy": float(layer["entropy"]),
            }
        )
    frame = pd.DataFrame(rows)
    numeric = frame.drop(columns=["item_id"]).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Geometry extraction contains non-finite values")
    if frame["item_id"].duplicated().any():
        raise ValueError("Geometry extraction contains duplicate item identifiers")
    frame.attrs["source_sha256"] = actual_hash
    frame.attrs["spectral_layer"] = spectral_layer
    return frame


def _spline_classifier(n_knots: int) -> Pipeline:
    return Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=5_000, solver="lbfgs")),
        ]
    )


def _linear_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=5_000, solver="lbfgs")),
        ]
    )


def _residual_pipeline(n_knots: int) -> tuple[Pipeline, Pipeline]:
    nuisance_model = Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=3,
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    return nuisance_model, _linear_classifier()


def _safe_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    if np.unique(labels).size != 2:
        return math.nan
    return float(roc_auc_score(labels, predictions))


def _cross_fitted_predictions(
    frame: pd.DataFrame,
    labels: np.ndarray,
    *,
    seed: int,
    n_splits: int,
    n_knots: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    methods = {
        "length": (["proof_token_count"], _spline_classifier(n_knots)),
        "hfer": (["hfer"], _linear_classifier()),
        "length_plus_hfer": (
            ["proof_token_count", "hfer"],
            _spline_classifier(n_knots),
        ),
        "length_plus_mean_logprob": (
            ["proof_token_count", "mean_logprob"],
            _spline_classifier(n_knots),
        ),
    }
    predictions = {
        name: np.full(len(frame), np.nan, dtype=np.float64) for name in methods
    }
    predictions["hfer_residualized_length"] = np.full(
        len(frame), np.nan, dtype=np.float64
    )
    fold_rows: list[dict[str, Any]] = []

    for fold, (train, test) in enumerate(splitter.split(frame, labels)):
        fold_record: dict[str, Any] = {
            "seed": seed,
            "fold": fold,
            "n_train": int(train.size),
            "n_test": int(test.size),
        }
        for name, (columns, estimator) in methods.items():
            fitted = clone(estimator).fit(frame.iloc[train][columns], labels[train])
            probability = fitted.predict_proba(frame.iloc[test][columns])[:, 1]
            predictions[name][test] = probability
            fold_record[f"auroc.{name}"] = _safe_auc(labels[test], probability)

        nuisance, residual_classifier = _residual_pipeline(n_knots)
        length_train = frame.iloc[train][["proof_token_count"]]
        length_test = frame.iloc[test][["proof_token_count"]]
        hfer_train = frame.iloc[train]["hfer"].to_numpy()
        hfer_test = frame.iloc[test]["hfer"].to_numpy()
        nuisance.fit(length_train, hfer_train)
        residual_train = hfer_train - nuisance.predict(length_train)
        residual_test = hfer_test - nuisance.predict(length_test)
        residual_classifier.fit(residual_train[:, None], labels[train])
        probability = residual_classifier.predict_proba(residual_test[:, None])[:, 1]
        predictions["hfer_residualized_length"][test] = probability
        fold_record["auroc.hfer_residualized_length"] = _safe_auc(
            labels[test], probability
        )
        fold_rows.append(fold_record)

    prediction_frame = pd.DataFrame(
        {
            "item_id": frame["item_id"],
            "label": labels,
            "seed": seed,
            **predictions,
        }
    )
    if prediction_frame.drop(columns=["item_id"]).isna().any().any():
        raise RuntimeError("Cross-fitting left missing predictions")
    return prediction_frame, fold_rows


def _bootstrap_auc_difference(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    observed = _safe_auc(labels, first) - _safe_auc(labels, second)
    differences: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, labels.size, labels.size)
        if np.unique(labels[indices]).size != 2:
            continue
        differences.append(
            _safe_auc(labels[indices], first[indices])
            - _safe_auc(labels[indices], second[indices])
        )
    if not differences:
        raise RuntimeError("No valid stratified composition appeared in bootstrap")
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "valid_resamples": len(differences),
    }


def _correlations(frame: pd.DataFrame) -> dict[str, float]:
    controls = [
        "proof_token_count",
        "proof_text_length",
        "mean_logprob",
        "perplexity",
        "mean_entropy",
        "max_entropy",
    ]
    return {
        f"hfer__{column}": float(frame["hfer"].corr(frame[column], method="spearman"))
        for column in controls
    }


def audit_geometry_extraction(
    input_path: str | Path,
    output_dir: str | Path,
    config: GeometryAuditConfig | None = None,
) -> dict[str, Any]:
    """Run a fully cross-fitted audit and persist every test prediction."""
    config = config or GeometryAuditConfig()
    frame = load_geometry_extraction(
        input_path,
        spectral_layer=config.spectral_layer,
        expected_sha256=config.expected_sha256,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_predictions: list[pd.DataFrame] = []
    all_folds: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    method_names = [
        "length",
        "hfer",
        "length_plus_hfer",
        "hfer_residualized_length",
        "length_plus_mean_logprob",
    ]
    for label_field in config.label_fields:
        labels = frame[label_field].to_numpy(dtype=np.int64)
        seed_summaries: list[dict[str, Any]] = []
        for seed in config.seeds:
            predictions, fold_rows = _cross_fitted_predictions(
                frame,
                labels,
                seed=seed,
                n_splits=config.n_splits,
                n_knots=config.n_knots,
            )
            predictions.insert(1, "label_field", label_field)
            all_predictions.append(predictions)
            for row in fold_rows:
                row["label_field"] = label_field
            all_folds.extend(fold_rows)
            seed_summary: dict[str, Any] = {"seed": seed}
            for method in method_names:
                seed_summary[f"auroc.{method}"] = _safe_auc(
                    labels, predictions[method].to_numpy()
                )
            seed_summary["delta_auc.length_plus_hfer_minus_length"] = (
                seed_summary["auroc.length_plus_hfer"]
                - seed_summary["auroc.length"]
            )
            seed_summaries.append(seed_summary)

        label_summary: dict[str, Any] = {
            "n": int(labels.size),
            "n_valid": int(labels.sum()),
            "n_invalid": int(labels.size - labels.sum()),
            "seeds": seed_summaries,
            "aggregate": {},
        }
        for key in seed_summaries[0]:
            if key == "seed":
                continue
            values = np.asarray([row[key] for row in seed_summaries], dtype=np.float64)
            label_summary["aggregate"][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }

        canonical = all_predictions[-1]
        label_summary["paired_bootstrap"] = _bootstrap_auc_difference(
            labels,
            canonical["length_plus_hfer"].to_numpy(),
            canonical["length"].to_numpy(),
            seed=config.seeds[-1],
            n_resamples=config.bootstrap_resamples,
        )
        summaries[label_field] = label_summary

    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    predictions_frame.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(all_folds).to_csv(output / "fold_metrics.csv", index=False)
    frame.to_csv(output / "flattened_public_scores.csv", index=False)

    plot_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for label_field, label_summary in summaries.items():
        for method in method_names:
            metric = label_summary["aggregate"][f"auroc.{method}"]
            row = {
                "status": "executed_public_cache_reanalysis",
                "dataset": "geometry_of_reason_public_llama8b_cache",
                "model": "Meta-Llama-3.1-8B-Instruct (upstream extraction)",
                "condition": label_field,
                "score": method,
                "metric": "auroc",
                "value": metric["mean"],
                "std": metric["std"],
                "n": label_summary["n"],
                "seeds": ",".join(str(seed) for seed in config.seeds),
                "evidence_scope": "observational_not_causal",
            }
            plot_rows.append(row)
            table_rows.append(row)
    pd.DataFrame(plot_rows).to_csv(output / "plot_metrics.csv", index=False)
    atomic_write_json(
        output / "summary_rows.json",
        {"tables": {"table2_reproduction": table_rows}},
    )

    result: dict[str, Any] = {
        "analysis_status": "exploratory_reanalysis_of_public_cache",
        "scope_boundary": (
            "Observational cross-fitting does not identify causal validity; "
            "paired G0/G1 transformations are not present in this public cache."
        ),
        "source": {
            "path": str(Path(input_path)),
            "upstream_url": (
                "https://github.com/vcnoel/geometry-of-reason/blob/"
                "41b74913eefecbba14c0fcce8aaed15eaf94b139/"
                "data/results/rebuttal/llama8b_full_extraction.json"
            ),
            "upstream_revision": "41b74913eefecbba14c0fcce8aaed15eaf94b139",
            "sha256": frame.attrs["source_sha256"],
            "spectral_layer": config.spectral_layer,
            "records": len(frame),
        },
        "config": asdict(config),
        "correlations_spearman": _correlations(frame),
        "labels": summaries,
    }
    atomic_write_json(output / "metrics.json", result)
    atomic_write_json(output / "config.resolved.json", asdict(config))
    atomic_write_json(output / "environment.json", capture_environment())
    return result


def format_audit_summary(result: dict[str, Any]) -> Iterable[str]:
    for label_field, summary in result["labels"].items():
        aggregate = summary["aggregate"]
        yield f"{label_field} (n={summary['n']}):"
        for name in (
            "auroc.length",
            "auroc.hfer",
            "auroc.length_plus_hfer",
            "auroc.hfer_residualized_length",
            "auroc.length_plus_mean_logprob",
            "delta_auc.length_plus_hfer_minus_length",
        ):
            statistics = aggregate[name]
            yield f"  {name}: {statistics['mean']:.4f} ± {statistics['std']:.4f}"

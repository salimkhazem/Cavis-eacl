"""Deterministic semantic-versus-nuisance toy experiment for CAVIS.

The construction intentionally gives two scores *identical* observational
performance in the training environment.  Their behaviour diverges only after
an intervention:

* the semantic score is unchanged by validity-preserving nuisance flips;
* the nuisance score changes under those flips;
* for validity-changing pairs with nuisance held fixed, only the semantic
  score orders the valid member above the invalid member.

This is a didactic construction, not evidence about a language model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ToyConfig:
    """Parameters for the deterministic two-dimensional construction."""

    n_items: int = 400
    semantic_margin: float = 1.0
    noise_std: float = 0.35
    alpha: float = 0.10
    threshold: float = 0.0
    seed: int = 17

    def validate(self) -> None:
        if self.n_items < 4 or self.n_items % 2:
            raise ValueError("n_items must be an even integer of at least four")
        if self.semantic_margin <= 0:
            raise ValueError("semantic_margin must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")


@dataclass(frozen=True)
class ToyResult:
    """Arrays and summary metrics emitted by :func:`run_toy`."""

    config: ToyConfig
    train: dict[str, np.ndarray]
    shifted_test: dict[str, np.ndarray]
    invariance: dict[str, np.ndarray]
    pairs: dict[str, np.ndarray]
    metrics: dict[str, dict[str, float]]

    def metrics_json(self) -> dict[str, Any]:
        return {
            "experiment": "semantic_nuisance_toy",
            "config": asdict(self.config),
            "metrics": self.metrics,
            "caveat": (
                "Synthetic construction for theorem illustration; not an "
                "empirical language-model result."
            ),
        }


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Return tie-aware binary AUROC using the Mann--Whitney identity."""

    labels = np.asarray(y, dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    if labels.shape != values.shape:
        raise ValueError("y and score must have the same shape")
    if not np.all(np.isin(labels, (-1, 1))):
        raise ValueError("labels must be encoded as -1 and +1")
    positive = values[labels == 1]
    negative = values[labels == -1]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both classes are required")

    # Pairwise evaluation is transparent and entirely adequate for this small
    # illustrative experiment.  Ties receive half credit.
    differences = positive[:, None] - negative[None, :]
    return float(np.mean(differences > 0) + 0.5 * np.mean(differences == 0))


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile, including the +infinity case."""

    radii = np.asarray(values, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    k = int(np.ceil((radii.size + 1) * (1 - alpha)))
    if k > radii.size:
        return float("inf")
    return float(np.partition(radii, k - 1)[k - 1])


def _decision(score: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(score) > threshold


def run_toy(config: ToyConfig | None = None) -> ToyResult:
    """Generate the CAVIS semantic/nuisance proof-of-concept.

    The training nuisance coordinate is an exact copy of the semantic
    coordinate.  Consequently, the two scalar scores have exactly equal
    observational AUROC for every seed.  In the shifted environment, the
    nuisance correlation reverses.  Separate intervention arrays provide the
    quantities needed for InvFlip, sensitivity, and CertifiedPairRate.
    """

    cfg = config or ToyConfig()
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)

    labels = np.tile(np.array([-1, 1], dtype=np.int8), cfg.n_items // 2)
    rng.shuffle(labels)

    train_noise = rng.normal(0.0, cfg.noise_std, cfg.n_items)
    semantic_train = labels * cfg.semantic_margin + train_noise
    nuisance_train = semantic_train.copy()

    test_noise = rng.normal(0.0, cfg.noise_std, cfg.n_items)
    semantic_test = labels * cfg.semantic_margin + test_noise
    nuisance_test = -labels * cfg.semantic_margin + test_noise

    # Validity-preserving intervention: keep semantics and label fixed while
    # reflecting the nuisance coordinate.  Both base scores are evaluated at
    # the same fixed zero threshold.
    inv_noise = rng.normal(0.0, cfg.noise_std, cfg.n_items)
    inv_semantic_before = labels * cfg.semantic_margin + inv_noise
    inv_semantic_after = inv_semantic_before.copy()
    inv_nuisance_before = labels * cfg.semantic_margin + inv_noise
    inv_nuisance_after = -inv_nuisance_before

    # Validity-changing matched pairs: validity changes but nuisance is held
    # fixed.  Shared noise makes the semantic margin exactly 2 * margin.
    n_pairs = cfg.n_items // 2
    pair_semantic_noise = rng.normal(0.0, cfg.noise_std, n_pairs)
    pair_nuisance = rng.normal(0.0, 1.0, n_pairs)
    sem_positive = cfg.semantic_margin + pair_semantic_noise
    sem_negative = -cfg.semantic_margin + pair_semantic_noise
    nui_positive = pair_nuisance.copy()
    nui_negative = pair_nuisance.copy()

    semantic_radii = np.abs(inv_semantic_after - inv_semantic_before)
    nuisance_radii = np.abs(inv_nuisance_after - inv_nuisance_before)
    q_semantic = finite_sample_quantile(semantic_radii, cfg.alpha)
    q_nuisance = finite_sample_quantile(nuisance_radii, cfg.alpha)
    d_semantic = sem_positive - sem_negative
    d_nuisance = nui_positive - nui_negative

    metrics = {
        "semantic_score": {
            "train_auroc": binary_auc(labels, semantic_train),
            "shifted_test_auroc": binary_auc(labels, semantic_test),
            "inv_flip": float(
                np.mean(
                    _decision(inv_semantic_before, cfg.threshold)
                    != _decision(inv_semantic_after, cfg.threshold)
                )
            ),
            "sensitivity": float(np.mean(d_semantic > 0)),
            "certified_pair_rate": float(np.mean(d_semantic > (2.0 * q_semantic))),
            "q_alpha": q_semantic,
        },
        "nuisance_score": {
            "train_auroc": binary_auc(labels, nuisance_train),
            "shifted_test_auroc": binary_auc(labels, nuisance_test),
            "inv_flip": float(
                np.mean(
                    _decision(inv_nuisance_before, cfg.threshold)
                    != _decision(inv_nuisance_after, cfg.threshold)
                )
            ),
            "sensitivity": float(np.mean(d_nuisance > 0)),
            "certified_pair_rate": float(np.mean(d_nuisance > (2.0 * q_nuisance))),
            "q_alpha": q_nuisance,
        },
    }

    return ToyResult(
        config=cfg,
        train={
            "label": labels,
            "semantic": semantic_train,
            "nuisance": nuisance_train,
        },
        shifted_test={
            "label": labels,
            "semantic": semantic_test,
            "nuisance": nuisance_test,
        },
        invariance={
            "label": labels,
            "semantic_before": inv_semantic_before,
            "semantic_after": inv_semantic_after,
            "nuisance_before": inv_nuisance_before,
            "nuisance_after": inv_nuisance_after,
        },
        pairs={
            "semantic_positive": sem_positive,
            "semantic_negative": sem_negative,
            "nuisance_positive": nui_positive,
            "nuisance_negative": nui_negative,
        },
        metrics=metrics,
    )


def points_as_rows(result: ToyResult) -> list[dict[str, float | int | str]]:
    """Flatten train/test points for an auditable CSV artifact."""

    rows: list[dict[str, float | int | str]] = []
    for split_name, split in (
        ("train", result.train),
        ("shifted_test", result.shifted_test),
    ):
        for index, (label, semantic, nuisance) in enumerate(
            zip(
                split["label"],
                split["semantic"],
                split["nuisance"],
                strict=True,
            )
        ):
            rows.append(
                {
                    "experiment": "semantic_nuisance_toy",
                    "split": split_name,
                    "item_id": index,
                    "label": int(label),
                    "semantic": float(semantic),
                    "nuisance": float(nuisance),
                }
            )
    return rows

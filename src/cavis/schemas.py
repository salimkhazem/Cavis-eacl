"""Typed, serializable records shared by the CAVIS pipeline.

The project deliberately uses standard-library dataclasses instead of a runtime
validation framework.  The records are immutable, validate the invariants that
matter for the statistical code, and expose JSON-compatible ``to_dict``
methods.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ValidityDecision(str, Enum):
    """Three-way prediction returned by an invariance certificate."""

    VALID = "valid"
    INVALID = "invalid"
    ABSTAIN = "abstain"


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha_like(value: str | None, field_name: str) -> None:
    if value is not None:
        _require_nonempty(value, field_name)


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """One model/metric extraction for a transformed reasoning item.

    ``scores`` maps stable metric names (for example ``"hfer"`` or
    ``"mean_log_likelihood"``) to scalar values.  Hashes identify the exact
    model input and optional persisted artifact.
    """

    item_id: str
    dataset: str
    model_id: str
    model_revision: str
    transformation_id: str
    label: int
    token_length: int
    scores: Mapping[str, float]
    seed: int
    input_hash: str
    artifact_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "item_id",
            "dataset",
            "model_id",
            "model_revision",
            "transformation_id",
            "input_hash",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        _require_sha_like(self.artifact_hash, "artifact_hash")
        if self.label not in (-1, 0, 1):
            raise ValueError("label must use {-1, +1} or {0, 1} encoding")
        if self.token_length < 0:
            raise ValueError("token_length must be non-negative")
        if not self.scores:
            raise ValueError("scores must contain at least one metric")
        clean_scores = {str(name): float(value) for name, value in self.scores.items()}
        if any(not name for name in clean_scores):
            raise ValueError("score names must be non-empty")
        if any(not math.isfinite(value) for value in clean_scores.values()):
            raise ValueError("all scores must be finite")
        object.__setattr__(self, "scores", clean_scores)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairRecord:
    """A validity-changing, nuisance-controlled positive/negative pair."""

    pair_id: str
    positive_item_id: str
    negative_item_id: str
    intervention_type: str
    token_length_delta: int
    theorem_id: str | None = None
    evidence_id: str | None = None
    mechanically_verified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "pair_id",
            "positive_item_id",
            "negative_item_id",
            "intervention_type",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        if self.positive_item_id == self.negative_item_id:
            raise ValueError("positive and negative items must be distinct")
        if self.theorem_id is not None:
            _require_nonempty(self.theorem_id, "theorem_id")
        if self.evidence_id is not None:
            _require_nonempty(self.evidence_id, "evidence_id")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransformEvidence:
    """Audit trail for a semantics-preserving or corrupting transform."""

    transformation_id: str
    command: Sequence[str]
    lean_version: str
    return_code: int
    source_hash: str
    target_hash: str
    mechanically_verified: bool
    human_validation: str | None = None
    stdout_hash: str | None = None
    stderr_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "transformation_id",
            "lean_version",
            "source_hash",
            "target_hash",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        if not self.command or any(not str(part) for part in self.command):
            raise ValueError("command must contain at least one non-empty argument")
        if self.mechanically_verified and self.return_code != 0:
            raise ValueError("a failed command cannot be mechanically verified")
        if self.human_validation is not None:
            _require_nonempty(self.human_validation, "human_validation")
        _require_sha_like(self.stdout_hash, "stdout_hash")
        _require_sha_like(self.stderr_hash, "stderr_hash")
        object.__setattr__(self, "command", tuple(str(part) for part in self.command))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload


@dataclass(frozen=True, slots=True)
class CertificateResult:
    """A calibrated interval and its three-way threshold decision."""

    score: float
    alpha: float
    quantile: float
    lower: float
    upper: float
    threshold: float
    decision: ValidityDecision
    certified: bool
    calibration_size: int

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.calibration_size <= 0:
            raise ValueError("calibration_size must be positive")
        if not math.isfinite(self.score) or not math.isfinite(self.threshold):
            raise ValueError("score and threshold must be finite")
        if math.isnan(self.quantile) or self.quantile < 0.0:
            raise ValueError("quantile must be non-negative and not NaN")
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("certificate interval is malformed")
        decision = ValidityDecision(self.decision)
        object.__setattr__(self, "decision", decision)
        if self.certified != (decision is not ValidityDecision.ABSTAIN):
            raise ValueError("certified must be false exactly when the decision abstains")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        payload = asdict(self)
        payload["decision"] = self.decision.value
        return payload

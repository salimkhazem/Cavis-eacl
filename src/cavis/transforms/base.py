"""Shared transformation records.

Naming a transform ``G0`` or ``G1`` records its design intent, not a semantic
fact.  All results start as ``unverified`` and carry no semantic-validity assertion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from cavis.data.io import sha256_text

TransformFamily = Literal["g0", "g1"]
ExpectedLeanOutcome = Literal["compile", "reject"]


@dataclass(frozen=True, slots=True)
class TransformSpec:
    name: str
    family: TransformFamily
    expected_lean_outcome: ExpectedLeanOutcome
    description: str


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Output of a deterministic source-to-source transformation."""

    item_id: str
    source: str
    target: str
    spec: TransformSpec
    seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)
    evidence_state: str = "unverified"
    semantic_status: str = "not_established"

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise ValueError(f"{self.spec.name} produced an unchanged target")
        if self.evidence_state != "unverified":
            raise ValueError("Construction cannot mark a transform as verified")
        if self.semantic_status != "not_established":
            raise ValueError("Construction cannot establish a semantic assertion")

    @property
    def source_sha256(self) -> str:
        return sha256_text(self.source)

    @property
    def target_sha256(self) -> str:
        return sha256_text(self.target)

    @property
    def transformation_id(self) -> str:
        """Content-addressed ID suitable for score and evidence records."""

        return (
            f"{self.item_id}:{self.spec.name}:s{self.seed}:"
            f"{self.target_sha256[:16]}"
        )


class LeanTransform(Protocol):
    spec: TransformSpec

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        ...

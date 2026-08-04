"""Canonical, immutable records emitted by all dataset adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ReasoningExample:
    """A single reasoning trace with optional step-level supervision.

    ``valid`` is deliberately tri-state.  Missing or ambiguous annotations are
    represented by ``None`` rather than silently coerced to a binary label.
    Likewise, an unlabeled suffix in a process-supervision dataset remains
    ``None`` in ``step_labels``.
    """

    item_id: str
    group_id: str
    dependence_id: str
    dataset: str
    problem: str
    steps: tuple[str, ...]
    step_labels: tuple[bool | None, ...]
    valid: bool | None
    source_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must be non-empty")
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if not self.dependence_id:
            raise ValueError("dependence_id must be non-empty")
        if len(self.steps) != len(self.step_labels):
            raise ValueError("steps and step_labels must have the same length")

    @property
    def reasoning_text(self) -> str:
        """Return the trace in a stable teacher-forcing representation."""

        return "\n".join(self.steps)

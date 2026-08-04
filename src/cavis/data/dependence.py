"""Canonical statistical dependence identities for prepared datasets.

``group_id`` remains source provenance.  ``dependence_id`` is the
dataset-scoped unit used for splitting, calibration, resampling, and
selection.  Lean identities are derived from original base declarations and
must be propagated unchanged to every transformed descendant.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cavis.transforms.lexical import (
    declaration_header_end,
    identifier_spans,
    lean_tokens,
    theorem_declaration,
)

from .io import canonical_json_hash

GEOMETRY_DEPENDENCE_NAMESPACE = "geometry_minif2f::lean_statement"


@dataclass(frozen=True, slots=True)
class LeanStatementIdentity:
    """Identity and auditable provenance for one Lean proposition."""

    dependence_id: str
    theorem_name: str
    statement_sha256: str


@dataclass(frozen=True, slots=True)
class DependenceDiagnostics:
    """Validation summary emitted when enriching prepared Lean rows."""

    base_rows: int
    source_groups: int
    dependence_groups: int
    names: int
    aliased_statement_groups: int


def scoped_group_dependence_id(dataset: str, group_id: str) -> str:
    """Return a dataset-scoped dependence ID for non-Lean datasets."""

    dataset = str(dataset).strip()
    group_id = str(group_id).strip()
    if not dataset or not group_id:
        raise ValueError("dataset and group_id must be non-empty")
    return f"{dataset}::group::{group_id}"


def geometry_statement_identity(source: str) -> LeanStatementIdentity:
    """Parse one supported Lean base file into a proposition identity.

    The lightweight lexer deliberately fails closed for declaration syntaxes
    it cannot identify fully.  In particular, namespace blocks, qualified
    declaration names, and quoted identifiers need a real Lean parser rather
    than silent truncation.
    """

    if not source.strip():
        raise ValueError("Lean source must be non-empty")
    keyword, name = theorem_declaration(source)
    declarations = [
        span
        for span in identifier_spans(source)
        if span.text in {"theorem", "lemma", "example"}
    ]
    if len(declarations) != 1 or declarations[0].start != keyword.start:
        raise ValueError("Lean source must contain exactly one theorem or lemma")
    if any(
        span.text == "namespace"
        for span in identifier_spans(source[: keyword.start])
    ):
        raise ValueError("Lean namespace declarations require a full name parser")
    header_end = declaration_header_end(source, name.end)
    between = source[keyword.end : name.start]
    tail_tokens = lean_tokens(source[name.end:header_end])
    if "«" in between or (tail_tokens and tail_tokens[0] in {".", "»"}):
        raise ValueError("qualified or quoted Lean declaration names are unsupported")
    theorem_name = name.text
    if not theorem_name:
        raise ValueError("Lean theorem name must be non-empty")
    statement_tokens = lean_tokens(source[name.end:header_end])
    if not statement_tokens:
        raise ValueError("Lean theorem statement must be non-empty")
    statement_sha256 = canonical_json_hash(list(statement_tokens))
    return LeanStatementIdentity(
        dependence_id=(
            f"{GEOMETRY_DEPENDENCE_NAMESPACE}::{statement_sha256}"
        ),
        theorem_name=theorem_name,
        statement_sha256=statement_sha256,
    )


def enrich_prepared_dependence_ids(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], DependenceDiagnostics]:
    """Add canonical Lean dependence fields to old prepared rows.

    Only base rows are parsed.  Their identity is propagated by source
    ``group_id`` to G0/G1 descendants.  The join fails closed if one source
    group maps to multiple statements or one theorem name maps to multiple
    normalized statements.
    """

    if not rows:
        raise ValueError("prepared rows must be non-empty")
    group_identities: dict[str, LeanStatementIdentity] = {}
    name_hashes: dict[str, set[str]] = defaultdict(set)
    hash_names: dict[str, set[str]] = defaultdict(set)
    base_rows = 0
    for raw in rows:
        if str(raw.get("transform_kind", "base")) != "base":
            continue
        base_rows += 1
        group_id = str(raw.get("group_id", "")).strip()
        if not group_id:
            raise ValueError("every base row requires a non-empty group_id")
        identity = geometry_statement_identity(str(raw.get("reasoning", "")))
        previous = group_identities.get(group_id)
        if (
            previous is not None
            and previous.dependence_id != identity.dependence_id
        ):
            raise ValueError(
                f"source group {group_id!r} maps to multiple Lean statements"
            )
        if previous is None or identity.theorem_name < previous.theorem_name:
            group_identities[group_id] = identity
        name_hashes[identity.theorem_name].add(identity.statement_sha256)
        hash_names[identity.statement_sha256].add(identity.theorem_name)
    if base_rows == 0:
        raise ValueError("prepared rows contain no base rows")
    collisions = {
        name: hashes for name, hashes in name_hashes.items() if len(hashes) > 1
    }
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(
            "Lean theorem names map to multiple normalized statements: "
            f"{names}"
        )

    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        group_id = str(row.get("group_id", "")).strip()
        try:
            identity = group_identities[group_id]
        except KeyError as exc:
            raise ValueError(
                f"row group {group_id!r} has no parsed base declaration"
            ) from exc
        existing = row.get("dependence_id")
        if existing is not None and str(existing) != identity.dependence_id:
            raise ValueError(
                f"row {row.get('item_id')!r} has a conflicting dependence_id"
            )
        existing_source_group = row.get("source_group_id")
        if (
            existing_source_group is not None
            and str(existing_source_group) != group_id
        ):
            raise ValueError(
                f"row {row.get('item_id')!r} has a conflicting source_group_id"
            )
        row["source_group_id"] = group_id
        row["dependence_id"] = identity.dependence_id
        row["theorem_name"] = identity.theorem_name
        row["statement_sha256"] = identity.statement_sha256
        output.append(row)
    diagnostics = DependenceDiagnostics(
        base_rows=base_rows,
        source_groups=len(group_identities),
        dependence_groups=len(hash_names),
        names=len(name_hashes),
        aliased_statement_groups=sum(len(names) > 1 for names in hash_names.values()),
    )
    return output, diagnostics


__all__ = [
    "DependenceDiagnostics",
    "GEOMETRY_DEPENDENCE_NAMESPACE",
    "LeanStatementIdentity",
    "enrich_prepared_dependence_ids",
    "geometry_statement_identity",
    "scoped_group_dependence_id",
]

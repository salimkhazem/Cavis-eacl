from __future__ import annotations

import pytest

from cavis.data.dependence import (
    enrich_prepared_dependence_ids,
    geometry_statement_identity,
    scoped_group_dependence_id,
)


def _lean(name: str, statement: str = "True", proof: str = "trivial") -> str:
    return f"theorem {name} : {statement} := by\n  {proof}\n"


def test_geometry_identity_uses_statement_not_filename_or_name() -> None:
    first = geometry_statement_identity(_lean("first"))
    alias = geometry_statement_identity(
        "theorem second -- rendering-only comment\n : True := by\n  trivial\n"
    )
    assert first.dependence_id == alias.dependence_id
    assert first.statement_sha256 == alias.statement_sha256
    assert first.theorem_name == "first"
    assert alias.theorem_name == "second"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_lean("Foo.bar"), "qualified or quoted"),
        ("theorem «foo» : True := by trivial\n", "qualified or quoted"),
        (
            "namespace Foo\n"
            "theorem bar : True := by trivial\n"
            "end Foo\n",
            "namespace",
        ),
        (
            "theorem first : True := by trivial\n"
            "theorem second : True := by trivial\n",
            "exactly one",
        ),
    ],
)
def test_geometry_identity_fails_closed_on_unsupported_declarations(
    source: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        geometry_statement_identity(source)


def test_enrichment_parses_bases_only_and_propagates_to_descendants() -> None:
    rows = [
        {
            "item_id": "base-a",
            "group_id": "source-a",
            "transform_kind": "base",
            "reasoning": _lean("same"),
        },
        {
            "item_id": "g1-a",
            "group_id": "source-a",
            "transform_kind": "g1",
            "reasoning": _lean("renamed", "False"),
        },
        {
            "item_id": "base-b",
            "group_id": "source-b",
            "transform_kind": "base",
            "reasoning": _lean("alias"),
        },
    ]
    enriched, diagnostics = enrich_prepared_dependence_ids(rows)
    assert len({row["dependence_id"] for row in enriched}) == 1
    assert enriched[1]["theorem_name"] == "same"
    assert enriched[1]["source_group_id"] == "source-a"
    assert diagnostics.base_rows == 2
    assert diagnostics.source_groups == 2
    assert diagnostics.dependence_groups == 1
    assert diagnostics.aliased_statement_groups == 1


def test_enrichment_rejects_name_to_multiple_statements() -> None:
    rows = [
        {
            "item_id": "first",
            "group_id": "first",
            "transform_kind": "base",
            "reasoning": _lean("collision", "True"),
        },
        {
            "item_id": "second",
            "group_id": "second",
            "transform_kind": "base",
            "reasoning": _lean("collision", "False"),
        },
    ]
    with pytest.raises(ValueError, match="multiple normalized statements"):
        enrich_prepared_dependence_ids(rows)


def test_non_geometry_dependence_ids_are_dataset_scoped() -> None:
    first = scoped_group_dependence_id("processbench_math", "problem-1")
    second = scoped_group_dependence_id("prm800k", "problem-1")
    assert first == "processbench_math::group::problem-1"
    assert first != second

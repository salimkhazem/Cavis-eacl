"""Conservative syntactic contracts for G0 eligibility."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .lexical import (
    declaration_header_end,
    identifier_spans,
    lean_tokens,
    rewrite_identifiers,
    theorem_declaration,
)


@dataclass(frozen=True, slots=True)
class ContractCheck:
    verified: bool
    reason: str


def _round_trip_identifier(
    source: str,
    target: str,
    *,
    old_name: str,
    new_name: str,
) -> bool:
    if new_name in {span.text for span in identifier_spans(source)}:
        return False
    reverted, count = rewrite_identifiers(target, {new_name: old_name})
    return count > 0 and lean_tokens(reverted) == lean_tokens(source)


def _check_declaration_rename(
    source: str, target: str, parameters: Mapping[str, Any]
) -> ContractCheck:
    old_name = str(parameters.get("old_name", ""))
    new_name = str(parameters.get("new_name", ""))
    if not old_name or not new_name or old_name == new_name:
        return ContractCheck(False, "missing_or_invalid_name_mapping")
    try:
        _, source_name = theorem_declaration(source)
        _, target_name = theorem_declaration(target)
    except ValueError:
        return ContractCheck(False, "declaration_not_found")
    if source_name.text != old_name or target_name.text != new_name:
        return ContractCheck(False, "declaration_mapping_mismatch")
    # Lean theorems are not recursive.  Rewriting a same-named reference in the
    # proof could instead alter a global dependency, so eligibility requires
    # the declaration to be the only source occurrence.
    old_occurrences = sum(
        span.text == old_name for span in identifier_spans(source)
    )
    if old_occurrences != 1:
        return ContractCheck(False, "declaration_name_has_nonlocal_occurrences")
    if not _round_trip_identifier(
        source, target, old_name=old_name, new_name=new_name
    ):
        return ContractCheck(False, "token_round_trip_failed")
    return ContractCheck(True, "exact_declaration_alpha_equivalence")


def _check_local_alpha(
    source: str, target: str, parameters: Mapping[str, Any]
) -> ContractCheck:
    old_name = str(parameters.get("old_name", ""))
    new_name = str(parameters.get("new_name", ""))
    binder_offset = parameters.get("binder_offset")
    if (
        not old_name
        or not new_name
        or old_name == new_name
        or not isinstance(binder_offset, int)
    ):
        return ContractCheck(False, "missing_or_invalid_alpha_mapping")
    try:
        _, declaration_name = theorem_declaration(source)
        header_end = declaration_header_end(source, declaration_name.end)
    except ValueError:
        return ContractCheck(False, "declaration_header_not_found")
    binder_matches = [
        span
        for span in identifier_spans(source)
        if span.start == binder_offset and span.text == old_name
    ]
    if (
        len(binder_matches) != 1
        or not declaration_name.end <= binder_offset < header_end
    ):
        return ContractCheck(False, "recorded_binder_not_in_header")
    if not _round_trip_identifier(
        source, target, old_name=old_name, new_name=new_name
    ):
        return ContractCheck(False, "token_round_trip_failed")

    # Reject common shadowing binders in the proof.  This is deliberately
    # conservative: uncertain alpha rewrites remain ineligible.
    proof = source[header_end:]
    escaped = re.escape(old_name)
    shadow_pattern = re.compile(
        rf"\b(?:fun|let|have|intro|intros|cases|rcases)\s+{escaped}\b|"
        rf"λ\s*{escaped}\b"
    )
    if shadow_pattern.search(proof):
        return ContractCheck(False, "possible_shadowing_binder")
    return ContractCheck(True, "capture_avoiding_local_alpha_equivalence")


def check_g0_contract(
    transform_name: str,
    source: str,
    target: str,
    parameters: Mapping[str, Any],
) -> ContractCheck:
    """Validate the exact rewrite contract for one supported G0 family."""

    if source == target:
        return ContractCheck(False, "target_unchanged")
    if transform_name in {"whitespace", "comments"}:
        try:
            equal = lean_tokens(source) == lean_tokens(target)
        except ValueError:
            return ContractCheck(False, "lexical_error")
        return ContractCheck(
            equal,
            (
                "token_identity_modulo_comments_and_whitespace"
                if equal
                else "code_tokens_changed"
            ),
        )
    if transform_name in {"declaration_rename", "theorem_rename"}:
        return _check_declaration_rename(source, target, parameters)
    if transform_name in {"local_alpha_rename", "alpha_rename"}:
        return _check_local_alpha(source, target, parameters)
    return ContractCheck(False, f"unsupported_g0_contract:{transform_name}")


def declaration_statement_changed(source: str, target: str) -> ContractCheck:
    """Check that a G1 candidate changes the declaration header, not just proof text."""

    try:
        _, source_name = theorem_declaration(source)
        _, target_name = theorem_declaration(target)
        source_end = declaration_header_end(source, source_name.end)
        target_end = declaration_header_end(target, target_name.end)
        source_header = lean_tokens(source[source_name.end:source_end])
        target_header = lean_tokens(target[target_name.end:target_end])
    except ValueError:
        return ContractCheck(False, "declaration_header_not_found")
    changed = source_header != target_header
    return ContractCheck(changed, "statement_changed" if changed else "statement_unchanged")

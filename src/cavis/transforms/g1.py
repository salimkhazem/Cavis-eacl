"""Candidate validity-changing (G1) Lean transformations."""

from __future__ import annotations

import re

from .base import TransformResult, TransformSpec
from .lexical import (
    apply_edit,
    choose_deterministically,
    code_matches,
    declaration_header_end,
    theorem_declaration,
)


def _header_bounds(source: str) -> tuple[int, int]:
    _, name = theorem_declaration(source)
    return name.end, declaration_header_end(source, name.end)


class RelationFlipCorruption:
    spec = TransformSpec(
        name="comparison_flip",
        family="g1",
        expected_lean_outcome="reject",
        description="Flip one relation in the theorem header.",
    )
    _pattern = re.compile(r"(?<![:<>=!])(?:≤|≥|≠|=|<|>)(?![=>])")
    _replacement = {"≤": "≥", "≥": "≤", "≠": "=", "=": "≠", "<": ">", ">": "<"}

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        start, end = _header_bounds(source)
        candidates = code_matches(source, self._pattern, start=start, end=end)
        chosen = choose_deterministically(
            candidates, seed=seed, item_id=item_id, namespace=self.spec.name
        )
        old = chosen.group(0)
        new = self._replacement[old]
        target = apply_edit(source, chosen.start(), chosen.end(), new)
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={
                "old_relation": old,
                "new_relation": new,
                "offset": chosen.start(),
            },
        )


class NumericLiteralCorruption:
    spec = TransformSpec(
        name="numeric_literal",
        family="g1",
        expected_lean_outcome="reject",
        description="Change one numeric literal in the theorem header.",
    )
    _pattern = re.compile(r"(?<![\w'₀-₉])\d+(?![\w'₀-₉])")

    @staticmethod
    def _same_width_neighbor(value: str) -> str:
        number = int(value)
        width = len(value)
        for delta in (1, -1, 2, -2):
            candidate = number + delta
            if candidate >= 0 and len(str(candidate)) == width:
                return str(candidate)
        # ``9`` and similar boundary cases: change the last digit but keep width.
        replacement_digit = "1" if value[-1] == "0" else str((int(value[-1]) + 1) % 10)
        candidate = value[:-1] + replacement_digit
        return candidate if candidate != value else ("0" * (width - 1) + "1")

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        start, end = _header_bounds(source)
        candidates = code_matches(source, self._pattern, start=start, end=end)
        chosen = choose_deterministically(
            candidates, seed=seed, item_id=item_id, namespace=self.spec.name
        )
        old = chosen.group(0)
        new = self._same_width_neighbor(old)
        target = apply_edit(source, chosen.start(), chosen.end(), new)
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={
                "old_literal": old,
                "new_literal": new,
                "offset": chosen.start(),
            },
        )


class ProofStepDeletionCorruption:
    spec = TransformSpec(
        name="proof_step_deletion",
        family="g1",
        expected_lean_outcome="reject",
        description="Delete one non-structural tactic line from the proof.",
    )
    _structural = re.compile(
        r"^\s*(?:begin|end|by|{|}|\[|\]|--.*|/-.*-/)\s*,?\s*$"
    )

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        _, end = _header_bounds(source)
        candidates: list[re.Match[str]] = []
        line_pattern = re.compile(r"(?m)^[^\n]*\S[^\n]*(?:\n|$)")
        for match in line_pattern.finditer(source, end):
            line = match.group(0).rstrip("\n")
            if not self._structural.fullmatch(line):
                candidates.append(match)
        chosen = choose_deterministically(
            candidates, seed=seed, item_id=item_id, namespace=self.spec.name
        )
        old = chosen.group(0)
        # Preserve a newline to minimize downstream layout differences.
        replacement = "\n" if old.endswith("\n") else ""
        target = apply_edit(source, chosen.start(), chosen.end(), replacement)
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={
                "deleted_line": old.rstrip("\n"),
                "offset": chosen.start(),
            },
        )


class ArithmeticOperatorCorruption:
    spec = TransformSpec(
        name="arithmetic_operator_flip",
        family="g1",
        expected_lean_outcome="reject",
        description="Change one arithmetic operator in the theorem header.",
    )
    _pattern = re.compile(r"(?<![:/+\-*])(?:\+|-|\*|/)(?![=/+\-*])")
    _replacement = {"+": "-", "-": "+", "*": "/", "/": "*"}

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        start, end = _header_bounds(source)
        candidates = code_matches(source, self._pattern, start=start, end=end)
        chosen = choose_deterministically(
            candidates, seed=seed, item_id=item_id, namespace=self.spec.name
        )
        old = chosen.group(0)
        new = self._replacement[old]
        target = apply_edit(source, chosen.start(), chosen.end(), new)
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={
                "old_operator": old,
                "new_operator": new,
                "offset": chosen.start(),
            },
        )


class PremiseMutationCorruption:
    spec = TransformSpec(
        name="premise_mutation",
        family="g1",
        expected_lean_outcome="reject",
        description="Mutate a relation or literal inside an explicit premise binder.",
    )
    _binder = re.compile(r"\([^()\n]*:[^()\n]*\)")
    _relation = RelationFlipCorruption._pattern
    _number = NumericLiteralCorruption._pattern

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        start, end = _header_bounds(source)
        premise_ranges = [
            (match.start(), match.end())
            for match in self._binder.finditer(source, start, end)
        ]
        candidates: list[re.Match[str]] = []
        for lower, upper in premise_ranges:
            candidates.extend(
                code_matches(source, self._relation, start=lower, end=upper)
            )
            candidates.extend(
                code_matches(source, self._number, start=lower, end=upper)
            )
        chosen = choose_deterministically(
            candidates, seed=seed, item_id=item_id, namespace=self.spec.name
        )
        old = chosen.group(0)
        if old in RelationFlipCorruption._replacement:
            new = RelationFlipCorruption._replacement[old]
            mutation_type = "relation"
        else:
            new = NumericLiteralCorruption._same_width_neighbor(old)
            mutation_type = "numeric_literal"
        target = apply_edit(source, chosen.start(), chosen.end(), new)
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={
                "mutation_type": mutation_type,
                "old_value": old,
                "new_value": new,
                "offset": chosen.start(),
            },
        )

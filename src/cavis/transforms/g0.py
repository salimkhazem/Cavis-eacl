"""Candidate validity-preserving (G0) Lean transformations."""

from __future__ import annotations

import hashlib
import random
import re

from .base import TransformResult, TransformSpec
from .lexical import (
    apply_edit,
    choose_deterministically,
    declaration_header_end,
    identifier_spans,
    rewrite_identifiers,
    theorem_declaration,
)


def _rng(seed: int, item_id: str, namespace: str) -> random.Random:
    digest = hashlib.sha256(
        f"{namespace}\0{seed}\0{item_id}".encode()
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


class WhitespaceTransform:
    spec = TransformSpec(
        name="whitespace",
        family="g0",
        expected_lean_outcome="compile",
        description="Change blank lines and trailing spaces without editing tokens.",
    )

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        rng = _rng(seed, item_id, self.spec.name)
        lines = source.replace("\r\n", "\n").replace("\r", "\n").splitlines()
        target_lines: list[str] = []
        changed = 0
        for line in lines:
            if not line.strip():
                repeats = 1 + rng.randrange(3)
                target_lines.extend([""] * repeats)
                changed += repeats != 1
            else:
                spaces = 1 + rng.randrange(3)
                target_lines.append(line.rstrip() + (" " * spaces))
                changed += 1
        target = "\n".join(target_lines) + "\n"
        if target == source:
            target = source.rstrip("\n") + "  \n"
            changed += 1
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={"edited_lines": changed},
        )


class CommentTransform:
    spec = TransformSpec(
        name="comments",
        family="g0",
        expected_lean_outcome="compile",
        description="Insert deterministic block comments before the declaration.",
    )
    _phrases = (
        "CAVIS semantic-invariance candidate",
        "LeanTwin rendering control",
        "proof content intentionally unchanged",
        "interventional audit marker",
    )

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        keyword, _ = theorem_declaration(source)
        rng = _rng(seed, item_id, self.spec.name)
        count = 1 + rng.randrange(2)
        phrases = rng.sample(self._phrases, k=count)
        payload = "\n".join(
            f"/- {phrase}; seed={seed}; marker={index} -/"
            for index, phrase in enumerate(phrases)
        )
        target = apply_edit(source, keyword.start, keyword.start, payload + "\n")
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={"comments": count, "insertion_offset": keyword.start},
        )


class TheoremRenameTransform:
    spec = TransformSpec(
        name="declaration_rename",
        family="g0",
        expected_lean_outcome="compile",
        description="Rename the declaration and exact self-references.",
    )

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        _, name = theorem_declaration(source)
        suffix = hashlib.sha256(
            f"{seed}\0{item_id}\0{name.text}".encode()
        ).hexdigest()[:10]
        replacement = f"cavis_{suffix}_{name.text}"
        target, occurrences = rewrite_identifiers(source, {name.text: replacement})
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={
                "old_name": name.text,
                "new_name": replacement,
                "occurrences": occurrences,
            },
        )


class AlphaRenameTransform:
    spec = TransformSpec(
        name="local_alpha_rename",
        family="g0",
        expected_lean_outcome="compile",
        description="Conservatively rename one explicitly typed local binder.",
    )

    _binder = re.compile(r"[\(\{]([^(){}]*?):[^(){}]*?[\)\}]")
    _reserved = frozenset(
        {
            "Type",
            "Prop",
            "Sort",
            "theorem",
            "lemma",
            "by",
            "begin",
            "end",
            "fun",
            "let",
            "in",
        }
    )

    def apply(self, source: str, *, item_id: str, seed: int) -> TransformResult:
        _, declaration_name = theorem_declaration(source)
        header_end = declaration_header_end(source, declaration_name.end)
        header = source[declaration_name.end:header_end]
        header_offset = declaration_name.end
        identifiers = list(identifier_spans(source))
        by_span = {(span.start, span.end): span for span in identifiers}
        candidates = []
        for binder in self._binder.finditer(header):
            left_start = header_offset + binder.start(1)
            left_end = header_offset + binder.end(1)
            for (start, end), token in by_span.items():
                if left_start <= start and end <= left_end:
                    if token.text not in self._reserved and token.text != "_":
                        candidates.append(token)
        chosen = choose_deterministically(
            candidates,
            seed=seed,
            item_id=item_id,
            namespace=self.spec.name,
        )
        assert hasattr(chosen, "text")
        suffix = hashlib.sha256(
            f"{seed}\0{item_id}\0{chosen.text}".encode()
        ).hexdigest()[:8]
        replacement = f"cavis_{chosen.text}_{suffix}"
        target, occurrences = rewrite_identifiers(
            source, {chosen.text: replacement}
        )
        return TransformResult(
            item_id=item_id,
            source=source,
            target=target,
            spec=self.spec,
            seed=seed,
            parameters={
                "old_name": chosen.text,
                "new_name": replacement,
                "binder_offset": chosen.start,
                "occurrences": occurrences,
            },
        )

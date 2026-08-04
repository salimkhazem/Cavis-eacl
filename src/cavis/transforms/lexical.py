"""Small Lean-aware lexical helpers.

This is intentionally not presented as a complete Lean parser.  It only keeps
rewrites out of line comments, nested block comments, and quoted strings.  The
external Lean checker remains the authority on whether a candidate compiles.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CodeSpan:
    start: int
    end: int
    text: str


def _identifier_start(character: str) -> bool:
    return character == "_" or character.isalpha() or unicodedata.category(
        character
    ).startswith("L")


def _identifier_continue(character: str) -> bool:
    category = unicodedata.category(character)
    return (
        _identifier_start(character)
        or character.isdigit()
        or character == "'"
        or category.startswith(("M", "N"))
    )


def code_spans(source: str) -> Iterator[CodeSpan]:
    """Yield code spans, excluding Lean comments and string literals."""

    index = 0
    code_start = 0
    length = len(source)
    while index < length:
        if source.startswith("--", index):
            if code_start < index:
                yield CodeSpan(code_start, index, source[code_start:index])
            newline = source.find("\n", index + 2)
            if newline < 0:
                return
            index = newline + 1
            code_start = index
            continue
        if source.startswith("/-", index):
            if code_start < index:
                yield CodeSpan(code_start, index, source[code_start:index])
            depth = 1
            index += 2
            while index < length and depth:
                if source.startswith("/-", index):
                    depth += 1
                    index += 2
                elif source.startswith("-/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            code_start = index
            continue
        if source[index] == '"':
            if code_start < index:
                yield CodeSpan(code_start, index, source[code_start:index])
            index += 1
            escaped = False
            while index < length:
                character = source[index]
                index += 1
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    break
            code_start = index
            continue
        index += 1
    if code_start < length:
        yield CodeSpan(code_start, length, source[code_start:length])


def identifier_spans(source: str) -> Iterator[CodeSpan]:
    for span in code_spans(source):
        relative = 0
        while relative < len(span.text):
            character = span.text[relative]
            if not _identifier_start(character):
                relative += 1
                continue
            end = relative + 1
            while end < len(span.text) and _identifier_continue(span.text[end]):
                end += 1
            yield CodeSpan(
                span.start + relative,
                span.start + end,
                span.text[relative:end],
            )
            relative = end


def lean_tokens(source: str) -> tuple[str, ...]:
    """Tokenize enough Lean syntax to check exact rendering-only rewrites.

    Comments and whitespace are omitted; quoted strings are retained verbatim.
    Operators are emitted character-by-character, which is sufficient for
    equality checks and deliberately avoids pretending to be a full parser.
    """

    tokens: list[str] = []
    index = 0
    length = len(source)
    while index < length:
        if source[index].isspace():
            index += 1
            continue
        if source.startswith("--", index):
            newline = source.find("\n", index + 2)
            index = length if newline < 0 else newline + 1
            continue
        if source.startswith("/-", index):
            depth = 1
            index += 2
            while index < length and depth:
                if source.startswith("/-", index):
                    depth += 1
                    index += 2
                elif source.startswith("-/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            if depth:
                raise ValueError("Unterminated Lean block comment")
            continue
        if source[index] == '"':
            start = index
            index += 1
            escaped = False
            while index < length:
                character = source[index]
                index += 1
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    break
            else:
                raise ValueError("Unterminated Lean string literal")
            tokens.append(source[start:index])
            continue
        if _identifier_start(source[index]):
            end = index + 1
            while end < length and _identifier_continue(source[end]):
                end += 1
            tokens.append(source[index:end])
            index = end
            continue
        if source[index].isdigit():
            end = index + 1
            while end < length and source[end].isdigit():
                end += 1
            tokens.append(source[index:end])
            index = end
            continue
        tokens.append(source[index])
        index += 1
    return tuple(tokens)


def rewrite_identifiers(source: str, replacements: Mapping[str, str]) -> tuple[str, int]:
    """Replace complete identifiers outside comments and strings."""

    edits = [
        (span.start, span.end, replacements[span.text])
        for span in identifier_spans(source)
        if span.text in replacements
    ]
    target = source
    for start, end, replacement in reversed(edits):
        target = target[:start] + replacement + target[end:]
    return target, len(edits)


def theorem_declaration(source: str) -> tuple[CodeSpan, CodeSpan]:
    """Return keyword and theorem-name spans for the first declaration."""

    identifiers = list(identifier_spans(source))
    for index, token in enumerate(identifiers[:-1]):
        if token.text in {"theorem", "lemma", "example"}:
            if token.text == "example":
                # Anonymous examples have no declaration identifier.
                raise ValueError("Anonymous Lean examples cannot be renamed")
            return token, identifiers[index + 1]
    raise ValueError("No theorem or lemma declaration found")


def declaration_header_end(source: str, start: int) -> int:
    """Find the declaration's ``:=`` outside comments and strings."""

    for span in code_spans(source):
        if span.end <= start:
            continue
        begin = max(start, span.start)
        relative = source.find(":=", begin, span.end)
        if relative >= 0:
            return relative
    # Lean 3 command-style proofs can start with a top-level ``begin`` after
    # the proposition.  It is only a fallback and must still pass Lean.
    match = re.search(r"(?m)^\s*begin\b", source[start:])
    if match:
        return start + match.start()
    raise ValueError("Could not locate theorem proof delimiter")


def code_matches(
    source: str, pattern: re.Pattern[str], *, start: int = 0, end: int | None = None
) -> list[re.Match[str]]:
    """Return regex matches constrained to actual code spans."""

    limit = len(source) if end is None else end
    matches: list[re.Match[str]] = []
    for span in code_spans(source):
        lower = max(start, span.start)
        upper = min(limit, span.end)
        if lower >= upper:
            continue
        matches.extend(pattern.finditer(source, lower, upper))
    return matches


def apply_edit(source: str, start: int, end: int, replacement: str) -> str:
    if not 0 <= start <= end <= len(source):
        raise ValueError("Invalid source edit span")
    return source[:start] + replacement + source[end:]


def choose_deterministically(
    values: list[CodeSpan] | list[re.Match[str]],
    *,
    seed: int,
    item_id: str,
    namespace: str,
) -> CodeSpan | re.Match[str]:
    if not values:
        raise ValueError(f"No candidate found for {namespace}")
    import hashlib

    digest = hashlib.sha256(
        f"{namespace}\0{seed}\0{item_id}".encode()
    ).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]

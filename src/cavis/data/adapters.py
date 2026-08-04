"""Adapters for the public datasets used by the CAVIS audit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .dependence import (
    geometry_statement_identity,
    scoped_group_dependence_id,
)
from .io import canonical_json_hash, iter_jsonl, read_json, read_parquet_rows, sha256_text
from .records import ReasoningExample


def _as_text_sequence(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must contain only strings")
    return tuple(value)


def _geometry_group_id(
    filename: str, label: bool, valid_stems: frozenset[str]
) -> str:
    stem = Path(filename).stem
    if label:
        return stem
    match = re.fullmatch(r"(.+)_([0-9])", stem)
    if match and match.group(1) in valid_stems:
        return match.group(1)
    return stem


def load_geometry(
    extraction_path: str | Path,
    *,
    proof_root: str | Path | None = None,
    label_field: str = "label_corrected",
) -> list[ReasoningExample]:
    """Load the public Geometry of Reason extraction.

    The adapter keeps all published scalar and spectral fields in metadata.
    If ``proof_root`` is supplied, source lookup checks both
    ``{valid,invalid}/`` and the root itself.  Missing proof text is an error,
    preventing accidental evaluation on filenames alone.
    """

    if label_field not in {"label_original", "label_corrected", "is_valid"}:
        raise ValueError(f"Unsupported Geometry label field: {label_field}")
    rows = read_json(extraction_path)
    if not isinstance(rows, list):
        raise ValueError("Geometry extraction must be a JSON array")

    def parse_label(row: Mapping[str, Any]) -> bool:
        value = row.get(label_field)
        if label_field == "is_valid":
            if not isinstance(value, bool):
                raise ValueError("Geometry is_valid must be boolean")
            return value
        if value not in {"valid", "invalid"}:
            raise ValueError(f"Unexpected Geometry label: {value!r}")
        return value == "valid"

    parsed_labels = [parse_label(row) for row in rows]
    valid_stems = frozenset(
        Path(str(row["file"])).stem
        for row, label in zip(rows, parsed_labels, strict=True)
        if label
    )
    root = Path(proof_root) if proof_root is not None else None
    if root is None:
        raise ValueError(
            "Geometry loading requires proof_root to derive dependence identities"
        )
    output: list[ReasoningExample] = []
    name_hashes: dict[str, set[str]] = {}
    for row, valid in zip(rows, parsed_labels, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("Every Geometry row must be an object")
        filename = str(row.get("file", ""))
        if not filename.endswith(".lean"):
            raise ValueError(f"Geometry row has invalid Lean filename: {filename!r}")
        proof = ""
        proof_path: Path | None = None
        candidates = (
            root / ("valid" if valid else "invalid") / filename,
            root / ("invalid" if valid else "valid") / filename,
            root / filename,
        )
        proof_path = next((path for path in candidates if path.is_file()), None)
        if proof_path is None:
            raise FileNotFoundError(
                f"Could not find {filename} below proof root {root}"
            )
        proof = proof_path.read_text(encoding="utf-8")
        identity = geometry_statement_identity(proof)
        hashes = name_hashes.setdefault(identity.theorem_name, set())
        hashes.add(identity.statement_sha256)
        if len(hashes) > 1:
            raise ValueError(
                f"Lean theorem name {identity.theorem_name!r} maps to "
                "multiple normalized statements"
            )
        source_group_id = _geometry_group_id(filename, valid, valid_stems)

        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"file", "label_original", "label_corrected", "is_valid"}
        }
        metadata.update(
            {
                "filename": filename,
                "label_field": label_field,
                "label_original": row.get("label_original"),
                "label_corrected": row.get("label_corrected"),
                "proof_path": str(proof_path) if proof_path is not None else None,
                "source_group_id": source_group_id,
                "dependence_id": identity.dependence_id,
                "theorem_name": identity.theorem_name,
                "statement_sha256": identity.statement_sha256,
            }
        )
        source_value: Any = {"row": row, "proof": proof}
        output.append(
            ReasoningExample(
                item_id=Path(filename).stem,
                group_id=source_group_id,
                dependence_id=identity.dependence_id,
                dataset="geometry_minif2f",
                problem="",
                steps=(proof,),
                step_labels=(valid,),
                valid=valid,
                source_hash=canonical_json_hash(source_value),
                metadata=metadata,
            )
        )
    return output


def _processbench_step_labels(label: int, n_steps: int) -> tuple[bool | None, ...]:
    if label == -1:
        return (True,) * n_steps
    if not 0 <= label < n_steps:
        raise ValueError(f"ProcessBench label {label} outside [-1, {n_steps - 1}]")
    # The benchmark identifies the first erroneous step.  It does not establish
    # labels for the causally downstream suffix, so those remain unknown.
    return (True,) * label + (False,) + (None,) * (n_steps - label - 1)


def load_processbench(
    path: str | Path, *, subset: str | None = None
) -> list[ReasoningExample]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("ProcessBench input must be a JSON array")
    dataset_name = subset or Path(path).stem.lower()
    output: list[ReasoningExample] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Every ProcessBench row must be an object")
        item_id = str(row.get("id", ""))
        if not item_id or item_id in seen:
            raise ValueError(f"Missing or duplicate ProcessBench id: {item_id!r}")
        seen.add(item_id)
        steps = _as_text_sequence(row.get("steps"), field="steps")
        label = row.get("label")
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError("ProcessBench label must be an integer first-error index")
        labels = _processbench_step_labels(label, len(steps))
        dataset = f"processbench_{dataset_name}"
        output.append(
            ReasoningExample(
                item_id=item_id,
                group_id=item_id,
                dependence_id=scoped_group_dependence_id(dataset, item_id),
                dataset=dataset,
                problem=str(row.get("problem", "")),
                steps=steps,
                step_labels=labels,
                valid=label == -1,
                source_hash=canonical_json_hash(row),
                metadata={
                    "generator": row.get("generator"),
                    "first_error_step": None if label == -1 else label,
                    "final_answer_correct": row.get("final_answer_correct"),
                },
            )
        )
    return output


def _rating_to_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > 0:
            return True
        if value < 0:
            return False
        return None
    return None


def _prm_annotation_labels(
    generated_steps: tuple[str, ...], annotation_steps: Any
) -> tuple[bool | None, ...]:
    labels: list[bool | None] = [None] * len(generated_steps)
    if not isinstance(annotation_steps, Sequence):
        return tuple(labels)
    for index, annotation in enumerate(annotation_steps):
        if index >= len(labels) or not isinstance(annotation, Mapping):
            break
        completions = annotation.get("completions", [])
        if not isinstance(completions, Sequence):
            continue
        chosen = annotation.get("chosen_completion")
        candidate: Mapping[str, Any] | None = None
        if isinstance(chosen, int) and not isinstance(chosen, bool):
            if 0 <= chosen < len(completions) and isinstance(
                completions[chosen], Mapping
            ):
                candidate = completions[chosen]
        if candidate is None:
            # Recover the label of the actually generated step rather than
            # selecting a counterfactual annotator completion.
            candidate = next(
                (
                    completion
                    for completion in completions
                    if isinstance(completion, Mapping)
                    and completion.get("text") == generated_steps[index]
                ),
                None,
            )
        if candidate is not None:
            labels[index] = _rating_to_label(candidate.get("rating"))
    return tuple(labels)


def load_prm800k(path: str | Path) -> list[ReasoningExample]:
    """Load a PRM800K JSONL phase while preserving unlabeled suffixes."""

    output: list[ReasoningExample] = []
    for row_number, row in enumerate(iter_jsonl(path)):
        question = row.get("question")
        label = row.get("label")
        if not isinstance(question, Mapping) or not isinstance(label, Mapping):
            raise ValueError(f"Malformed PRM800K row {row_number}")
        steps = _as_text_sequence(
            question.get("pre_generated_steps", ()), field="pre_generated_steps"
        )
        labels = _prm_annotation_labels(steps, label.get("steps"))
        if any(value is False for value in labels):
            valid: bool | None = False
        elif steps and all(value is True for value in labels):
            valid = True
        else:
            valid = None
        item_id = f"phase2-{row_number:06d}"
        problem = str(question.get("problem", ""))
        group_id = sha256_text(problem)[:20] or item_id
        output.append(
            ReasoningExample(
                item_id=item_id,
                group_id=group_id,
                dependence_id=scoped_group_dependence_id("prm800k", group_id),
                dataset="prm800k",
                problem=problem,
                steps=steps,
                step_labels=labels,
                valid=valid,
                source_hash=canonical_json_hash(row),
                metadata={
                    "row_number": row_number,
                    "generation": row.get("generation"),
                    "finish_reason": label.get("finish_reason"),
                    "quality_control": row.get("is_quality_control_question"),
                    "initial_screening": row.get(
                        "is_initial_screening_question"
                    ),
                    "ground_truth_answer": question.get("ground_truth_answer"),
                    "pre_generated_answer": question.get("pre_generated_answer"),
                },
            )
        )
    return output


def _math_shepherd_rows(path: str | Path) -> Iterable[Mapping[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return read_parquet_rows(path)
    if suffix == ".json":
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError("Math-Shepherd JSON must be an array")
        return rows
    if suffix == ".jsonl":
        return iter_jsonl(path)
    raise ValueError("Math-Shepherd input must be .parquet, .json, or .jsonl")


def load_math_shepherd(path: str | Path) -> list[ReasoningExample]:
    output: list[ReasoningExample] = []
    for row_number, row in enumerate(_math_shepherd_rows(path)):
        if not isinstance(row, Mapping):
            raise ValueError(f"Malformed Math-Shepherd row {row_number}")
        steps = _as_text_sequence(row.get("completions"), field="completions")
        raw_labels = row.get("labels")
        if not isinstance(raw_labels, Sequence) or isinstance(
            raw_labels, (str, bytes)
        ):
            raise ValueError("Math-Shepherd labels must be a sequence")
        labels = tuple(_rating_to_label(value) for value in raw_labels)
        if len(steps) != len(labels):
            raise ValueError("Math-Shepherd completions/labels length mismatch")
        if any(value is False for value in labels):
            valid: bool | None = False
        elif steps and all(value is True for value in labels):
            valid = True
        else:
            valid = None
        problem = str(row.get("prompt", ""))
        item_id = str(row.get("id") or f"math-shepherd-{row_number:06d}")
        group_id = sha256_text(problem)[:20] or item_id
        output.append(
            ReasoningExample(
                item_id=item_id,
                group_id=group_id,
                dependence_id=scoped_group_dependence_id(
                    "math_shepherd", group_id
                ),
                dataset="math_shepherd",
                problem=problem,
                steps=steps,
                step_labels=labels,
                valid=valid,
                source_hash=canonical_json_hash(row),
                metadata={"row_number": row_number},
            )
        )
    return output

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cavis.reproducibility.io import write_jsonl
from cavis.schemas import ScoreRecord

from .spectral import compute_spectral_scores
from .token import token_scores


@dataclass(frozen=True)
class ExtractionConfig:
    model_id: str
    revision: str
    layers: tuple[int, ...] = (-1,)
    dtype: str = "bfloat16"
    device: str = "cuda"
    max_length: int = 2048
    batch_token_budget: int = 2048
    batch_max_examples: int = 8
    max_graph_tokens: int = 512
    high_frequency_fraction: float = 0.5
    quantization: str | None = None
    trust_remote_code: bool = False
    save_pooled_hidden: bool = False

    def __post_init__(self) -> None:
        if len(self.revision) != 40:
            raise ValueError("model revision must be a pinned 40-character commit")
        if not self.layers:
            raise ValueError("at least one extraction layer is required")
        if self.max_length < 2 or self.batch_token_budget < 2:
            raise ValueError("token limits must be at least two")
        if self.batch_max_examples <= 0:
            raise ValueError("batch_max_examples must be positive")
        if self.quantization not in {None, "4bit"}:
            raise ValueError("quantization must be None or '4bit'")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact_filename(item_id: str, transformation_id: str) -> str:
    """Return a portable, fixed-width filename for one hidden-state artifact."""

    identity = f"{item_id}\0{transformation_id}"
    return f"{_sha256_text(identity)}.npy"


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _parse_json_mapping(value: Any, *, field_name: str, item_id: str) -> dict[str, Any]:
    if _is_missing(value):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} for {item_id} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} for {item_id} must be a mapping")
    return dict(value)


def _literal_bool(value: Any, *, field_name: str, item_id: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{field_name} for {item_id} must be a literal boolean")


def _parse_string_list(value: Any, *, field_name: str, item_id: str) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} for {item_id} is not valid JSON") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} for {item_id} must be a JSON string list")
    return value


def _reasoning_target_mask(
    offsets: Sequence[Sequence[int]], *, reasoning_start: int
) -> list[bool]:
    """Select real tokens whose character span overlaps the reasoning text.

    Tokenizing the prompt separately and counting its tokens is not exact:
    byte-pair tokenizers may merge across the prompt/reasoning boundary. Fast
    tokenizer offsets let us define the span once in the concatenated text.
    Special tokens conventionally have the empty offset ``(0, 0)`` and are
    therefore excluded.
    """

    if reasoning_start < 0:
        raise ValueError("reasoning_start must be non-negative")
    mask: list[bool] = []
    for index, offset in enumerate(offsets):
        if len(offset) != 2:
            raise ValueError(f"offset {index} must contain exactly two integers")
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < start:
            raise ValueError(f"offset {index} is invalid: {(start, end)}")
        mask.append(end > max(start, reasoning_start))
    return mask


def _normalize_row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize prepared Parquet provenance to the frozen protocol contract.

    Prepared LeanTwin uses ``semantic_variant_id`` for the semantic root shared
    by a base and its G0 descendants.  The evaluator needs an identifier for
    each exact scored text, so the normalized ``semantic_variant_id`` is the
    unique row ``item_id`` and the source value is retained as
    ``semantic_root_id``.
    """

    item_id = str(row["item_id"])
    metadata = _parse_json_mapping(
        row.get("metadata_json"),
        field_name="metadata_json",
        item_id=item_id,
    )
    metadata.update(
        _parse_json_mapping(
            row.get("metadata"),
            field_name="metadata",
            item_id=item_id,
        )
    )
    is_prepared = any(
        field in row
        for field in (
            "transform_kind",
            "parent_variant_id",
            "pair_ids_json",
            "cavis_eligible",
        )
    )
    if not is_prepared:
        return metadata

    source_kind = str(row.get("transform_kind", "base"))
    if source_kind not in {"base", "g0", "g1"}:
        raise ValueError(f"transform_kind for {item_id} is unsupported: {source_kind!r}")
    variant_kind = "g0" if source_kind == "g0" else "base"
    parent_id = row.get("parent_variant_id")
    parent_id = None if _is_missing(parent_id) else str(parent_id)
    pair_id = row.get("pair_id")
    pair_id = None if _is_missing(pair_id) else str(pair_id)
    pair_side = row.get("pair_side")
    pair_side = None if _is_missing(pair_side) else str(pair_side)
    positive_id = row.get("positive_semantic_variant_id")
    positive_id = None if _is_missing(positive_id) else str(positive_id)
    semantic_root = row.get("semantic_variant_id", item_id)
    semantic_root = item_id if _is_missing(semantic_root) else str(semantic_root)
    dependence_id = row.get("dependence_id")
    if _is_missing(dependence_id) or not str(dependence_id).strip():
        raise ValueError(
            f"dependence_id for prepared row {item_id} must be non-empty"
        )
    source_group_id = row.get("source_group_id", row.get("group_id"))
    if _is_missing(source_group_id) or not str(source_group_id).strip():
        raise ValueError(
            f"source_group_id for prepared row {item_id} must be non-empty"
        )

    normalized: dict[str, Any] = {
        "semantic_variant_id": item_id,
        "semantic_root_id": semantic_root,
        "group_id": str(row["group_id"]),
        "source_group_id": str(source_group_id),
        "dependence_id": str(dependence_id),
        "variant_kind": variant_kind,
        "source_transform_kind": source_kind,
        "g0_parent_id": parent_id if variant_kind == "g0" else None,
        # G0 descendants inherit pair columns for provenance but are not pair
        # sides. Pair discovery happens only on G1 roots.
        "g1_pair_id": pair_id if source_kind == "g1" else None,
        "g1_side": pair_side if source_kind == "g1" else None,
        "g1_positive_id": positive_id if source_kind == "g1" else None,
        "pair_ids": _parse_string_list(
            row.get("pair_ids_json"),
            field_name="pair_ids_json",
            item_id=item_id,
        ),
    }
    for field in ("mechanically_verified", "cavis_eligible"):
        normalized[field] = _literal_bool(
            row.get(field),
            field_name=field,
            item_id=item_id,
        )
    for field in (
        "semantic_status",
        "evidence_state",
        "syntactic_contract_verified",
        "compiler_outcomes_match",
        "paired_validity_approved",
        "source_hash",
        "target_hash",
        "transform_name",
        "theorem_name",
        "statement_sha256",
    ):
        if field in row and not _is_missing(row[field]):
            value = row[field]
            if field in {
                "syntactic_contract_verified",
                "compiler_outcomes_match",
                "paired_validity_approved",
            }:
                value = _literal_bool(value, field_name=field, item_id=item_id)
            normalized[field] = value
    metadata.update(normalized)
    return metadata


def _normalize_layer(index: int, n_layers: int) -> int:
    normalized = index if index >= 0 else n_layers + index
    if normalized < 0 or normalized >= n_layers:
        raise IndexError(f"Layer {index} is invalid for a {n_layers}-layer model")
    return normalized


def _greedy_batches(
    examples: Sequence[dict[str, Any]],
    *,
    token_budget: int,
    max_examples: int,
) -> Iterator[list[dict[str, Any]]]:
    """Pack adjacent examples without changing their deterministic order."""
    current: list[dict[str, Any]] = []
    current_max = 0
    for example in examples:
        length = len(example["input_ids"])
        candidate_max = max(current_max, length)
        candidate_size = len(current) + 1
        if current and (
            candidate_max * candidate_size > token_budget
            or candidate_size > max_examples
        ):
            yield current
            current = []
            current_max = 0
        current.append(example)
        current_max = max(current_max, length)
    if current:
        yield current


class HFScoreExtractor:
    """No-gradient teacher-forced extractor with immutable model revisions."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def load(self) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:  # pragma: no cover - requires optional extra
            raise RuntimeError(
                "HF extraction requires `uv sync --extra hf` "
                "(and `--extra quantization` for 4-bit Linux runs)."
            ) from exc

        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if self.config.dtype not in dtype_by_name:
            raise ValueError(f"Unsupported dtype {self.config.dtype!r}")
        dtype = dtype_by_name[self.config.dtype]
        if not self.config.device.startswith("cuda") and dtype != torch.float32:
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has neither pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype,
            "attn_implementation": "eager",
        }
        if self.config.quantization == "4bit":
            if not self.config.device.startswith("cuda"):
                raise RuntimeError("4-bit extraction is supported only on CUDA")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": self.config.device}

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not load {self.config.model_id}@{self.config.revision}. "
                "If this is a gated Meta model, accept its license and run `hf auth login`."
            ) from exc
        if self.config.quantization is None:
            model.to(self.config.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._torch, self._tokenizer, self._model = torch, tokenizer, model

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _prepare(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self._tokenizer is None:
            raise RuntimeError("load() must be called before extraction")
        prepared: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row_index, raw in enumerate(rows):
            row = dict(raw)
            item_id = str(row.get("item_id", "")).strip()
            reasoning = str(row.get("reasoning", row.get("text", "")))
            prompt = str(row.get("prompt", ""))
            if not item_id or not reasoning:
                raise ValueError(f"Row {row_index} requires non-empty item_id and reasoning/text")
            transformation_id = str(row.get("transformation_id", "base"))
            identity = (item_id, transformation_id)
            if identity in seen:
                raise ValueError(f"Duplicate item/transformation identity: {identity}")
            seen.add(identity)

            separator = "\n" if prompt else ""
            full_text = f"{prompt.rstrip()}{separator}{reasoning}"
            prefix = f"{prompt.rstrip()}{separator}" if prompt else ""
            tokenized = self._tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.config.max_length,
                return_offsets_mapping=True,
            )
            offsets = tokenized.pop("offset_mapping", None)
            if offsets is None:
                raise RuntimeError(
                    "The configured fast tokenizer did not return offset mappings"
                )
            target_mask = _reasoning_target_mask(
                offsets,
                reasoning_start=len(prefix),
            )
            input_ids = tokenized["input_ids"]
            if sum(target_mask[1:]) == 0:
                raise ValueError(f"No scoreable reasoning token remains for {item_id}")
            prepared.append(
                {
                    "row": row,
                    "full_text": full_text,
                    "input_ids": input_ids,
                    "attention_mask": tokenized["attention_mask"],
                    "target_mask": target_mask,
                    "truncated": len(input_ids) >= self.config.max_length,
                }
            )
        return prepared

    def extract(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        seed: int,
        artifact_dir: str | Path | None = None,
    ) -> Iterator[ScoreRecord]:
        if self._model is None:
            self.load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        torch = self._torch
        prepared = self._prepare(rows)
        batches = _greedy_batches(
            prepared,
            token_budget=self.config.batch_token_budget,
            max_examples=self.config.batch_max_examples,
        )
        artifact_root = Path(artifact_dir) if artifact_dir is not None else None
        if artifact_root is not None:
            artifact_root.mkdir(parents=True, exist_ok=True)

        for batch in batches:
            padded = self._tokenizer.pad(
                {
                    "input_ids": [example["input_ids"] for example in batch],
                    "attention_mask": [example["attention_mask"] for example in batch],
                },
                padding=True,
                return_tensors="pt",
            )
            input_ids = padded["input_ids"].to(self.config.device)
            attention_mask = padded["attention_mask"].to(self.config.device)
            target_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            for row_index, example in enumerate(batch):
                length = len(example["target_mask"])
                target_mask[row_index, :length] = torch.tensor(
                    example["target_mask"], dtype=torch.bool, device=self.config.device
                )

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            if outputs.attentions is None or outputs.hidden_states is None:
                raise RuntimeError(
                    "Model did not expose attentions/hidden states; eager attention is required"
                )
            sequence_scores = token_scores(outputs.logits, input_ids, target_mask)
            n_layers = len(outputs.attentions)
            layers = tuple(_normalize_layer(layer, n_layers) for layer in self.config.layers)

            for row_index, example in enumerate(batch):
                row = example["row"]
                scores = sequence_scores[row_index].as_dict()
                valid_length = int(attention_mask[row_index].sum().item())
                mask = (
                    target_mask[row_index, :valid_length]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
                layer_payloads: list[dict[str, float]] = []
                for layer in layers:
                    attention = outputs.attentions[layer][
                        row_index, :, :valid_length, :valid_length
                    ]
                    hidden = outputs.hidden_states[layer + 1][
                        row_index, :valid_length, :
                    ]
                    spectral = compute_spectral_scores(
                        attention,
                        hidden,
                        mask=mask,
                        high_frequency_fraction=self.config.high_frequency_fraction,
                        max_graph_tokens=self.config.max_graph_tokens,
                    )
                    payload = spectral.as_dict()
                    layer_payloads.append(
                        {
                            key: float(value)
                            for key, value in payload.items()
                            if key not in {"subsampled", "n_graph_tokens"}
                        }
                    )
                    for key, value in payload.items():
                        if isinstance(value, bool):
                            scores[f"layer_{layer}.{key}"] = float(value)
                        else:
                            scores[f"layer_{layer}.{key}"] = float(value)
                for key in layer_payloads[0]:
                    scores[f"layers_mean.{key}"] = float(
                        np.mean([payload[key] for payload in layer_payloads])
                    )

                artifact_hash: str | None = None
                row_metadata = _normalize_row_metadata(row)
                metadata: dict[str, Any] = {
                    **dict(row_metadata),
                    "truncated": example["truncated"],
                    "requested_layers": list(self.config.layers),
                    "resolved_layers": list(layers),
                    "extraction": asdict(self.config),
                }
                provenance_fields = (
                    "group_id",
                    "source_group_id",
                    "dependence_id",
                    "theorem_name",
                    "statement_sha256",
                    "theorem_id",
                    "problem_id",
                    "source_revision",
                )
                for field in provenance_fields:
                    if field in row and row[field] is not None:
                        metadata[field] = row[field]
                if self.config.save_pooled_hidden:
                    if artifact_root is None:
                        raise ValueError("artifact_dir is required when save_pooled_hidden=true")
                    pooled = (
                        outputs.hidden_states[-1][row_index, :valid_length][
                            target_mask[row_index, :valid_length]
                        ]
                        .float()
                        .mean(dim=0)
                        .cpu()
                        .numpy()
                    )
                    artifact_name = _artifact_filename(
                        str(row["item_id"]),
                        str(row.get("transformation_id", "base")),
                    )
                    artifact_path = artifact_root / artifact_name
                    _atomic_save_array(artifact_path, pooled)
                    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                    metadata["pooled_hidden_path"] = (
                        Path(artifact_root.name) / artifact_name
                    ).as_posix()

                yield ScoreRecord(
                    item_id=str(row["item_id"]),
                    dataset=str(row.get("dataset", "unknown")),
                    model_id=self.config.model_id,
                    model_revision=self.config.revision,
                    transformation_id=str(row.get("transformation_id", "base")),
                    label=int(row["label"]),
                    token_length=sequence_scores[row_index].n_tokens,
                    scores=scores,
                    seed=seed,
                    input_hash=_sha256_text(example["full_text"]),
                    artifact_hash=artifact_hash,
                    metadata=metadata,
                )
            del outputs

    def extract_to_jsonl(
        self,
        rows: Iterable[Mapping[str, Any]],
        output_path: str | Path,
        *,
        seed: int,
        artifact_dir: str | Path | None = None,
    ) -> list[ScoreRecord]:
        records = list(self.extract(rows, seed=seed, artifact_dir=artifact_dir))
        write_jsonl(output_path, (record.to_dict() for record in records))
        return records


__all__ = ["ExtractionConfig", "HFScoreExtractor", "_greedy_batches"]

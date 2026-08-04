#!/usr/bin/env python3
"""Run the frozen CAVIS protocol on long-form ScoreRecord JSONL.

Example:

    python scripts/evaluate_cavis.py \
      --scores results/runs/extraction/scores.jsonl \
      --score-name hfer \
      --dataset geometry_minif2f \
      --model-id meta-llama/Llama-3.2-1B-Instruct \
      --output-dir results/runs/cavis_hfer_seed17

The strict metadata contract is documented in
``cavis.evaluation.protocol.ProtocolRow`` and the module docstring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from cavis.evaluation.protocol import PROTOCOL_VERSION, run_frozen_protocol


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen CAVIS score/model/dataset slice."
    )
    parser.add_argument("--scores", type=Path, required=True, help="ScoreRecord JSONL")
    parser.add_argument("--score-name", required=True, help="Key inside each scores map")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dataset", help="Required when JSONL contains multiple datasets")
    parser.add_argument("--model-id", help="Required when JSONL contains multiple models")
    parser.add_argument("--model-revision")
    parser.add_argument("--record-seed", type=int, help="Filter extraction seed")
    return parser.parse_args()


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
        records.append(payload)
    if not records:
        raise ValueError(f"{path} contains no JSON objects")
    return records, hashlib.sha256(raw).hexdigest()


def _select(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filters = {
        "dataset": args.dataset,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "seed": args.record_seed,
    }
    selected = [
        record
        for record in records
        if all(value is None or record.get(key) == value for key, value in filters.items())
    ]
    if not selected:
        active = {key: value for key, value in filters.items() if value is not None}
        raise ValueError(f"no records match filters: {active}")
    return selected


def _json_safe(value: Any) -> Any:
    """Convert non-finite IEEE values to explicit JSON strings."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_outputs(
    output_dir: Path,
    *,
    metrics: dict[str, Any],
    per_item: tuple[dict[str, Any], ...],
    input_path: Path,
    input_sha256: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **metrics,
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_sha256,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    safe_rows = [_json_safe(dict(row)) for row in per_item]
    with (output_dir / "per_item.jsonl").open("w", encoding="utf-8") as handle:
        for row in safe_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    columns = sorted({key for row in safe_rows for key in row})
    with (output_dir / "per_item.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in safe_rows:
            flat = {
                key: (
                    json.dumps(value, sort_keys=True, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(flat)


def main() -> int:
    args = _arguments()
    try:
        records, input_sha256 = _load_jsonl(args.scores)
        selected = _select(records, args)
        result = run_frozen_protocol(
            selected,
            score_name=args.score_name,
            split_seed=args.split_seed,
            alpha=args.alpha,
        )
        _write_outputs(
            args.output_dir,
            metrics=dict(result.metrics),
            per_item=tuple(dict(row) for row in result.per_item),
            input_path=args.scores,
            input_sha256=input_sha256,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "metrics": str((args.output_dir / "metrics.json").resolve()),
        "per_item_csv": str((args.output_dir / "per_item.csv").resolve()),
        "per_item_jsonl": str((args.output_dir / "per_item.jsonl").resolve()),
        "n_selected_records": len(selected),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

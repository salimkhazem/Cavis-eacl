from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cavis.reproducibility.io import atomic_write_json, write_jsonl
from cavis.scores.probe import (
    PROBE_FIT_SCOPE,
    defensible_probe_fit_mask,
    fit_nested_linear_probe,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the supervised probe ceiling.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    args = parser.parse_args()
    rows = _read_jsonl(args.scores)
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for row in rows:
        metadata = row.get("metadata", {})
        artifact = metadata.get("pooled_hidden_path")
        dependence_id = metadata.get("dependence_id")
        if not artifact or not dependence_id:
            raise ValueError(
                "Every row needs metadata.pooled_hidden_path and "
                "metadata.dependence_id"
            )
        artifact_path = Path(str(artifact))
        if not artifact_path.is_absolute():
            artifact_path = args.scores.parent / artifact_path
        features.append(np.load(artifact_path, allow_pickle=False))
        labels.append(1 if int(row["label"]) == 1 else 0)
        groups.append(str(dependence_id))
    matrix = np.stack(features)
    fit_mask = defensible_probe_fit_mask(rows)
    logits, parameters, assignments = fit_nested_linear_probe(
        matrix,
        labels,
        groups,
        split_seed=args.split_seed,
        fit_mask=fit_mask,
        c_grid=args.c_grid,
    )
    for row, logit, fit_eligible in zip(rows, logits, fit_mask, strict=True):
        probe_split = assignments[str(row["metadata"]["dependence_id"])]
        row["scores"] = {**row["scores"], "linear_probe_logit": float(logit)}
        row["metadata"] = {
            **row["metadata"],
            "probe_split": probe_split,
            "probe_fit_scope": PROBE_FIT_SCOPE,
            "probe_fit_eligible": bool(fit_eligible),
            "probe_used_for_fit": bool(
                fit_eligible and probe_split == "train"
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scores_with_probe.jsonl", rows)
    atomic_write_json(args.output_dir / "probe_parameters.json", parameters.to_dict())
    print(
        f"Wrote {len(rows)} probe logits; C={parameters.c}; "
        f"inner AUROC={parameters.selection_auroc:.4f}"
    )


if __name__ == "__main__":
    main()

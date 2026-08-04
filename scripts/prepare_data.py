#!/usr/bin/env python3
"""Materialize deterministic Parquet inputs declared in configs/data."""

from __future__ import annotations

import argparse
from pathlib import Path

from cavis.data.preparation import prepare_from_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest.lock"))
    parser.add_argument("--config-dir", type=Path, default=Path("configs/data"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Dataset key to prepare (repeatable); default prepares all.",
    )
    parser.add_argument(
        "--build-leantwin",
        action="store_true",
        help=(
            "Create unverified G0/G1 candidates. Without this flag Geometry "
            "contains base proofs only."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--transform-seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/cache/prepared/preparation_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = prepare_from_configs(
        config_dir=args.config_dir,
        manifest_path=args.manifest,
        project_root=args.project_root,
        selected=tuple(args.only),
        build_leantwin=args.build_leantwin,
        sample_seed=args.sample_seed,
        transform_seed=args.transform_seed,
        force=args.force,
        output_manifest=args.output_manifest,
    )
    for summary in summaries:
        cache = "cached" if summary.cached else "written"
        print(
            f"{summary.dataset_key}: {cache} {summary.record_count} rows -> "
            f"{summary.output_path} ({summary.output_sha256[:12]})"
        )
        if summary.semantic_status == "contains_unverified_candidates":
            print(
                "  WARNING: LeanTwin candidates remain semantic_status="
                "not_established until external evidence is joined."
            )


if __name__ == "__main__":
    main()

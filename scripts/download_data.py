#!/usr/bin/env python3
"""Download public audit artifacts at revisions pinned in data_manifest.lock."""

from __future__ import annotations

import argparse
from pathlib import Path

from cavis.data.download import artifact_is_current, download_artifact, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest.lock"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Artifact name to fetch (repeatable); default fetches all.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check cache state; never access the network.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = load_manifest(args.manifest)
    requested = set(args.only)
    unknown = requested.difference(spec.name for spec in specs)
    if unknown:
        raise SystemExit(f"Unknown artifact(s): {', '.join(sorted(unknown))}")
    selected = [spec for spec in specs if not requested or spec.name in requested]
    failed = False
    for spec in selected:
        if args.check:
            current = artifact_is_current(spec, args.data_root)
            print(f"{spec.name}: {'ok' if current else 'missing-or-mismatched'}")
            failed |= not current
        else:
            destination = download_artifact(spec, data_root=args.data_root)
            print(f"{spec.name}: {destination}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

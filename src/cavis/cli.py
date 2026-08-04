from __future__ import annotations

import argparse
from pathlib import Path

from cavis import __version__
from cavis.certificates import InvarianceCalibrator
from cavis.reproducibility.manifest import validate_manifest, validate_model_configs


def _validate(args: argparse.Namespace) -> None:
    manifest = validate_manifest(args.manifest)
    errors = validate_model_configs(manifest, args.model_configs)
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"valid: {len(manifest['sources'])} sources, "
        f"{len(manifest['models'])} immutable models"
    )


def _certificate(args: argparse.Namespace) -> None:
    calibrator = InvarianceCalibrator(alpha=args.alpha).fit(args.radii)
    result = calibrator.predict_one(args.score, args.threshold)
    import json

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cavis")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(required=True)

    validate = subparsers.add_parser("validate", help="validate immutable manifests")
    validate.add_argument("--manifest", type=Path, default=Path("data_manifest.lock"))
    validate.add_argument("--model-configs", type=Path, default=Path("configs/model"))
    validate.set_defaults(function=_validate)

    certificate = subparsers.add_parser(
        "certificate", help="compute one invariance certificate"
    )
    certificate.add_argument("--radii", type=float, nargs="+", required=True)
    certificate.add_argument("--alpha", type=float, default=0.1)
    certificate.add_argument("--score", type=float, required=True)
    certificate.add_argument("--threshold", type=float, required=True)
    certificate.set_defaults(function=_certificate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()


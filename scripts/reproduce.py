from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cavis.reproducibility.environment import capture_environment
from cavis.reproducibility.io import atomic_write_json, sha256_file
from cavis.reproducibility.manifest import validate_manifest, validate_model_configs
from scripts.make_figures import generate_figures
from scripts.make_matrix import generate_matrix
from scripts.make_tables import generate_tables

ROOT = Path(__file__).resolve().parents[1]


def _run_tests() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q"]
    process = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            "CPU tests failed during reconstruction:\n"
            f"{process.stdout}\n{process.stderr}"
        )
    return {
        "command": command,
        "return_code": process.returncode,
        "summary": process.stdout.strip().splitlines()[-1],
    }


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def reproduce(*, run_tests: bool) -> dict[str, Any]:
    manifest = validate_manifest(ROOT / "data_manifest.lock")
    config_errors = validate_model_configs(manifest, ROOT / "configs/model")
    if config_errors:
        raise RuntimeError("Model configuration mismatch:\n" + "\n".join(config_errors))

    matrix_rows = generate_matrix(ROOT / "results/tables/experiment_matrix.csv")
    table_manifest = generate_tables(
        spec_path=ROOT / "artifacts/specs/experiment_design.json",
        results_dir=ROOT / "results",
        output_dir=ROOT / "artifacts/tables",
    )
    figure_manifest = generate_figures(
        results_dir=ROOT / "results",
        output_dir=ROOT / "artifacts/figures",
        seed=17,
    )
    test_report = _run_tests() if run_tests else {"status": "skipped_by_flag"}

    audit_metrics = ROOT / "results/runs/geometry_cpu_audit/metrics.json"
    if audit_metrics.is_file():
        payload = json.loads(audit_metrics.read_text(encoding="utf-8"))
        audit_status: dict[str, Any] = {
            "status": payload["analysis_status"],
            "scope_boundary": payload["scope_boundary"],
            "source_sha256": payload["source"]["sha256"],
            "metrics_sha256": sha256_file(audit_metrics),
        }
    else:
        audit_status = {"status": "not_available"}

    generated = [
        *(ROOT / "artifacts/tables").glob("*"),
        *(ROOT / "artifacts/figures").glob("*"),
        ROOT / "results/tables/experiment_matrix.csv",
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "cpu_cache_reconstruction_no_model_inference",
        "environment": capture_environment(ROOT),
        "manifest": {
            "sources": len(manifest["sources"]),
            "models": len(manifest["models"]),
            "sha256": sha256_file(ROOT / "data_manifest.lock"),
            "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        },
        "tests": test_report,
        "public_geometry_audit": audit_status,
        "matrix": {
            "rows": len(matrix_rows),
            "run_status_counts": {
                status: sum(row["run_status"] == status for row in matrix_rows)
                for status in sorted({row["run_status"] for row in matrix_rows})
            },
        },
        "tables": table_manifest["tables"],
        "figures": figure_manifest["figures"],
        "generated_hashes": _file_hashes(generated),
        "gpu_scope": (
            "Missing extraction caches are reported as not_run. "
            "This CPU command neither fabricates nor launches GPU outputs."
        ),
    }
    atomic_write_json(ROOT / "results/reproduction_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only CAVIS reconstruction.")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    report = reproduce(run_tests=not args.skip_tests)
    print(
        "Reconstruction complete: "
        f"{report['matrix']['rows']} matrix rows; "
        "report=results/reproduction_report.json"
    )


if __name__ == "__main__":
    main()

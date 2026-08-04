from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cavis.data.io import canonical_json_hash, sha256_file
from cavis.reproducibility import atomic_write_json
from cavis.reproducibility.config import load_yaml
from cavis.transforms.resource_exclusion import (
    verify_resource_exclusion_report,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    experiment_key: str
    dataset_key: str
    model_key: str
    experiment_config: Path
    compute_config: Path
    model_config: Path
    input_path: Path
    input_kind: str
    output_path: Path
    physical_device: str
    seed: int
    max_base_items: int | None = None
    max_transforms_per_family: int | None = None
    precision_tag: str = "bfloat16"
    quantization: str | None = None
    execution_preflight: tuple[tuple[str, str], ...] = ()

    def command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "scripts.extract_scores",
            "--model-config",
            str(self.model_config),
            "--input",
            str(self.input_path),
            "--output",
            str(self.output_path),
            "--seed",
            str(self.seed),
            "--device",
            "cuda",
        ]
        if self.max_base_items is not None:
            command.extend(["--max-base-items", str(self.max_base_items)])
        if self.max_transforms_per_family is not None:
            command.extend(
                [
                    "--max-transforms-per-family",
                    str(self.max_transforms_per_family),
                ]
            )
        if self.quantization is not None:
            command.extend(["--quantization", self.quantization])
        return command


def _signature_payload(job: Job) -> dict[str, Any]:
    implementation_paths = (
        ROOT / "scripts/run_sweep.py",
        ROOT / "scripts/extract_scores.py",
        ROOT / "src/cavis/scores/extractor.py",
        ROOT / "src/cavis/scores/spectral.py",
        ROOT / "src/cavis/scores/token.py",
        ROOT / "src/cavis/schemas.py",
    )
    normalized_command = ["<python>", *job.command()[1:]]
    payload = {
        "schema_version": 1,
        "experiment_key": job.experiment_key,
        "dataset_key": job.dataset_key,
        "model_key": job.model_key,
        "input_kind": job.input_kind,
        "input_path": str(job.input_path),
        "input_sha256": sha256_file(job.input_path),
        "model_config_path": str(job.model_config),
        "model_config_sha256": sha256_file(job.model_config),
        "experiment_config_sha256": sha256_file(job.experiment_config),
        "compute_config_sha256": sha256_file(job.compute_config),
        "physical_device": job.physical_device,
        "seed": job.seed,
        "max_base_items": job.max_base_items,
        "max_transforms_per_family": job.max_transforms_per_family,
        "precision_tag": job.precision_tag,
        "quantization": job.quantization,
        "execution_preflight": dict(job.execution_preflight),
        "command": normalized_command,
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in implementation_paths
        },
    }
    payload["signature"] = canonical_json_hash(payload)
    return payload


def _index_configs(directory: Path, section: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.yaml")):
        payload = load_yaml(path)
        value = payload[section]
        key = value["key"]
        if key in index:
            raise ValueError(f"Duplicate {section} key {key}")
        index[key] = (path, value)
    return index


def build_jobs(
    experiment_path: Path,
    compute_path: Path,
    *,
    result_root: Path,
) -> list[Job]:
    experiment = load_yaml(experiment_path)["experiment"]
    compute = load_yaml(compute_path)["compute"]
    models = _index_configs(Path("configs/model"), "model")
    datasets = _index_configs(Path("configs/data"), "dataset")
    assignment: dict[str, str] = {}
    for worker in compute["workers"]:
        for model_key in worker["model_keys"]:
            if model_key in assignment:
                raise ValueError(f"Model {model_key} assigned to multiple workers")
            assignment[model_key] = str(worker["cuda_visible_device"])
    seed = int(compute["extraction_seed"])
    experiment_key = str(experiment["key"])
    require_verified = bool(experiment.get("require_verified_leantwin", False))
    eligible_path_field = experiment.get("eligible_path_field", "eligible_path")
    if not isinstance(eligible_path_field, str) or not eligible_path_field.strip():
        raise ValueError("experiment.eligible_path_field must be a non-empty string")
    max_base_items = (
        int(experiment["max_base_items"])
        if experiment.get("max_base_items") is not None
        else None
    )
    max_transforms = (
        int(experiment["transformations_per_family"])
        if experiment.get("transformations_per_family") is not None
        else None
    )
    configured_precision = experiment.get("precision")
    if isinstance(configured_precision, list) and "4bit_secondary" in configured_precision:
        precision_variants = (
            ("bfloat16", None),
            ("4bit_secondary", "4bit"),
        )
    else:
        precision_variants = (("bfloat16", None),)

    jobs: list[Job] = []
    for dataset_key in experiment["datasets"]:
        if dataset_key not in datasets:
            raise KeyError(f"No data config for {dataset_key}")
        _, dataset = datasets[dataset_key]
        use_verified = dataset_key == "geometry_leantwin" and require_verified
        input_field = eligible_path_field if use_verified else "prepared_path"
        if input_field not in dataset:
            raise KeyError(f"Data config {dataset_key} lacks {input_field}")
        for model_key in experiment["models"]:
            if model_key not in models:
                raise KeyError(f"No model config for {model_key}")
            if model_key not in assignment:
                raise KeyError(f"No GPU assignment for {model_key}")
            model_path, _ = models[model_key]
            for precision_tag, quantization in precision_variants:
                output_path = (
                    result_root
                    / experiment_key
                    / dataset_key
                    / model_key
                )
                if len(precision_variants) > 1:
                    output_path /= precision_tag
                jobs.append(
                    Job(
                        experiment_key=experiment_key,
                        dataset_key=dataset_key,
                        model_key=model_key,
                        experiment_config=experiment_path,
                        compute_config=compute_path,
                        model_config=model_path,
                        input_path=Path(dataset[input_field]),
                        input_kind="verified" if use_verified else "prepared",
                        output_path=output_path,
                        physical_device=assignment[model_key],
                        seed=seed,
                        max_base_items=max_base_items,
                        max_transforms_per_family=max_transforms,
                        precision_tag=precision_tag,
                        quantization=quantization,
                    )
                )
    return jobs


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    source_hash = sha256_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {label} must be a JSON object")
    if sha256_file(path) != source_hash:
        raise ValueError(f"{path}: {label} changed while it was read")
    return value, source_hash


def _require_current_file(
    metadata: Any,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(metadata, dict):
        raise ValueError(f"Formal execution preflight lacks {label} metadata")
    path_value = metadata.get("path")
    expected_hash = metadata.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Formal execution preflight lacks {label}.path")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise ValueError(f"Formal execution preflight lacks {label}.sha256")
    path = Path(path_value)
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError(
            f"Formal execution preflight {label} is missing or changed: {path}"
        )
    return path, expected_hash


def _resolve_workspace_path(path: Path) -> Path:
    """Resolve repository-relative artifact paths independently of aliases."""

    return (path if path.is_absolute() else ROOT / path).resolve()


def _jobs_requiring_formal_preflight(jobs: list[Job]) -> list[Job]:
    """Identify formal-artifact consumers from resolved paths, not field names."""

    datasets = _index_configs(ROOT / "configs/data", "dataset")
    known_formal_paths = {
        _resolve_workspace_path(Path(str(dataset["formal_eligible_path"])))
        for _, dataset in datasets.values()
        if isinstance(dataset.get("formal_eligible_path"), str)
        and str(dataset["formal_eligible_path"]).strip()
    }
    ordinary_leantwin_path: Path | None = None
    leantwin = datasets.get("geometry_leantwin")
    if leantwin is not None:
        ordinary_value = leantwin[1].get("eligible_path")
        if isinstance(ordinary_value, str) and ordinary_value.strip():
            ordinary_leantwin_path = _resolve_workspace_path(
                Path(ordinary_value)
            )

    formal_jobs: list[Job] = []
    for job in jobs:
        input_path = _resolve_workspace_path(job.input_path)
        known_formal_artifact = input_path in known_formal_paths
        nonpilot_verified_leantwin = (
            job.dataset_key == "geometry_leantwin"
            and job.input_kind == "verified"
            and (
                ordinary_leantwin_path is None
                or input_path != ordinary_leantwin_path
            )
        )
        if known_formal_artifact or nonpilot_verified_leantwin:
            formal_jobs.append(job)
    return formal_jobs


def enforce_execution_preflight(
    experiment_path: Path,
    jobs: list[Job],
) -> list[Job]:
    """Fail closed before GPU work on a hash-bound formal LeanTwin artifact."""

    experiment = load_yaml(experiment_path)["experiment"]
    configured = experiment.get("execution_preflight")
    if configured is None:
        if _jobs_requiring_formal_preflight(jobs):
            raise ValueError(
                "Experiments consuming the resolved formal eligible artifact "
                "must configure experiment.execution_preflight; changing the "
                "dataset field alias does not bypass this requirement"
            )
        return jobs
    if not isinstance(configured, dict):
        raise ValueError("experiment.execution_preflight must be a mapping")
    design_value = configured.get("formal_design_report")
    eligibility_value = configured.get("formal_eligibility_report")
    merge_value = configured.get("human_validation_merge_report")
    if (
        not isinstance(design_value, str)
        or not design_value
        or not isinstance(eligibility_value, str)
        or not eligibility_value
        or not isinstance(merge_value, str)
        or not merge_value
    ):
        raise ValueError(
            "Formal execution preflight requires design, eligibility, and "
            "human-validation merge reports"
        )
    design_path = Path(design_value)
    eligibility_path = Path(eligibility_value)
    merge_path = Path(merge_value)
    design, design_hash = _read_json_object(
        design_path,
        label="formal design report",
    )
    eligibility, eligibility_hash = _read_json_object(
        eligibility_path,
        label="formal eligibility report",
    )
    merge, merge_hash = _read_json_object(
        merge_path,
        label="human-validation merge report",
    )
    if (
        design.get("schema_version")
        != "cavis.formal_post_adjudication_design.v1"
        or design.get("passes") is not True
        or design.get("ready_for_merge") is not True
    ):
        raise ValueError("Formal design report is not a passing frozen gate")
    design_inputs = design.get("inputs")
    provenance = eligibility.get("provenance")
    if not isinstance(design_inputs, dict) or not isinstance(provenance, dict):
        raise ValueError("Formal execution reports have incomplete provenance")
    if provenance.get("evidence_mode") != "reused_read_only":
        raise ValueError("Formal eligibility must reuse compiler evidence read-only")
    if (
        provenance.get("validator_worktrees_clean") is not True
        or provenance.get("environment_validated_before_and_after") is not True
    ):
        raise ValueError(
            "Formal eligibility lacks before/after clean-validator checks"
        )
    source_path, source_hash = _require_current_file(
        design_inputs.get("eligible_parquet"),
        label="eligible_parquet",
    )
    evidence_path, evidence_hash = _require_current_file(
        design_inputs.get("compiler_evidence"),
        label="compiler_evidence",
    )
    resource_path, resource_hash = _require_current_file(
        design_inputs.get("resource_exclusion"),
        label="resource_exclusion",
    )
    _, observed_resource_hash = verify_resource_exclusion_report(
        report_path=resource_path,
        retained_source_path=source_path,
        retained_evidence_path=evidence_path,
    )
    if observed_resource_hash != resource_hash:
        raise ValueError(
            "Formal resource-exclusion report does not match the frozen design"
        )
    recorded_design_path = provenance.get("evidence_binding_report_path")
    if (
        not isinstance(recorded_design_path, str)
        or Path(recorded_design_path).resolve() != design_path.resolve()
        or provenance.get("evidence_binding_report_sha256") != design_hash
    ):
        raise ValueError("Formal eligibility is not bound to this design report")
    if (
        provenance.get("input_path") is None
        or Path(str(provenance["input_path"])).resolve()
        != source_path.resolve()
        or provenance.get("input_sha256") != source_hash
        or provenance.get("bound_input_sha256") != source_hash
        or provenance.get("evidence_path") is None
        or Path(str(provenance["evidence_path"])).resolve()
        != evidence_path.resolve()
        or provenance.get("evidence_sha256") != evidence_hash
        or provenance.get("bound_evidence_sha256") != evidence_hash
        or provenance.get("formal_resource_exclusion_report_path") is None
        or Path(
            str(provenance["formal_resource_exclusion_report_path"])
        ).resolve()
        != resource_path.resolve()
        or provenance.get("formal_resource_exclusion_report_sha256")
        != resource_hash
    ):
        raise ValueError(
            "Formal eligibility input/evidence does not match the frozen design"
        )
    validations_value = provenance.get("human_validations_path")
    validations_hash = provenance.get("human_validations_sha256")
    if (
        not isinstance(validations_value, str)
        or not validations_value
        or not isinstance(validations_hash, str)
        or not validations_hash
    ):
        raise ValueError("Formal eligibility lacks combined human validations")
    validations_path = Path(validations_value)
    if (
        not validations_path.is_file()
        or sha256_file(validations_path) != validations_hash
    ):
        raise ValueError("Combined human validations are missing or changed")
    merge_inputs = merge.get("inputs")
    merge_output = merge.get("output")
    if (
        merge.get("schema_version") != "cavis.human_validation_merge.v1"
        or not isinstance(merge_inputs, dict)
        or not isinstance(merge_output, dict)
    ):
        raise ValueError("Human-validation merge report is incomplete")
    merge_resource_path, merge_resource_hash = _require_current_file(
        merge_inputs.get("formal_resource_exclusion"),
        label="formal_resource_exclusion",
    )
    if (
        merge_resource_path.resolve() != resource_path.resolve()
        or merge_resource_hash != resource_hash
    ):
        raise ValueError(
            "Human-validation merge is not bound to the exact formal "
            "resource exclusion"
        )
    merge_design = merge_inputs.get("formal_design_report")
    if (
        not isinstance(merge_design, dict)
        or merge_design.get("path") is None
        or Path(str(merge_design["path"])).resolve() != design_path.resolve()
        or merge_design.get("sha256") != design_hash
        or merge_output.get("path") is None
        or Path(str(merge_output["path"])).resolve() != validations_path.resolve()
        or merge_output.get("sha256") != validations_hash
        or provenance.get("human_validation_merge_report_path") is None
        or Path(
            str(provenance["human_validation_merge_report_path"])
        ).resolve()
        != merge_path.resolve()
        or provenance.get("human_validation_merge_report_sha256") != merge_hash
        or provenance.get("bound_human_validations_sha256") != validations_hash
    ):
        raise ValueError(
            "Formal eligibility is not bound to the exact verified "
            "human-validation merge"
        )

    formal_jobs = _jobs_requiring_formal_preflight(jobs)
    if not formal_jobs:
        raise ValueError(
            "Configured formal execution preflight has no verified LeanTwin job"
        )
    output_value = provenance.get("output_path")
    output_hash = provenance.get("output_sha256")
    if not isinstance(output_value, str) or not isinstance(output_hash, str):
        raise ValueError("Formal eligibility output provenance is incomplete")
    formal_output = Path(output_value)
    if (
        not formal_output.is_file()
        or sha256_file(formal_output) != output_hash
        or any(
            job.input_path.resolve() != formal_output.resolve()
            for job in formal_jobs
        )
    ):
        raise ValueError(
            "GPU jobs do not consume the exact hash-bound formal eligible Parquet"
        )
    if (
        sha256_file(design_path) != design_hash
        or sha256_file(eligibility_path) != eligibility_hash
        or sha256_file(merge_path) != merge_hash
        or sha256_file(resource_path) != resource_hash
    ):
        raise ValueError("A formal preflight report changed during verification")

    preflight = tuple(
        sorted(
            {
                str(design_path): design_hash,
                str(eligibility_path): eligibility_hash,
                str(merge_path): merge_hash,
                str(source_path): source_hash,
                str(evidence_path): evidence_hash,
                str(resource_path): resource_hash,
                str(validations_path): validations_hash,
                str(formal_output): output_hash,
            }.items()
        )
    )
    return [replace(job, execution_preflight=preflight) for job in jobs]


def _run_device_queue(device: str, jobs: list[Job], *, resume: bool) -> list[str]:
    completed: list[str] = []
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for job in jobs:
        for path_value, expected_hash in job.execution_preflight:
            path = Path(path_value)
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(
                    f"Execution preflight artifact changed before GPU job: {path}"
                )
        environment["PYTHONHASHSEED"] = str(job.seed)
        score_path = job.output_path / "scores.jsonl"
        signature_path = job.output_path / "job_signature.json"
        run_manifest_path = job.output_path / "run_manifest.json"
        if resume and score_path.is_file():
            if not signature_path.is_file():
                raise RuntimeError(
                    f"Existing scores lack a job signature: {score_path}. "
                    "Pass --no-resume to replace them intentionally."
                )
            existing = json.loads(signature_path.read_text(encoding="utf-8"))
            expected = _signature_payload(job)
            if existing != expected:
                raise RuntimeError(
                    f"Existing scores have a stale signature: {score_path}. "
                    "Pass --no-resume to replace them intentionally."
                )
            if not run_manifest_path.is_file():
                raise RuntimeError(
                    f"Existing scores lack an output manifest: {score_path}. "
                    "Pass --no-resume to replace them intentionally."
                )
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if (
                run_manifest.get("status") != "completed"
                or run_manifest.get("scores_sha256") != sha256_file(score_path)
                or run_manifest.get("input_sha256") != expected["input_sha256"]
                or run_manifest.get("model_revision")
                != load_yaml(job.model_config)["model"]["revision"]
            ):
                raise RuntimeError(
                    f"Existing scores fail output-integrity checks: {score_path}. "
                    "Pass --no-resume to replace them intentionally."
                )
            completed.append(
                f"SKIP {job.experiment_key}/{job.dataset_key}/{job.model_key}"
            )
            continue
        if not job.input_path.is_file():
            raise FileNotFoundError(
                f"{job.input_kind.title()} data missing: {job.input_path}. "
                "Run the documented data/Lean verification target first."
            )
        signature = _signature_payload(job)
        subprocess.run(job.command(), check=True, env=environment)
        if not run_manifest_path.is_file():
            raise RuntimeError(
                f"Extractor completed without an output manifest: {run_manifest_path}"
            )
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if (
            run_manifest.get("status") != "completed"
            or run_manifest.get("scores_sha256") != sha256_file(score_path)
            or run_manifest.get("input_sha256") != signature["input_sha256"]
        ):
            raise RuntimeError(f"Extractor output failed integrity checks: {score_path}")
        atomic_write_json(signature_path, signature)
        completed.append(
            f"DONE {job.experiment_key}/{job.dataset_key}/{job.model_key}"
        )
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or execute a deterministic model×dataset extraction sweep."
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--compute",
        type=Path,
        default=Path("configs/compute/workstation.yaml"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/runs/extractions"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    jobs = build_jobs(args.experiment, args.compute, result_root=args.result_root)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if args.execute:
        jobs = enforce_execution_preflight(args.experiment, jobs)
    for job in jobs:
        command = " ".join(shlex.quote(part) for part in job.command())
        print(f"CUDA_VISIBLE_DEVICES={job.physical_device} {command}")
    if not args.execute:
        print(f"Dry run: {len(jobs)} jobs. Pass --execute to launch.")
        return

    queues: dict[str, list[Job]] = {}
    for job in jobs:
        queues.setdefault(job.physical_device, []).append(job)
    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = {
            executor.submit(
                _run_device_queue,
                device,
                queue,
                resume=not args.no_resume,
            ): device
            for device, queue in queues.items()
        }
        for future in as_completed(futures):
            device = futures[future]
            for message in future.result():
                print(f"[GPU {device}] {message}")


if __name__ == "__main__":
    main()

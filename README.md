# CAVIS

CAVIS is a reproducible audit framework for internal reasoning-verifier scores. It evaluates observational discrimination, stability under validity-preserving transformations, directional response to matched validity-changing edits, and conformal stability-or-abstention certificates.

This is the Full implementation of the paper ** Reasoning or Rendering? Transformation-Specific Robustness Certificates for Internal Reasoning Verifiers **

## Quick start

Python 3.10--3.12 and `uv` are supported.

```bash
uv sync --frozen --extra dev --extra plots
make test
make reproduce
```

`make reproduce` is CPU-only. It validates the pinned manifest, runs the test suite, and rebuilds the experiment matrix, tables, and figures from the released compact outputs. Generated files are written to:

```text
artifacts/figures/
artifacts/tables/
results/tables/
results/reproduction_report.json
```

## Released evaluation inputs

The compact records needed for CPU reconstruction are located under:

```text
results/runs/evaluations/formal_audit/
results/runs/evaluations/external_audit/
results/runs/evaluations/diagnostics/
results/runs/geometry_cpu_audit/
results/runs/toy/
```

Each executed extraction also has a sanitized environment receipt and run manifest. Raw score arrays are omitted because they are substantially larger; the full GPU route below reconstructs them.

## Data preparation

All remote resources and model revisions are fixed in `data_manifest.lock`.
To download and prepare the public datasets:

```bash
uv sync --frozen --extra dev --extra plots --extra hf
make data
make data-check
```

Some model repositories require prior access approval and a local Hugging Face token. No credential is stored in this repository.

The Lean validation environment is pinned separately:

```bash
make lean-environment
```

This command requires `elan` and installs the specified Lean 3 toolchain.

## GPU execution

Install the inference dependencies:

```bash
make install-gpu
```

All sweep targets are dry runs by default. Set `EXECUTE=1` to launch them:

```bash
make pilot-candidates EXECUTE=1
make formal-audit EXECUTE=1
make external-audit EXECUTE=1
make ablations EXECUTE=1
```

The workstation mapping is defined in `configs/compute/workstation.yaml`. Override `CUDA_VISIBLE_DEVICES` or the compute configuration when using a different machine.

After extraction, run:

```bash
make evaluate-formal
make evaluate-external
make evaluate-ablations
make diagnostics
make reproduce
```

## Determinism and provenance

- Dataset partitions are grouped and seeded with `17`, `42`, and `97`.
- Model and dataset revisions are immutable entries in `data_manifest.lock`.
- Every extraction stores its resolved configuration, environment, runtime, revision, and input hashes.
- Evaluation choices are selected on train/calibration groups and frozen before test evaluation.
- Formal pairs are hash-bound to compiler evidence and consensus validation.
- Tables and figures are generated from CSV/JSON records, never transcribed by the rendering scripts.

The independent validation procedure is documented in `docs/HUMAN_VALIDATION.md`.

CUDA kernels and quantized inference can still vary across driver or hardware versions. Environment receipts make such differences visible.

## Main commands

```text
make help                 list supported targets
make install              install CPU and development dependencies
make install-gpu          install inference dependencies
make test                 validate the manifest and run unit tests
make lint                 run static checks
make data                 download and prepare public datasets
make public-audit         rebuild the public Geometry audit
make formal-audit         plan or execute the seven-model formal sweep
make external-audit       plan or execute the external transfer sweep
make evaluate-formal      evaluate frozen formal score caches
make evaluate-external    evaluate frozen external score caches
make tables               rebuild deterministic tables
make figures              rebuild deterministic figures
make reproduce            rebuild all CPU outputs
make release-check        validate anonymity and file integrity
```

## Repository layout

```text
configs/                  fixed data, model, compute, and experiment settings
src/cavis/                reusable library modules
scripts/                  command-line entry points
tests/                    deterministic unit and integration tests
data/cache/               compact formal evidence included in this release
results/runs/             compact evaluation records and run receipts
artifacts/specs/          fixed rendering specifications
artifacts/tables/         generated CSV and LaTeX tables
artifacts/figures/        generated PDF/PNG figures and editable diagram source
```

## Evaluation scope

Formal CAVIS metrics apply only to transformations that satisfy the released eligibility and provenance protocol. External benchmark rows provide observational transfer measurements because those datasets do not contain audited transformation pairs. Conformal guarantees are marginal over a new
exchangeable dependence group and do not imply conditional validity after selection or non-abstention.

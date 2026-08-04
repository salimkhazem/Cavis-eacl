UV ?= uv
PY := $(UV) run python
EXECUTE ?= 0
EXEC_FLAG := $(if $(filter 1 true yes,$(EXECUTE)),--execute,)

GEOMETRY_JSON := data/external/geometry_of_reason/data/results/rebuttal/llama8b_full_extraction.json
GEOMETRY_SHA256 := b8a69d6250dbd624df1c38eb2ee536d8502c15b29b37bce188f9b9be32796475
FORMAL_INPUT := data/cache/prepared/geometry_leantwin.formal.eligible.parquet
FORMAL_EXCLUSION := data/cache/geometry_leantwin/formal_dependence_exclusion.json
FORMAL_DESIGN := data/cache/geometry_leantwin/formal_design_report.json
FORMAL_ELIGIBILITY := data/cache/geometry_leantwin/formal_eligibility_report.json
FORMAL_MERGE := data/cache/geometry_leantwin/combined_human_validations_report.json
FORMAL_ALPHA ?= 0.10
FORMAL_CALIBRATION_SIZE ?= 15
LEAN_CWD ?= data/external/minif2f_lean3_environment
LEAN_TOOLCHAIN ?= leanprover-community/lean:3.42.1

.PHONY: help install install-gpu test lint data data-check data-leantwin \
	lean-environment public-audit pilot-candidates formal-audit external-audit \
	ablations evaluate-formal evaluate-external evaluate-ablations diagnostics \
	matrix tables figures reproduce release-hashes release-check

help:
	@echo "CAVIS reproducibility targets"
	@echo "  make install              install CPU, plotting, and test dependencies"
	@echo "  make install-gpu          install inference dependencies"
	@echo "  make test                 validate configuration and run CPU tests"
	@echo "  make data                 download and prepare pinned public datasets"
	@echo "  make public-audit         rebuild the public Geometry audit"
	@echo "  make formal-audit         plan the formal sweep; add EXECUTE=1 to run"
	@echo "  make external-audit       plan the transfer sweep; add EXECUTE=1 to run"
	@echo "  make evaluate-formal      evaluate formal score caches"
	@echo "  make evaluate-external    evaluate external score caches"
	@echo "  make tables               rebuild generated tables"
	@echo "  make figures              rebuild generated figures"
	@echo "  make reproduce            rebuild all CPU outputs"
	@echo "  make release-check        validate anonymity and file integrity"

install:
	$(UV) sync --frozen --extra dev --extra plots

install-gpu:
	$(UV) sync --frozen --extra dev --extra plots --extra hf --extra quantization

test:
	$(PY) -m scripts.validate_manifest
	$(UV) run pytest -q

lint:
	$(UV) run ruff check src scripts tests

data:
	$(PY) -m scripts.download_data
	$(PY) -m scripts.prepare_data --build-leantwin \
		--output-manifest results/data_preparation_manifest.json

data-check:
	$(PY) -m scripts.download_data --check
	$(PY) -m scripts.validate_manifest

data-leantwin:
	$(PY) -m scripts.download_data --only geometry_of_reason
	$(PY) -m scripts.prepare_data --only geometry_leantwin --build-leantwin --force

lean-environment:
	$(PY) -m scripts.download_data --only minif2f_lean3_environment
	@command -v elan >/dev/null 2>&1 || \
		(echo "elan is required for Lean validation" >&2; exit 2)
	@elan run "$(LEAN_TOOLCHAIN)" lean --version >/dev/null 2>&1 || \
		elan toolchain install "$(LEAN_TOOLCHAIN)"
	cd "$(LEAN_CWD)" && \
		elan run "$(LEAN_TOOLCHAIN)" leanpkg configure && \
		elan run "$(LEAN_TOOLCHAIN)" leanpkg build

public-audit:
	$(PY) -m scripts.download_data --only geometry_of_reason
	$(PY) -m scripts.audit_geometry \
		--input "$(GEOMETRY_JSON)" \
		--output results/runs/geometry_cpu_audit \
		--expected-sha256 "$(GEOMETRY_SHA256)"

pilot-candidates:
	$(PY) -m scripts.run_sweep \
		--experiment configs/experiment/pilot_candidates.yaml $(EXEC_FLAG)

formal-audit:
	$(PY) -m scripts.run_sweep \
		--experiment configs/experiment/formal_audit.yaml $(EXEC_FLAG)

external-audit:
	$(PY) -m scripts.run_sweep \
		--experiment configs/experiment/external_audit.yaml $(EXEC_FLAG)

ablations:
	$(PY) -m scripts.run_sweep \
		--experiment configs/experiment/ablations.yaml $(EXEC_FLAG)

evaluate-formal:
	$(PY) -m scripts.evaluate_sweep --experiment formal_audit \
		--dependence-input "$(FORMAL_INPUT)" \
		--exclude-selection "$(FORMAL_EXCLUSION)" \
		--formal-design-report "$(FORMAL_DESIGN)" \
		--formal-eligibility-report "$(FORMAL_ELIGIBILITY)" \
		--formal-merge-report "$(FORMAL_MERGE)" \
		--alpha "$(FORMAL_ALPHA)" \
		--calibration-size "$(FORMAL_CALIBRATION_SIZE)"

evaluate-external:
	$(PY) -m scripts.evaluate_sweep --experiment external_audit

evaluate-ablations:
	$(PY) -m scripts.run_ablations \
		--dependence-input "$(FORMAL_INPUT)" \
		--exclude-selection "$(FORMAL_EXCLUSION)" \
		--formal-design-report "$(FORMAL_DESIGN)" \
		--formal-eligibility-report "$(FORMAL_ELIGIBILITY)" \
		--formal-merge-report "$(FORMAL_MERGE)"

diagnostics:
	$(PY) -m scripts.analyze_formal_results

matrix:
	$(PY) -m scripts.make_matrix

tables:
	$(PY) -m scripts.make_tables

figures:
	$(PY) -m scripts.make_figures

reproduce:
	$(PY) -m scripts.reproduce

release-hashes:
	$(PY) -m scripts.verify_release --write

release-check:
	$(PY) -m scripts.verify_release

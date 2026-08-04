from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cavis.data import (
    assign_grouped_splits,
    deterministic_sample,
    load_geometry,
    load_math_shepherd,
    load_prm800k,
    load_processbench,
    proportional_stratified_sample,
)
from cavis.data.download import load_manifest
from cavis.data.io import read_json, sha256_text
from cavis.data.preparation import (
    materialize_leantwin_candidates,
    prepare_dataset,
)
from cavis.data.records import ReasoningExample


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_grouped_splits_are_order_invariant_and_leak_free() -> None:
    groups = [f"theorem-{index // 3}" for index in range(90)]
    forward = assign_grouped_splits(groups, seed=17)
    backward = assign_grouped_splits(reversed(groups), seed=17)
    assert forward == backward
    assert set(forward) == set(groups)
    assert set(forward.values()).issubset({"train", "calibration", "test"})
    row_splits = [forward[group] for group in groups]
    for group in set(groups):
        assert (
            len(
                {
                    split
                    for split, value in zip(row_splits, groups, strict=True)
                    if value == group
                }
            )
            == 1
        )


@dataclass(frozen=True)
class _Sample:
    name: str
    label: bool


def test_deterministic_stratified_sample_is_order_invariant() -> None:
    values = [_Sample(str(index), bool(index % 2)) for index in range(20)]
    first = deterministic_sample(
        values, 7, seed=97, key=lambda row: row.name, strata=lambda row: row.label
    )
    second = deterministic_sample(
        list(reversed(values)),
        7,
        seed=97,
        key=lambda row: row.name,
        strata=lambda row: row.label,
    )
    assert first == second
    counts = {label: sum(row.label == label for row in first) for label in (False, True)}
    assert abs(counts[False] - counts[True]) <= 1


def test_proportional_stratified_sample_preserves_prevalence() -> None:
    values = [
        _Sample(str(index), index >= 80)
        for index in range(100)
    ]
    selected = proportional_stratified_sample(
        values,
        25,
        seed=17,
        key=lambda row: row.name,
        strata=lambda row: row.label,
    )
    assert len(selected) == 25
    assert sum(row.label for row in selected) == 5


def test_processbench_adapter_preserves_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "gsm8k.json"
    _write_json(
        path,
        [
            {
                "id": "gsm8k-0",
                "generator": "model",
                "problem": "1+1?",
                "steps": ["one", "bad", "downstream"],
                "final_answer_correct": False,
                "label": 1,
            },
            {
                "id": "gsm8k-1",
                "generator": "model",
                "problem": "2+2?",
                "steps": ["four"],
                "final_answer_correct": True,
                "label": -1,
            },
        ],
    )
    rows = load_processbench(path, subset="gsm8k")
    assert rows[0].step_labels == (True, False, None)
    assert rows[0].valid is False
    assert rows[1].step_labels == (True,)
    assert rows[1].valid is True
    assert rows[0].source_hash != rows[1].source_hash
    assert rows[0].dependence_id == "processbench_gsm8k::group::gsm8k-0"


def test_geometry_adapter_groups_generated_variant(tmp_path: Path) -> None:
    extraction = tmp_path / "geometry.json"
    _write_json(
        extraction,
        [
            {
                "file": "foo.lean",
                "label_original": "valid",
                "label_corrected": "valid",
                "is_valid": True,
                "proof_token_count": 10,
                "spectral": {"layer_0": {"hfer": 0.2}},
            },
            {
                "file": "foo_1.lean",
                "label_original": "invalid",
                "label_corrected": "invalid",
                "is_valid": False,
                "proof_token_count": 11,
                "spectral": {"layer_0": {"hfer": 0.3}},
            },
        ],
    )
    proof_root = tmp_path / "proofs"
    (proof_root / "valid").mkdir(parents=True)
    (proof_root / "invalid").mkdir(parents=True)
    (proof_root / "valid" / "foo.lean").write_text(
        "theorem foo : True := by trivial\n", encoding="utf-8"
    )
    (proof_root / "invalid" / "foo_1.lean").write_text(
        "theorem foo : True := by contradiction\n", encoding="utf-8"
    )
    rows = load_geometry(extraction, proof_root=proof_root)
    assert [row.group_id for row in rows] == ["foo", "foo"]
    assert rows[0].dependence_id == rows[1].dependence_id
    assert rows[0].dependence_id.startswith(
        "geometry_minif2f::lean_statement::"
    )
    assert [row.valid for row in rows] == [True, False]
    assert rows[0].steps[0].startswith("theorem")


def test_prm800k_adapter_does_not_select_counterfactual_completion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "phase2_test.jsonl"
    row = {
        "generation": 4,
        "is_quality_control_question": False,
        "is_initial_screening_question": False,
        "question": {
            "problem": "p",
            "pre_generated_steps": ["actual first", "actual bad", "unlabeled"],
            "ground_truth_answer": "x",
            "pre_generated_answer": "y",
        },
        "label": {
            "steps": [
                {
                    "completions": [{"text": "actual first", "rating": 1}],
                    "chosen_completion": 0,
                },
                {
                    "completions": [
                        {"text": "actual bad", "rating": -1},
                        {"text": "counterfactual repair", "rating": 1},
                    ],
                    "chosen_completion": None,
                },
            ],
            "finish_reason": "found_error",
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    parsed = load_prm800k(path)
    assert parsed[0].step_labels == (True, False, None)
    assert parsed[0].valid is False


def test_math_shepherd_json_adapter(tmp_path: Path) -> None:
    path = tmp_path / "math.json"
    _write_json(
        path,
        [
            {
                "prompt": "question",
                "completions": ["ok", "wrong"],
                "labels": [True, False],
            }
        ],
    )
    row = load_math_shepherd(path)[0]
    assert row.step_labels == (True, False)
    assert row.valid is False
    assert row.dataset == "math_shepherd"


def test_unsafe_serialization_is_refused(tmp_path: Path) -> None:
    unsafe = tmp_path / "payload.pkl"
    unsafe.write_bytes(b"not actually a pickle")
    with pytest.raises(ValueError, match="unsafe"):
        read_json(unsafe)


def test_manifest_rejects_mutable_revisions(tmp_path: Path) -> None:
    path = tmp_path / "data_manifest.lock"
    _write_json(
        path,
        {
            "artifacts": {
                "dataset": {
                    "kind": "huggingface_dataset",
                    "repo_id": "org/repo",
                    "revision": "main",
                    "destination": "data/raw/repo",
                }
            }
        },
    )
    with pytest.raises(ValueError, match="immutable"):
        load_manifest(path)


def test_manifest_accepts_full_revision(tmp_path: Path) -> None:
    path = tmp_path / "data_manifest.lock"
    revision = "a" * 40
    _write_json(
        path,
        {
            "artifacts": {
                "dataset": {
                    "kind": "huggingface_dataset",
                    "repo_id": "org/repo",
                    "revision": revision,
                    "destination": "data/raw/repo",
                    "include": ["*.json"],
                }
            }
        },
    )
    spec = load_manifest(path)[0]
    assert spec.revision == revision
    assert spec.include == ("*.json",)


def test_prepare_processbench_writes_canonical_parquet_and_uses_cache(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "data" / "external" / "processbench"
    source_root.mkdir(parents=True)
    _write_json(
        source_root / "gsm8k.json",
        [
            {
                "id": f"gsm8k-{index}",
                "problem": f"problem {index}",
                "steps": ["correct"] if index % 2 else ["error"],
                "label": -1 if index % 2 else 0,
                "generator": "model",
                "final_answer_correct": bool(index % 2),
            }
            for index in range(10)
        ],
    )
    manifest = tmp_path / "data_manifest.lock"
    _write_json(
        manifest,
        {
            "sources": {
                "processbench": {
                    "kind": "huggingface_dataset",
                    "repo_id": "Qwen/ProcessBench",
                    "revision": "a" * 40,
                    "destination": "data/external/processbench",
                }
            }
        },
    )
    config = tmp_path / "configs" / "data" / "processbench_gsm8k.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        """dataset:
  key: processbench_gsm8k
  source: processbench
  subset: gsm8k
  split:
    seeds: [17, 42, 97]
  prepared_path: data/cache/prepared/processbench_gsm8k.parquet
""",
        encoding="utf-8",
    )
    first = prepare_dataset(
        config,
        manifest_path=manifest,
        project_root=tmp_path,
        build_leantwin=False,
    )
    second = prepare_dataset(
        config,
        manifest_path=manifest,
        project_root=tmp_path,
        build_leantwin=False,
    )
    assert first.record_count == 10
    assert first.cached is False
    assert second.cached is True
    assert first.output_sha256 == second.output_sha256
    import pyarrow.parquet as pq

    rows = pq.read_table(tmp_path / first.output_path).to_pylist()
    required = {
        "item_id",
        "dataset",
        "prompt",
        "reasoning",
        "label",
        "transformation_id",
        "group_id",
        "source_group_id",
        "dependence_id",
        "pair_id",
        "metadata_json",
        "split_s17",
    }
    assert required.issubset(rows[0])
    assert {row["dataset"] for row in rows} == {"processbench_gsm8k"}
    assert all(
        row["dependence_id"].startswith("processbench_gsm8k::group::")
        for row in rows
    )


def test_leantwin_candidates_remain_explicitly_unverified() -> None:
    source = """theorem demo (x : Nat) (h : x = 2) : x + 1 = 3 :=
by
  omega
"""
    example = ReasoningExample(
        item_id="demo",
        group_id="demo",
        dependence_id="geometry_minif2f::lean_statement::demo",
        dataset="geometry_leantwin",
        problem="",
        steps=(source,),
        step_labels=(True,),
        valid=True,
        source_hash="a" * 64,
        metadata={"proof_path": "/absolute/machine-specific/proof.lean"},
    )
    rows, counts, skipped = materialize_leantwin_candidates(
        [example],
        split_seeds=(17, 42, 97),
        transform_seed=17,
        transform_names=(
            "whitespace",
            "comments",
            "declaration_rename",
            "local_alpha_rename",
            "comparison_flip",
            "arithmetic_operator_flip",
            "premise_mutation",
        ),
    )
    assert len(rows) == 20
    assert not skipped
    assert sum(counts.values()) == 19
    assert "/absolute/" not in rows[0]["metadata_json"]
    for row in rows:
        assert row["target_hash"] == sha256_text(row["reasoning"])
        assert (
            row["dependence_id"]
            == "geometry_minif2f::lean_statement::demo"
        )
        assert row["source_group_id"] == "demo"
        assert row["upstream_source_hash"] == "a" * 64
        assert row["semantic_status"] == "not_established"
        assert row["evidence_state"] == "unverified"
        assert row["mechanically_verified"] is False
        assert row["split_s17"] == rows[0]["split_s17"]
        assert row["semantic_variant_id"]
        assert "parent_variant_id" in row
        assert row["transform_kind"] in {"base", "g0", "g1"}
    g1_roots = [row for row in rows if row["transform_kind"] == "g1"]
    assert len(g1_roots) == 3
    for root in g1_roots:
        descendants = [
            row
            for row in rows
            if row["pair_id"] == root["pair_id"]
            and row["transform_kind"] == "g0"
            and row["pair_side"] == "negative"
        ]
        assert len(descendants) == 4
        assert all(
            row["semantic_variant_id"] == root["semantic_variant_id"]
            and row["parent_variant_id"] == root["item_id"]
            for row in descendants
        )
    positive_g0 = [
        row
        for row in rows
        if row["transform_kind"] == "g0" and row["pair_side"] == "positive"
    ]
    assert len(positive_g0) == 4
    assert all(len(json.loads(row["pair_ids_json"])) == 3 for row in positive_g0)

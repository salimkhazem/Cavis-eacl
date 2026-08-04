# LeanTwin paired-validity review

Compiler rejection is necessary evidence for a G1 candidate but does not prove that the edited item is a valid matched negative. A human reviewer must inspect the exact positive/negative pair and provide one JSON object per line:

```json
{
  "pair_id": "theorem-id::comparison_flip::s17",
  "decision": "approved",
  "policy": "paired_process_validity_v1",
  "positive_hash": "64-lowercase-hex-characters",
  "negative_hash": "64-lowercase-hex-characters",
  "reviewed_statement_change": true,
  "validator": "reviewer-pseudonym",
  "notes": "Why the edit creates the intended invalid process while retaining a meaningful matched pair."
}
```

The eligibility join rejects missing fields, empty reviewer notes, duplicate `pair_id` values, non-approved decisions, and any hash mismatch. The review file itself is hashed into `eligibility_report.json`. Reviewer identities may be stable pseudonyms in an anonymous artifact; the camera-ready release should state the review protocol and adjudication procedure.

Recommended procedure:

1. Two reviewers independently inspect each candidate without model scores.
2. Run `make adjudicate-pilot-reviews` on the two complete, independent ledgers. The command preserves the source files, canonicalizes only the accepted decision aliases in its outputs, and hashes every input.
3. Disagreements are written to the exclusion ledger with reason `reviewer_disagreement`; the report is marked `ready_for_lean=false` and the command exits nonzero.
4. Disagreements are adjudicated before any confirmatory score analysis. Rerun the command and proceed only when the report says `ready_for_lean=true`.
5. Record exclusions with a typed reason in a separate ledger; never silently drop a failed or ambiguous pair.
6. Freeze and hash the final JSONL before running `make pilot EXECUTE=1`.

The confirmatory review is a distinct second stage. Run `make review-formal`; its `prepare-formal-evidence` prerequisite compiles on CPU into a raw `cavis.lean_evidence.v2` ledger and then creates a retained compiler ledger plus formal review-source Parquet. After the policy freeze, at most one new
attempt is scheduled for each non-terminal target under the fixed timeout budget. The v2 ledger does not encode historical attempt counts, and prior exploratory attempts may exist; a compatible terminal timeout is reused without retry. A timeout remains an execution error, never a Lean rejection
or an invalid label; its complete canonical dependence (including aliases and siblings) is removed before selection. The separate `lean_resource_exclusion.json` binds the budget, raw and retained hashes, closure, attrition, environment, and implementation. Archive it with the review artifacts. Non-timeout errors remain fatal, and a compatible rerun does not retry a terminal timeout. Legacy compiler evidence is neither reused nor overwritten. Selection uses the normalized Lean statement `dependence_id`; `group_id` is traceability metadata only. Two reviewers then use `make review-formal-interactive`, followed by `make adjudicate-formal-reviews`. Each reviewer file must start as an independent copy of the untouched pending queue.

Do not replace the pilot approval ledger with the formal ledger. Run `make validate-formal-design` first. During score-blind construction, the original exact split-by-family equalities were found infeasible before formal review, GPU inference, or score access. The amended blinded design retains
hard quotas of 120 pairs, 40 per transformation, and one pair per canonical dependence. Its split cells are selected by a frozen minimum-L1 MILP with at least seven calibration and seven test items per transformation and seed; the exact achieved cells are recorded rather than stated as a priori equalities. The
gate records human-review attrition and requires at least 15 unanimously approved calibration and 15 test dependencies for each of seeds 17, 42, and 97, and at least three calibration and three test approvals per G1 family under every seed. It inspects no model scores. If it fails, do not run the GPU sweep or lower the threshold post hoc.

Then run `make merge-human-validations`, which fails on duplicate or conflicting pair IDs, pairs outside the frozen formal selection, or any hash mismatch among the queue, selection, adjudication, approved ledger and design report. Finally run `make verify-formal-leantwin`. Archive
the merge report, formal eligibility report and eligible-Parquet hash. The join verifies that the combined ledger is exactly the sorted union of the frozen pilot and unanimously approved formal ledgers, and binds that ledger and merge-report SHA-256 into the eligibility report. The formal join is
written to `geometry_leantwin.formal.eligible.parquet` and `formal_eligibility_report.json`, while the pilot `geometry_leantwin.eligible.parquet` and `eligibility_report.json` remain immutable. The join reuses `lean_evidence_v2.formal.jsonl` without compiling or rewriting it only after its path and SHA-256 have been bound by the passing formal-design report. The final join reads the same hash-bound formal review-source Parquet. Incomplete, changed or provenance-mismatched evidence fails closed. Every v2 row records the exact mathlib commit and clean-worktree status. The validator independently checks both and rejects modifications or untracked files. Only the `formal_audit` and `ablations` configs route extraction to the formal artifact, and their evaluation removes all pilot `dependence_id` values before fitting.

The implementation enforces the machine-checkable fields but cannot enforce reviewer independence; that remains a documented study-protocol obligation. Both the Make target and the Python sweep launcher fail closed on missing or changed gate artifacts before formal GPU execution.
The queue/selection, post-review design report, and validation-merge report also record and revalidate their Git HEAD plus exact implementation-file SHA-256 values. Unrelated untracked result caches do not invalidate this fingerprint; tracked source changes do.

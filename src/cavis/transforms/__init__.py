"""Deterministic Lean transformation families used by LeanTwin."""

from .base import TransformResult, TransformSpec
from .contracts import (
    ContractCheck,
    check_g0_contract,
    declaration_statement_changed,
)
from .eligibility import (
    LeanCompileEvidence,
    join_cavis_eligibility,
)
from .g0 import (
    AlphaRenameTransform,
    CommentTransform,
    TheoremRenameTransform,
    WhitespaceTransform,
)
from .g1 import (
    ArithmeticOperatorCorruption,
    NumericLiteralCorruption,
    PremiseMutationCorruption,
    ProofStepDeletionCorruption,
    RelationFlipCorruption,
)
from .verify import (
    VerificationEvidence,
    to_core_transform_evidence,
    verify_with_command,
)

TRANSFORM_REGISTRY = {
    "whitespace": WhitespaceTransform,
    "comments": CommentTransform,
    "declaration_rename": TheoremRenameTransform,
    "local_alpha_rename": AlphaRenameTransform,
    "comparison_flip": RelationFlipCorruption,
    "arithmetic_operator_flip": ArithmeticOperatorCorruption,
    "premise_mutation": PremiseMutationCorruption,
    # Backward-compatible aliases used in early exploratory configs.
    "theorem_rename": TheoremRenameTransform,
    "alpha_rename": AlphaRenameTransform,
    "relation_flip": RelationFlipCorruption,
    "numeric_literal": NumericLiteralCorruption,
    "proof_step_deletion": ProofStepDeletionCorruption,
}


def make_transform(name: str):
    """Instantiate a configured transform by stable name."""

    try:
        constructor = TRANSFORM_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(TRANSFORM_REGISTRY))
        raise ValueError(f"Unknown transform {name!r}; choose one of: {choices}") from exc
    return constructor()

__all__ = [
    "AlphaRenameTransform",
    "ArithmeticOperatorCorruption",
    "CommentTransform",
    "ContractCheck",
    "LeanCompileEvidence",
    "NumericLiteralCorruption",
    "PremiseMutationCorruption",
    "ProofStepDeletionCorruption",
    "RelationFlipCorruption",
    "TheoremRenameTransform",
    "TransformResult",
    "TransformSpec",
    "VerificationEvidence",
    "WhitespaceTransform",
    "check_g0_contract",
    "declaration_statement_changed",
    "join_cavis_eligibility",
    "make_transform",
    "to_core_transform_evidence",
    "verify_with_command",
]

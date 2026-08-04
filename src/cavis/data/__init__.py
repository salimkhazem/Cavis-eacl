"""Dataset adapters and deterministic split utilities for CAVIS.

The adapters deliberately return a small, model-agnostic record instead of a
``datasets.Dataset``.  This keeps audit inputs inspectable and makes it
possible to hash every source row before model code is involved.
"""

from .adapters import (
    load_geometry,
    load_math_shepherd,
    load_prm800k,
    load_processbench,
)
from .dependence import (
    DependenceDiagnostics,
    LeanStatementIdentity,
    enrich_prepared_dependence_ids,
    geometry_statement_identity,
    scoped_group_dependence_id,
)
from .records import ReasoningExample
from .splits import (
    assign_grouped_splits,
    deterministic_sample,
    proportional_stratified_sample,
    stable_hash,
)

__all__ = [
    "ReasoningExample",
    "DependenceDiagnostics",
    "LeanStatementIdentity",
    "assign_grouped_splits",
    "deterministic_sample",
    "enrich_prepared_dependence_ids",
    "geometry_statement_identity",
    "load_geometry",
    "load_math_shepherd",
    "load_prm800k",
    "load_processbench",
    "proportional_stratified_sample",
    "scoped_group_dependence_id",
    "stable_hash",
]

"""Tie-aware Retrieval Metrics (TRM).

A lightweight evaluation library for retrieval systems that properly handles
tied relevance scores. Reports expected metric values, score range, and bias
to quantify ordering uncertainty among tied candidates.

Reference
---------
Kisu Yang, Yoonna Jang, Hwanseok Jang, Kenneth Choi,
Isabelle Augenstein, Heuiseok Lim.
"Reliable Evaluation Protocol for Low-Precision Retrieval." ACL 2026.
"""

from .evaluator import (
    DEFAULT_K_LIST,
    SUPPORTED_METRICS,
    EvaluationOutput,
    TieAwareResult,
    evaluate,
)
from .metrics import build_tie_groups

__version__ = "0.1.0"

__all__ = [
    "evaluate",
    "build_tie_groups",
    "EvaluationOutput",
    "TieAwareResult",
    "SUPPORTED_METRICS",
    "DEFAULT_K_LIST",
    "__version__",
]

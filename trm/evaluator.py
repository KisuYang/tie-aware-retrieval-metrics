"""Tie-aware Retrieval Metric (TRM) evaluator.

Provides a single entry-point :func:`evaluate` that computes expected values,
extrema (max/min), range, bias, and tie-oblivious scores for standard
retrieval metrics under tied relevance scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .metrics import (
    EXPECTED_METRIC_FNS,
    POINT_METRIC_FNS,
    build_tie_groups,
    _extrema_preds,
    _oblivious_preds,
)


SUPPORTED_METRICS = list(EXPECTED_METRIC_FNS.keys())
DEFAULT_K_LIST = [1, 3, 5, 10, 20, 50, 100]


@dataclass
class TieAwareResult:
    """Container for tie-aware evaluation results at a single cutoff."""

    expected: float
    maximum: float
    minimum: float
    oblivious: float

    @property
    def range(self) -> float:
        """Score range: ``M_max - M_min``  (Equation 4)."""
        return self.maximum - self.minimum

    @property
    def bias(self) -> float:
        """Score bias: ``M_obl - E[M]``  (Equation 5)."""
        return self.oblivious - self.expected


@dataclass
class EvaluationOutput:
    """Aggregated evaluation results over multiple queries.

    Attributes
    ----------
    metrics : dict
        ``metrics[metric_name][k]`` returns a :class:`TieAwareResult` with
        the macro-averaged expected, max, min, oblivious, range, and bias.
    per_query : dict
        ``per_query[metric_name][k]`` returns a list of per-query
        :class:`TieAwareResult` objects.
    k_list : list of int
        The cutoff values used for evaluation.
    """

    metrics: Dict[str, Dict[int, TieAwareResult]] = field(default_factory=dict)
    per_query: Dict[str, Dict[int, List[TieAwareResult]]] = field(
        default_factory=dict
    )
    k_list: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert aggregated results to a flat dictionary.

        Returns a dict of the form::

            {
                "ndcg@10_expected": 0.74,
                "ndcg@10_oblivious": 0.75,
                "ndcg@10_range": 0.02,
                "ndcg@10_bias": 0.01,
                ...
            }
        """
        out: Dict[str, float] = {}
        for metric_name, k_results in self.metrics.items():
            for k, result in k_results.items():
                prefix = f"{metric_name}@{k}"
                out[f"{prefix}_expected"] = result.expected
                out[f"{prefix}_oblivious"] = result.oblivious
                out[f"{prefix}_maximum"] = result.maximum
                out[f"{prefix}_minimum"] = result.minimum
                out[f"{prefix}_range"] = result.range
                out[f"{prefix}_bias"] = result.bias
        return out

    def __repr__(self) -> str:
        lines = []
        for metric_name, k_results in self.metrics.items():
            for k, r in k_results.items():
                lines.append(
                    f"{metric_name}@{k}: "
                    f"E[M]={r.expected:.4f}  "
                    f"M_obl={r.oblivious:.4f}  "
                    f"range={r.range:.4f}  "
                    f"bias={r.bias:.4f}"
                )
        return "EvaluationOutput(\n  " + "\n  ".join(lines) + "\n)"


def evaluate(
    scores: List[List[float]],
    is_relevant: List[List[bool]],
    metrics: Optional[List[str]] = None,
    k_list: Optional[List[int]] = None,
) -> EvaluationOutput:
    """Compute tie-aware retrieval metrics over a set of queries.

    Parameters
    ----------
    scores : list of list of float
        Per-query relevance scores for each candidate document.
        ``scores[i][j]`` is the score of the *j*-th document for query *i*.
    is_relevant : list of list of bool
        Per-query relevance labels.
        ``is_relevant[i][j]`` indicates whether document *j* is relevant to
        query *i*.
    metrics : list of str, optional
        Which metrics to compute. Supported values:
        ``"ndcg"``, ``"mrr"``, ``"map"``, ``"recall"``, ``"precision"``,
        ``"f1"``, ``"hits"``.
        Defaults to ``["ndcg", "mrr", "map", "recall"]``.
    k_list : list of int, optional
        Cutoff values. Defaults to ``[1, 3, 5, 10, 20, 50, 100]``.

    Returns
    -------
    EvaluationOutput
        Object containing macro-averaged and per-query results with
        expected values, extrema, range, and bias for each metric at
        each cutoff.

    Examples
    --------
    >>> from trm import evaluate
    >>> scores = [[0.97, 0.97, 0.97, 0.99, 0.95]]
    >>> is_relevant = [[True, False, True, True, False]]
    >>> result = evaluate(scores, is_relevant, metrics=["recall"], k_list=[3])
    >>> print(f"E[Recall@3] = {result.metrics['recall'][3].expected:.4f}")
    E[Recall@3] = 0.8889
    """
    if metrics is None:
        metrics = ["ndcg", "mrr", "map", "recall"]
    if k_list is None:
        k_list = DEFAULT_K_LIST

    unknown = set(metrics) - EXPECTED_METRIC_FNS.keys()
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(SUPPORTED_METRICS)}"
        )
    if len(scores) != len(is_relevant):
        raise ValueError(
            f"scores ({len(scores)}) and is_relevant ({len(is_relevant)}) "
            "must have the same number of queries."
        )

    n_queries = len(scores)

    # Initialize accumulators
    per_query: Dict[str, Dict[int, List[TieAwareResult]]] = {
        m: {k: [] for k in k_list} for m in metrics
    }

    for q_scores, q_rels in zip(scores, is_relevant):
        groups = build_tie_groups(q_scores, q_rels)
        n_pos = sum(q_rels)

        for k in k_list:
            max_preds = _extrema_preds(groups, k, optimistic=True)
            min_preds = _extrema_preds(groups, k, optimistic=False)
            obl_preds = _oblivious_preds(q_rels, q_scores, k)

            for m in metrics:
                exp_fn = EXPECTED_METRIC_FNS[m]
                point_fn = POINT_METRIC_FNS[m]

                result = TieAwareResult(
                    expected=exp_fn(groups, k),
                    maximum=point_fn(max_preds, n_pos),
                    minimum=point_fn(min_preds, n_pos),
                    oblivious=point_fn(obl_preds, n_pos),
                )
                per_query[m][k].append(result)

    # Macro-average
    agg: Dict[str, Dict[int, TieAwareResult]] = {}
    for m in metrics:
        agg[m] = {}
        for k in k_list:
            query_results = per_query[m][k]
            if n_queries == 0:
                agg[m][k] = TieAwareResult(0.0, 0.0, 0.0, 0.0)
            else:
                agg[m][k] = TieAwareResult(
                    expected=sum(r.expected for r in query_results) / n_queries,
                    maximum=sum(r.maximum for r in query_results) / n_queries,
                    minimum=sum(r.minimum for r in query_results) / n_queries,
                    oblivious=sum(r.oblivious for r in query_results) / n_queries,
                )

    return EvaluationOutput(metrics=agg, per_query=per_query, k_list=k_list)

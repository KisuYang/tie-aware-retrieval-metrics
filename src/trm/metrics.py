"""Closed-form tie-aware retrieval metric computations.

Implements the expected value, maximum, and minimum of standard retrieval
metrics (nDCG, MRR, MAP, Recall, Precision, F1, Hits) in the presence of
tied relevance scores, following McSherry & Najork (2008) and the TRM
formulation in Yang et al. (2025).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Tie-group utilities
# ---------------------------------------------------------------------------

def build_tie_groups(
    scores: Sequence[float],
    is_relevant: Sequence[bool],
) -> List[Tuple[int, int]]:
    """Return ``(group_size, num_relevant)`` pairs sorted by descending score.

    Parameters
    ----------
    scores : sequence of float
        Relevance scores for each document.
    is_relevant : sequence of bool
        Whether each document is relevant to the query.

    Returns
    -------
    list of (int, int)
        Each element is ``(|G_n|, r_n)`` for tie group *n* in descending
        score order.
    """
    if len(scores) != len(is_relevant):
        raise ValueError(
            f"scores ({len(scores)}) and is_relevant ({len(is_relevant)}) "
            "must have the same length."
        )
    if len(scores) == 0:
        return []

    paired = sorted(
        zip(scores, is_relevant), key=lambda x: x[0], reverse=True
    )

    groups: List[Tuple[int, int]] = []
    cur_score = paired[0][0]
    cur_size = 0
    cur_rel = 0
    for s, r in paired:
        if s == cur_score:
            cur_size += 1
            cur_rel += int(r)
        else:
            groups.append((cur_size, cur_rel))
            cur_score = s
            cur_size = 1
            cur_rel = int(r)
    groups.append((cur_size, cur_rel))
    return groups


def _group_params(
    groups: List[Tuple[int, int]], k: int
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Compute per-group (|G_n|, r_n, t_n) and total relevant N+.

    ``t_n`` is the number of items from group *n* that appear in the top-*k*
    list (Equation 8 in the paper).
    """
    n_plus = sum(r for _, r in groups)
    result: List[Tuple[int, int, int]] = []
    cumulative = 0
    for g_size, r_n in groups:
        t_n = max(0, min(g_size, k - cumulative))
        result.append((g_size, r_n, t_n))
        cumulative += g_size
        if cumulative >= k:
            break
    return result, n_plus


# ---------------------------------------------------------------------------
# DCG weight helpers
# ---------------------------------------------------------------------------

def _dcg_weight(rank: int) -> float:
    """``1 / log2(rank + 1)`` for 1-based *rank*."""
    return 1.0 / math.log2(rank + 1)


def _sum_dcg_weights(a: int, b: int) -> float:
    """Sum of DCG weights from rank *a* to rank *b* inclusive (1-based)."""
    return sum(_dcg_weight(r) for r in range(a, b + 1))


# ---------------------------------------------------------------------------
# Expected metrics (closed-form)
# ---------------------------------------------------------------------------

def expected_hits(groups: List[Tuple[int, int]], k: int) -> float:
    """E[Hits@k]  (Equation 9)."""
    params, _ = _group_params(groups, k)
    return sum(r / g * t for g, r, t in params if t > 0)


def expected_recall(groups: List[Tuple[int, int]], k: int) -> float:
    """E[Recall@k]  (Equation 10)."""
    params, n_plus = _group_params(groups, k)
    if n_plus == 0:
        return 0.0
    return sum(r / g * t for g, r, t in params if t > 0) / n_plus


def expected_precision(groups: List[Tuple[int, int]], k: int) -> float:
    """E[Precision@k]  (Equation 11)."""
    params, _ = _group_params(groups, k)
    return sum(r / g * t for g, r, t in params if t > 0) / k


def expected_f1(groups: List[Tuple[int, int]], k: int) -> float:
    """E[F1@k]  (Equation 12)."""
    params, n_plus = _group_params(groups, k)
    if n_plus == 0:
        return 0.0
    hits = sum(r / g * t for g, r, t in params if t > 0)
    return 2.0 * hits / (k + n_plus)


def expected_ndcg(groups: List[Tuple[int, int]], k: int) -> float:
    """E[nDCG@k]  (Equations 14-16)."""
    params, n_plus = _group_params(groups, k)
    if n_plus == 0:
        return 0.0

    dcg = 0.0
    cumulative = 0
    for g_size, r_n, t_n in params:
        if t_n > 0:
            p_n = r_n / g_size
            dcg += p_n * _sum_dcg_weights(cumulative + 1, cumulative + t_n)
        cumulative += g_size

    idcg = _sum_dcg_weights(1, min(n_plus, k))
    return dcg / idcg if idcg > 0 else 0.0


def expected_mrr(groups: List[Tuple[int, int]], k: int) -> float:
    """E[RR@k]  (Equations 17-21)."""
    n_plus = sum(r for _, r in groups)
    if n_plus == 0:
        return 0.0

    # Find n* — first group with r_n > 0
    cumulative = 0
    for g_size, r_n in groups:
        if r_n > 0:
            c_prev = cumulative
            break
        cumulative += g_size
    else:
        return 0.0

    if k <= c_prev:
        return 0.0

    u = min(g_size - 1, k - c_prev - 1)
    rr = 0.0
    for t in range(u + 1):
        rank_t = c_prev + t + 1
        pi_t = (
            math.comb(g_size - r_n, t) / math.comb(g_size, t)
            if t > 0
            else 1.0
        )
        lambda_t = r_n / (g_size - t)
        rr += (1.0 / rank_t) * pi_t * lambda_t

    return rr


def expected_ap(groups: List[Tuple[int, int]], k: int) -> float:
    """E[AP@k]  (Equations 22-24)."""
    params, n_plus = _group_params(groups, k)
    if n_plus == 0:
        return 0.0

    ap = 0.0
    cumulative = 0
    rel_before = 0  # R_{n-1}
    for g_size, r_n, t_n in params:
        if r_n > 0 and t_n > 0:
            p_n = r_n / g_size
            for t in range(t_n):
                rank = cumulative + t + 1
                # Expected relevant items before position j within the group
                if g_size == 1:
                    inside = 0.0
                else:
                    inside = t * (r_n - 1) / (g_size - 1)
                a_nt = rel_before + inside + 1
                ap += p_n * a_nt / rank
        cumulative += g_size
        rel_before += r_n

    return ap / n_plus


# ---------------------------------------------------------------------------
# Extrema (max / min)
# ---------------------------------------------------------------------------

def _extrema_preds(
    groups: List[Tuple[int, int]], k: int, optimistic: bool
) -> List[bool]:
    """Build the best-case or worst-case binary prediction list."""
    preds: List[bool] = []
    for g_size, r_n in groups:
        neg = g_size - r_n
        if optimistic:
            preds.extend([True] * r_n + [False] * neg)
        else:
            preds.extend([False] * neg + [True] * r_n)
        if len(preds) >= k:
            break
    return preds[:k]


def _oblivious_preds(
    is_relevant: Sequence[bool],
    scores: Sequence[float],
    k: int,
) -> List[bool]:
    """Build the tie-oblivious prediction list (index-preserving order)."""
    paired = sorted(
        enumerate(zip(scores, is_relevant)),
        key=lambda x: (-x[1][0], x[0]),
    )
    return [rel for _, (_, rel) in paired[:k]]


# ---------------------------------------------------------------------------
# Point metric functions (for max/min/oblivious computation)
# ---------------------------------------------------------------------------

def _recall_fn(preds: Sequence[bool], n_pos: int) -> float:
    return 0.0 if n_pos == 0 else sum(preds) / n_pos


def _precision_fn(preds: Sequence[bool], _n_pos: int) -> float:
    return 0.0 if len(preds) == 0 else sum(preds) / len(preds)


def _f1_fn(preds: Sequence[bool], n_pos: int) -> float:
    h = sum(preds)
    if h == 0:
        return 0.0
    return 2.0 * h / (len(preds) + n_pos)


def _hits_fn(preds: Sequence[bool], _n_pos: int) -> float:
    return float(sum(preds))


def _ndcg_fn(preds: Sequence[bool], n_pos: int) -> float:
    if n_pos == 0:
        return 0.0
    dcg = sum(_dcg_weight(i + 1) for i, f in enumerate(preds) if f)
    idcg = _sum_dcg_weights(1, min(n_pos, len(preds)))
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_fn(preds: Sequence[bool], _n_pos: int) -> float:
    for i, flag in enumerate(preds, 1):
        if flag:
            return 1.0 / i
    return 0.0


def _ap_fn(preds: Sequence[bool], n_pos: int) -> float:
    if n_pos == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for i, flag in enumerate(preds, 1):
        if flag:
            hits += 1
            ap += hits / i
    return ap / n_pos


POINT_METRIC_FNS = {
    "recall": _recall_fn,
    "precision": _precision_fn,
    "f1": _f1_fn,
    "hits": _hits_fn,
    "ndcg": _ndcg_fn,
    "mrr": _mrr_fn,
    "map": _ap_fn,
}

EXPECTED_METRIC_FNS = {
    "recall": expected_recall,
    "precision": expected_precision,
    "f1": expected_f1,
    "hits": expected_hits,
    "ndcg": expected_ndcg,
    "mrr": expected_mrr,
    "map": expected_ap,
}

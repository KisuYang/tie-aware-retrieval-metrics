"""Tests for tie-aware retrieval metrics."""

import math
import pytest

from trm import evaluate, build_tie_groups
from trm.metrics import (
    expected_recall,
    expected_precision,
    expected_f1,
    expected_hits,
    expected_ndcg,
    expected_mrr,
    expected_ap,
)


# ---------------------------------------------------------------------------
# build_tie_groups
# ---------------------------------------------------------------------------

class TestBuildTieGroups:
    def test_no_ties(self):
        groups = build_tie_groups([0.9, 0.8, 0.7], [True, False, True])
        assert groups == [(1, 1), (1, 0), (1, 1)]

    def test_all_tied(self):
        groups = build_tie_groups([0.5, 0.5, 0.5], [True, False, True])
        assert groups == [(3, 2)]

    def test_empty(self):
        groups = build_tie_groups([], [])
        assert groups == []

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            build_tie_groups([0.1, 0.2], [True])


# ---------------------------------------------------------------------------
# No-tie cases: expected == oblivious (deterministic)
# ---------------------------------------------------------------------------

class TestNoTies:
    """When all scores are distinct, E[M] should match the oblivious score."""

    scores = [[0.9, 0.7, 0.5, 0.3, 0.1]]
    rels = [[True, False, True, False, True]]

    def test_recall(self):
        result = evaluate(self.scores, self.rels, ["recall"], [3])
        r = result.metrics["recall"][3]
        assert r.expected == pytest.approx(r.oblivious)
        assert r.range == pytest.approx(0.0)
        assert r.bias == pytest.approx(0.0)

    def test_ndcg(self):
        result = evaluate(self.scores, self.rels, ["ndcg"], [3])
        r = result.metrics["ndcg"][3]
        assert r.expected == pytest.approx(r.oblivious)
        assert r.range == pytest.approx(0.0)

    def test_mrr(self):
        result = evaluate(self.scores, self.rels, ["mrr"], [5])
        r = result.metrics["mrr"][5]
        assert r.expected == pytest.approx(1.0)

    def test_map(self):
        result = evaluate(self.scores, self.rels, ["map"], [5])
        r = result.metrics["map"][5]
        assert r.expected == pytest.approx(r.oblivious)
        assert r.range == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Paper Figure 1 example
# ---------------------------------------------------------------------------

class TestFigure1:
    """Reproduce the example from Figure 1 of the paper.

    5 documents: d67 (score=0.99), d5 (0.97, relevant), d20 (0.97),
    d98 (0.97, relevant), d44 (0.95).
    Tie group G2 = {d5, d20, d98} with 2 relevant out of 3.
    E[Recall@3] = 2/3
    """

    scores = [[0.99, 0.97, 0.97, 0.97, 0.95]]
    rels = [[False, True, False, True, False]]

    def test_expected_recall_at_3(self):
        result = evaluate(self.scores, self.rels, ["recall"], [3])
        r = result.metrics["recall"][3]
        assert r.expected == pytest.approx(2.0 / 3.0)

    def test_range_recall_at_3(self):
        result = evaluate(self.scores, self.rels, ["recall"], [3])
        r = result.metrics["recall"][3]
        # max: both relevant in top-3 -> recall = 1.0
        assert r.maximum == pytest.approx(1.0)
        # min: only one relevant in top-3 -> recall = 0.5
        assert r.minimum == pytest.approx(0.5)
        assert r.range == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# All relevant / all irrelevant edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_relevant(self):
        scores = [[0.5, 0.5, 0.5]]
        rels = [[True, True, True]]
        result = evaluate(scores, rels, ["recall", "ndcg", "mrr"], [2])
        for m in ["recall", "ndcg", "mrr"]:
            r = result.metrics[m][2]
            assert r.range == pytest.approx(0.0)

    def test_no_relevant(self):
        scores = [[0.5, 0.5, 0.5]]
        rels = [[False, False, False]]
        result = evaluate(scores, rels, ["recall", "ndcg", "mrr", "map"], [2])
        for m in ["recall", "ndcg", "mrr", "map"]:
            r = result.metrics[m][2]
            assert r.expected == pytest.approx(0.0)
            assert r.range == pytest.approx(0.0)

    def test_k_larger_than_list(self):
        scores = [[0.9, 0.5]]
        rels = [[True, True]]
        result = evaluate(scores, rels, ["recall"], [10])
        r = result.metrics["recall"][10]
        assert r.expected == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Multi-query aggregation
# ---------------------------------------------------------------------------

class TestMultiQuery:
    def test_two_queries(self):
        scores = [
            [0.9, 0.8, 0.7],
            [0.5, 0.5, 0.5],
        ]
        rels = [
            [True, False, True],  # no ties
            [True, False, False],  # all tied
        ]
        result = evaluate(scores, rels, ["recall"], [2])
        # Query 0: recall@2 = 1/2, Query 1: E[recall@2] = (1/3*2)/1 = 2/3
        expected_avg = (0.5 + 2.0 / 3.0) / 2
        r = result.metrics["recall"][2]
        assert r.expected == pytest.approx(expected_avg)


# ---------------------------------------------------------------------------
# Closed-form expected metric unit tests
# ---------------------------------------------------------------------------

class TestExpectedMetrics:
    """Test individual expected metric functions with known results."""

    def test_expected_ndcg_no_tie(self):
        # [relevant, irrelevant] -> nDCG@1 = 1.0
        groups = [(1, 1), (1, 0)]
        assert expected_ndcg(groups, 1) == pytest.approx(1.0)

    def test_expected_mrr_tie(self):
        # Two docs tied, one relevant -> E[RR@1] = 0.5
        groups = [(2, 1)]
        assert expected_mrr(groups, 1) == pytest.approx(0.5)

    def test_expected_mrr_tie_k2(self):
        # Two docs tied, one relevant -> E[RR@2] = 0.5*1 + 0.5*0.5 = 0.75
        groups = [(2, 1)]
        assert expected_mrr(groups, 2) == pytest.approx(0.75)

    def test_expected_ap_no_tie(self):
        # [rel, irrel, rel] -> AP@3 = (1/1 + 2/3) / 2 = 5/6
        groups = [(1, 1), (1, 0), (1, 1)]
        assert expected_ap(groups, 3) == pytest.approx(5.0 / 6.0)

    def test_expected_hits(self):
        # 3 docs tied, 2 relevant, k=2 -> E[Hits@2] = 2*(2/3) = 4/3
        groups = [(3, 2)]
        assert expected_hits(groups, 2) == pytest.approx(4.0 / 3.0)


# ---------------------------------------------------------------------------
# to_dict output
# ---------------------------------------------------------------------------

class TestToDict:
    def test_keys(self):
        result = evaluate(
            [[0.5, 0.3]], [[True, False]], ["recall"], [1]
        )
        d = result.to_dict()
        assert "recall@1_expected" in d
        assert "recall@1_range" in d
        assert "recall@1_bias" in d
        assert "recall@1_oblivious" in d
        assert "recall@1_maximum" in d
        assert "recall@1_minimum" in d


# ---------------------------------------------------------------------------
# Unknown metric error
# ---------------------------------------------------------------------------

class TestErrors:
    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate([[0.5]], [[True]], metrics=["nonexistent"])

    def test_query_count_mismatch(self):
        with pytest.raises(ValueError, match="same number"):
            evaluate([[0.5]], [[True], [False]])


# ---------------------------------------------------------------------------
# per_query access
# ---------------------------------------------------------------------------

class TestPerQuery:
    def test_per_query_length(self):
        scores = [[0.9, 0.8], [0.5, 0.3]]
        rels = [[True, False], [False, True]]
        result = evaluate(scores, rels, ["ndcg"], [1])
        assert len(result.per_query["ndcg"][1]) == 2
import math

import numpy as np
import pytest

from evaluation import all_metrics, ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from retrieval import search


def test_recall_counts_relevant_in_top_k():
    ranked = ["a", "x", "b", "y", "z", "c"]
    assert recall_at_k(ranked, {"a", "b", "c"}, 5) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, {"a", "b", "c"}, 6) == 1.0
    assert recall_at_k(ranked, {"q"}, 5) == 0.0


def test_recall_is_capped_when_more_relevant_than_k():
    ranked = ["a", "b", "c", "d", "e"]
    relevant = set("abcdefghij")
    assert recall_at_k(ranked, relevant, 5) == 0.5


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank_at_k(["a", "x"], {"a"}, 5) == 1.0
    assert reciprocal_rank_at_k(["x", "y", "a", "b"], {"a", "b"}, 5) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(["x", "y", "z", "w", "v", "a"], {"a"}, 5) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "x", "y", "z"], {"a", "b"}, 5) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # Relevant at ranks 2 and 4; two relevant docs exist.
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["x", "a", "y", "b", "z"], {"a", "b"}, 5) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_uses_at_most_k_relevant():
    # 8 relevant docs but only 5 slots: filling all 5 slots is a perfect score.
    assert ndcg_at_k(list("abcde"), set("abcdefgh"), 5) == pytest.approx(1.0)


def test_ndcg_order_matters():
    better = ndcg_at_k(["a", "x", "y", "z", "w"], {"a"}, 5)
    worse = ndcg_at_k(["x", "y", "z", "w", "a"], {"a"}, 5)
    assert better > worse > 0


def test_metrics_ignore_results_beyond_k():
    assert all_metrics(["x"] * 5 + ["a"], {"a"}) == {
        "recall@5": 0.0, "mrr@5": 0.0, "ndcg@5": 0.0,
    }


def test_search_returns_top_k_by_dot_product():
    docs = np.eye(4, dtype=np.float32)
    query = np.array([0.1, 0.9, 0.0, 0.5], dtype=np.float32)
    indices, scores = search(query, docs, k=2)
    assert indices.tolist() == [1, 3]
    assert scores.tolist() == pytest.approx([0.9, 0.5])

import math

import pytest

from rerank_ja_bench.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

RANKED = ["a", "b", "c", "d", "e", "f"]


def test_hit_at_k():
    assert hit_at_k(RANKED, {"c"}, 3) == 1.0
    assert hit_at_k(RANKED, {"d"}, 3) == 0.0
    assert hit_at_k(RANKED, {"z"}, 6) == 0.0


def test_recall_at_k_counts_fraction_of_relevant():
    # 正解3つのうち上位5件に入るのは b と e の2つ
    assert recall_at_k(RANKED, {"b", "e", "z"}, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, {"b", "e", "z"}, 1) == 0.0


def test_recall_at_k_rejects_empty_relevant():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)


def test_reciprocal_rank_uses_first_relevant_only():
    assert reciprocal_rank(RANKED, {"c", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"z"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a"}, {"b"}, 10) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # 並び：x(0) a(2) b(1)。DCG = 2/log2(3) + 1/log2(4)、IDCG = 2/log2(2) + 1/log2(3)
    dcg = 2 / math.log2(3) + 1 / math.log2(4)
    idcg = 2 / 1 + 1 / math.log2(3)
    assert ndcg_at_k(["x", "a", "b"], {"a"}, {"b"}, 10) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_includes_labels_outside_candidates():
    # 正解 z が候補にないので、a を1位にしても 1.0 にならない
    dcg = 2 / 1
    idcg = 2 / 1 + 2 / math.log2(3)
    assert ndcg_at_k(["a", "b"], {"a", "z"}, set(), 10) == pytest.approx(dcg / idcg)


def test_ndcg_respects_cutoff():
    # 正解が11位にあると nDCG@10 は 0
    ranked = [f"x{i}" for i in range(10)] + ["a"]
    assert ndcg_at_k(ranked, {"a"}, set(), 10) == 0.0


def test_ndcg_partial_overlapping_relevant_counts_as_relevant():
    assert ndcg_at_k(["a"], {"a"}, {"a"}, 10) == pytest.approx(1.0)

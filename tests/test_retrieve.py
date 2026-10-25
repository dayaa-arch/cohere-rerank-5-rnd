import numpy as np

from rerank_ja_bench.retrieve import coverage, top_k


def test_top_k_orders_by_score_and_breaks_ties_by_index():
    q = np.array([[1.0, 0.0]])
    docs = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8], [1.0, 0.0]])
    idx, scores = top_k(q, docs, 3)
    # 文書1と3は同点なので、添字の小さい1が先
    assert idx.tolist() == [[1, 3, 2]]
    assert np.allclose(scores, [[1.0, 1.0, 0.6]])


def test_coverage_skips_no_answer_and_lists_misses():
    qa = [
        {"id": "q1", "type": "fact", "relevant_chunk_ids": ["a"]},
        {"id": "q2", "type": "multi", "relevant_chunk_ids": ["b", "c"]},
        {"id": "q3", "type": "no_answer", "relevant_chunk_ids": []},
    ]
    candidates = {"q1": ["x", "y"], "q2": ["b", "x"], "q3": ["a"]}
    cov = coverage(qa, candidates, 2)
    assert cov["n"] == 2
    assert cov["hit@2"] == 0.5
    assert cov["recall@2"] == 0.25
    assert cov["missed"] == ["q1"]
    assert cov["by_type"]["multi"]["recall@2"] == 0.5

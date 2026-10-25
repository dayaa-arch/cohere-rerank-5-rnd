from rerank_ja_bench.qa import EXPECTED_COUNTS, validate


def make_qa():
    """種類ごとの想定問数どおりに並べた、正しい評価データ。"""
    qa = []
    for t, n in EXPECTED_COUNTS.items():
        for _ in range(n):
            rel = (
                [] if t == "no_answer" else ["0001-01", "0001-02"] if t == "multi" else ["0001-01"]
            )
            qa.append(
                {
                    "id": f"q{len(qa) + 1:02d}",
                    "type": t,
                    "relevant_chunk_ids": rel,
                    "partial_chunk_ids": [],
                }
            )
    return qa


CHUNK_IDS = {"0001-01", "0001-02", "0001-03"}


def test_valid_data_has_no_errors():
    assert validate(make_qa(), CHUNK_IDS) == []


def test_unknown_chunk_id_is_reported():
    qa = make_qa()
    qa[0]["relevant_chunk_ids"] = ["9999-01"]
    assert any("9999-01" in e for e in validate(qa, CHUNK_IDS))


def test_no_answer_with_labels_is_reported():
    qa = make_qa()
    qa[-1]["relevant_chunk_ids"] = ["0001-01"]
    assert any("答えがない問題" in e for e in validate(qa, CHUNK_IDS))


def test_multi_with_single_chunk_is_reported():
    qa = make_qa()
    multi = next(q for q in qa if q["type"] == "multi")
    multi["relevant_chunk_ids"] = ["0001-01"]
    assert any("複数箇所" in e for e in validate(qa, CHUNK_IDS))


def test_overlap_between_relevant_and_partial_is_reported():
    qa = make_qa()
    qa[0]["partial_chunk_ids"] = ["0001-01"]
    assert any("両方" in e for e in validate(qa, CHUNK_IDS))


def test_wrong_type_count_is_reported():
    qa = make_qa()[1:]
    assert any("type fact" in e for e in validate(qa, CHUNK_IDS))

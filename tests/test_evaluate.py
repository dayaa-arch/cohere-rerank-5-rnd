import random

import pytest

from rerank_ja_bench.evaluate import (
    bootstrap_ci,
    compress,
    cost_per_call,
    covered,
    merge_spans,
    percentile,
    question_rows,
    summarize,
)


def test_merge_spans_sorts_and_joins_overlaps():
    spans = [{"start": 10, "end": 15}, {"start": 0, "end": 5}, {"start": 5, "end": 8}]
    assert merge_spans(spans) == [(0, 8), (10, 15)]


def test_compress_keeps_span_text_in_document_order():
    text = "あいうえおかきくけこ"
    spans = [{"start": 5, "end": 7, "score": 0.9}, {"start": 0, "end": 2, "score": 0.5}]
    assert compress(text, spans) == "あいかき"
    assert compress(text, None) == ""


def test_covered_requires_whole_range_in_one_span():
    assert covered(2, 4, [(0, 5)])
    assert not covered(4, 7, [(0, 5), (6, 9)])


def test_cost_per_call_by_model():
    assert cost_per_call("rerank-v5.0-pro", {"input_tokens": 1_000_000}) == pytest.approx(0.05)
    assert cost_per_call("rerank-v4.0-pro", {"search_units": 1}) == pytest.approx(0.0025)
    assert cost_per_call(None, {}) is None


def test_percentile_matches_linear_interpolation():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10, 20, 30, 40, 50], 0.95) == pytest.approx(48.0)


def test_bootstrap_ci_contains_mean_of_constant():
    assert bootstrap_ci([0.5] * 10, random.Random(0)) == (0.5, 0.5)


def make_record(cond, qid, rep, ranked, model=None, spans=None, latency=100.0, billed=None):
    return {
        "condition": cond,
        "qid": qid,
        "rep": rep,
        "latency_ms": latency,
        "params": {"model": model, "relevant_spans": spans} if model else {},
        "candidate_ids": sorted(c for c, _, _ in ranked),
        "ranked": [{"chunk_id": c, "score": s, "spans": sp} for c, s, sp in ranked],
        "billed": billed or {},
    }


def test_question_rows_and_summary_on_hand_made_data():
    chunks = {"a": "前置き。答えはここ。", "b": "関係ない文。", "c": "別の文。"}
    qa = [
        {
            "id": "q1",
            "type": "fact",
            "relevant_chunk_ids": ["a"],
            "partial_chunk_ids": [],
            "evidence": [{"chunk_id": "a", "text": "答えはここ"}],
        },
        {"id": "q2", "type": "no_answer", "relevant_chunk_ids": [], "partial_chunk_ids": []},
    ]
    span_a = [{"start": 4, "end": 10, "score": 0.9}]  # 「答えはここ。」だけを残す
    model, spans = "rerank-v5.0-pro", {"mode": "enabled", "threshold": 0.4}
    records = {
        "C2": {
            "q1": {
                rep: make_record(
                    "C2",
                    "q1",
                    rep,
                    [("b", 0.6, None), ("a", 0.9, span_a), ("c", 0.1, None)],
                    model,
                    spans,
                    latency=100.0 * rep,
                    billed={"input_tokens": 1000},
                )
                for rep in (1, 2, 3)
            },
            "q2": {
                1: make_record(
                    "C2",
                    "q2",
                    1,
                    [("b", 0.2, None), ("c", 0.1, None)],
                    model,
                    spans,
                    50.0,
                    {"input_tokens": 1000},
                )
            },
        }
    }
    rows = question_rows(qa, chunks, records, count_tokens=len)
    q1 = next(r for r in rows if r["qid"] == "q1")
    assert q1["mrr"] == 0.5
    assert q1["hit@1"] == 0.0 and q1["hit@5"] == 1.0
    assert q1["latency_ms_median"] == 200.0
    assert q1["cost_usd"] == pytest.approx(0.00005)
    # 上位5件（b, a, c）の全文 6+10+4=20字、残るのは a のスパン6字だけ
    assert q1["top5_tokens_full"] == 20
    assert q1["top5_tokens_compressed"] == 6
    assert q1["token_reduction"] == pytest.approx(0.7)
    assert q1["evidence_retained"] == 1.0
    # しきい値 0.50 以上は b(0.6) と a(0.9)、そのうち正解は a
    assert q1["n_above"] == 2 and q1["tp_above"] == 1

    s = summarize(rows, ["C2"])[0]
    assert s["threshold_precision"] == 0.5
    assert s["threshold_recall"] == 1.0
    assert s["no_answer_all_below"] == 1.0
    assert s["answerable_any_above"] == 1.0
    assert s["latency_p50_ms"] == pytest.approx(125.0)

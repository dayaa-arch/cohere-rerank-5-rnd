import json

import httpx

from rerank_ja_bench.rerankers.cohere import CohereReranker, parse_results
from rerank_ja_bench.run import cost_usd, load_done, plan


def test_plan_rotates_condition_order_per_question():
    order = plan(["q1", "q2", "q3"], ["A", "B", "C"], reps=1)
    assert [c for _, q, c in order if q == "q1"] == ["A", "B", "C"]
    assert [c for _, q, c in order if q == "q2"] == ["B", "C", "A"]
    assert [c for _, q, c in order if q == "q3"] == ["C", "A", "B"]


def test_plan_repeats_all_questions_per_rep():
    order = plan(["q1", "q2"], ["A"], reps=3)
    assert [(rep, q) for rep, q, _ in order] == [
        (1, "q1"),
        (1, "q2"),
        (2, "q1"),
        (2, "q2"),
        (3, "q1"),
        (3, "q2"),
    ]


def test_load_done_reads_saved_records(tmp_path):
    (tmp_path / "C.jsonl").write_text(
        json.dumps({"condition": "C", "qid": "q1", "rep": 1})
        + "\n"
        + json.dumps({"condition": "C", "qid": "q2", "rep": 2})
        + "\n"
    )
    assert load_done(tmp_path) == {("C", "q1", 1), ("C", "q2", 2)}


def test_cost_usd_by_billing_unit():
    assert cost_usd("rerank-v5.0-pro", 2_000_000, 0) == 0.1
    assert cost_usd("rerank-v4.0-pro", 0, 120) == 0.3


def test_parse_results_keeps_missing_spans_as_none():
    raw = {
        "results": [
            {
                "index": 2,
                "relevance_score": 0.9,
                "relevant_spans": [{"start": 0, "end": 3, "score": 0.8}],
            },
            {"index": 0, "relevance_score": 0.1},
        ]
    }
    ranked = parse_results(raw)
    assert [d.index for d in ranked] == [2, 0]
    assert ranked[0].spans[0].end == 3
    assert ranked[1].spans is None


def test_cohere_reranker_sends_spans_and_retries_on_429(monkeypatch):
    monkeypatch.setattr("rerank_ja_bench.rerankers.cohere.time.sleep", lambda s: None)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(429, json={"message": "rate limited"})
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.7},
                    {"index": 0, "relevance_score": 0.2},
                ],
                "meta": {"billed_units": {"input_tokens": 42}},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    r = CohereReranker(
        "C2", "rerank-v5.0-pro", {"mode": "enabled", "threshold": 0.4}, client=client
    )
    result = r.rerank("質問", ["文書0", "文書1"])
    assert len(calls) == 2
    assert calls[1]["relevant_spans"] == {"mode": "enabled", "threshold": 0.4}
    assert calls[1]["documents"] == ["文書0", "文書1"]
    assert [d.index for d in result.ranked] == [1, 0]
    assert result.billed == {"input_tokens": 42}


def test_cohere_reranker_omits_spans_when_disabled():
    r = CohereReranker("C", "rerank-v5.0-pro", client=httpx.Client())
    assert "relevant_spans" not in r.request_body("q", ["d"])

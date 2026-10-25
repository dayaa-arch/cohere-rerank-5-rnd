import json
from pathlib import Path

from rerank_ja_bench.page_experiment import candidate_pages, page_text


def test_candidate_pages_dedupes_in_first_seen_order():
    assert candidate_pages(["2505-02", "2798-04", "2505-01", "2801-04", "2798-01"]) == [
        "2505",
        "2798",
        "2801",
    ]


def test_page_text_joins_title_headings_and_blocks():
    doc = {
        "title": "表題",
        "sections": [
            {"heading": "概要", "blocks": ["一文目。", "二文目。"]},
            {"heading": "概要 > 詳細", "blocks": ["三文目。"]},
        ],
    }
    assert page_text(doc) == "表題\n概要\n一文目。\n二文目。\n概要 > 詳細\n三文目。"


def test_every_evidence_string_is_found_in_its_page():
    docs = {
        json.loads(line)["doc_id"]: page_text(json.loads(line))
        for line in Path("data/corpus/docs.jsonl").read_text().splitlines()
    }
    for line in Path("data/qa.jsonl").read_text().splitlines():
        for ev in json.loads(line).get("evidence", []):
            assert ev["text"] in docs[ev["chunk_id"].split("-")[0]]

"""評価データ data/qa.jsonl の読み込み・検証と、人が確認するための一覧の出力。

    uv run python -m rerank_ja_bench.qa check    # 形式と問数を検証する
    uv run python -m rerank_ja_bench.qa review   # docs/phase2_qa_review.md を作る

qa.jsonl の各行：
- id, type, question, relevant_chunk_ids（仕様書の必須項目）
- partial_chunk_ids：答えの一部だけを含むチャンク。nDCG で関連度1として使う
- answer：確認用の想定回答（答えがない問題は null）
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

QA_PATH = Path("data/qa.jsonl")
CHUNKS_PATH = Path("data/chunks.jsonl")
REVIEW_PATH = Path("docs/phase2_qa_review.md")

EXPECTED_COUNTS = {"fact": 12, "paraphrase": 12, "multi": 8, "no_answer": 8}
PARTIAL_PREVIEW_CHARS = 150
TYPE_LABELS = {
    "fact": "素直な事実検索",
    "paraphrase": "言い換え",
    "multi": "複数箇所にまたがる",
    "no_answer": "答えがコーパスにない",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(qa: list[dict], chunk_ids: set[str]) -> list[str]:
    """問題点を文字列のリストで返す。空なら問題なし。"""
    errors = []
    ids = [q["id"] for q in qa]
    for dup, n in Counter(ids).items():
        if n > 1:
            errors.append(f"{dup}: id が重複している")
    for q in qa:
        qid = q["id"]
        rel = q["relevant_chunk_ids"]
        par = q.get("partial_chunk_ids", [])
        if q["type"] not in EXPECTED_COUNTS:
            errors.append(f"{qid}: 未知の type {q['type']!r}")
        for cid in rel + par:
            if cid not in chunk_ids:
                errors.append(f"{qid}: chunk_id {cid} が chunks.jsonl にない")
        if set(rel) & set(par):
            errors.append(f"{qid}: relevant と partial の両方に入っているチャンクがある")
        if q["type"] == "no_answer" and (rel or par):
            errors.append(f"{qid}: 答えがない問題に正解チャンクが付いている")
        if q["type"] != "no_answer" and not rel:
            errors.append(f"{qid}: 正解チャンクがない")
        if q["type"] == "multi" and len(rel) < 2:
            errors.append(f"{qid}: 複数箇所の問題なのに正解チャンクが1つしかない")
    counts = Counter(q["type"] for q in qa)
    for t, n in EXPECTED_COUNTS.items():
        if counts.get(t, 0) != n:
            errors.append(f"type {t} は {n} 問の想定だが {counts.get(t, 0)} 問ある")
    return errors


def render_review(qa: list[dict], chunks: dict[str, dict]) -> str:
    lines = [
        "# 評価データ案（40問）の確認用一覧",
        "",
        (
            "`data/qa.jsonl` から自動で作った一覧。直すときは `data/qa.jsonl` を直し、"
            "`uv run python -m rerank_ja_bench.qa review` で作り直す。"
        ),
        "",
        "- ◎ 正解チャンク（`relevant_chunk_ids`）：答えを含む。Recall@5、MRR、nDCG（関連度2）に使う",
        (
            "- ○ 部分正解チャンク（`partial_chunk_ids`）：答えの一部や関連する記述だけを含む。"
            "nDCG（関連度1）にだけ使う。本文は先頭150字だけ表示する"
        ),
        "",
    ]
    for t, label in TYPE_LABELS.items():
        group = [q for q in qa if q["type"] == t]
        lines += [f"## {label}（{len(group)}問）", ""]
        for q in group:
            lines += [f"### {q['id']}　{q['question']}", ""]
            if q.get("answer"):
                lines += [f"想定回答：{q['answer']}", ""]
            marks = [("◎", cid) for cid in q["relevant_chunk_ids"]]
            marks += [("○", cid) for cid in q.get("partial_chunk_ids", [])]
            if not marks:
                lines += ["正解チャンクなし（コーパスに答えがない想定）", ""]
            for mark, cid in marks:
                c = chunks[cid]
                body = c["text"].split("\n", 1)[1].replace("\n", " / ")
                # 正解チャンクは全文を、部分正解チャンクは先頭だけを出す
                if mark == "○" and len(body) > PARTIAL_PREVIEW_CHARS:
                    body = body[:PARTIAL_PREVIEW_CHARS] + "…"
                lines += [f"- {mark} `{cid}`（{c['title']}｜{c['heading']}）", f"  > {body}"]
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "review"])
    args = parser.parse_args()

    qa = load_jsonl(QA_PATH)
    chunks = {c["chunk_id"]: c for c in load_jsonl(CHUNKS_PATH)}
    errors = validate(qa, set(chunks))
    for e in errors:
        print("NG", e)
    if errors:
        raise SystemExit(1)
    print(f"OK: {len(qa)} questions {dict(Counter(q['type'] for q in qa))}")
    if args.command == "review":
        REVIEW_PATH.write_text(render_review(qa, chunks) + "\n")
        print(f"wrote {REVIEW_PATH}")


if __name__ == "__main__":
    main()

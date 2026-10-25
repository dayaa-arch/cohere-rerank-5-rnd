"""追加実験：ページ単位の文書で Context Focus のトークン削減を測り直す。

本実行（チャンク単位）では、チャンクが短くスパンでほとんど削れなかった。
ここでは、1段目の上位20チャンクを元のページにまとめ、ページ全文をリランカーに渡す。

    uv run python -m rerank_ja_bench.page_experiment run --dry-run
    uv run python -m rerank_ja_bench.page_experiment run
    uv run python -m rerank_ja_bench.page_experiment evaluate

- 候補：data/candidates.jsonl の上位20チャンクの親ページを、最初に出てきた順に重複なく並べる
- 条件：rerank-v5.0-pro と rerank-v5.0-fast、Context Focus threshold 0.4、40問×1回
- max_tokens_per_doc は 8192（最長のページが既定の 4096 を超えるため）
- 出力：results/2026-10-25/page_cf/（raw/、per_question.csv、summary.csv、summary.md）
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from rerank_ja_bench.evaluate import (
    OFFICIAL_THRESHOLDS,
    compress,
    covered,
    merge_spans,
    ruri_token_counter,
    write_csv,
)
from rerank_ja_bench.metrics import hit_at_k, reciprocal_rank
from rerank_ja_bench.qa import CHUNKS_PATH, QA_PATH, load_jsonl
from rerank_ja_bench.rerankers.cohere import CohereReranker
from rerank_ja_bench.retrieve import CANDIDATES_PATH
from rerank_ja_bench.run import PRICE_PER_M_TOKENS, TOKENS_PER_CHAR, load_done

DOCS_PATH = Path("data/corpus/docs.jsonl")
OUT_DIR = Path("results/2026-10-25/page_cf")
MAIN_RAW_DIR = Path("results/2026-10-25/raw")
MAX_TOKENS_PER_DOC = 8192
TOP_K_CONTEXT = 5
SPANS = {"mode": "enabled", "threshold": 0.4}
CONDITIONS = {
    "P_t04": ("rerank-v5.0-pro", "C2_t04"),  # 比べるチャンク単位の条件
    "F_t04": ("rerank-v5.0-fast", "D2_t04"),
}


def page_text(doc: dict) -> str:
    """ページ全文。タイトル、見出し、本文のブロックを改行でつなぐ。"""
    lines = [doc["title"]]
    for s in doc["sections"]:
        lines.append(s["heading"])
        lines.extend(s["blocks"])
    return "\n".join(lines)


def candidate_pages(chunk_ids: list[str]) -> list[str]:
    """チャンクID（"2505-02" など）の並びを、親ページの番号の並びに直す。"""
    return list(dict.fromkeys(c.split("-")[0] for c in chunk_ids))


def run(dry_run: bool) -> None:
    load_dotenv()
    qa = load_jsonl(QA_PATH)
    pages = {d["doc_id"]: page_text(d) for d in load_jsonl(DOCS_PATH)}
    cands = {r["id"]: [c["chunk_id"] for c in r["candidates"]] for r in load_jsonl(CANDIDATES_PATH)}
    raw_dir = OUT_DIR / "raw"
    done = load_done(raw_dir) if raw_dir.exists() else set()
    todo = [(q, c) for q in qa for c in CONDITIONS if (c, q["id"], 1) not in done]

    n_pages = [len(candidate_pages(cands[q["id"]])) for q in qa]
    chars = sum(
        sum(len(pages[p]) for p in candidate_pages(cands[q["id"]]))
        + len(candidate_pages(cands[q["id"]])) * len(q["question"])
        for q in qa
    )
    est = sum(chars * TOKENS_PER_CHAR / 1e6 * PRICE_PER_M_TOKENS[m] for m, _ in CONDITIONS.values())
    print(
        f"plan: {len(todo)} calls ({len(done)} done); pages per question "
        f"median {statistics.median(n_pages)}, max {max(n_pages)}; "
        f"tokens/pass ≈ {chars * TOKENS_PER_CHAR:,.0f}; est ${est:.4f}"
    )
    if dry_run:
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    rerankers = {
        c: CohereReranker(c, m, SPANS, max_tokens_per_doc=MAX_TOKENS_PER_DOC)
        for c, (m, _) in CONDITIONS.items()
    }
    for i, (q, cond) in enumerate(todo, start=1):
        r = rerankers[cond]
        page_ids = candidate_pages(cands[q["id"]])
        started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        result = r.rerank(q["question"], [pages[p] for p in page_ids])
        record = {
            "condition": cond,
            "qid": q["id"],
            "rep": 1,
            "started_at": started_at,
            "latency_ms": round(result.latency_ms, 1),
            "params": r.params,
            "candidate_ids": page_ids,
            "ranked": [
                {
                    "chunk_id": page_ids[d.index],
                    "score": d.score,
                    "spans": None
                    if d.spans is None
                    else [{"start": s.start, "end": s.end, "score": s.score} for s in d.spans],
                }
                for d in result.ranked
            ],
            "billed": result.billed,
            "raw": result.raw,
        }
        with (raw_dir / f"{cond}.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if i == 1 or i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} billed {result.billed}", flush=True)
        time.sleep(0.2)


def evaluate() -> None:
    qa = load_jsonl(QA_PATH)
    pages = {d["doc_id"]: page_text(d) for d in load_jsonl(DOCS_PATH)}
    chunks = {c["chunk_id"]: c["text"] for c in load_jsonl(CHUNKS_PATH)}
    count = ruri_token_counter()
    rows = []
    for cond, (model, chunk_cond) in CONDITIONS.items():
        page_recs = {r["qid"]: r for r in load_jsonl(OUT_DIR / "raw" / f"{cond}.jsonl")}
        chunk_recs = {
            r["qid"]: r for r in load_jsonl(MAIN_RAW_DIR / f"{chunk_cond}.jsonl") if r["rep"] == 1
        }
        for q in qa:
            rec = page_recs[q["id"]]
            ranked_pages = [d["chunk_id"] for d in rec["ranked"]]
            rel_pages = list(dict.fromkeys(c.split("-")[0] for c in q["relevant_chunk_ids"]))
            row = {
                "condition": cond,
                "qid": q["id"],
                "type": q["type"],
                "n_pages": len(ranked_pages),
                "billed_input_tokens": rec["billed"].get("input_tokens"),
                "latency_ms": rec["latency_ms"],
                "top1_score": rec["ranked"][0]["score"],
                "n_above": sum(d["score"] >= OFFICIAL_THRESHOLDS[model] for d in rec["ranked"]),
            }
            if rel_pages and hit_at_k(ranked_pages, rel_pages, len(ranked_pages)):
                row["page_hit@1"] = hit_at_k(ranked_pages, rel_pages, 1)
                row["page_mrr"] = reciprocal_rank(ranked_pages, rel_pages)
            if q["relevant_chunk_ids"]:
                # ページ単位：上位5ページの全文と、スパンだけに削った本文
                top = rec["ranked"][:TOP_K_CONTEXT]
                row["page_top5_tokens_full"] = sum(count(pages[d["chunk_id"]]) for d in top)
                row["page_top5_tokens_compressed"] = sum(
                    count(compress(pages[d["chunk_id"]], d["spans"])) for d in top
                )
                # チャンク単位（本実行の同じモデル・同じ threshold）：上位5件の全文とスパン
                ctop = chunk_recs[q["id"]]["ranked"][:TOP_K_CONTEXT]
                row["chunk_top5_tokens_full"] = sum(count(chunks[d["chunk_id"]]) for d in ctop)
                row["chunk_top5_tokens_compressed"] = sum(
                    count(compress(chunks[d["chunk_id"]], d["spans"])) for d in ctop
                )
                # 根拠の文字列が、LLMに渡す本文（上位5件をスパンで削ったもの）に残っているか
                pspans = {d["chunk_id"]: merge_spans(d["spans"] or []) for d in top}
                cspans = {d["chunk_id"]: merge_spans(d["spans"] or []) for d in ctop}
                ev = q.get("evidence", [])
                page_kept = chunk_kept = page_in_top = 0
                for e in ev:
                    p = e["chunk_id"].split("-")[0]
                    if p in pspans:
                        page_in_top += 1
                        start = pages[p].index(e["text"])
                        page_kept += covered(start, start + len(e["text"]), pspans[p])
                    if e["chunk_id"] in cspans:
                        start = chunks[e["chunk_id"]].index(e["text"])
                        chunk_kept += covered(start, start + len(e["text"]), cspans[e["chunk_id"]])
                row["n_evidence"] = len(ev)
                row["evidence_page_in_top5"] = page_in_top
                row["evidence_kept_page"] = page_kept
                row["evidence_kept_chunk"] = chunk_kept
            rows.append(row)

    summary = []
    for cond, (model, chunk_cond) in CONDITIONS.items():
        g = [r for r in rows if r["condition"] == cond]
        ans = [r for r in g if "page_top5_tokens_full" in r]
        acc = [r for r in g if "page_mrr" in r]
        na = [r for r in g if r["type"] == "no_answer"]
        n_ev = sum(r["n_evidence"] for r in ans)

        def mean(key: str, rs: list[dict] = ans) -> float:
            return statistics.mean(r[key] for r in rs)

        summary.append(
            {
                "condition": cond,
                "model": model,
                "compared_chunk_condition": chunk_cond,
                "pages_per_question": statistics.mean(r["n_pages"] for r in g),
                "billed_input_tokens_per_question": statistics.mean(
                    r["billed_input_tokens"] for r in g
                ),
                "cost_per_question_usd": statistics.mean(r["billed_input_tokens"] for r in g)
                / 1e6
                * PRICE_PER_M_TOKENS[model],
                "latency_p50_ms": statistics.median(r["latency_ms"] for r in g),
                "page_top5_tokens_full": mean("page_top5_tokens_full"),
                "page_top5_tokens_compressed": mean("page_top5_tokens_compressed"),
                "page_token_reduction": 1
                - mean("page_top5_tokens_compressed") / mean("page_top5_tokens_full"),
                "chunk_top5_tokens_full": mean("chunk_top5_tokens_full"),
                "chunk_top5_tokens_compressed": mean("chunk_top5_tokens_compressed"),
                "n_evidence": n_ev,
                "evidence_reached_llm_page": sum(r["evidence_kept_page"] for r in ans) / n_ev,
                "evidence_reached_llm_chunk": sum(r["evidence_kept_chunk"] for r in ans) / n_ev,
                "evidence_retained_given_page_in_top5": sum(r["evidence_kept_page"] for r in ans)
                / sum(r["evidence_page_in_top5"] for r in ans),
                "page_hit@1": statistics.mean(r["page_hit@1"] for r in acc),
                "page_mrr": statistics.mean(r["page_mrr"] for r in acc),
                "no_answer_all_below": sum(r["n_above"] == 0 for r in na) / len(na),
                "no_answer_top1_max": max(r["top1_score"] for r in na),
            }
        )
    write_csv(OUT_DIR / "per_question.csv", rows)
    write_csv(OUT_DIR / "summary.csv", summary)
    (OUT_DIR / "summary.md").write_text(render_summary(summary))
    for s in summary:
        print(
            json.dumps(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
                ensure_ascii=False,
            )
        )


def render_summary(summary: list[dict]) -> str:
    """チャンク単位（本実行）とページ単位（追加実験）の比較表。"""
    lines = [
        "# 追加実験：ページ単位の Context Focus（threshold 0.4）",
        "",
        (
            "LLMに渡す本文は、どちらも上位5件をスパンだけに削ったもの。トークン数は ruri-v3 のトークナイザーで数えた。"
            "根拠が残った割合は、答えのある32問の根拠の文字列のうち、削ったあとの上位5件に丸ごと残ったものの割合。"
            "削減率は32問の合計トークンの比（本実行の summary.md は31問の問題ごとの削減率の平均なので、チャンクの値が少し違う）。"
        ),
        "",
        "| モデル | 単位 | 上位5件の全文（トークン） | スパンだけに削った本文（トークン） | 削減率 | 根拠が残った割合 | 1問あたり課金トークン |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        chunk_red = 1 - s["chunk_top5_tokens_compressed"] / s["chunk_top5_tokens_full"]
        lines.append(
            f"| `{s['model']}` | チャンク（本実行 {s['compared_chunk_condition']}） "
            f"| {s['chunk_top5_tokens_full']:,.0f} | {s['chunk_top5_tokens_compressed']:,.0f} "
            f"| {chunk_red * 100:.1f}％ | {s['evidence_reached_llm_chunk'] * 100:.1f}％ | — |"
        )
        lines.append(
            f"| `{s['model']}` | ページ（追加実験） | {s['page_top5_tokens_full']:,.0f} "
            f"| {s['page_top5_tokens_compressed']:,.0f} | {s['page_token_reduction'] * 100:.1f}％ "
            f"| {s['evidence_reached_llm_page'] * 100:.1f}％ "
            f"| {s['billed_input_tokens_per_question']:,.0f} |"
        )
    lines += [
        "",
        "| モデル | 候補のページ数（平均） | 1問あたり費用（ドル） | 遅延 p50（ms） | 答えなし8問で全候補がしきい値未満 | 答えなしの1位の最大 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        lines.append(
            f"| `{s['model']}` | {s['pages_per_question']:.2f} | {s['cost_per_question_usd']:.6f} "
            f"| {s['latency_p50_ms']:,.0f} | {s['no_answer_all_below'] * 100:.1f}％ "
            f"| {s['no_answer_top1_max']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "evaluate"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        run(args.dry_run)
    else:
        evaluate()


if __name__ == "__main__":
    main()

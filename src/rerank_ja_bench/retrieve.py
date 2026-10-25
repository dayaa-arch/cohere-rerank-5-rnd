"""1段目の検索。全チャンクを埋め込み、質問ごとに上位 TOP_K 件を保存する。

    uv run python -m rerank_ja_bench.retrieve

出力：
- data/candidates.jsonl：質問ごとの候補（全条件で使い回す。リランカーにはこの順で渡す）
- data/candidates_meta.json：モデル、リビジョン、ライブラリの版、実行日時、所要時間、
  正解が上位 TOP_K 件に入った割合
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from rerank_ja_bench.metrics import hit_at_k, recall_at_k
from rerank_ja_bench.qa import CHUNKS_PATH, QA_PATH, load_jsonl

MODEL_ID = "cl-nagoya/ruri-v3-310m"
MODEL_REVISION = "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"
QUERY_PREFIX = "検索クエリ: "
DOCUMENT_PREFIX = "検索文書: "
TOP_K = 20
BATCH_SIZE = 16

CANDIDATES_PATH = Path("data/candidates.jsonl")
META_PATH = Path("data/candidates_meta.json")


def top_k(query_emb: np.ndarray, doc_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """正規化済みの埋め込みで内積（コサイン類似度）を取り、上位 k 件の添字と類似度を返す。

    類似度が同じときは添字の小さい（chunks.jsonl で先の）チャンクを上にする。
    """
    scores = query_emb @ doc_emb.T
    order = np.lexsort((np.arange(scores.shape[1])[None, :].repeat(len(scores), 0), -scores))
    idx = order[:, :k]
    return idx, np.take_along_axis(scores, idx, axis=1)


def coverage(qa: list[dict], candidates: dict[str, list[str]], k: int) -> dict:
    """答えのある問題について、正解が上位 k 件に入った割合を種類ごとに集計する。"""
    rows = []
    for q in qa:
        if not q["relevant_chunk_ids"]:
            continue
        ranked = candidates[q["id"]]
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "hit": hit_at_k(ranked, q["relevant_chunk_ids"], k),
                "recall": recall_at_k(ranked, q["relevant_chunk_ids"], k),
            }
        )
    by_type: dict[str, dict] = {}
    for t in dict.fromkeys(r["type"] for r in rows):
        group = [r for r in rows if r["type"] == t]
        by_type[t] = {
            "n": len(group),
            f"hit@{k}": sum(r["hit"] for r in group) / len(group),
            f"recall@{k}": sum(r["recall"] for r in group) / len(group),
        }
    return {
        "n": len(rows),
        f"hit@{k}": sum(r["hit"] for r in rows) / len(rows),
        f"recall@{k}": sum(r["recall"] for r in rows) / len(rows),
        "by_type": by_type,
        "missed": [r["id"] for r in rows if r["hit"] == 0.0],
    }


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)
    qa = load_jsonl(QA_PATH)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")

    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
    load_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    doc_emb = model.encode(
        [DOCUMENT_PREFIX + c["text"] for c in chunks],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    doc_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_emb = model.encode(
        [QUERY_PREFIX + q["question"] for q in qa],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_sec = time.perf_counter() - t0

    idx, scores = top_k(query_emb, doc_emb, TOP_K)
    candidates = {}
    with CANDIDATES_PATH.open("w") as f:
        for q, row_idx, row_scores in zip(qa, idx, scores, strict=True):
            cands = [
                {"rank": r, "chunk_id": chunks[i]["chunk_id"], "score": round(float(s), 6)}
                for r, (i, s) in enumerate(zip(row_idx, row_scores, strict=True), start=1)
            ]
            candidates[q["id"]] = [c["chunk_id"] for c in cands]
            f.write(json.dumps({"id": q["id"], "candidates": cands}, ensure_ascii=False) + "\n")

    cov = coverage(qa, candidates, TOP_K)
    meta = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "query_prefix": QUERY_PREFIX,
        "document_prefix": DOCUMENT_PREFIX,
        "top_k": TOP_K,
        "similarity": "cosine (normalized embeddings, dot product)",
        "n_chunks": len(chunks),
        "n_questions": len(qa),
        "started_at": started_at,
        "versions": {
            "python": platform.python_version(),
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
        },
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "seconds": {
            "model_load": round(load_sec, 2),
            "encode_chunks": round(doc_sec, 2),
            "encode_queries": round(query_sec, 2),
        },
        "coverage": cov,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    print(f"wrote {CANDIDATES_PATH} and {META_PATH}")
    print(f"encode: chunks {doc_sec:.1f}s ({len(chunks)}), queries {query_sec:.1f}s ({len(qa)})")
    print(f"hit@{TOP_K} {cov[f'hit@{TOP_K}']:.3f}  recall@{TOP_K} {cov[f'recall@{TOP_K}']:.3f}")
    for t, v in cov["by_type"].items():
        print(
            f"  {t:<10} n={v['n']:>2}  hit {v[f'hit@{TOP_K}']:.3f}  recall {v[f'recall@{TOP_K}']:.3f}"
        )
    print(f"missed: {cov['missed']}")


if __name__ == "__main__":
    main()

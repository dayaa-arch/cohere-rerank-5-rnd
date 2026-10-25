"""条件A：リランカーなし。1段目の検索順をそのまま返す。"""

from __future__ import annotations

from rerank_ja_bench.rerankers.base import RankedDoc, Reranker, RerankResult


class NoReranker(Reranker):
    def __init__(self) -> None:
        self.name = "none"
        self.params = {}

    def rerank(self, query: str, documents: list[str]) -> RerankResult:
        return RerankResult(
            ranked=[RankedDoc(index=i, score=None) for i in range(len(documents))],
            latency_ms=0.0,
        )

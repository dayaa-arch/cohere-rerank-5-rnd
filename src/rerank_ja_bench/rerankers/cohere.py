"""Cohere Rerank（v2 API）。生の応答を残すため、SDK ではなく HTTP で直接呼ぶ。

APIキーは環境変数 COHERE_API_KEY から読む。キーはログにも結果にも書かない。
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from rerank_ja_bench.rerankers.base import RankedDoc, Reranker, RerankResult, Span

ENDPOINT = "https://api.cohere.com/v2/rerank"
CLIENT_NAME = "rerank-ja-bench"
RETRY_STATUS = {429, 500, 502, 503, 504}


class CohereReranker(Reranker):
    billable = True

    def __init__(
        self,
        name: str,
        model: str,
        relevant_spans: dict[str, Any] | None = None,
        client: httpx.Client | None = None,
        max_retries: int = 4,
    ) -> None:
        self.name = name
        self.model = model
        self.relevant_spans = relevant_spans
        self.params = {"model": model, "relevant_spans": relevant_spans}
        self.max_retries = max_retries
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            key = os.environ.get("COHERE_API_KEY")
            if not key:
                raise RuntimeError("環境変数 COHERE_API_KEY が設定されていない")
            self._client = httpx.Client(
                timeout=60,
                headers={"Authorization": f"Bearer {key}", "X-Client-Name": CLIENT_NAME},
            )
        return self._client

    def request_body(self, query: str, documents: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "query": query, "documents": documents}
        if self.relevant_spans is not None:
            body["relevant_spans"] = self.relevant_spans
        return body

    def rerank(self, query: str, documents: list[str]) -> RerankResult:
        body = self.request_body(query, documents)
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            res = self.client.post(ENDPOINT, json=body)
            latency_ms = (time.perf_counter() - t0) * 1000
            if res.status_code not in RETRY_STATUS or attempt == self.max_retries:
                break
            time.sleep(2 ** (attempt + 1))
        res.raise_for_status()
        raw = res.json()
        return RerankResult(
            ranked=parse_results(raw),
            latency_ms=latency_ms,
            raw=raw,
            billed=(raw.get("meta") or {}).get("billed_units") or {},
        )


def parse_results(raw: dict[str, Any]) -> list[RankedDoc]:
    ranked = []
    for item in raw["results"]:
        spans = item.get("relevant_spans")
        ranked.append(
            RankedDoc(
                index=item["index"],
                score=item["relevance_score"],
                spans=None
                if spans is None
                else [Span(start=s["start"], end=s["end"], score=s["score"]) for s in spans],
            )
        )
    return ranked

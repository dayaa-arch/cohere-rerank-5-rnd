"""条件E：ruri-v3 の日本語リランカー（ローカル、CPU）。"""

from __future__ import annotations

import time

from rerank_ja_bench.rerankers.base import RankedDoc, Reranker, RerankResult

MODEL_ID = "cl-nagoya/ruri-v3-reranker-310m"
MODEL_REVISION = "bb46934ee9ed09f850b9fcff17501b3ef7ddb2b3"


class RuriReranker(Reranker):
    def __init__(self, batch_size: int = 20) -> None:
        self.name = "ruri-v3-reranker-310m"
        self.params = {"model": MODEL_ID, "revision": MODEL_REVISION, "device": "cpu"}
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(MODEL_ID, revision=MODEL_REVISION, device="cpu")
        return self._model

    def rerank(self, query: str, documents: list[str]) -> RerankResult:
        model = self.model  # 読み込み時間を遅延に含めない
        t0 = time.perf_counter()
        scores = model.predict(
            [(query, d) for d in documents], batch_size=self.batch_size, show_progress_bar=False
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        order = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), i))
        return RerankResult(
            ranked=[RankedDoc(index=i, score=float(scores[i])) for i in order],
            latency_ms=latency_ms,
            raw={"scores": [float(s) for s in scores]},
        )

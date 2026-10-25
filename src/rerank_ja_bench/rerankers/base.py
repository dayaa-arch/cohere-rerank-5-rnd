"""リランカーの共通インターフェース。

新しいリランカー（Nemotron など）は Reranker を継承し、rerank() を実装すれば run.py から使える。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """文書中の該当箇所。start・end は Unicode のコードポイント単位で、範囲は [start, end)。"""

    start: int
    end: int
    score: float


@dataclass
class RankedDoc:
    index: int  # 渡した documents の中での位置（0始まり）
    score: float | None
    spans: list[Span] | None = None  # Context Focus を使わない条件では None


@dataclass
class RerankResult:
    ranked: list[RankedDoc]  # スコアの高い順
    latency_ms: float
    raw: dict[str, Any] = field(default_factory=dict)  # 生の応答（保存用）
    billed: dict[str, Any] = field(default_factory=dict)  # 課金量（API の応答にあれば）


class Reranker(ABC):
    name: str
    params: dict[str, Any]
    billable: bool = False  # True なら API 呼び出しで課金が発生する

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> RerankResult: ...

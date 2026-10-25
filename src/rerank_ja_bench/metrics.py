"""検索結果の順位に対する指標。

どの関数も、ranked（チャンクIDを上位から並べたリスト）と正解ラベルを受け取る。
- relevant：正解（関連度2）。Hit・Recall・MRR はこれだけを見る
- partial：部分正解（関連度1）。nDCG だけで使う
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

GAIN_RELEVANT = 2
GAIN_PARTIAL = 1


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """上位 k 件に正解が1つでも入れば 1.0、なければ 0.0。"""
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """正解のうち、上位 k 件に入った割合。"""
    if not relevant:
        raise ValueError("relevant が空の問題には使えない")
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """最初の正解の順位の逆数。正解がなければ 0.0。平均すると MRR になる。"""
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    ranked: Sequence[str],
    relevant: Collection[str],
    partial: Collection[str],
    k: int,
) -> float:
    """関連度つきの nDCG@k。利得は正解2・部分正解1をそのまま使い、log2(順位+1) で割る。

    理想の並びは、ラベルの付いたチャンクすべて（候補に入っていないものも含む）から作る。
    """
    relevant, partial = set(relevant), set(partial) - set(relevant)

    def gain(cid: str) -> int:
        if cid in relevant:
            return GAIN_RELEVANT
        if cid in partial:
            return GAIN_PARTIAL
        return 0

    dcg = sum(gain(cid) / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1))
    ideal = sorted([GAIN_RELEVANT] * len(relevant) + [GAIN_PARTIAL] * len(partial), reverse=True)
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal[:k], start=1))
    if idcg == 0:
        raise ValueError("ラベルのない問題には使えない")
    return dcg / idcg

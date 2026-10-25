"""summary.csv と per_question.csv から、比較表（Markdown）とグラフを作る。API は呼ばない。

    uv run python -m rerank_ja_bench.report --run results/2026-10-25

出力（--run のディレクトリ）：
- summary.md：比較表
- fig_accuracy.png：条件ごとの MRR と nDCG@10（95％信頼区間つき）
- fig_tradeoff.png：nDCG@10 と遅延、nDCG@10 と1問あたりの費用
- cases.md：条件によって最初の正解の順位が大きく変わった問題の一覧（メモの材料）
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib_fontja  # noqa: F401  日本語フォント（IPAexゴシック）を有効にする

from rerank_ja_bench.qa import QA_PATH, load_jsonl

LABELS = {
    "A": "A リランカーなし",
    "B": "B Rerank 4 Pro",
    "C": "C Rerank 5 Pro",
    "D": "D Rerank 5 Fast",
    "C2_t01": "C2 5 Pro＋CF 0.1",
    "C2_t04": "C2 5 Pro＋CF 0.4",
    "D2_t01": "D2 5 Fast＋CF 0.1",
    "D2_t04": "D2 5 Fast＋CF 0.4",
    "E": "E ruri-v3-reranker",
}
MODELS = {
    "A": "—",
    "B": "`rerank-v4.0-pro`",
    "C": "`rerank-v5.0-pro`",
    "D": "`rerank-v5.0-fast`",
    "C2_t01": "`rerank-v5.0-pro`（Context Focus 0.1）",
    "C2_t04": "`rerank-v5.0-pro`（Context Focus 0.4）",
    "D2_t01": "`rerank-v5.0-fast`（Context Focus 0.1）",
    "D2_t04": "`rerank-v5.0-fast`（Context Focus 0.4）",
    "E": "`cl-nagoya/ruri-v3-reranker-310m`（CPU）",
}
# 公式発表（提供元の自社評価、日本語、RCP-nDCG@10）。実測値と混ぜないため別の表に出す
OFFICIAL_JA = {"rerank-v5.0-pro": 89.4, "rerank-v4.0-pro": 89.3, "rerank-v5.0-fast": 86.8}

# グラフに出す条件。C2・D2 は順位が C・D と同じかを表で示し、点が重なるのでグラフには出さない
FIG_CONDITIONS = ["A", "B", "C", "D", "E"]

# 色：1系列なので参照パレットの slot 1（青）だけを使い、基準線と文字は灰色・墨色にする
SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("", None):
                r[k] = None
            elif v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def fmt(v, digits=3, pct=False) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}％" if pct else f"{v:.{digits}f}"


def ci(s: dict, m: str) -> str:
    return f"{fmt(s[m])}（{fmt(s[m + '_ci_low'], 2)}〜{fmt(s[m + '_ci_high'], 2)}）"


def render_markdown(
    summary: list[dict], meta: dict, run_meta: dict, paired: list[dict] | None = None
) -> str:
    by = {s["condition"]: s for s in summary}
    run = run_meta["runs"][-1] if run_meta.get("runs") else {}
    n_acc = int(summary[0]["n_accuracy"])
    lines = [
        "# 比較表：Cohere Rerank 5 日本語RAG検証（実測）",
        "",
        (
            f"実行日：{run.get('started_at', '')[:10]}。コーパスは国税庁タックスアンサー（源泉所得税）345チャンク、"
            f"質問40問。1段目は `cl-nagoya/ruri-v3-310m` の上位20件を全条件で共通に使った。"
        ),
        "",
        (
            f"精度の指標は、答えがあり1段目の上位20件に正解が入った {n_acc} 問で計算した"
            f"（除外：{', '.join(meta['accuracy_excluded_not_in_top20']) or 'なし'}）。"
            "かっこ内は95％信頼区間（問題を復元抽出するブートストラップ）。"
        ),
        "",
        "## 精度",
        "",
        "| 条件 | モデル | Hit@1 | Recall@5 | MRR | nDCG@10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c, s in by.items():
        lines.append(
            f"| {LABELS.get(c, c)} | {MODELS.get(c, '')} | {fmt(s['hit@1'])} | {fmt(s['recall@5'])} "
            f"| {ci(s, 'mrr')} | {ci(s, 'ndcg@10')} |"
        )
    if paired:
        lines += [
            "",
            "## 条件の差（同じ問題どうしの差の平均と95％信頼区間）",
            "",
            ("信頼区間が0をまたぐ差は、この問題数では誤差と区別できない。"
            "「良い／悪い」は、その指標で前者が上回った／下回った問題の数。"),
            "",
            "| 比較 | Hit@1 の差 | MRR の差 | nDCG@10 の差 | MRR で良い／悪い |",
            "| --- | --- | --- | --- | --- |",
        ]
        by_pair: dict[tuple[str, str], dict[str, dict]] = {}
        for d in paired:
            by_pair.setdefault((d["a"], d["b"]), {})[d["metric"]] = d

        def diff(d: dict) -> str:
            return f"{d['diff']:+.3f}（{d['ci_low']:+.2f}〜{d['ci_high']:+.2f}）"

        for (a, b), ms in by_pair.items():
            mrr = ms["mrr"]
            lines.append(
                f"| {a} − {b} | {diff(ms['hit@1'])} | {diff(mrr)} | {diff(ms['ndcg@10'])} "
                f"| {int(mrr['a_better'])}／{int(mrr['b_better'])} |"
            )
    stable = [
        s for s in summary if s.get("rank_stable_rate") is not None and s["rank_stable_rate"] < 1
    ]
    if stable:
        lines += [
            "",
            "3回の実行で順位がすべて同じだった問題の割合は、"
            + "、".join(
                f"{LABELS.get(s['condition'], s['condition'])} が {s['rank_stable_rate'] * 100:.1f}％"
                for s in stable
            )
            + "で、ほかの条件はすべて100％だった。精度は1回目の順位で計算した。",
        ]
    lines += [
        "",
        "## 速さと費用",
        "",
        (
            "遅延は、問題ごとに3回の中央値を取り、40問の p50・p95 を出した。Cohere はネットワーク込み、"
            "E はCPU（4スレッド）でのローカル計算の時間で、性質が違うため直接は比べない。"
            "費用は応答の課金量（実測）に単価を掛けた。v5 の単価は公式ブログ、v4 は第三者サイトの値。"
        ),
        "",
        "| 条件 | p50（ms） | p95（ms） | 1問あたり費用（ドル） | 1問あたり課金量 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c, s in by.items():
        if s.get("latency_p50_ms") is None:
            continue
        if s.get("billed_search_units_per_question"):
            billed = f"{s['billed_search_units_per_question']:.0f} 検索単位"
        elif s.get("billed_input_tokens_per_question"):
            billed = f"{s['billed_input_tokens_per_question']:,.0f} トークン"
        else:
            billed = "—"
        cost = s.get("cost_per_question_usd")
        lines.append(
            f"| {LABELS.get(c, c)} | {s['latency_p50_ms']:,.0f} | {s['latency_p95_ms']:,.0f} "
            f"| {'—' if cost is None else f'{cost:.6f}'} | {billed} |"
        )
    cf = [c for c in by if by[c].get("token_reduction") is not None]
    if cf:
        lines += [
            "",
            "## Context Focus：トークン削減と正解箇所の保持",
            "",
            (
                f"上位5件の本文（トークン数は {meta['token_counter']} で数えた）を、スパンだけに削ったときの減り方。"
                "スパンが1つも返らない文書は、LLMに渡さない（0トークン）として数えた。"
                "保持率は、上位5件に入った正解チャンクの根拠の文字列が、スパンに丸ごと入っている割合。"
            ),
            "",
            "| 条件 | トークン削減率 | 上位5件のうちスパンなしの文書（平均） | 正解箇所の保持率 | 順位が Context Focus なしと同じ問題 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in cf:
            s = by[c]
            lines.append(
                f"| {LABELS.get(c, c)} | {fmt(s['token_reduction'], pct=True)} "
                f"| {s['top5_docs_without_spans']:.2f}件 | {fmt(s['evidence_retained'], pct=True)} "
                f"| {fmt(s.get('same_ranking_as_base'), pct=True)} |"
            )
    th = [c for c in ("C", "D") if by.get(c, {}).get("threshold") is not None]
    if th:
        lines += [
            "",
            "## 較正スコア：公式しきい値での分離",
            "",
            (
                "公式ブログのしきい値（Pro 0.50、Fast 0.45）以上を「関連あり」とみなした。"
                "適合率・再現率は、40問×候補20件の組のうち、正解チャンク（部分正解は含めない）を正しく拾えたかで数えた。"
            ),
            "",
            "| 条件 | しきい値 | 適合率 | 適合率（部分正解も正解とみなす） | 再現率 | 答えなし8問で全候補が下回った割合 | 答えあり32問で1件以上が上回った割合 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in th:
            s = by[c]
            lines.append(
                f"| {LABELS.get(c, c)} | {s['threshold']:.2f} | {fmt(s['threshold_precision'])} "
                f"| {fmt(s.get('threshold_precision_with_partial'))} "
                f"| {fmt(s['threshold_recall'])} | {fmt(s['no_answer_all_below'], pct=True)} "
                f"| {fmt(s['answerable_any_above'], pct=True)} |"
            )
    sep = [s for s in summary if s.get("no_answer_top1_max") is not None]
    if sep:
        lines += [
            "",
            "### 1位のスコアで答えの有無を分けられるか",
            "",
            (f"答えなし8問の1位のスコアの最大値と、精度の対象 {n_acc} 問の1位のスコアの最小値。"
            "最大値が最小値を下回れば、その間のどこかにしきい値を引けば分けられる（しきい値はこのデータで事後に決めることになる）。"),
            "",
            "| 条件 | 答えなしの最大 | 答えありの最小 | 答えありの中央値 | 間の幅 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for s in sep:
            gap = s["answerable_top1_min"] - s["no_answer_top1_max"]
            lines.append(
                f"| {LABELS.get(s['condition'], s['condition'])} | {fmt(s['no_answer_top1_max'])} "
                f"| {fmt(s['answerable_top1_min'])} | {fmt(s['answerable_top1_median'])} | {gap:+.3f} |"
            )
    lines += [
        "",
        "## 参考：公式発表の日本語スコア（提供元の自社評価、実測ではない）",
        "",
        "| モデル | 日本語 RCP-nDCG@10 |",
        "| --- | --- |",
    ]
    for m, v in OFFICIAL_JA.items():
        lines.append(f"| `{m}` | {v} |")
    lines += ["", "出典：[Cohere公式ブログ：Rerank 5](https://cohere.com/blog/rerank-5)", ""]
    return "\n".join(lines)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy(summary: list[dict], path: Path) -> None:
    conds = [s for s in summary if s["condition"] in FIG_CONDITIONS]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, facecolor=SURFACE)
    ys = list(range(len(conds)))[::-1]
    for ax, m, title in zip(axes, ["mrr", "ndcg@10"], ["MRR", "nDCG@10"], strict=True):
        _style(ax)
        base = next(s for s in conds if s["condition"] == "A")[m]
        ax.axvline(base, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
        for y, s in zip(ys, conds, strict=True):
            lo, hi = s[m + "_ci_low"], s[m + "_ci_high"]
            ax.plot([lo, hi], [y, y], color=SERIES, linewidth=2, solid_capstyle="round")
            ax.plot(
                s[m], y, "o", color=SERIES, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2
            )
            ax.annotate(
                f"{s[m]:.3f}",
                (hi, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        ax.set_xlim(min(s[m + "_ci_low"] for s in conds) - 0.05, 1.04)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([LABELS.get(s["condition"], s["condition"]) for s in conds], color=TEXT)
    n = int(conds[0]["n_accuracy"])
    fig.suptitle(
        f"条件ごとの精度（{n}問、線は95％信頼区間、破線はリランカーなし）",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_tradeoff(summary: list[dict], path: Path) -> None:
    shown = [s for s in summary if s["condition"] in FIG_CONDITIONS]
    api = [s for s in shown if s.get("cost_per_question_usd") is not None]
    with_latency = [s for s in shown if s.get("latency_p50_ms") is not None]
    base = next(s for s in summary if s["condition"] == "A")["ndcg@10"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, facecolor=SURFACE)
    panels = [
        (axes[0], with_latency, "latency_p50_ms", "遅延 p50（ms、対数目盛）", True),
        (axes[1], api, "cost_per_question_usd", "1問あたり費用（ドル）", False),
    ]
    for ax, rows, key, xlabel, logx in panels:
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.axhline(base, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
        for s in rows:
            local = s["condition"] == "E"
            ax.plot(
                s[key],
                s["ndcg@10"],
                "o",
                markersize=8,
                color=SURFACE if local else SERIES,
                markeredgecolor=SERIES,
                markeredgewidth=2,
            )
            ax.annotate(
                LABELS.get(s["condition"], s["condition"]),
                (s[key], s["ndcg@10"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=TEXT,
            )
        if logx:
            ax.set_xscale("log")
            ticks = [t for t in (200, 500, 1000, 2000, 5000, 10000, 20000)]
            lo, hi = min(s[key] for s in rows), max(s[key] for s in rows)
            ax.set_xticks([t for t in ticks if lo / 2 <= t <= hi * 2])
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(lo / 1.6, hi * 2.2)
        else:
            ax.set_xlim(0, max(s[key] for s in rows) * 1.3)
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    axes[0].set_ylabel("nDCG@10", color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle(
        "精度と遅延・費用（白抜きはCPUでのローカル計算、破線はリランカーなし）",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def render_cases(per_q: list[dict], qa: dict[str, dict], top: int = 10) -> str:
    """最初の正解の順位が条件によって大きく変わった問題（主要5条件の比較）。"""
    focus = ["A", "B", "C", "D", "E"]
    by_q: dict[str, dict[str, dict]] = {}
    for r in per_q:
        if r.get("first_relevant_rank") is not None and r["condition"] in focus:
            by_q.setdefault(r["qid"], {})[r["condition"]] = r
    spread = []
    for qid, conds in by_q.items():
        rr = [c["mrr"] for c in conds.values()]
        spread.append((max(rr) - min(rr), qid))
    lines = [
        "# 条件による差が大きい問題（自動抽出）",
        "",
        "最初の正解の順位（1位が最良）を条件ごとに並べた。逆順位の最大と最小の差が大きい順。",
        "",
        "| 問題 | 種類 | 質問 | " + " | ".join(focus) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in focus) + " |",
    ]
    for d, qid in sorted(spread, reverse=True)[:top]:
        q = qa[qid]
        ranks = [
            str(int(by_q[qid][c]["first_relevant_rank"])) if c in by_q[qid] else "—" for c in focus
        ]
        lines.append(f"| {qid} | {q['type']} | {q['question']} | " + " | ".join(ranks) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)
    summary = read_csv(run_dir / "summary.csv")
    per_q = read_csv(run_dir / "per_question.csv")
    meta = json.loads((run_dir / "evaluation_meta.json").read_text())
    run_meta = json.loads((run_dir / "run_meta.json").read_text())
    qa = {q["id"]: q for q in load_jsonl(QA_PATH)}

    paired_path = run_dir / "paired_diff.csv"
    paired = read_csv(paired_path) if paired_path.exists() else None
    (run_dir / "summary.md").write_text(render_markdown(summary, meta, run_meta, paired))
    plot_accuracy(summary, run_dir / "fig_accuracy.png")
    plot_tradeoff(summary, run_dir / "fig_tradeoff.png")
    (run_dir / "cases.md").write_text(render_cases(per_q, qa))
    print(f"wrote {run_dir}/summary.md, fig_accuracy.png, fig_tradeoff.png, cases.md")


if __name__ == "__main__":
    main()

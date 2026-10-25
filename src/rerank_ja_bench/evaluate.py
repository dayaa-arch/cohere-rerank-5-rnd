"""保存済みの生の応答から、問題ごとの指標と条件ごとの集計を作る。API は呼ばない。

    uv run python -m rerank_ja_bench.evaluate --run results/2026-10-25

出力（--run のディレクトリ）：
- per_question.csv：条件×問題ごとの指標
- summary.csv：条件ごとの集計
- evaluation_meta.json：集計の前提（トークナイザー、しきい値、除外した問題など）

前提：
- 精度は1回目の応答の順位で計算する。3回の順位が同じかどうかは rank_stable に残す
- 精度の対象は、答えがあり、1段目の上位20件に正解が入った問題（入らない問題は件数だけ記録）
- 遅延は問題ごとに3回の中央値を取り、その p50・p95 を出す
- 費用は応答の billed_units（実測）に単価を掛ける
- トークン削減率は、上位5件の本文と、Context Focus のスパンだけを残した本文のトークン数を比べる。
  スパンが1つもない文書は、LLMに渡さない（0トークン）として数える
- 正解箇所の保持率は、上位5件に入った正解チャンクの evidence が、スパンで覆われて残っている割合
- しきい値での分離は、公式しきい値（Pro 0.50、Fast 0.45）以上を「関連あり」とみなして数える
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from rerank_ja_bench.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from rerank_ja_bench.qa import CHUNKS_PATH, QA_PATH, load_jsonl
from rerank_ja_bench.run import PRICE_PER_1K_SEARCHES, PRICE_PER_M_TOKENS

TOP_K_CONTEXT = 5
OFFICIAL_THRESHOLDS = {"rerank-v5.0-pro": 0.50, "rerank-v5.0-fast": 0.45}
TOKENIZER_ID = "cl-nagoya/ruri-v3-310m"
TOKENIZER_REVISION = "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
ACCURACY_METRICS = ["hit@1", "hit@5", "recall@5", "mrr", "ndcg@10"]
# Context Focus ありの条件と、同じモデルで Context Focus なしの条件
SPANS_BASE = {"C2_t01": "C", "C2_t04": "C", "D2_t01": "D", "D2_t04": "D"}


def load_records(raw_dir: Path) -> dict[str, dict[str, dict[int, dict]]]:
    """records[条件][問題][回] = 1回分の記録。"""
    records: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(raw_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                records[r["condition"]][r["qid"]][r["rep"]] = r
    return records


def merge_spans(spans: list[dict]) -> list[tuple[int, int]]:
    """スパンを位置順に並べ、重なりやつながりをまとめる。"""
    merged: list[list[int]] = []
    for s in sorted(spans, key=lambda s: s["start"]):
        if merged and s["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s["end"])
        else:
            merged.append([s["start"], s["end"]])
    return [(a, b) for a, b in merged]


def compress(text: str, spans: list[dict] | None) -> str:
    """スパンの部分だけを位置順につなげた本文。スパンがなければ空文字列。"""
    if not spans:
        return ""
    return "".join(text[a:b] for a, b in merge_spans(spans))


def covered(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    """[start, end) がまとめたスパンのどれか1つに丸ごと入っているか。"""
    return any(a <= start and end <= b for a, b in ranges)


def cost_per_call(model: str | None, billed: dict) -> float | None:
    if model in PRICE_PER_M_TOKENS:
        return billed.get("input_tokens", 0) / 1e6 * PRICE_PER_M_TOKENS[model]
    if model in PRICE_PER_1K_SEARCHES:
        return billed.get("search_units", 0) / 1000 * PRICE_PER_1K_SEARCHES[model]
    return None


def percentile(values: list[float], p: float) -> float:
    """線形補間の百分位数（numpy の既定と同じ）。"""
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    """平均の95％信頼区間（問題を復元抽出するブートストラップ）。"""
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_SAMPLES)
    )
    return means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


def question_rows(
    qa: list[dict],
    chunks: dict[str, str],
    records: dict[str, dict[str, dict[int, dict]]],
    count_tokens,
) -> list[dict]:
    rows = []
    for cond, by_question in records.items():
        for q in qa:
            reps = by_question.get(q["id"], {})
            if not reps:
                continue
            first = reps[min(reps)]
            ranked = [d["chunk_id"] for d in first["ranked"]]
            candidates = first["candidate_ids"]
            rel, par = q["relevant_chunk_ids"], q["partial_chunk_ids"]
            model = first["params"].get("model")
            row = {
                "condition": cond,
                "qid": q["id"],
                "type": q["type"],
                "n_reps": len(reps),
                "rank_stable": all(
                    [d["chunk_id"] for d in r["ranked"]] == ranked for r in reps.values()
                ),
                "in_top20": bool(rel) and hit_at_k(candidates, rel, len(candidates)) == 1.0,
                "ranking": " ".join(ranked),
            }
            if row["in_top20"]:
                row.update(
                    {
                        "hit@1": hit_at_k(ranked, rel, 1),
                        "hit@5": hit_at_k(ranked, rel, 5),
                        "recall@5": recall_at_k(ranked, rel, 5),
                        "mrr": reciprocal_rank(ranked, rel),
                        "ndcg@10": ndcg_at_k(ranked, rel, par, 10),
                        "first_relevant_rank": next(i for i, c in enumerate(ranked, 1) if c in rel),
                    }
                )
            latencies = [r["latency_ms"] for r in reps.values()]
            if model:
                row["latency_ms_median"] = statistics.median(latencies)
                costs = [cost_per_call(model, r["billed"]) for r in reps.values()]
                if all(c is not None for c in costs):
                    row["cost_usd"] = statistics.mean(costs)
                    row["billed_input_tokens"] = statistics.mean(
                        r["billed"].get("input_tokens", 0) for r in reps.values()
                    )
                    row["billed_search_units"] = statistics.mean(
                        r["billed"].get("search_units", 0) for r in reps.values()
                    )

            top = first["ranked"][:TOP_K_CONTEXT]
            if first["params"].get("relevant_spans") and row["in_top20"]:
                full = sum(count_tokens(chunks[d["chunk_id"]]) for d in top)
                comp = sum(count_tokens(compress(chunks[d["chunk_id"]], d["spans"])) for d in top)
                row["top5_tokens_full"] = full
                row["top5_tokens_compressed"] = comp
                row["token_reduction"] = 1 - comp / full
                row["top5_docs_without_spans"] = sum(1 for d in top if not d["spans"])
                kept = total = 0
                spans_by_chunk = {d["chunk_id"]: merge_spans(d["spans"] or []) for d in top}
                for ev in q.get("evidence", []):
                    if ev["chunk_id"] not in spans_by_chunk:
                        continue
                    total += 1
                    start = chunks[ev["chunk_id"]].index(ev["text"])
                    kept += covered(start, start + len(ev["text"]), spans_by_chunk[ev["chunk_id"]])
                if total:
                    row["evidence_in_top5"] = total
                    row["evidence_retained"] = kept / total

            threshold = OFFICIAL_THRESHOLDS.get(model)
            if threshold is not None:
                above = [d["chunk_id"] for d in first["ranked"] if d["score"] >= threshold]
                row["threshold"] = threshold
                row["n_above"] = len(above)
                row["tp_above"] = sum(1 for c in above if c in rel)
                row["n_relevant_in_candidates"] = sum(1 for c in candidates if c in rel)
                row["top1_score"] = first["ranked"][0]["score"]
            rows.append(row)
    return rows


def summarize(rows: list[dict], conditions: list[str]) -> list[dict]:
    rng = random.Random(BOOTSTRAP_SEED)
    out = []
    for cond in conditions:
        g = [r for r in rows if r["condition"] == cond]
        acc = [r for r in g if r["in_top20"]]
        s: dict = {"condition": cond, "n_questions": len(g), "n_accuracy": len(acc)}
        s["rank_stable_rate"] = sum(r["rank_stable"] for r in g) / len(g)
        base = SPANS_BASE.get(cond)
        if base:
            base_rank = {r["qid"]: r["ranking"] for r in rows if r["condition"] == base}
            same = [r["ranking"] == base_rank.get(r["qid"]) for r in g if r["qid"] in base_rank]
            if same:
                s["same_ranking_as_base"] = sum(same) / len(same)
        for m in ACCURACY_METRICS:
            vals = [r[m] for r in acc]
            s[m] = statistics.mean(vals)
            if m in ("mrr", "ndcg@10"):
                s[f"{m}_ci_low"], s[f"{m}_ci_high"] = bootstrap_ci(vals, rng)
        lat = [r["latency_ms_median"] for r in g if "latency_ms_median" in r]
        if lat:
            s["latency_p50_ms"] = percentile(lat, 0.50)
            s["latency_p95_ms"] = percentile(lat, 0.95)
        billed = [r for r in g if "cost_usd" in r]
        if billed:
            s["cost_per_question_usd"] = statistics.mean(r["cost_usd"] for r in billed)
            s["billed_input_tokens_per_question"] = statistics.mean(
                r["billed_input_tokens"] for r in billed
            )
            s["billed_search_units_per_question"] = statistics.mean(
                r["billed_search_units"] for r in billed
            )
        red = [r for r in g if "token_reduction" in r]
        if red:
            s["token_reduction"] = statistics.mean(r["token_reduction"] for r in red)
            s["top5_docs_without_spans"] = statistics.mean(
                r["top5_docs_without_spans"] for r in red
            )
            ev = [r for r in red if "evidence_retained" in r]
            s["evidence_retained"] = sum(
                r["evidence_retained"] * r["evidence_in_top5"] for r in ev
            ) / sum(r["evidence_in_top5"] for r in ev)
        thr = [r for r in g if "threshold" in r]
        if thr:
            tp = sum(r["tp_above"] for r in thr)
            n_above = sum(r["n_above"] for r in thr)
            n_rel = sum(r["n_relevant_in_candidates"] for r in thr)
            s["threshold"] = thr[0]["threshold"]
            s["threshold_precision"] = tp / n_above if n_above else None
            s["threshold_recall"] = tp / n_rel if n_rel else None
            na = [r for r in thr if r["type"] == "no_answer"]
            s["no_answer_all_below"] = sum(r["n_above"] == 0 for r in na) / len(na)
            ans = [r for r in thr if r["type"] != "no_answer"]
            s["answerable_any_above"] = sum(r["n_above"] > 0 for r in ans) / len(ans)
        out.append(s)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def ruri_token_counter():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="results/<日付> のディレクトリ")
    args = parser.parse_args()
    run_dir = Path(args.run)

    qa = load_jsonl(QA_PATH)
    chunks = {c["chunk_id"]: c["text"] for c in load_jsonl(CHUNKS_PATH)}
    records = load_records(run_dir / "raw")
    order = ["A", "B", "C", "D", "C2_t01", "C2_t04", "D2_t01", "D2_t04", "E"]
    conditions = [c for c in order if c in records] + [c for c in records if c not in order]

    rows = question_rows(qa, chunks, records, ruri_token_counter())
    summary = summarize(rows, conditions)
    write_csv(run_dir / "per_question.csv", rows)
    write_csv(run_dir / "summary.csv", summary)
    excluded = sorted({r["qid"] for r in rows if r["type"] != "no_answer" and not r["in_top20"]})
    meta = {
        "accuracy_excluded_not_in_top20": excluded,
        "context_top_k": TOP_K_CONTEXT,
        "token_counter": f"{TOKENIZER_ID}@{TOKENIZER_REVISION} の tokenizer（特殊トークンなし）",
        "docs_without_spans": "0トークン（LLMに渡さない）として数える",
        "official_thresholds": OFFICIAL_THRESHOLDS,
        "cost_source": "応答の billed_units（実測）× 単価（v5：公式ブログ、v4：第三者サイト）",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "ci": 0.95},
    }
    (run_dir / "evaluation_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"wrote {run_dir}/per_question.csv, summary.csv ({len(conditions)} conditions)")


if __name__ == "__main__":
    main()

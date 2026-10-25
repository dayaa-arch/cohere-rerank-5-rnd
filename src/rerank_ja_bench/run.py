"""本実行。全条件で、保存済みの候補20件を同じ順でリランカーに渡し、生の応答をすべて保存する。

    uv run python -m rerank_ja_bench.run --dry-run            # 呼び出し回数と費用の見積もりだけ
    uv run python -m rerank_ja_bench.run --conditions A,B,C   # 一部の条件だけ
    uv run python -m rerank_ja_bench.run                      # 全条件

- 結果は results/<日付>/raw/<条件>.jsonl に1呼び出し1行で追記する。
- 同じ出力先で再実行すると、保存済みの（条件、問題、回）は飛ばす。途中で止まっても二重に課金しない。
- 遅延の測定がネットワークの時間帯に偏らないよう、問題ごとに条件の順番を回しながら交互に呼ぶ。
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from rerank_ja_bench.qa import CHUNKS_PATH, QA_PATH, load_jsonl
from rerank_ja_bench.rerankers.base import Reranker
from rerank_ja_bench.rerankers.cohere import CohereReranker
from rerank_ja_bench.rerankers.none import NoReranker
from rerank_ja_bench.retrieve import CANDIDATES_PATH

# 疎通確認（results/2026-10-25/smoke/）の課金トークン数から求めた、1文字あたりのトークン数。
# クエリを文書ごとに数える前提で 916 / (992 + 3 × 50) ≒ 0.80。見積もりにだけ使う。
TOKENS_PER_CHAR = 0.80
# 単価（ドル）。v5 は100万トークンあたりで公式ブログの値、v4 は1,000検索単位あたりで第三者サイトの値（要確認）
PRICE_PER_M_TOKENS = {"rerank-v5.0-pro": 0.05, "rerank-v5.0-fast": 0.02}
PRICE_PER_1K_SEARCHES = {"rerank-v4.0-pro": 2.50}


def _spans(threshold: float) -> dict:
    return {"mode": "enabled", "threshold": threshold}


CONDITIONS: dict[str, Callable[[], Reranker]] = {
    "A": NoReranker,
    "B": lambda: CohereReranker("B", "rerank-v4.0-pro"),
    "C": lambda: CohereReranker("C", "rerank-v5.0-pro"),
    "D": lambda: CohereReranker("D", "rerank-v5.0-fast"),
    "C2_t01": lambda: CohereReranker("C2_t01", "rerank-v5.0-pro", _spans(0.1)),
    "C2_t04": lambda: CohereReranker("C2_t04", "rerank-v5.0-pro", _spans(0.4)),
    "D2_t01": lambda: CohereReranker("D2_t01", "rerank-v5.0-fast", _spans(0.1)),
    "D2_t04": lambda: CohereReranker("D2_t04", "rerank-v5.0-fast", _spans(0.4)),
}


def _ruri() -> Reranker:
    from rerank_ja_bench.rerankers.ruri import RuriReranker

    return RuriReranker()


CONDITIONS["E"] = _ruri


def plan(qids: list[str], conditions: list[str], reps: int) -> list[tuple[int, str, str]]:
    """（回、問題、条件）の実行順。問題ごとに条件の順番を1つずつずらす。"""
    order = []
    for rep in range(1, reps + 1):
        for i, qid in enumerate(qids):
            shift = i % len(conditions)
            for cond in conditions[shift:] + conditions[:shift]:
                order.append((rep, qid, cond))
    return order


def load_done(raw_dir: Path) -> set[tuple[str, str, int]]:
    done = set()
    for path in raw_dir.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["qid"], r["rep"]))
    return done


def estimate(qa: list[dict], chunks: dict[str, str], candidates: dict[str, list[str]]) -> dict:
    """1回分（全問×候補20件）の課金量の見積もり。"""
    chars = sum(
        sum(len(chunks[c]) for c in candidates[q["id"]])
        + len(candidates[q["id"]]) * len(q["question"])
        for q in qa
    )
    return {"tokens_per_pass": round(chars * TOKENS_PER_CHAR), "searches_per_pass": len(qa)}


def cost_usd(model: str, tokens: float, searches: int) -> float:
    if model in PRICE_PER_M_TOKENS:
        return tokens / 1e6 * PRICE_PER_M_TOKENS[model]
    if model in PRICE_PER_1K_SEARCHES:
        return searches / 1000 * PRICE_PER_1K_SEARCHES[model]
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rpm", type=float, default=10, help="課金のある API の毎分の呼び出し上限")
    parser.add_argument("--max-calls", type=int, default=900, help="課金のある呼び出しの上限")
    parser.add_argument("--out", default=f"results/{date.today().isoformat()}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv()  # .env に COHERE_API_KEY があれば読む（設定済みの環境変数は上書きしない）

    conditions = args.conditions.split(",")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")

    qa = load_jsonl(QA_PATH)
    chunks = {c["chunk_id"]: c["text"] for c in load_jsonl(CHUNKS_PATH)}
    candidates = {
        r["id"]: [c["chunk_id"] for c in r["candidates"]] for r in load_jsonl(CANDIDATES_PATH)
    }
    out = Path(args.out)
    raw_dir = out / "raw"
    done = load_done(raw_dir) if raw_dir.exists() else set()
    todo = [
        s
        for s in plan([q["id"] for q in qa], conditions, args.reps)
        if (s[2], s[1], s[0]) not in done
    ]
    rerankers = {c: CONDITIONS[c]() for c in conditions}

    est = estimate(qa, chunks, candidates)
    billable_todo = [s for s in todo if rerankers[s[2]].billable]
    print(f"plan: {len(todo)} calls ({len(billable_todo)} billable), {len(done)} already done")
    total = 0.0
    for c in conditions:
        r = rerankers[c]
        n = sum(1 for s in todo if s[2] == c)
        model = r.params.get("model", "")
        usd = cost_usd(model, est["tokens_per_pass"] * n / len(qa), n) if r.billable else 0.0
        total += usd
        print(f"  {c:<7} {model:<32} calls {n:>4}  est ${usd:.4f}")
    print(f"  total est ${total:.4f}  (tokens/pass ≈ {est['tokens_per_pass']:,})")
    if args.dry_run:
        return
    if len(billable_todo) > args.max_calls:
        raise SystemExit(f"billable calls {len(billable_todo)} > --max-calls {args.max_calls}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
    run_info = {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "conditions": {c: {"class": type(r).__name__, **r.params} for c, r in rerankers.items()},
        "reps": args.reps,
        "rpm": args.rpm,
        "candidates": str(CANDIDATES_PATH),
        "versions": {"python": platform.python_version(), "httpx": httpx.__version__},
    }
    meta["runs"].append(run_info)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    qtext = {q["id"]: q["question"] for q in qa}
    min_interval = 60.0 / args.rpm
    last_billable = 0.0
    billed_total: dict[str, float] = {}
    seen_billing: set[str] = set()
    for n, (rep, qid, cond) in enumerate(todo, start=1):
        r = rerankers[cond]
        if r.billable:
            wait = last_billable + min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last_billable = time.monotonic()
        cand_ids = candidates[qid]
        started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        result = r.rerank(qtext[qid], [chunks[c] for c in cand_ids])
        record = {
            "condition": cond,
            "qid": qid,
            "rep": rep,
            "started_at": started_at,
            "latency_ms": round(result.latency_ms, 1),
            "params": r.params,
            "candidate_ids": cand_ids,
            "ranked": [
                {
                    "chunk_id": cand_ids[d.index],
                    **{k: v for k, v in asdict(d).items() if k != "index"},
                }
                for d in result.ranked
            ],
            "billed": result.billed,
            "raw": result.raw,
        }
        with (raw_dir / f"{cond}.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for k, v in result.billed.items():
            billed_total[k] = billed_total.get(k, 0) + (v or 0)
        if r.billable and cond not in seen_billing:
            # 課金単位が想定（v5 はトークン、v4 は検索単位）と合っているかを最初の1回で確かめる
            seen_billing.add(cond)
            print(f"  first call {cond}: billed {result.billed}", flush=True)
        if n % 20 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} done  billed so far {billed_total}", flush=True)

    run_info["ended_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    run_info["billed_total"] = billed_total
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {raw_dir}")


if __name__ == "__main__":
    main()

"""評価データから比較表までを1コマンドで作り直す。

    uv run python -m rerank_ja_bench.pipeline --run results/2026-10-25

既定では、保存済みの候補（data/candidates.jsonl）と生の応答（<run>/raw/）を使い、
API は呼ばない。チャンク分割と評価データの検証、集計、比較表とグラフの作成を順に行う。

- --retrieve：1段目の検索をやり直す（候補が変わると、保存済みの応答とは対応しなくなる）
- --call-api：本実行を行う（課金あり）。保存済みの（条件、問題、回）は飛ばす
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def step(*args: str) -> None:
    cmd = [sys.executable, "-m", *args]
    print("$", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="results/<日付> のディレクトリ")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--call-api", action="store_true")
    args = parser.parse_args()

    step("rerank_ja_bench.chunk")
    step("rerank_ja_bench.qa", "check")
    if args.retrieve:
        step("rerank_ja_bench.retrieve")
    if args.call_api:
        step("rerank_ja_bench.run", "--out", args.run)
    step("rerank_ja_bench.evaluate", "--run", args.run)
    step("rerank_ja_bench.report", "--run", args.run)


if __name__ == "__main__":
    main()

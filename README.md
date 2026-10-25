# cohere-rerank-5-rnd

日本語RAGで Cohere Rerank 5 に乗り換える価値があるかを、40問の小さな評価で確かめる検証リポジトリ。仕様は [docs/spec.md](docs/spec.md) にある。

現在はフェーズ3（1段目の検索）まで。報告は [docs/phase1_report.md](docs/phase1_report.md)、[docs/phase2_report.md](docs/phase2_report.md)、[docs/phase3_report.md](docs/phase3_report.md) を参照。コーパスの出典とライセンスは [data/corpus/README.md](data/corpus/README.md) にある。再現手順、実行日、モデルID、コーパスの出典とライセンスは、本実行のあとにここへ書く。

## セットアップ

```bash
uv sync
cp .env.example .env   # COHERE_API_KEY を書く（.env はコミットしない）
```

APIキーは環境変数 `COHERE_API_KEY` から読む。

## 評価データを作り直す

```bash
uv run python -m rerank_ja_bench.corpus fetch     # 国税庁のページを取得（data/corpus/raw/）
uv run python -m rerank_ja_bench.corpus extract   # 本文を抽出（data/corpus/docs.jsonl）
uv run python -m rerank_ja_bench.chunk            # チャンク分割（data/chunks.jsonl）
uv run python -m rerank_ja_bench.qa check         # 評価データの検証
uv run python -m rerank_ja_bench.retrieve         # 1段目の検索（data/candidates.jsonl）
uv run pytest
```

## ディレクトリ

```text
data/corpus/            元の文書と出典・ライセンス
data/chunks.jsonl       チャンク（フェーズ2）
data/qa.jsonl           質問と正解チャンクID（フェーズ2）
data/candidates.jsonl   1段目の上位20件（フェーズ3、全条件で共通）
src/rerank_ja_bench/    チャンク分割・検索・リランカーのアダプター・評価・集計
results/YYYY-MM-DD/     生の応答、問題ごとの結果、summary.csv
tests/                  指標計算のテスト
```

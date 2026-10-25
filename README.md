# cohere-rerank-5-rnd

日本語RAGで Cohere Rerank 5 に乗り換える価値があるかを、40問の小さな評価で確かめる検証リポジトリ。仕様は [docs/spec.md](docs/spec.md) にある。

現在はフェーズ1（準備）まで。結果は [docs/phase1_report.md](docs/phase1_report.md) を参照。再現手順、実行日、モデルID、コーパスの出典とライセンスは、本実行のあとにここへ書く。

## セットアップ

```bash
uv sync
cp .env.example .env   # COHERE_API_KEY を書く（.env はコミットしない）
```

APIキーは環境変数 `COHERE_API_KEY` から読む。

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

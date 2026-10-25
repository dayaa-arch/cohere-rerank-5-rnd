# cohere-rerank-5-rnd

日本語RAGで Cohere Rerank 5 に乗り換える価値があるかを、40問の小さな評価で確かめる検証リポジトリ。仕様は [docs/spec.md](docs/spec.md) にある。

現在はフェーズ5（集計）まで。結果は [docs/phase5_report.md](docs/phase5_report.md) と、比較表 [results/2026-10-25/summary.md](results/2026-10-25/summary.md) を参照。

## 結果の要約（2026-10-25 実行、実測）

| 条件 | MRR | nDCG@10 | 遅延 p50 | 1問あたり費用 |
| --- | --- | --- | --- | --- |
| リランカーなし | 0.801 | 0.860 | — | — |
| `rerank-v4.0-pro` | 0.930 | 0.939 | 499 ms | 0.0025 ドル |
| `rerank-v5.0-pro` | 0.962 | 0.945 | 434 ms | 0.00018 ドル |
| `rerank-v5.0-fast` | 0.946 | 0.932 | 416 ms | 0.00011 ドル |
| `cl-nagoya/ruri-v3-reranker-310m`（CPU） | 0.941 | 0.928 | 7,789 ms | — |

- どのリランカーも、リランカーなしより明らかに精度が上がる。モデルどうしの差は、31問では誤差の範囲
- Context Focus は、短いチャンク（平均約280字）ではトークンを1.5〜10％しか減らさなかった。正解箇所は100％残った
- 公式しきい値（Pro 0.50、Fast 0.45）で、答えのない8問はすべての候補を弾けた
- `rerank-v4.0-pro` の単価は第三者サイトの値（1,000検索単位あたり2.50ドル）で、公式には確かめていない

## 実行条件

| 項目 | 内容 |
| --- | --- |
| 実行日 | 2026-10-25（本実行 09:15〜09:41 UTC） |
| リランカー | `rerank-v4.0-pro`、`rerank-v5.0-pro`、`rerank-v5.0-fast`（Cohere v2 API `https://api.cohere.com/v2/rerank`）、`cl-nagoya/ruri-v3-reranker-310m`（リビジョン `bb46934e`） |
| Context Focus | `relevant_spans`（`mode: enabled`、`threshold` 0.1 と 0.4） |
| 1段目の検索 | `cl-nagoya/ruri-v3-310m`（リビジョン `18b60fb8`）、コサイン類似度の上位20件 |
| 評価データ | 40問（事実12、言い換え12、複数箇所8、答えなし8）。精度は1段目の上位20件に正解が入った31問で計算 |
| 回数 | 各条件40問×3回。精度は1回目、遅延は3回の中央値 |
| 環境 | Python 3.13、httpx 0.28、sentence-transformers 6.2、torch 2.14（CPU、4スレッド） |

モデルのパラメータ、ライブラリの版、課金量の合計は [results/2026-10-25/run_meta.json](results/2026-10-25/run_meta.json) に記録してある。

## コーパスの出典とライセンス

国税庁ホームページ「タックスアンサー（よくある税の質問）」の源泉所得税65項目（2026-10-25取得、令和8年4月1日現在法令等）。[公共データ利用規約（第1.0版）](https://www.nta.go.jp/chuijiko/copy.htm)（CC BY 4.0 と互換）に従って利用している。

`data/corpus/docs.jsonl` と `data/chunks.jsonl` は、「タックスアンサー（源泉所得税）」（国税庁）（https://www.nta.go.jp/taxes/shiraberu/taxanswer/code/index.htm）を加工して作成した。加工の内容は [data/corpus/README.md](data/corpus/README.md) にある。

## 再現手順

```bash
uv sync
uv run python -m rerank_ja_bench.pipeline --run results/2026-10-25   # 比較表とグラフまで（API は呼ばない）
uv run pytest                                                          # 指標の計算などのテスト
```

`pipeline` は、チャンク分割、評価データの検証、集計、比較表とグラフの作成を順に行う。保存済みの候補（`data/candidates.jsonl`）と生の応答（`results/2026-10-25/raw/`）を使うので、API キーは要らない。

最初から作り直す場合（課金あり）：

```bash
cp .env.example .env   # COHERE_API_KEY を書く（.env はコミットしない）。環境変数でもよい
uv run python -m rerank_ja_bench.corpus fetch     # 国税庁のページを取得（data/corpus/raw/）
uv run python -m rerank_ja_bench.corpus extract   # 本文を抽出（data/corpus/docs.jsonl）
uv run python -m rerank_ja_bench.pipeline --run results/<日付> --retrieve --call-api
```

`--call-api` の前に `uv run python -m rerank_ja_bench.run --dry-run` で呼び出し回数と費用の見積もりを確かめられる。今回の実費は0.40ドルだった。

## ディレクトリ

```text
data/corpus/                元の文書（HTML）、抽出した本文、出典とライセンス
data/chunks.jsonl           チャンク（345件）
data/qa.jsonl               質問、正解・部分正解のチャンクID、根拠の文字列
data/candidates.jsonl       1段目の上位20件（全条件で共通）
src/rerank_ja_bench/        チャンク分割・検索・リランカーのアダプター・本実行・集計・比較表
results/2026-10-25/         生の応答（raw/）、問題ごとの結果、summary.csv、比較表、グラフ
docs/                       仕様書とフェーズごとの報告
tests/                      指標の計算などのテスト
```

リランカーは `src/rerank_ja_bench/rerankers/` の共通インターフェース（`Reranker.rerank()`）で呼ぶ。別のリランカーは、ここにアダプターを1つ足し、`run.py` の `CONDITIONS` に登録すれば比べられる。

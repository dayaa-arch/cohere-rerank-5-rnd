# コーパス：国税庁タックスアンサー（源泉所得税）

出典：国税庁ホームページ「タックスアンサー（よくある税の質問）」源泉所得税の65項目
（https://www.nta.go.jp/taxes/shiraberu/taxanswer/code/index.htm）

ライセンス：[公共データ利用規約（第1.0版）](https://www.nta.go.jp/chuijiko/copy.htm)。CC BY 4.0 と互換。

取得日：2026-10-25（項目ごとの取得日時は `manifest.json`）。各ページは「令和8年4月1日現在法令等」の内容。

## ファイル

| ファイル | 内容 |
| --- | --- |
| `raw/{番号}.html` | 取得したページそのまま（出典：国税庁ホームページ、各ページのURLは `https://www.nta.go.jp/taxes/shiraberu/taxanswer/gensen/{番号}.htm`） |
| `manifest.json` | 取得元、ライセンス、取得日時 |
| `docs.jsonl` | `raw/` から本文を取り出したもの |

`docs.jsonl` と `../chunks.jsonl` は、「タックスアンサー（源泉所得税）」（国税庁）（https://www.nta.go.jp/taxes/shiraberu/taxanswer/code/index.htm）を加工して作成した。加工の内容は次のとおり。

- 「対象税目」「根拠法令等」「関連リンク」「関連コード」「QAリンク」「お問い合わせ先」の節を除いた
- 表は1行ずつ「セル | セル」の形の文に直した。画像の算式（No.2517 など）は取り込んでいない
- 400字を目安にチャンクに分け、先頭に「ページ名｜見出し」を付けた（`src/rerank_ja_bench/chunk.py`）

再取得：`uv run python -m rerank_ja_bench.corpus fetch && uv run python -m rerank_ja_bench.corpus extract`。ページは税制改正で更新されるので、検証には保存済みの `raw/` を使う。

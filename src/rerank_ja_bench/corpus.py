"""国税庁タックスアンサー（源泉所得税）の取得と本文抽出。

    uv run python -m rerank_ja_bench.corpus fetch    # data/corpus/raw/*.html を取得
    uv run python -m rerank_ja_bench.corpus extract  # data/corpus/docs.jsonl を作る

出典：国税庁ホームページ（公共データ利用規約（第1.0版））。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://www.nta.go.jp"
INDEX_URL = f"{BASE_URL}/taxes/shiraberu/taxanswer/code/index.htm"
CATEGORY = "gensen"

CORPUS_DIR = Path("data/corpus")
RAW_DIR = CORPUS_DIR / "raw"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
DOCS_PATH = CORPUS_DIR / "docs.jsonl"

# 本文として使わない節。対象税目は全ページ同じ、根拠法令等は条文番号の列、残りは他ページへのリンク集。
SKIP_SECTIONS = {"対象税目", "根拠法令等", "関連リンク", "関連コード", "QAリンク", "お問い合わせ先"}

USER_AGENT = "rerank-ja-bench/0.1 (research; contact via GitHub dayaa-arch/cohere-rerank-5-rnd)"


def doc_url(code: str) -> str:
    return f"{BASE_URL}/taxes/shiraberu/taxanswer/{CATEGORY}/{code}.htm"


def list_codes(client: httpx.Client) -> list[str]:
    """コード一覧ページから、分野 CATEGORY の項目番号を掲載順に返す。"""
    html = client.get(INDEX_URL).raise_for_status().text
    codes = re.findall(rf'href="[^"]*taxanswer/{CATEGORY}/(\d{{4}})\.htm"', html)
    return list(dict.fromkeys(codes))


def fetch(delay: float = 1.0) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched = {}
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True
    ) as client:
        codes = list_codes(client)
        for code in codes:
            res = client.get(doc_url(code)).raise_for_status()
            (RAW_DIR / f"{code}.html").write_bytes(res.content)
            fetched[code] = datetime.now(UTC).isoformat(timespec="seconds")
            time.sleep(delay)
    manifest = {
        "source": "国税庁ホームページ タックスアンサー（源泉所得税）",
        "index_url": INDEX_URL,
        "license": "公共データ利用規約（第1.0版）",
        "license_url": "https://www.nta.go.jp/chuijiko/copy.htm",
        "fetched_at": fetched,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"fetched {len(codes)} pages -> {RAW_DIR}")


def _clean(text: str) -> str:
    text = text.replace("　", " ")
    return re.sub(r"\s+", " ", text).strip()


def _table_rows(table: Tag) -> list[str]:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean(c.get_text()) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def extract_doc(html: str, code: str) -> dict:
    """1ページから、見出しごとの本文ブロック（段落・箇条・表の行）を取り出す。"""
    # 閉じタグのない <p> があるため、HTML5 と同じく次の <p> で段落を閉じる lxml を使う
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("#bodyArea")
    title = _clean(body.find("h1").get_text())
    title = re.sub(r"^No\.\d{4}\s*", "", title)

    as_of = None
    sections: list[dict] = []
    h2: str | None = None
    current: dict | None = None

    for el in body.find_all(["h2", "h3", "h4", "p", "li", "table"]):
        # 表の中の段落や入れ子の箇条は、外側の要素でまとめて扱う
        if el.find_parent("table") is not None:
            continue
        if el.name == "li" and el.find_parent("li") is not None:
            continue
        if el.name == "p" and el.find_parent("li") is not None:
            continue
        if el.name in ("h2", "h3", "h4"):
            heading = _clean(el.get_text())
            if el.name == "h2":
                h2 = heading
                path = [heading]
            else:
                path = [h2, heading] if h2 else [heading]
            current = {
                "heading": " > ".join(p for p in path if p),
                "skip": h2 in SKIP_SECTIONS,
                "blocks": [],
            }
            sections.append(current)
            continue
        if el.name == "table":
            blocks = _table_rows(el)
        else:
            blocks = [_clean(el.get_text())]
        for text in blocks:
            if not text:
                continue
            m = re.fullmatch(r"\[(.+現在法令等)\]", text)
            if m and as_of is None:
                as_of = m.group(1)
                continue
            if current is None:
                continue
            current["blocks"].append(text)

    return {
        "doc_id": code,
        "title": title,
        "url": doc_url(code),
        "as_of": as_of,
        "sections": [
            {"heading": s["heading"], "blocks": s["blocks"]}
            for s in sections
            if not s["skip"] and s["blocks"]
        ],
    }


def extract() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    codes = list(manifest["fetched_at"])
    with DOCS_PATH.open("w") as f:
        for code in codes:
            html = (RAW_DIR / f"{code}.html").read_text(encoding="utf-8")
            doc = extract_doc(html, code)
            doc["fetched_at"] = manifest["fetched_at"][code]
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"extracted {len(codes)} docs -> {DOCS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fetch", "extract"])
    args = parser.parse_args()
    {"fetch": fetch, "extract": extract}[args.command]()


if __name__ == "__main__":
    main()

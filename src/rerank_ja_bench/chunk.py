"""docs.jsonl をチャンクに分ける。全条件で同じチャンクを使う。

    uv run python -m rerank_ja_bench.chunk   # data/corpus/docs.jsonl -> data/chunks.jsonl

分け方：
- 節（h2/h3 の見出し）をまたがない。
- 節の中のブロック（段落・箇条・表の行）を、本文が MAX_CHARS 字を超えない範囲で順に詰める。
- 1ブロックが MAX_CHARS 字を超えるときは、句点「。」の直後で切る。
- 節の最後のチャンクが MIN_TAIL_CHARS 字未満なら、直前のチャンクに足しても
  MAX_CHARS + MIN_TAIL_CHARS 字以内に収まる場合に限り、直前のチャンクにまとめる。
- チャンクの先頭行に「ページ名｜見出し」を付ける（どのリランカーにも同じ文字列を渡す）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MAX_CHARS = 400
MIN_TAIL_CHARS = 80

DOCS_PATH = Path("data/corpus/docs.jsonl")
CHUNKS_PATH = Path("data/chunks.jsonl")


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """max_chars を超える文を句点の直後で切る。句点がなければ max_chars ごとに切る。"""
    if len(text) <= max_chars:
        return [text]
    sentences = [s for s in re.split(r"(?<=。)", text) if s]
    pieces: list[str] = []
    buf = ""
    for s in sentences:
        while len(s) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(s[:max_chars])
            s = s[max_chars:]
        if buf and len(buf) + len(s) > max_chars:
            pieces.append(buf)
            buf = ""
        buf += s
    if buf:
        pieces.append(buf)
    return pieces


def pack_blocks(
    blocks: list[str], max_chars: int = MAX_CHARS, min_tail: int = MIN_TAIL_CHARS
) -> list[str]:
    """1つの節のブロックを、改行でつないだ本文のまとまりに詰める。"""
    units = [u for b in blocks for u in split_long(b, max_chars)]
    bodies: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for u in units:
        add = len(u) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            bodies.append(cur)
            cur, cur_len = [], 0
            add = len(u)
        cur.append(u)
        cur_len += add
    if cur:
        bodies.append(cur)

    texts = ["\n".join(b) for b in bodies]
    if len(texts) >= 2 and len(texts[-1]) < min_tail:
        merged = texts[-2] + "\n" + texts[-1]
        if len(merged) <= max_chars + min_tail:
            texts[-2:] = [merged]
    return texts


def chunk_doc(doc: dict) -> list[dict]:
    chunks = []
    for section in doc["sections"]:
        for body in pack_blocks(section["blocks"]):
            n = len(chunks) + 1
            chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}-{n:02d}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "heading": section["heading"],
                    "text": f"{doc['title']}｜{section['heading']}\n{body}",
                }
            )
    return chunks


def main() -> None:
    docs = [json.loads(line) for line in DOCS_PATH.read_text().splitlines()]
    chunks = [c for d in docs for c in chunk_doc(d)]
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    lengths = sorted(len(c["text"]) for c in chunks)
    print(
        f"{len(docs)} docs -> {len(chunks)} chunks -> {CHUNKS_PATH} "
        f"(chars: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]})"
    )


if __name__ == "__main__":
    main()

from rerank_ja_bench.chunk import chunk_doc, pack_blocks, split_long


def test_split_long_keeps_short_text():
    assert split_long("短い文。", max_chars=10) == ["短い文。"]


def test_split_long_cuts_after_period():
    text = "あいうえお。かきくけこ。さしすせそ。"
    assert split_long(text, max_chars=12) == ["あいうえお。かきくけこ。", "さしすせそ。"]


def test_split_long_without_period_cuts_by_length():
    assert split_long("あ" * 25, max_chars=10) == ["あ" * 10, "あ" * 10, "あ" * 5]


def test_split_long_never_loses_text():
    text = "長い文" * 50 + "。短い。" + "い" * 30 + "。"
    assert "".join(split_long(text, max_chars=40)) == text
    assert all(len(p) <= 40 for p in split_long(text, max_chars=40))


def test_pack_blocks_respects_max_chars():
    blocks = ["あ" * 30, "い" * 30, "う" * 30]
    # 30 + 改行 + 30 = 61 は 70 以内、3つ目を足すと 92 になるので分かれる
    assert pack_blocks(blocks, max_chars=70, min_tail=0) == [
        "あ" * 30 + "\n" + "い" * 30,
        "う" * 30,
    ]


def test_pack_blocks_merges_short_tail():
    blocks = ["あ" * 60, "い" * 10]
    assert pack_blocks(blocks, max_chars=65, min_tail=20) == ["あ" * 60 + "\n" + "い" * 10]


def test_chunk_doc_does_not_cross_sections_and_numbers_ids():
    doc = {
        "doc_id": "9999",
        "title": "テスト",
        "sections": [
            {"heading": "概要", "blocks": ["一つ目。"]},
            {"heading": "概要 > 詳細", "blocks": ["二つ目。"]},
        ],
    }
    chunks = chunk_doc(doc)
    assert [c["chunk_id"] for c in chunks] == ["9999-01", "9999-02"]
    assert chunks[0]["text"] == "テスト｜概要\n一つ目。"
    assert chunks[1]["text"] == "テスト｜概要 > 詳細\n二つ目。"

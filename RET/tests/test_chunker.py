"""Chunker smoke tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunker import build_split_blocks
from models import Metadata


def test_build_split_blocks_basic():
    meta = Metadata(
        id="pmid_12345",
        title="Statin therapy for secondary prevention",
        year="2024",
        source_type="JAMA",
        text_summary="A summary of statin evidence.",
        text="Paragraph one.\n\nParagraph two with more content.",
        image_summary=["Figure 1: survival curve."],
    )
    blocks = build_split_blocks(meta)
    assert blocks
    types = {b.chunk_type for b in blocks}
    assert "title" in types
    assert "summary" in types
    assert "text" in types
    assert "image_desc" in types
    # 每个 block 都有 paper_id 和有效文本
    for b in blocks:
        assert b.paper_id == "pmid_12345"
        assert b.text.strip()
        assert b.id.startswith("pmid_12345_")


def test_chunk_text_no_chunk_exceeds_size():
    from chunker import chunk_text
    from config import CHUNK_SIZE

    # 构造一个很长的段落
    long_text = "Sentence one. " * 1000
    chunks = chunk_text(long_text)
    assert chunks
    for c in chunks:
        # token 数不应超过上限
        import bge_tokenizer
        assert bge_tokenizer.count_tokens(c) <= CHUNK_SIZE


def test_year_normalized_to_int():
    meta = Metadata(
        id="pmid_99999",
        title="Test",
        year="2020",
        text="Some text.",
    )
    blocks = build_split_blocks(meta)
    text_blocks = [b for b in blocks if b.chunk_type == "text"]
    assert text_blocks
    assert text_blocks[0].metadata.get("year") == 2020

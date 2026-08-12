"""
文本分块模块
- text 全文：按段落边界语义分割，512 tokens/chunk, 50 overlap
- title / text_summary / image_summary：各一条 chunk
"""
from __future__ import annotations

import re
import tiktoken
from typing import Any

from config import CHUNK_SIZE, CHUNK_OVERLAP, TOKENIZER_ENCODING, SOURCE_TYPE_BLACKLIST

_tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def _split_paragraphs(text: str) -> list[str]:
    """按双换行或单换行分段，保留段落完整性"""
    # 先按双换行分
    raw = re.split(r"\n{2,}", text)
    paragraphs = []
    for block in raw:
        block = block.strip()
        if not block:
            continue
        # 太长的段落再按单换行分
        if _count_tokens(block) > CHUNK_SIZE:
            lines = block.split("\n")
            for line in lines:
                line = line.strip()
                if line:
                    paragraphs.append(line)
        else:
            paragraphs.append(block)
    return paragraphs


def chunk_text(text: str) -> list[str]:
    """
    将全文 text 切分为 chunks。
    策略：按段落边界合并，超过 CHUNK_SIZE 则另起新 chunk，
    overlap 取上一 chunk 的最后一个段落。
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = _count_tokens(para)

        if current_len + para_len > CHUNK_SIZE and current:
            # 当前 chunk 已满，保存
            chunks.append("\n".join(current))

            # overlap: 取上一个 chunk 的最后一段作为开头
            if CHUNK_OVERLAP > 0 and len(current) >= 1:
                overlap_para = current[-1]
                current = [overlap_para]
                current_len = _count_tokens(overlap_para)
            else:
                current = []
                current_len = 0

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n".join(current))

    return chunks


def clean_source_type(raw: str | None) -> str | None:
    """清洗 source_type 字段"""
    if not raw:
        return None
    val = str(raw).strip()
    if not val:
        return None
    low = val.lower()
    for bad in SOURCE_TYPE_BLACKLIST:
        if bad in low:
            return None
    # 截断异常长的（>200 chars 大概率不是期刊名）
    if len(val) > 200:
        return None
    return val


def build_chunks(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """
    输入：一篇论文的 metadata dict
    输出：chunk 列表，每个 chunk 含 id, document, chunk_type, metadata
    """
    paper_id = str(metadata.get("id", ""))
    title = metadata.get("title") or ""
    text_summary = metadata.get("text_summary") or ""
    text = metadata.get("text") or ""
    image_summary: list[str] = metadata.get("image_summary") or []
    year = metadata.get("year")
    source_type = clean_source_type(metadata.get("source_type"))

    chunks: list[dict[str, Any]] = []

    # 1) title chunk
    if title.strip():
        chunks.append({
            "id": f"{paper_id}_title",
            "document": title.strip(),
            "chunk_type": "title",
            "metadata": {
                "paper_id": paper_id,
                "chunk_type": "title",
                "year": year,
                "source_type": source_type,
            },
        })

    # 2) text_summary chunk
    if text_summary.strip():
        chunks.append({
            "id": f"{paper_id}_summary",
            "document": text_summary.strip(),
            "chunk_type": "summary",
            "metadata": {
                "paper_id": paper_id,
                "chunk_type": "summary",
                "year": year,
                "source_type": source_type,
            },
        })

    # 3) text chunks
    text_chunks = chunk_text(text)
    for idx, tc in enumerate(text_chunks):
        chunks.append({
            "id": f"{paper_id}_text_{idx}",
            "document": tc,
            "chunk_type": "text",
            "metadata": {
                "paper_id": paper_id,
                "chunk_type": "text",
                "chunk_index": idx,
                "year": year,
                "source_type": source_type,
            },
        })

    # 4) image_summary chunks（文本层，每个图描述一条）
    for idx, img_desc in enumerate(image_summary):
        if img_desc and img_desc.strip():
            chunks.append({
                "id": f"{paper_id}_img_{idx}",
                "document": img_desc.strip(),
                "chunk_type": "image_desc",
                "metadata": {
                    "paper_id": paper_id,
                    "chunk_type": "image_desc",
                    "image_index": idx,
                    "year": year,
                    "source_type": source_type,
                },
            })

    return chunks

"""
文本分块模块
- text 全文：按段落边界语义分割，512 tokens/chunk, 50 overlap
- title / text_summary / image_summary：各一条 chunk

改造重点：
1. 统一返回 SplitBlock 对象（不再是 dict）
2. SplitBlock 本身不包含 embedding，embedding 由外部向量库管理
3. 使用真实 BGE-M3 tokenizer 统计 tokens
4. 按 token 实现滑动窗口 overlap
5. 超长段落按 sentence > whitespace > char 强制切分，保证单 chunk 不超过 CHUNK_SIZE
"""
from __future__ import annotations

import re
from typing import Any

import bge_tokenizer
from config import CHUNK_SIZE, CHUNK_OVERLAP, SOURCE_TYPE_BLACKLIST
from models import Metadata, SplitBlock


_tokenizer_count = bge_tokenizer.count_tokens


def _count_tokens(text: str) -> int:
    return _tokenizer_count(text)


def _split_paragraphs(text: str) -> list[str]:
    """按双换行或单换行分段，保留段落完整性"""
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


def _split_by_sentences(text: str) -> list[str]:
    """按句子边界切分，保留标点。"""
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _split_by_char(text: str, max_tokens: int) -> list[str]:
    """按字符切分，保证每段不超过 max_tokens 个 token。"""
    pieces: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if _count_tokens(candidate) > max_tokens and current:
            pieces.append(current)
            current = char
        else:
            current = candidate
    if current:
        pieces.append(current)
    if not pieces:
        pieces = [text]
    return pieces


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """
    将长文本切分为每段不超过 max_tokens 的小块。
    优先按句子边界，其次空白，最后字符。
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    sentences = _split_by_sentences(text)
    if len(sentences) > 1:
        pieces: list[str] = []
        current = ""
        current_tokens = 0
        for sent in sentences:
            sent_tokens = _count_tokens(sent)
            if sent_tokens > max_tokens:
                # 单句超长：先 flush current，再递归切分该句
                if current:
                    pieces.append(current)
                    current = ""
                    current_tokens = 0
                pieces.extend(_split_long_text(sent, max_tokens))
            elif current_tokens + sent_tokens <= max_tokens:
                current = f"{current} {sent}".strip() if current else sent
                current_tokens += sent_tokens
            else:
                pieces.append(current)
                current = sent
                current_tokens = sent_tokens
        if current:
            pieces.append(current)
        return pieces

    # 无句子边界：按空白切分
    words = text.split()
    if len(words) > 1:
        pieces = []
        current = ""
        current_tokens = 0
        for word in words:
            word_tokens = _count_tokens(word)
            if word_tokens > max_tokens:
                if current:
                    pieces.append(current)
                    current = ""
                    current_tokens = 0
                pieces.extend(_split_long_text(word, max_tokens))
            elif current_tokens + word_tokens <= max_tokens:
                current = f"{current} {word}".strip() if current else word
                current_tokens += word_tokens
            else:
                pieces.append(current)
                current = word
                current_tokens = word_tokens
        if current:
            pieces.append(current)
        return pieces if len(pieces) > 1 else _split_by_char(text, max_tokens)

    # 单个超长词：按字符切
    return _split_by_char(text, max_tokens)


def _overlap_units(units: list[str], max_overlap_tokens: int) -> list[str]:
    """从 units 末尾取尽可能多的单元，使其总 token 数不超过 max_overlap_tokens。"""
    overlap: list[str] = []
    overlap_tokens = 0
    for unit in reversed(units):
        unit_tokens = _count_tokens(unit)
        if overlap_tokens + unit_tokens <= max_overlap_tokens:
            overlap.insert(0, unit)
            overlap_tokens += unit_tokens
        else:
            break
    return overlap


def chunk_text(text: str) -> list[str]:
    """
    将全文 text 切分为 chunks。
    策略：按段落边界合并，超过 CHUNK_SIZE 则另起新 chunk，
    overlap 按真实 token 数从上一 chunk 末尾截取。
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    # 将所有段落进一步拆成不超过 CHUNK_SIZE 的单元
    units: list[str] = []
    for para in paragraphs:
        units.extend(_split_long_text(para, CHUNK_SIZE))

    if not units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens(unit)

        # 防御：单元不应超过 CHUNK_SIZE；若超过则再切分
        if unit_tokens > CHUNK_SIZE:
            sub_units = _split_long_text(unit, CHUNK_SIZE)
            for sub in sub_units:
                sub_tokens = _count_tokens(sub)
                if current_tokens + sub_tokens > CHUNK_SIZE and current_units:
                    chunks.append("\n".join(current_units))
                    current_units = _overlap_units(current_units, CHUNK_OVERLAP)
                    current_tokens = sum(_count_tokens(u) for u in current_units)
                current_units.append(sub)
                current_tokens += sub_tokens
            continue

        if current_tokens + unit_tokens > CHUNK_SIZE and current_units:
            chunks.append("\n".join(current_units))
            current_units = _overlap_units(current_units, CHUNK_OVERLAP)
            current_tokens = sum(_count_tokens(u) for u in current_units)

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunks.append("\n".join(current_units))

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


def _parse_year(year: Any) -> int | None:
    """将 year 解析为 int；无法解析时返回 None。"""
    if year is None:
        return None
    if isinstance(year, int):
        return year
    try:
        return int(str(year).strip())
    except (ValueError, TypeError):
        return None


def build_split_blocks(metadata: Metadata) -> list[SplitBlock]:
    """
    输入：一篇论文的 Metadata
    输出：SplitBlock 列表，每个 block 含 id, paper_id, chunk_type, text, position, metadata
    """
    paper_id = str(metadata.id) if metadata.id else ""
    if not paper_id:
        raise ValueError("Metadata.id cannot be empty")

    title = metadata.title or ""
    text_summary = metadata.text_summary or ""
    text = metadata.text or ""
    image_summary: list[str] = metadata.image_summary or []
    year = _parse_year(metadata.year)
    source_type = clean_source_type(metadata.source_type)

    base_meta = {
        "year": year,
        "source_type": source_type,
    }

    blocks: list[SplitBlock] = []

    # 1) title block
    if title.strip():
        blocks.append(SplitBlock(
            id=f"{paper_id}_title",
            paper_id=paper_id,
            chunk_type="title",
            text=title.strip(),
            position=0,
            metadata={**base_meta, "chunk_type": "title"},
        ))

    # 2) text_summary block
    if text_summary.strip():
        blocks.append(SplitBlock(
            id=f"{paper_id}_summary",
            paper_id=paper_id,
            chunk_type="summary",
            text=text_summary.strip(),
            position=0,
            metadata={**base_meta, "chunk_type": "summary"},
        ))

    # 3) text blocks
    text_chunks = chunk_text(text)
    for idx, tc in enumerate(text_chunks):
        blocks.append(SplitBlock(
            id=f"{paper_id}_text_{idx}",
            paper_id=paper_id,
            chunk_type="text",
            text=tc,
            position=idx,
            metadata={**base_meta, "chunk_type": "text", "chunk_index": idx},
        ))

    # 4) image_summary blocks（文本层，每个图描述一条）
    for idx, img_desc in enumerate(image_summary):
        if img_desc and img_desc.strip():
            blocks.append(SplitBlock(
                id=f"{paper_id}_img_{idx}",
                paper_id=paper_id,
                chunk_type="image_desc",
                text=img_desc.strip(),
                position=idx,
                metadata={**base_meta, "chunk_type": "image_desc", "image_index": idx},
            ))

    return blocks


def build_split_blocks_from_dict(obj: dict[str, Any]) -> list[SplitBlock]:
    """兼容旧接口：从 dict 构建 SplitBlock。"""
    return build_split_blocks(Metadata.from_dict(obj))


# 兼容旧名
build_chunks = build_split_blocks_from_dict

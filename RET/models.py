"""
RAG 核心数据模型
===============
与架构图 (arch.png) 对应：
- Metadata：一篇文献的元数据
- SplitBlock：文献拆分后的标准分块，不包含 embedding

原则：
1. SplitBlock 只保存业务字段，向量只存储在向量数据库中。
2. Metadata 不保存 split_blocks 列表的冗余副本，避免全量加载进内存。
3. 图片 base64 只在详情页按需读取，检索时只加载摘要/文本分词。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SplitBlock:
    """文献拆分后的一个标准分块。"""

    id: str                       # 全局唯一 chunk id: {paper_id}_{chunk_type}_{idx}
    paper_id: str                 # 所属文献 id
    chunk_type: str               # title | summary | text | image_desc
    text: str                     # 该分块的文本内容
    position: int = 0             # 在同类 chunk 中的序号
    metadata: dict[str, Any] = field(default_factory=dict)  # 附加元数据（年份、来源等）

    def to_chroma_dict(self) -> dict[str, Any]:
        """转换为 ChromaDB upsert 所需的 dict 格式。"""
        return {
            "id": self.id,
            "document": self.text,
            "chunk_type": self.chunk_type,
            "paper_id": self.paper_id,
            "position": self.position,
            **self.metadata,
        }


@dataclass
class Metadata:
    """文献元数据，与架构图中的 class Metadata 对应。"""

    title: str | None = None
    year: str | None = None
    source_type: str | None = None
    id: str | None = None
    text_summary: str | None = None
    text: str | None = None
    image_summary: list[str] | None = None
    image_base_64: list[str] | None = None
    evidence_level: str | None = None
    last_retrieved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 可保存的 dict。"""
        return {
            "title": self.title,
            "year": self.year,
            "source_type": self.source_type,
            "id": self.id,
            "text_summary": self.text_summary,
            "text": self.text,
            "image_summary": self.image_summary or [],
            "image_base_64": self.image_base_64 or [],
            "evidence_level": self.evidence_level,
            "last_retrieved_at": self.last_retrieved_at,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Metadata":
        """从 JSON dict 加载。"""
        return cls(
            title=obj.get("title"),
            year=obj.get("year"),
            source_type=obj.get("source_type"),
            id=str(obj.get("id")) if obj.get("id") is not None else None,
            text_summary=obj.get("text_summary"),
            text=obj.get("text"),
            image_summary=obj.get("image_summary") or [],
            image_base_64=obj.get("image_base_64") or [],
            evidence_level=obj.get("evidence_level"),
            last_retrieved_at=obj.get("last_retrieved_at"),
        )

    @property
    def index_text(self) -> str:
        """用于 BM25 / 关键词检索的完整文本（不加载 image_base_64）。"""
        parts: list[str] = []
        if self.title:
            parts.extend([self.title, self.title])
        if self.text_summary:
            parts.append(self.text_summary)
        if self.text:
            parts.append(self.text)
        if self.image_summary:
            parts.extend(s for s in self.image_summary if s)
        return "\n".join(parts)


__all__ = ["Metadata", "SplitBlock"]

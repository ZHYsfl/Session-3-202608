"""搜索结果的统一 Metadata 数据模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Metadata:
    """一条搜索结果的标准化元数据。"""

    title: str | None = None
    year: str | None = None
    source_type: str | None = None
    id: str | None = None  # UUID
    text: str | None = None
    text_summary: str | None = None
    image_base_64: list[str] = field(default_factory=list)
    image_summary: list[str] = field(default_factory=list)
    evidence_level: str | None = None
    last_retrieved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        result = asdict(self)
        if self.last_retrieved_at is not None:
            result["last_retrieved_at"] = self.last_retrieved_at.isoformat()
        return result

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Metadata:
    title: str | None = None
    year: str | None = None
    source_type: str | None = None
    id: str | None = None
    text_summary: str | None = None
    text: str | None = None
    image_summary: list[str] | None = field(default=None)
    image_base_64: list[str] | None = field(default=None)
    evidence_level: str | None = None
    last_retrieved_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Metadata:
        """从 dict 构造；忽略未知键，缺失字段为 None。"""
        if not isinstance(data, dict):
            raise TypeError(f"Metadata.from_dict expects dict, got {type(data)!r}")
        known = {f.name for f in fields(cls)}
        kwargs = {k: data.get(k) for k in known}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        """序列化为 dict；保留全部字段键，缺失值为 null。"""
        return asdict(self)

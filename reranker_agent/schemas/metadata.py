"""Pydantic schemas shared by the CLI and rerankers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Metadata(BaseModel):
    """Paper metadata produced by an upstream retrieval pipeline."""

    model_config = ConfigDict(extra="allow")

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

    def llm_view(self) -> dict[str, Any]:
        """Return only fields explicitly allowed to leave the application."""
        return {
            "id": self.id,
            "title": self.title,
            "year": self.year,
            "source_type": self.source_type,
            "text_summary": self.text_summary,
            "image_summary": self.image_summary,
            "evidence_level": self.evidence_level,
        }


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k1: int = Field(gt=0)
    candidates: list[Metadata] = Field(min_length=1)
    top_k2: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_k_values(self) -> "RerankRequest":
        if self.top_k1 > len(self.candidates):
            raise ValueError("top_k1 cannot exceed the number of candidates")
        if self.top_k2 > self.top_k1:
            raise ValueError("top_k2 cannot exceed top_k1")
        return self


def normalize_candidate_ids(candidates: list[Metadata]) -> list[Metadata]:
    """Fill missing IDs without mutating caller-owned models; reject duplicates."""
    normalized: list[Metadata] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        paper_id = candidate.id or f"generated_paper_{index:04d}"
        if paper_id in seen:
            raise ValueError(f"Duplicate paper id: {paper_id}")
        seen.add(paper_id)
        normalized.append(candidate.model_copy(update={"id": paper_id}))
    return normalized


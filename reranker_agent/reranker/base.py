"""Common reranker interface and validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from models.llm_client import LLMClient
from schemas.metadata import Metadata, normalize_candidate_ids


class RankingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    rank: int = Field(gt=0)
    score: float
    reason: str


class RerankState(TypedDict, total=False):
    query: str
    candidates: list[Metadata]
    top_k2: int
    results: list[dict]


class BaseReranker(ABC):
    method: str

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def prepare(
        self, query: str, candidates: list[Metadata], top_k2: int
    ) -> tuple[str, list[Metadata], int]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if not candidates:
            raise ValueError("candidates must not be empty")
        if top_k2 <= 0 or top_k2 > len(candidates):
            raise ValueError("top_k2 must be between 1 and the candidate count")
        return query, normalize_candidate_ids(candidates), top_k2

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[Metadata], top_k2: int
    ) -> list[dict]:
        """Return id/rank/score/reason dictionaries ordered best first."""


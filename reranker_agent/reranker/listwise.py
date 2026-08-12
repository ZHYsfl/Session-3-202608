"""Sliding-window listwise ranking and global fusion in LangGraph."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from schemas.metadata import Metadata

from .base import BaseReranker
from .fusion import fuse_rankings, normalize_to_ten
from .prompts import LISTWISE_PROMPT


class ListwiseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranking: list[str] = Field(min_length=1)
    reason: str


class ListwiseState(TypedDict, total=False):
    query: str
    candidates: list[Metadata]
    top_k2: int
    windows: list[list[Metadata]]
    window_rankings: list[list[str]]
    window_reasons: list[str]
    results: list[dict]


def make_sliding_windows(
    candidates: list[Metadata], window_size: int, stride: int
) -> list[list[Metadata]]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if stride > window_size:
        raise ValueError("stride cannot exceed window_size because it would skip papers")
    if len(candidates) <= window_size:
        return [candidates]
    last_start = len(candidates) - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [candidates[start : start + window_size] for start in starts]


class ListwiseReranker(BaseReranker):
    method = "listwise"

    def __init__(
        self,
        llm_client,
        *,
        window_size: int = 5,
        stride: int = 2,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ) -> None:
        super().__init__(llm_client)
        if window_size <= 0 or stride <= 0 or stride > window_size:
            raise ValueError("Require window_size > 0 and 0 < stride <= window_size")
        if fusion not in {"rrf", "borda"}:
            raise ValueError("fusion must be 'rrf' or 'borda'")
        self.window_size = window_size
        self.stride = stride
        self.fusion = fusion
        self.rrf_k = rrf_k

        graph = StateGraph(ListwiseState)
        graph.add_node("create_sliding_windows", self._create_windows)
        graph.add_node("rank_each_window", self._rank_each_window)
        graph.add_node("fuse_window_rankings", self._fuse_windows)
        graph.add_edge(START, "create_sliding_windows")
        graph.add_edge("create_sliding_windows", "rank_each_window")
        graph.add_edge("rank_each_window", "fuse_window_rankings")
        graph.add_edge("fuse_window_rankings", END)
        self.graph = graph.compile()

    def _create_windows(self, state: ListwiseState) -> dict:
        return {
            "windows": make_sliding_windows(
                state["candidates"], self.window_size, self.stride
            )
        }

    def _rank_each_window(self, state: ListwiseState) -> dict:
        rankings: list[list[str]] = []
        reasons: list[str] = []
        for window in state["windows"]:
            prompt = LISTWISE_PROMPT.format(
                query=state["query"],
                candidates_json=json.dumps(
                    [paper.llm_view() for paper in window], ensure_ascii=False
                ),
            )
            decision = self.llm_client.complete_json(prompt, ListwiseDecision)
            expected = [paper.id for paper in window]
            if len(decision.ranking) != len(expected) or set(decision.ranking) != set(expected):
                raise ValueError(
                    "LLM listwise ranking must contain every window ID exactly once"
                )
            rankings.append(decision.ranking)
            reasons.append(decision.reason)
        return {"window_rankings": rankings, "window_reasons": reasons}

    def _fuse_windows(self, state: ListwiseState) -> dict:
        universe = [paper.id for paper in state["candidates"]]
        raw_scores = fuse_rankings(
            state["window_rankings"], universe, self.fusion, self.rrf_k
        )
        scores = normalize_to_ten(raw_scores)
        original_order = {paper_id: index for index, paper_id in enumerate(universe)}
        ordered_ids = sorted(
            universe, key=lambda paper_id: (-raw_scores[paper_id], original_order[paper_id])
        )[: state["top_k2"]]
        appearances: dict[str, list[int]] = defaultdict(list)
        for window_index, ranking in enumerate(state["window_rankings"], start=1):
            for paper_id in ranking:
                appearances[paper_id].append(window_index)
        results = [
            {
                "id": paper_id,
                "rank": rank,
                "score": scores[paper_id],
                "reason": (
                    f"{self.fusion.upper()} fusion across sliding windows "
                    f"{appearances[paper_id]}; raw score={raw_scores[paper_id]:.6f}."
                ),
            }
            for rank, paper_id in enumerate(ordered_ids, start=1)
        ]
        return {"results": results}

    def rerank(
        self, query: str, candidates: list[Metadata], top_k2: int
    ) -> list[dict]:
        query, candidates, top_k2 = self.prepare(query, candidates, top_k2)
        state = self.graph.invoke(
            {"query": query, "candidates": candidates, "top_k2": top_k2}
        )
        return state["results"]


"""Independent LLM scoring implemented as a LangGraph workflow."""

from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from schemas.metadata import Metadata

from .base import BaseReranker
from .prompts import POINTWISE_PROMPT


class PointwiseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    score: float = Field(ge=0, le=10)
    reason: str


class PointwiseState(TypedDict, total=False):
    query: str
    candidates: list[Metadata]
    top_k2: int
    judgments: list[dict]
    results: list[dict]


class PointwiseReranker(BaseReranker):
    method = "pointwise"

    def __init__(self, llm_client) -> None:
        super().__init__(llm_client)
        graph = StateGraph(PointwiseState)
        graph.add_node("score_each_paper", self._score_each_paper)
        graph.add_node("sort_and_select", self._sort_and_select)
        graph.add_edge(START, "score_each_paper")
        graph.add_edge("score_each_paper", "sort_and_select")
        graph.add_edge("sort_and_select", END)
        self.graph = graph.compile()

    def _score_each_paper(self, state: PointwiseState) -> dict:
        judgments: list[dict] = []
        for original_index, paper in enumerate(state["candidates"]):
            prompt = POINTWISE_PROMPT.format(
                query=state["query"],
                paper_json=json.dumps(paper.llm_view(), ensure_ascii=False),
            )
            decision = self.llm_client.complete_json(prompt, PointwiseDecision)
            if decision.paper_id != paper.id:
                raise ValueError(
                    f"LLM returned paper_id {decision.paper_id!r}; expected {paper.id!r}"
                )
            judgments.append(
                {
                    "id": paper.id,
                    "score": decision.score,
                    "reason": decision.reason,
                    "original_index": original_index,
                }
            )
        return {"judgments": judgments}

    def _sort_and_select(self, state: PointwiseState) -> dict:
        ordered = sorted(
            state["judgments"], key=lambda item: (-item["score"], item["original_index"])
        )[: state["top_k2"]]
        results = [
            {
                "id": item["id"],
                "rank": rank,
                "score": round(float(item["score"]), 6),
                "reason": item["reason"],
            }
            for rank, item in enumerate(ordered, start=1)
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


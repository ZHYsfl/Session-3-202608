"""All-pairs comparison with win-count aggregation in LangGraph."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from schemas.metadata import Metadata

from .base import BaseReranker
from .prompts import PAIRWISE_PROMPT


class PairwiseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: str
    reason: str


class PairwiseState(TypedDict, total=False):
    query: str
    candidates: list[Metadata]
    top_k2: int
    comparisons: list[dict]
    results: list[dict]


class PairwiseReranker(BaseReranker):
    method = "pairwise"

    def __init__(self, llm_client, algorithm: str = "win_count") -> None:
        super().__init__(llm_client)
        if algorithm != "win_count":
            raise ValueError("This pairwise reranker implements the 'win_count' algorithm")
        self.algorithm = algorithm
        graph = StateGraph(PairwiseState)
        graph.add_node("compare_all_pairs", self._compare_all_pairs)
        graph.add_node("aggregate_wins", self._aggregate_wins)
        graph.add_edge(START, "compare_all_pairs")
        graph.add_edge("compare_all_pairs", "aggregate_wins")
        graph.add_edge("aggregate_wins", END)
        self.graph = graph.compile()

    def _compare_all_pairs(self, state: PairwiseState) -> dict:
        comparisons: list[dict] = []
        for paper_a, paper_b in combinations(state["candidates"], 2):
            prompt = PAIRWISE_PROMPT.format(
                query=state["query"],
                paper_a_json=json.dumps(paper_a.llm_view(), ensure_ascii=False),
                paper_b_json=json.dumps(paper_b.llm_view(), ensure_ascii=False),
            )
            decision = self.llm_client.complete_json(prompt, PairwiseDecision)
            allowed = {paper_a.id, paper_b.id}
            if decision.winner not in allowed:
                raise ValueError(
                    f"LLM winner {decision.winner!r} is not one of {sorted(allowed)}"
                )
            loser = paper_b.id if decision.winner == paper_a.id else paper_a.id
            comparisons.append(
                {"winner": decision.winner, "loser": loser, "reason": decision.reason}
            )
        return {"comparisons": comparisons}

    def _aggregate_wins(self, state: PairwiseState) -> dict:
        candidates = state["candidates"]
        wins = {paper.id: 0 for paper in candidates}
        notes: dict[str, list[str]] = defaultdict(list)
        for comparison in state["comparisons"]:
            wins[comparison["winner"]] += 1
            notes[comparison["winner"]].append(
                f"beat {comparison['loser']}: {comparison['reason']}"
            )
            notes[comparison["loser"]].append(
                f"lost to {comparison['winner']}: {comparison['reason']}"
            )

        denominator = max(len(candidates) - 1, 1)
        original_order = {paper.id: index for index, paper in enumerate(candidates)}
        ordered_ids = sorted(
            wins, key=lambda paper_id: (-wins[paper_id], original_order[paper_id])
        )[: state["top_k2"]]
        results = []
        for rank, paper_id in enumerate(ordered_ids, start=1):
            detail = "; ".join(notes[paper_id][:3]) or "Only candidate; no pair was required."
            results.append(
                {
                    "id": paper_id,
                    "rank": rank,
                    "score": round(10.0 * wins[paper_id] / denominator, 6)
                    if len(candidates) > 1
                    else 10.0,
                    "reason": f"Won {wins[paper_id]} of {len(candidates)-1} comparisons. {detail}",
                }
            )
        return {"results": results}

    def rerank(
        self, query: str, candidates: list[Metadata], top_k2: int
    ) -> list[dict]:
        query, candidates, top_k2 = self.prepare(query, candidates, top_k2)
        state = self.graph.invoke(
            {"query": query, "candidates": candidates, "top_k2": top_k2}
        )
        return state["results"]

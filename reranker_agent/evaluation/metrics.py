"""Standard information-retrieval metrics for one ranked query."""

from __future__ import annotations

import math
from typing import TypedDict


class GroundTruthItem(TypedDict):
    paper_id: str
    relevance: int


def _relevance_map(ground_truth: list[GroundTruthItem]) -> dict[str, int]:
    return {item["paper_id"]: int(item["relevance"]) for item in ground_truth}


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(
    predicted: list[str], ground_truth: list[GroundTruthItem], k: int
) -> float:
    _validate_k(k)
    relevant = {paper_id for paper_id, rel in _relevance_map(ground_truth).items() if rel > 0}
    hits = sum(1 for paper_id in predicted[:k] if paper_id in relevant)
    return hits / k


def recall_at_k(
    predicted: list[str], ground_truth: list[GroundTruthItem], k: int
) -> float:
    _validate_k(k)
    relevant = {paper_id for paper_id, rel in _relevance_map(ground_truth).items() if rel > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for paper_id in predicted[:k] if paper_id in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(
    predicted: list[str], ground_truth: list[GroundTruthItem]
) -> float:
    """Reciprocal rank for one query; average these values over queries for MRR."""
    relevant = {paper_id for paper_id, rel in _relevance_map(ground_truth).items() if rel > 0}
    for rank, paper_id in enumerate(predicted, start=1):
        if paper_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    predicted: list[str], ground_truth: list[GroundTruthItem], k: int
) -> float:
    _validate_k(k)
    relevance = _relevance_map(ground_truth)

    def dcg(values: list[int]) -> float:
        return sum(
            (2**rel - 1) / math.log2(position + 2)
            for position, rel in enumerate(values)
        )

    actual = [relevance.get(paper_id, 0) for paper_id in predicted[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg else 0.0


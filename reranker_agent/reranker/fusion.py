"""Global score aggregation for pairwise and sliding-window rankings."""

from __future__ import annotations

from collections import defaultdict


def borda_count(
    rankings: list[list[str]], universe: list[str]
) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        size = len(ranking)
        for position, paper_id in enumerate(ranking):
            scores[paper_id] += size - position
    return {paper_id: scores.get(paper_id, 0.0) for paper_id in universe}


def reciprocal_rank_fusion(
    rankings: list[list[str]], universe: list[str], rrf_k: int = 60
) -> dict[str, float]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, paper_id in enumerate(ranking, start=1):
            scores[paper_id] += 1.0 / (rrf_k + position)
    return {paper_id: scores.get(paper_id, 0.0) for paper_id in universe}


def fuse_rankings(
    rankings: list[list[str]],
    universe: list[str],
    method: str = "rrf",
    rrf_k: int = 60,
) -> dict[str, float]:
    if method == "rrf":
        return reciprocal_rank_fusion(rankings, universe, rrf_k)
    if method == "borda":
        return borda_count(rankings, universe)
    raise ValueError(f"Unsupported fusion method: {method}")


def normalize_to_ten(scores: dict[str, float]) -> dict[str, float]:
    """Map non-negative scores to [0, 10] while preserving ties and order."""
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0:
        return {paper_id: 0.0 for paper_id in scores}
    return {
        paper_id: round(10.0 * score / maximum, 6)
        for paper_id, score in scores.items()
    }


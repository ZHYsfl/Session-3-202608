from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
) -> list[int]:
    """
    Reciprocal Rank Fusion.

    score(d) = Σ 1 / (k + rank_i(d))，rank 从 1 开始。
    返回按融合分降序的文档 index 列表。
    """
    scores: dict[int, float] = defaultdict(float)

    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked, start=1):
            scores[doc_idx] += 1.0 / (k + rank)

    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

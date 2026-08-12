"""Convenience interface around ranking metrics."""

from __future__ import annotations

from .metrics import (
    GroundTruthItem,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class RankingEvaluator:
    def evaluate(
        self,
        predicted: list[str],
        ground_truth: list[GroundTruthItem],
        ks: list[int] | tuple[int, ...] = (1, 3, 5, 10),
    ) -> dict[str, float]:
        metrics: dict[str, float] = {"mrr": mean_reciprocal_rank(predicted, ground_truth)}
        for k in ks:
            metrics[f"precision@{k}"] = precision_at_k(predicted, ground_truth, k)
            metrics[f"recall@{k}"] = recall_at_k(predicted, ground_truth, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(predicted, ground_truth, k)
        return metrics


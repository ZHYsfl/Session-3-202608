from __future__ import annotations

import math
import unittest

from evaluation.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k
from models.llm_client import MockLLMClient
from reranker import ListwiseReranker, PairwiseReranker, PointwiseReranker
from reranker.listwise import make_sliding_windows
from schemas.metadata import Metadata


class CapturingMockClient(MockLLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_json(self, prompt, response_model):
        self.prompts.append(prompt)
        return super().complete_json(prompt, response_model)


def sample_papers() -> list[Metadata]:
    return [
        Metadata(
            id="p1",
            title="Deep learning intrusion detection",
            text_summary="Neural network cyber security experiment",
            text="FULL_TEXT_SECRET_9182",
            image_summary=["Network attack F1 chart"],
            image_base_64=["BASE64_SECRET_4721"],
            evidence_level="peer reviewed",
        ),
        Metadata(id="p2", title="Botanical field notes", text_summary="Plant survey"),
        Metadata(id="p3", title="Network anomaly detection", text_summary="Detects attacks"),
        Metadata(id="p4", title="Malware signatures", text_summary="Static analysis"),
    ]


class RerankerTests(unittest.TestCase):
    def test_sensitive_fields_never_enter_pointwise_prompts(self) -> None:
        client = CapturingMockClient()
        results = PointwiseReranker(client).rerank("deep network attack", sample_papers(), 2)
        combined = "\n".join(client.prompts)
        self.assertEqual(2, len(results))
        self.assertNotIn("FULL_TEXT_SECRET_9182", combined)
        self.assertNotIn("BASE64_SECRET_4721", combined)
        self.assertNotIn('"text":', combined)
        self.assertNotIn('"image_base_64":', combined)

    def test_all_reranker_workflows_return_top_k2(self) -> None:
        papers = sample_papers()
        pointwise = PointwiseReranker(MockLLMClient()).rerank("network attack", papers, 2)
        pairwise = PairwiseReranker(MockLLMClient()).rerank("network attack", papers, 2)
        listwise = ListwiseReranker(
            MockLLMClient(), window_size=3, stride=2
        ).rerank("network attack", papers, 2)
        for results in (pointwise, pairwise, listwise):
            self.assertEqual([1, 2], [item["rank"] for item in results])
            self.assertEqual(2, len({item["id"] for item in results}))

    def test_sliding_windows_cover_every_paper(self) -> None:
        papers = sample_papers() + [Metadata(id="p5"), Metadata(id="p6")]
        windows = make_sliding_windows(papers, window_size=3, stride=2)
        covered = {paper.id for window in windows for paper in window}
        self.assertEqual({paper.id for paper in papers}, covered)
        self.assertEqual(["p4", "p5", "p6"], [paper.id for paper in windows[-1]])


class MetricTests(unittest.TestCase):
    def test_metrics(self) -> None:
        predicted = ["x", "p1", "p2"]
        truth = [
            {"paper_id": "p1", "relevance": 3},
            {"paper_id": "p2", "relevance": 1},
        ]
        self.assertAlmostEqual(2 / 3, precision_at_k(predicted, truth, 3))
        self.assertAlmostEqual(1.0, recall_at_k(predicted, truth, 3))
        self.assertAlmostEqual(0.5, mean_reciprocal_rank(predicted, truth))
        expected_dcg = ((2**3 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4))
        ideal_dcg = ((2**3 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3))
        self.assertAlmostEqual(expected_dcg / ideal_dcg, ndcg_at_k(predicted, truth, 3))


if __name__ == "__main__":
    unittest.main()


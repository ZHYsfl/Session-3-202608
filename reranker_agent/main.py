"""Command-line entry point for the standalone reranking agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from models.llm_client import build_llm_client
from reranker import ListwiseReranker, PairwiseReranker, PointwiseReranker
from schemas.metadata import RerankRequest, normalize_candidate_ids

PROJECT_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_request(path: Path) -> RerankRequest:
    with path.open("r", encoding="utf-8") as handle:
        return RerankRequest.model_validate(json.load(handle))


def build_reranker(method: str, client, config: dict[str, Any], args):
    reranker_config = config.get("reranker", {})
    if method == "pointwise":
        return PointwiseReranker(client)
    if method == "pairwise":
        pair_config = reranker_config.get("pairwise", {})
        return PairwiseReranker(
            client, algorithm=pair_config.get("algorithm", "win_count")
        )
    list_config = reranker_config.get("listwise", {})
    return ListwiseReranker(
        client,
        window_size=args.window_size or int(list_config.get("window_size", 5)),
        stride=args.stride or int(list_config.get("stride", 2)),
        fusion=list_config.get("fusion", "rrf"),
        rrf_k=int(list_config.get("rrf_k", 60)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Agent paper reranker")
    parser.add_argument(
        "--method", choices=("pointwise", "pairwise", "listwise"), default="pointwise"
    )
    parser.add_argument(
        "--input", type=Path, default=PROJECT_DIR / "data" / "example_candidates.json"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.yaml")
    parser.add_argument("--query", help="Override the query from the input JSON")
    parser.add_argument("--top-k1", type=int, help="Use the first K1 input candidates")
    parser.add_argument("--top-k2", type=int, help="Override output result count")
    parser.add_argument("--window-size", type=int, help="Override listwise window size")
    parser.add_argument("--stride", type=int, help="Override listwise stride")
    parser.add_argument(
        "--provider", choices=("mock", "openai", "qwen", "compatible"),
        help="Override config llm.provider",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    request = load_request(args.input)

    query = args.query or request.query
    top_k1 = args.top_k1 if args.top_k1 is not None else request.top_k1
    top_k2 = args.top_k2 if args.top_k2 is not None else request.top_k2
    if top_k1 <= 0 or top_k1 > len(request.candidates):
        raise ValueError("top_k1 must be between 1 and the number of input candidates")
    if top_k2 <= 0 or top_k2 > top_k1:
        raise ValueError("top_k2 must be between 1 and top_k1")

    candidates = normalize_candidate_ids(request.candidates[:top_k1])
    llm_config = dict(config.get("llm", {}))
    if args.provider:
        llm_config["provider"] = args.provider
    client = build_llm_client(llm_config)
    reranker = build_reranker(args.method, client, config, args)
    ranking = reranker.rerank(query, candidates, top_k2)

    metadata_by_id = {paper.id: paper for paper in candidates}
    output = {
        "method": args.method,
        "top_k2": top_k2,
        "results": [
            {
                "metadata": metadata_by_id[item["id"]].model_dump(mode="json"),
                "rank": item["rank"],
                "score": item["score"],
                "reason": item["reason"],
            }
            for item in ranking
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


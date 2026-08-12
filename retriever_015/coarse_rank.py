from __future__ import annotations

from datetime import datetime
from typing import Any

from io_util import dump_metadatas, load_metadatas, load_rank_request, parse_rank_request
from metadata import Metadata
from retrievers import ScoreBreakdown, rank_indices, score_all

__all__ = [
    "coarse_rank",
    "coarse_rank_from_request",
    "load_metadatas",
    "dump_metadatas",
    "load_rank_request",
]


def _stamp_retrieved(items: list[Metadata]) -> list[Metadata]:
    now = datetime.now().isoformat(timespec="seconds")
    for m in items:
        m.last_retrieved_at = now
    return items


def _print_debug(breakdowns: list[ScoreBreakdown], top_indices: list[int]) -> None:
    by_idx = {b.index: b for b in breakdowns}
    print("--- score breakdown (ranked) ---")
    for rank, idx in enumerate(top_indices, start=1):
        b = by_idx[idx]
        print(
            f"{rank}. id={b.id} final={b.final:.4f} "
            f"relevance={b.relevance:.4f} (rel={b.rel:.4f} topic={b.topic:.4f} "
            f"aspect={b.aspect:.4f} joint={b.joint:.4f} constr={b.constraint:.4f}) "
            f"quality={b.quality:.4f} (ev={b.evidence:.4f} src={b.source_bonus:.4f}) "
            f"fields[t={b.title_score:.3f} s={b.summary_score:.3f} "
            f"x={b.text_score:.3f} i={b.image_score:.3f}]"
        )


def coarse_rank(
    metadatas: list[Metadata],
    query: str,
    k1: int,
    *,
    debug: bool = False,
) -> list[Metadata]:
    """
    粗排序：分字段相关性 + 主题完整度 + 证据质量次级加权，返回 top-k1。

    - 空列表或 k1 <= 0 → []
    - 空 query → 保持原序截取前 k1
    - k1 >= len → 返回全部排序结果
    - 返回条目的 last_retrieved_at 更新为当前时间戳
    """
    if not metadatas or k1 <= 0:
        return []

    if not query or not query.strip():
        return _stamp_retrieved(metadatas[:k1])

    breakdowns = score_all(metadatas, query)
    ranked = rank_indices(breakdowns)
    top = ranked[:k1]

    if debug:
        _print_debug(breakdowns, ranked)

    return _stamp_retrieved([metadatas[i] for i in top])


def coarse_rank_from_request(
    request: dict[str, Any],
    *,
    debug: bool = False,
) -> list[Metadata]:
    """从 {query, k1, metadatas} 字典执行粗排。metadatas 可为 dict 或 Metadata。"""
    metas, query, k1 = parse_rank_request(request)
    return coarse_rank(metas, query, k1, debug=debug)


if __name__ == "__main__":
    print("Use the CLI for file I/O:")
    print('  python cli.py --request examples/sample_request.json --output -')
    print("Or call coarse_rank(metadatas, query, k1) from Python.")
    print()
    samples = [
        Metadata(
            id="1",
            title="Attention Is All You Need",
            text_summary="Transformer architecture with self-attention for machine translation.",
            text="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
        ),
        Metadata(
            id="2",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            text_summary="Bidirectional encoder representations from transformers for language understanding.",
            text="BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
        ),
        Metadata(
            id="5",
            title="Neural Machine Translation by Jointly Learning to Align and Translate",
            text_summary="Attention mechanism for neural machine translation alignment.",
            text="We introduce an attention mechanism that allows the model to focus on relevant source words.",
        ),
    ]
    query = "transformer attention machine translation"
    for i, m in enumerate(coarse_rank(samples, query, 2), start=1):
        print(f"{i}. id={m.id} title={m.title}")

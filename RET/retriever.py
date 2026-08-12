"""
多路检索 + RRF 融合
- BM25 词汇检索
- Keyword 覆盖率检索
- 向量检索（title / summary / text / image_desc）
- RRF 排名融合
- 可选 MMR 重排
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from embedding import embed_single
from vector_store import query_all_chunk_types, aggregate_maxsim

# ─── 复用现有分词逻辑 ─────────────────────────────────────────

STOP_WORDS = {
    "的", "了", "和", "是", "在", "与", "对",
    "中", "及", "一个", "一种", "我们",
    "研究", "本文", "通过", "进行",
}

_PUNCT_RE = re.compile(r"[^\w一-鿿]+")
_SPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = _SPACE_RE.sub(" ", str(text).lower()).strip()
    result = []
    for w in jieba.lcut(text):
        w = w.strip()
        if w and w not in STOP_WORDS and not _PUNCT_RE.fullmatch(w):
            result.append(w)
    return result


# ─── Metadata 数据结构 ────────────────────────────────────────

@dataclass
class Metadata:
    title: str | None = None
    year: str | None = None
    source_type: str | None = None
    id: str | None = None
    text: str | None = None
    text_summary: str | None = None
    image_summary: list[str] | None = None
    image_base_64: list[str] | None = None
    evidence_level: list[str] | None = None
    last_retrieved_at: str | None = None


# ─── BM25 / Keyword 检索 ──────────────────────────────────────

# 缓存：避免每次查询都重新分词 480 篇全文
_bm25_cache: dict = {"metadata_id": None, "tokenized_docs": None, "bm25": None}


def _get_bm25(metadata_list: list[Metadata]) -> BM25Okapi:
    """获取缓存的 BM25 索引，仅在 metadata 变化时重建"""
    global _bm25_cache
    meta_id = id(metadata_list)
    if _bm25_cache["metadata_id"] == meta_id and _bm25_cache["bm25"] is not None:
        return _bm25_cache["bm25"]
    docs = [tokenize(build_document_text(m)) for m in metadata_list]
    bm25 = BM25Okapi(docs)
    _bm25_cache = {"metadata_id": meta_id, "tokenized_docs": docs, "bm25": bm25}
    return bm25


def _get_tokenized_docs(metadata_list: list[Metadata]) -> list[list[str]]:
    """获取缓存的 tokenized docs"""
    global _bm25_cache
    meta_id = id(metadata_list)
    if _bm25_cache["metadata_id"] == meta_id and _bm25_cache["tokenized_docs"] is not None:
        return _bm25_cache["tokenized_docs"]
    docs = [tokenize(build_document_text(m)) for m in metadata_list]
    bm25 = BM25Okapi(docs)
    _bm25_cache = {"metadata_id": meta_id, "tokenized_docs": docs, "bm25": bm25}
    return docs


def invalidate_bm25_cache():
    """热更新后调用，强制下次查询重建 BM25 索引"""
    global _bm25_cache
    _bm25_cache = {"metadata_id": None, "tokenized_docs": None, "bm25": None}


def build_document_text(m: Metadata) -> str:
    parts = []
    if m.title:
        parts.extend([m.title, m.title])
    if m.text_summary:
        parts.append(m.text_summary)
    if m.text:
        parts.append(m.text)
    if m.image_summary:
        parts.extend(s for s in m.image_summary if s)
    return "\n".join(parts)


def bm25_search(metadata_list: list[Metadata], question: str, keywords: str = "", top_k: int = 200) -> list[dict]:
    query_tokens = tokenize(keywords or question)
    if not query_tokens:
        return []
    bm25 = _get_bm25(metadata_list)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(
        [{"paper_id": metadata_list[i].id or str(i), "score": float(s), "index": i, "source": "bm25"}
         for i, s in enumerate(scores)],
        key=lambda x: x["score"], reverse=True,
    )
    for r, item in enumerate(ranked[:top_k], start=1):
        item["rank"] = r
    return ranked[:top_k]


def keyword_search(metadata_list: list[Metadata], question: str, keywords: str = "", top_k: int = 200) -> list[dict]:
    query_tokens = tokenize(keywords or question)
    if not query_tokens:
        return []
    q_set = set(query_tokens)
    cached_docs = _get_tokenized_docs(metadata_list)
    scores = []
    for i, m in enumerate(metadata_list):
        doc_tokens = set(cached_docs[i])
        title_tokens = set(tokenize(m.title or ""))
        if not doc_tokens and not title_tokens:
            scores.append((i, 0.0))
            continue
        doc_cov = len(q_set & doc_tokens) / len(q_set) if doc_tokens else 0.0
        title_cov = len(q_set & title_tokens) / len(q_set) if title_tokens else 0.0
        scores.append((i, 0.7 * doc_cov + 0.3 * title_cov))
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked = []
    for r, (i, s) in enumerate(scores[:top_k], start=1):
        ranked.append({
            "paper_id": metadata_list[i].id or str(i),
            "score": s, "index": i, "source": "keyword", "rank": r,
        })
    return ranked


# ─── 向量检索 ─────────────────────────────────────────────────

def _chunk_results_to_paper_scores(
    chunk_results: list[dict],
    source_name: str,
    top_k: int = 200,
) -> list[dict]:
    """
    将 chunk 级检索结果聚合为论文级，
    返回按 paper_id 聚合后的得分列表（MaxSim）。
    """
    agg = aggregate_maxsim(chunk_results)
    ranked = []
    for r, item in enumerate(agg[:top_k], start=1):
        ranked.append({
            "paper_id": item["paper_id"],
            "score": item["score"],
            "source": source_name,
            "rank": r,
            "best_chunk_id": item["best_chunk_id"],
            "best_document": item["document"],
        })
    return ranked


def vector_search(query: str, top_k_per_type: int = 200) -> dict[str, list[dict]]:
    """向量检索，返回各类型的论文级排名"""
    query_vec = embed_single(query)
    raw = query_all_chunk_types(query_vec, top_k_per_type=top_k_per_type)

    results = {}
    for ctype, chunks in raw.items():
        if ctype == "text":
            source = "vector_text"
        elif ctype == "title":
            source = "vector_title"
        elif ctype == "summary":
            source = "vector_summary"
        elif ctype == "image_desc":
            source = "vector_image"
        else:
            source = f"vector_{ctype}"
        results[source] = _chunk_results_to_paper_scores(chunks, source, top_k=top_k_per_type)
    return results


# ─── RRF 融合 ─────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    rrf_k: int = 60,
    top_k: int = 50,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
) -> list[dict]:
    """
    多路排名 RRF 融合。
    每个 ranked_list 的元素需包含: paper_id, rank, score, source
    可选 year / source_type 过滤。
    """
    fusion: dict[str, float] = {}
    paper_info: dict[str, dict] = {}

    for lst in ranked_lists:
        for item in lst:
            pid = item["paper_id"]
            if pid not in paper_info:
                paper_info[pid] = {
                    "paper_id": pid,
                    "sources": [],
                    "details": {},
                }
            fusion[pid] = fusion.get(pid, 0.0) + 1.0 / (rrf_k + item["rank"])
            paper_info[pid]["sources"].append(item["source"])
            paper_info[pid]["details"][item["source"]] = {
                "rank": item["rank"],
                "score": item.get("score", 0),
                "best_chunk_id": item.get("best_chunk_id", ""),
                "best_document": item.get("best_document", ""),
            }

    sorted_items = sorted(fusion.items(), key=lambda x: x[1], reverse=True)
    if not sorted_items:
        return []

    max_v = sorted_items[0][1]
    results = []
    for r, (pid, v) in enumerate(sorted_items, start=1):
        info = paper_info[pid]
        results.append({
            "paper_id": pid,
            "rrf_score": v,
            "normalized_score": v / max_v,
            "rank": r,
            "sources": info["sources"],
            "details": info["details"],
        })
        if len(results) >= top_k:
            break

    return results


# ─── MMR 多样性重排 ───────────────────────────────────────────

def mmr_rerank(
    results: list[dict],
    metadata_map: dict[str, Metadata],
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> list[dict]:
    """
    MMR 重排：平衡相关性与 source_type 多样性。
    相似度惩罚项基于 source_type 是否相同。
    """
    if len(results) <= 1:
        return results[:top_k]

    selected: list[dict] = [results[0]]
    remaining = results[1:]

    while remaining and len(selected) < top_k:
        best_score = -float("inf")
        best_idx = 0

        for i, candidate in enumerate(remaining):
            relevance = candidate["normalized_score"]
            # 多样性惩罚：与已选中论文 source_type 相同的计数
            diversity_penalty = 0.0
            candidate_src = metadata_map.get(candidate["paper_id"])
            candidate_src = candidate_src.source_type if candidate_src else None

            for sel in selected:
                sel_src = metadata_map.get(sel["paper_id"])
                sel_src = sel_src.source_type if sel_src else None
                if candidate_src and sel_src and candidate_src == sel_src:
                    diversity_penalty = max(diversity_penalty, 0.3)

            mmr = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr > best_score:
                best_score = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    # 重新编号
    for r, item in enumerate(selected, start=1):
        item["rank"] = r
    return selected


# ─── 主检索函数 ───────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    keywords: str = "",
    top_k: int = 10,
    metadata_list: list[Metadata] | None = None,
    use_mmr: bool = False,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
) -> list[dict]:
    """
    混合检索主入口。

    参数:
        query: 自然语言查询
        keywords: 可选关键词（空格分隔）
        top_k: 返回结果数
        metadata_list: Metadata 列表（用于 BM25/Keyword）
        use_mmr: 是否启用 MMR 多样性重排
        year_filter: 年份下限过滤
        source_type_filter: 来源过滤

    返回:
        [{paper_id, rank, rrf_score, normalized_score, sources, details, metadata}, ...]
    """
    if not query.strip():
        return []

    ranked_lists: list[list[dict]] = []

    # 1) BM25 + Keyword 词汇检索
    if metadata_list:
        ranked_lists.append(bm25_search(metadata_list, query, keywords))
        ranked_lists.append(keyword_search(metadata_list, query, keywords))

    # 2) 向量检索（4路）
    vec_results = vector_search(query)
    for source, pr in vec_results.items():
        if pr:
            ranked_lists.append(pr)

    # 3) RRF 融合
    fused = reciprocal_rank_fusion(
        *ranked_lists,
        year_filter=year_filter,
        source_type_filter=source_type_filter,
        top_k=max(top_k * 5, 50),  # 先取多些，再 MMR
    )

    # 4) 可选 MMR
    if use_mmr and metadata_list:
        meta_map = {m.id: m for m in metadata_list if m.id}
        fused = mmr_rerank(fused, meta_map, top_k=top_k)

    # 5) 截断 + 附加 metadata
    top = fused[:top_k]
    now = datetime.now(timezone.utc).isoformat()

    if metadata_list:
        meta_by_id = {m.id: m for m in metadata_list if m.id}
        for item in top:
            pid = item["paper_id"]
            m = meta_by_id.get(pid)
            if m:
                m.last_retrieved_at = now
                item["metadata"] = m
                item["title"] = m.title
                item["year"] = m.year
                item["source_type"] = m.source_type
                item["text_summary"] = m.text_summary
                item["image_summary"] = m.image_summary
                # image_base_64 附带返回
                item["image_base_64"] = m.image_base_64

    return top


# ─── 加载元数据 ───────────────────────────────────────────────

def load_metadata_from_dir(data_dir: str) -> list[Metadata]:
    items = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            obj = json.load(f)
        items.append(Metadata(
            id=obj.get("id"), title=obj.get("title"),
            year=obj.get("year"), source_type=obj.get("source_type"),
            text=obj.get("text"), text_summary=obj.get("text_summary"),
            image_summary=obj.get("image_summary"),
            image_base_64=obj.get("image_base_64"),
            evidence_level=obj.get("evidence_level"),
            last_retrieved_at=obj.get("last_retrieved_at"),
        ))
    return items

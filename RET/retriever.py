"""
多路检索 + RRF 融合
- BM25 词汇检索
- Keyword 覆盖率检索
- 向量检索（title / summary / text / image_desc）
- RRF 排名融合
- 可选 MMR 重排

改造重点：
1. Metadata 从 models.py 导入，统一数据结构
2. BM25 索引支持增量更新（add_document），避免热更新时重建全部
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from embedding import embed_single
from models import Metadata
from tokenizer_v2 import content_tokens
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


# ─── 增量 BM25 索引 ───────────────────────────────────────────

class IncrementalBM25:
    """
    增量 BM25 索引：新增文档时不必重建全部。
    注意：rank_bm25 的 BM25Okapi 不原生支持增量，这里通过重新实例化实现。
    但由于只保存了 tokenized_docs，重建成本比重新分词全文要低得多。
    """

    def __init__(self):
        self._tokenized_docs: list[list[str]] = []
        self._paper_ids: list[str] = []
        self._id_to_index: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    def add_document(self, paper_id: str, tokens: list[str]):
        """新增一篇文档；如果已存在则替换"""
        with self._lock:
            if paper_id in self._id_to_index:
                idx = self._id_to_index[paper_id]
                self._tokenized_docs[idx] = tokens
            else:
                idx = len(self._tokenized_docs)
                self._tokenized_docs.append(tokens)
                self._paper_ids.append(paper_id)
                self._id_to_index[paper_id] = idx
            self._bm25 = None  # 标记需要重建

    def add_documents(self, paper_ids: list[str], docs_tokens: list[list[str]]):
        """批量新增"""
        for pid, tokens in zip(paper_ids, docs_tokens):
            self.add_document(pid, tokens)

    def get_bm25(self) -> BM25Okapi:
        with self._lock:
            if self._bm25 is None:
                corpus = [doc if doc else ["__empty__"] for doc in self._tokenized_docs]
                self._bm25 = BM25Okapi(corpus)
            return self._bm25

    def get_docs(self) -> list[list[str]]:
        with self._lock:
            return list(self._tokenized_docs)

    def get_paper_ids(self) -> list[str]:
        with self._lock:
            return list(self._paper_ids)

    def index_of(self, paper_id: str) -> int | None:
        with self._lock:
            return self._id_to_index.get(paper_id)


_bm25_index = IncrementalBM25()


def build_document_text(m: Metadata) -> str:
    """构造用于 BM25 / keyword 的文档文本（不加载 image_base64）"""
    return m.index_text


def _tokens_for_metadata(m: Metadata) -> list[str]:
    return content_tokens(build_document_text(m))


def build_bm25_index(metadata_list: list[Metadata]):
    """用完整列表重建 BM25 索引（启动/全量重建时用）"""
    global _bm25_index
    new_index = IncrementalBM25()
    for m in metadata_list:
        if m.id:
            new_index.add_document(m.id, _tokens_for_metadata(m))
    _bm25_index = new_index


def add_to_bm25_index(metadata: Metadata):
    """增量添加单篇论文到 BM25 索引"""
    if metadata.id:
        _bm25_index.add_document(metadata.id, _tokens_for_metadata(metadata))


def invalidate_bm25_cache():
    """兼容旧接口：重新触发 BM25 索引重建"""
    global _bm25_index
    _bm25_index = IncrementalBM25()


# ─── BM25 / Keyword 检索 ──────────────────────────────────────

def bm25_search(metadata_list: list[Metadata], question: str, keywords: str = "", top_k: int = 200) -> list[dict]:
    query_tokens = tokenize(keywords or question)
    if not query_tokens:
        return []
    bm25 = _bm25_index.get_bm25()
    scores = bm25.get_scores(query_tokens)
    paper_ids = _bm25_index.get_paper_ids()
    ranked = sorted(
        [{"paper_id": paper_ids[i], "score": float(s), "index": i, "source": "bm25"}
         for i, s in enumerate(scores) if i < len(paper_ids)],
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
    paper_ids = _bm25_index.get_paper_ids()
    cached_docs = _bm25_index.get_docs()
    scores = []
    meta_by_id = {m.id: m for m in metadata_list if m.id}

    for i, pid in enumerate(paper_ids):
        m = meta_by_id.get(pid)
        if m is None:
            scores.append((pid, 0.0))
            continue
        doc_tokens = set(cached_docs[i])
        title_tokens = set(tokenize(m.title or ""))
        if not doc_tokens and not title_tokens:
            scores.append((pid, 0.0))
            continue
        doc_cov = len(q_set & doc_tokens) / len(q_set) if q_set else 0.0
        title_cov = len(q_set & title_tokens) / len(q_set) if q_set else 0.0
        scores.append((pid, 0.7 * doc_cov + 0.3 * title_cov))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked = []
    for r, (pid, s) in enumerate(scores[:top_k], start=1):
        ranked.append({
            "paper_id": pid,
            "score": s, "index": r, "source": "keyword", "rank": r,
        })
    return ranked


# ─── 向量检索 ─────────────────────────────────────────────────

def _chunk_results_to_paper_scores(
    chunk_results: list[dict],
    source_name: str,
    top_k: int = 200,
) -> list[dict]:
    """
    将 chunk 级检索结果聚合为论文级，返回按 paper_id 聚合后的得分列表（MaxSim）。
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


def vector_search(
    query: str,
    top_k_per_type: int = 200,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
) -> dict[str, list[dict]]:
    """向量检索，返回各类型的论文级排名"""
    query_vec = embed_single(query)
    raw = query_all_chunk_types(
        query_vec,
        top_k_per_type=top_k_per_type,
        year_filter=year_filter,
        source_type_filter=source_type_filter,
    )

    results = {}
    for ctype, chunks in raw.items():
        source_map = {
            "text": "vector_text",
            "title": "vector_title",
            "summary": "vector_summary",
            "image_desc": "vector_image",
        }
        source = source_map.get(ctype, f"vector_{ctype}")
        results[source] = _chunk_results_to_paper_scores(chunks, source, top_k=top_k_per_type)
    return results


# ─── RRF 融合 ─────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    rrf_k: int = 60,
    top_k: int = 50,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
    metadata_map: dict[str, Metadata] | None = None,
) -> list[dict]:
    """
    多路排名 RRF 融合。
    每个 ranked_list 的元素需包含: paper_id, rank, score, source
    可选 year / source_type 过滤。
    """
    metadata_map = metadata_map or {}
    fusion: dict[str, float] = {}
    paper_info: dict[str, dict] = {}

    for lst in ranked_lists:
        for item in lst:
            pid = item["paper_id"]
            meta = metadata_map.get(pid)

            # 过滤：年份下限
            if year_filter is not None:
                item_year = getattr(meta, "year", None)
                try:
                    if item_year is None or int(item_year) < year_filter:
                        continue
                except (ValueError, TypeError):
                    continue

            # 过滤：来源类型
            if source_type_filter:
                item_source_type = getattr(meta, "source_type", None)
                if item_source_type != source_type_filter:
                    continue

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

    metadata_list = metadata_list or []
    ranked_lists: list[list[dict]] = []

    # 1) BM25 + Keyword 词汇检索
    if metadata_list:
        bm25_res = bm25_search(metadata_list, query, keywords)
        kw_res = keyword_search(metadata_list, query, keywords)
        if bm25_res:
            ranked_lists.append(bm25_res)
        if kw_res:
            ranked_lists.append(kw_res)

    # 构建 metadata_map（供过滤和 MMR 使用）
    metadata_map = {m.id: m for m in metadata_list if m.id}

    # 2) 向量检索（4路）
    vec_results = vector_search(
        query,
        year_filter=year_filter,
        source_type_filter=source_type_filter,
    )
    for source, pr in vec_results.items():
        if pr:
            ranked_lists.append(pr)

    # 3) RRF 融合
    fused = reciprocal_rank_fusion(
        *ranked_lists,
        year_filter=year_filter,
        source_type_filter=source_type_filter,
        metadata_map=metadata_map,
        top_k=max(top_k * 5, 50),
    )

    # 4) 可选 MMR
    if use_mmr and metadata_list:
        fused = mmr_rerank(fused, metadata_map, top_k=top_k)

    # 5) 截断 + 附加 metadata
    top = fused[:top_k]
    now = datetime.now(timezone.utc).isoformat()

    for item in top:
        pid = item["paper_id"]
        m = metadata_map.get(pid)
        if m:
            m.last_retrieved_at = now
            item["metadata"] = m
            item["title"] = m.title
            item["year"] = m.year
            item["source_type"] = m.source_type
            item["text_summary"] = m.text_summary
            item["image_summary"] = m.image_summary

    return top


# ─── 加载元数据 ───────────────────────────────────────────────

def load_metadata_from_dir(data_dir: str) -> list[Metadata]:
    from load_metadata import load_metadata_from_dir as _load
    return _load(data_dir)

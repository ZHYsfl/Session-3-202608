"""
ChromaDB 向量库模块
- 单 collection 管理所有 chunk 类型
- 支持 metadata 过滤
- 支持增量 upsert（热更新）

改造重点：
1. SplitBlock 本身不存向量，只通过 chunk_id 与向量库关联。
2. 每个 batch 立即 upsert，避免全量加载到内存。
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_DIM
from models import SplitBlock

_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=ChromaSettings(anonymized_telemetry=False),
)


def get_collection():
    """获取或创建 collection"""
    return _client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def clear_collection():
    """清空 collection（用于重建）"""
    try:
        _client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    return get_collection()


CHROMA_MAX_BATCH = 4000  # Chroma 单次 upsert 上限约 5461，保守取 4000


def _normalize_year(value: Any) -> int | None:
    """将 year 归一化为 int；无法解析时返回 None。"""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            return None
    return None


def _to_chroma_records(blocks: list[SplitBlock], embeddings: list[list[float]]) -> tuple[list[str], list[list[float]], list[str], list[dict]]:
    ids = [b.id for b in blocks]
    docs = [b.text for b in blocks]
    metas = []
    for b in blocks:
        meta = {
            "paper_id": b.paper_id,
            "chunk_type": b.chunk_type,
            "position": b.position,
        }
        # 过滤掉 None / 非基本类型，避免 Chroma metadata 验证失败
        for k, v in b.metadata.items():
            if v is None:
                continue
            if k == "year":
                norm = _normalize_year(v)
                if norm is not None:
                    meta[k] = norm
            elif isinstance(v, (str, int, float, bool)):
                meta[k] = v
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                meta[k] = v
        metas.append(meta)
    return ids, embeddings, docs, metas


def upsert_blocks(blocks: list[SplitBlock], embeddings: list[list[float]]):
    """
    将 SplitBlock + embedding 批量写入 ChromaDB。
    要求 len(blocks) == len(embeddings)。
    """
    if not blocks:
        return
    if len(blocks) != len(embeddings):
        raise ValueError(f"blocks ({len(blocks)}) 和 embeddings ({len(embeddings)}) 数量不一致")

    col = get_collection()
    ids, embs, docs, metas = _to_chroma_records(blocks, embeddings)

    for i in range(0, len(blocks), CHROMA_MAX_BATCH):
        col.upsert(
            ids=ids[i : i + CHROMA_MAX_BATCH],
            embeddings=embs[i : i + CHROMA_MAX_BATCH],
            documents=docs[i : i + CHROMA_MAX_BATCH],
            metadatas=metas[i : i + CHROMA_MAX_BATCH],
        )


# 兼容旧接口（chunks dict）

def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """保留旧接口，传入 dict 列表也能 upsert。"""
    blocks = []
    for c in chunks:
        blocks.append(SplitBlock(
            id=c["id"],
            paper_id=c.get("metadata", {}).get("paper_id", ""),
            chunk_type=c.get("chunk_type", "text"),
            text=c.get("document", ""),
            position=c.get("metadata", {}).get("chunk_index", 0),
            metadata={k: v for k, v in (c.get("metadata") or {}).items() if k not in ("paper_id", "chunk_type")},
        ))
    upsert_blocks(blocks, embeddings)


def query_by_chunk_type(
    query_vec: list[float] | np.ndarray,
    chunk_type: str,
    top_k: int = 100,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    按 chunk_type 检索，可选元数据过滤。
    返回: [{paper_id, document, score, metadata}, ...]
    """
    col = get_collection()

    # ChromaDB where 要求顶层只能有一个操作符；多条件用 $and 组合。
    clauses: list[dict[str, Any]] = [{"chunk_type": chunk_type}]
    if year_filter is not None and isinstance(year_filter, int):
        clauses.append({"year": {"$gte": year_filter}})
    if source_type_filter and isinstance(source_type_filter, str):
        clauses.append({"source_type": source_type_filter})

    if len(clauses) == 1:
        where = clauses[0]
    else:
        where = {"$and": clauses}

    query_embeddings = [query_vec.tolist() if isinstance(query_vec, np.ndarray) else query_vec]
    try:
        results = col.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        # 若 metadata 类型不匹配或字段缺失导致 ChromaDB 过滤失败，
        # 降级为仅按 chunk_type 查询，避免直接抛错。
        import warnings
        warnings.warn(f"[vector_store] metadata filter failed ({exc}); falling back to chunk_type only.")
        results = col.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            where={"chunk_type": chunk_type},
            include=["documents", "metadatas", "distances"],
        )

    out = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            dist = results["distances"][0][i] if results["distances"] else 0.0
            sim = 1.0 - dist  # cosine distance → similarity
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc = results["documents"][0][i] if results["documents"] else ""
            out.append({
                "chunk_id": doc_id,
                "paper_id": meta.get("paper_id", ""),
                "document": doc,
                "score": float(sim),
                "chunk_type": chunk_type,
                "metadata": meta,
            })
    return out


def query_all_chunk_types(
    query_vec: list[float] | np.ndarray,
    top_k_per_type: int = 100,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    一次查询所有 chunk 类型，返回按类型分组的 dict。
    """
    results: dict[str, list[dict[str, Any]]] = {}
    for ctype in ("title", "summary", "text", "image_desc"):
        results[ctype] = query_by_chunk_type(
            query_vec,
            ctype,
            top_k=top_k_per_type,
            year_filter=year_filter,
            source_type_filter=source_type_filter,
        )
    return results


# ─── 聚合：MaxSim ───────────────────────────────────────────────────

def aggregate_maxsim(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    将 chunk 级结果按 paper_id 聚合，每篇论文取最大相似度。
    输入: query_by_chunk_type 返回的列表
    输出: [{paper_id, score, best_chunk, chunk_type, ...}, 按score降序]
    """
    paper_best: dict[str, dict[str, Any]] = {}
    for c in chunks:
        pid = c["paper_id"]
        if pid not in paper_best or c["score"] > paper_best[pid]["score"]:
            paper_best[pid] = {
                "paper_id": pid,
                "score": c["score"],
                "best_chunk_id": c["chunk_id"],
                "document": c["document"],
                "chunk_type": c["chunk_type"],
                "metadata": c["metadata"],
            }
    return sorted(paper_best.values(), key=lambda x: x["score"], reverse=True)


def get_existing_chunk_ids() -> set[str]:
    """获取当前 collection 中已有的所有 chunk id。"""
    col = get_collection()
    try:
        # chroma 的 get 可能在数据量大时占用内存，仅用于小规模热更新判断
        data = col.get(include=[])
        return set(data.get("ids", []))
    except Exception:
        return set()


def count_collection() -> int:
    """获取 collection 中的总条数。"""
    try:
        return get_collection().count()
    except Exception:
        return 0

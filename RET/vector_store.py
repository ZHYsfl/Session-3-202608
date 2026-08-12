"""
ChromaDB 向量库模块
- 单 collection 管理所有 chunk 类型
- 支持 metadata 过滤
"""
from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_DIM

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=ChromaSettings(anonymized_telemetry=False))


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


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """
    批量写入 chunks + embeddings 到 Chroma。
    自动按 Chroma batch size 上限分批。
    """
    if not chunks:
        return
    col = get_collection()

    for i in range(0, len(chunks), CHROMA_MAX_BATCH):
        batch_c = chunks[i : i + CHROMA_MAX_BATCH]
        batch_e = embeddings[i : i + CHROMA_MAX_BATCH]
        ids = [c["id"] for c in batch_c]
        documents = [c["document"] for c in batch_c]
        metadatas = [c["metadata"] for c in batch_c]
        col.upsert(ids=ids, embeddings=batch_e, documents=documents, metadatas=metadatas)


def query_by_chunk_type(
    query_vec: list[float],
    chunk_type: str,
    top_k: int = 100,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
) -> list[dict]:
    """
    按 chunk_type 检索，可选元数据过滤。
    返回: [{paper_id, document, score, metadata}, ...]
    """
    col = get_collection()

    where = {"chunk_type": chunk_type}
    # Chroma 的 where 子句
    # if year_filter:
    #     where["year"] = {"$gte": year_filter}
    # if source_type_filter:
    #     where["source_type"] = source_type_filter

    results = col.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
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
    query_vec: list[float],
    top_k_per_type: int = 100,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
) -> dict[str, list[dict]]:
    """
    一次查询所有 chunk 类型，返回按类型分组的 dict。
    """
    results = {}
    for ctype in ("title", "summary", "text", "image_desc"):
        results[ctype] = query_by_chunk_type(
            query_vec, ctype,
            top_k=top_k_per_type,
            year_filter=year_filter,
            source_type_filter=source_type_filter,
        )
    return results


# ─── 聚合：MaxSim ────────────────────────────────────────────

def aggregate_maxsim(chunks: list[dict]) -> list[dict]:
    """
    将 chunk 级结果按 paper_id 聚合，每篇论文取最大相似度。
    输入: query_by_chunk_type 返回的列表
    输出: [{paper_id, score, best_chunk, chunk_type, ...}, 按score降序]
    """
    paper_best: dict[str, dict] = {}
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

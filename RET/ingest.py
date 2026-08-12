"""
批量入库脚本
==================
标准 RAG 入库流程：
1. 读取 data/ 下所有 JSON
2. 分块（SplitBlock）
3. 每一小批（每篇论文）调用 embedding API
4. 立即 upsert 到 ChromaDB
5. 更新 index_manifest

改造重点：
- 不再累积全部 chunks 和 embeddings 到内存
- SplitBlock 不存储向量
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from chunker import build_split_blocks_from_dict
from config import CHROMA_PERSIST_DIR, INDEX_VERSION
from embedding import embed_batch
from index_store import add_paper_chunks, clear_manifest, get_index_version, set_index_version
from models import Metadata
from retriever import build_bm25_index
from vector_store import clear_collection, upsert_blocks

LOG_FILE = os.path.join(os.path.dirname(__file__), "ingest.log")


def _log(msg: str):
    """同时输出到 stdout 和日志文件"""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ingest(
    data_dir: str,
    reset: bool = True,
    skip_indexed: bool = True,
) -> dict[str, int]:
    """
    批量入库主函数。

    Args:
        data_dir: 数据目录
        reset: 是否清空旧库重建
        skip_indexed: 是否跳过已在 index_manifest 中的论文（增量入库）
    """
    data_dir = os.path.abspath(data_dir)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

    # 清空日志
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    _log(f"📂 发现 {len(files)} 个 JSON 文件")
    _log(f"💾 ChromaDB 路径: {CHROMA_PERSIST_DIR}")

    if reset:
        _log("🗑️  清空旧向量库...")
        clear_collection()
        _log("🗑️  清空索引清单...")
        clear_manifest()
        set_index_version(None)
    else:
        stored_version = get_index_version()
        if stored_version != INDEX_VERSION:
            # reset=False 时不自动清空：让用户显式选择 reset=True 来重建，
            # 这样支持中断后恢复，也避免误删已有索引。
            _log(
                f"⚠️ 索引版本不匹配 (stored={stored_version}, current={INDEX_VERSION})。"
                f"由于 reset=False，不清空现有数据，继续增量入库。"
                f"如需强制重建，请使用 reset=True。"
            )

    total_chunks = 0
    total_papers = 0
    skipped_papers = 0
    indexed_metas: list[Metadata] = []

    t_start = time.time()

    from index_store import is_paper_indexed

    for i, fname in enumerate(files):
        filepath = os.path.join(data_dir, fname)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            _log(f"  ⚠️  读取失败 {fname}: {e}")
            continue

        paper_id = obj.get("id", fname.replace(".json", ""))

        if skip_indexed and is_paper_indexed(paper_id):
            skipped_papers += 1
            continue

        try:
            blocks = build_split_blocks_from_dict(obj)
        except Exception as e:
            _log(f"  ⚠️  分块失败 {fname}: {e}")
            continue

        if not blocks:
            continue

        # 每篇论文一个 batch，不累积全部，立即 embedding + upsert
        texts = [b.text for b in blocks]
        try:
            vecs = embed_batch(texts)
            upsert_blocks(blocks, vecs)
            add_paper_chunks(paper_id, [b.id for b in blocks])
        except Exception as e:
            _log(f"  ❌ 入库失败 {fname}: {e}")
            continue

        total_papers += 1
        total_chunks += len(blocks)
        indexed_metas.append(Metadata.from_dict(obj))

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t_start
            _log(
                f"  ✅ [{i+1}/{len(files)}] "
                f"已入库 {total_papers} 篇 / 跳过 {skipped_papers} 篇 | "
                f"{total_chunks:>6} chunks | {elapsed:.0f}s"
            )

    from vector_store import count_collection
    total_count = count_collection()

    # 同步构建 BM25 索引（只使用已入库论文，避免全量加载未索引元数据）
    if indexed_metas:
        _log("📊 构建 BM25 索引...")
        build_bm25_index(indexed_metas)
        _log(f"   ✅ BM25 索引已构建 ({len(indexed_metas)} 篇)")

    # 记录索引版本
    set_index_version(INDEX_VERSION)

    elapsed = time.time() - t_start
    _log(f"\n{'='*50}")
    _log(f"✅ 入库完成！耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    _log(f"   论文数: {total_papers}")
    _log(f"   总 chunks: {total_count}")
    _log(f"   平均每篇: {total_count / max(total_papers, 1):.1f} chunks")
    _log(f"   索引版本: {INDEX_VERSION}")
    _log(f"{'='*50}")

    return {
        "total_papers": total_papers,
        "total_chunks": total_count,
        "skipped_papers": skipped_papers,
        "elapsed_seconds": int(elapsed),
    }


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "data")
    ingest(data_dir, reset=True, skip_indexed=True)

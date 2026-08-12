"""
批量入库脚本
1. 读取 data/ 下所有 JSON
2. 分块
3. 批量 embedding
4. 写入 ChromaDB
"""
from __future__ import annotations

import json
import os
import sys
import time

from config import CHROMA_PERSIST_DIR
from chunker import build_chunks, clean_source_type
from embedding import embed_batch
from vector_store import clear_collection, upsert_chunks

LOG_FILE = os.path.join(os.path.dirname(__file__), "ingest.log")


def _log(msg: str):
    """同时输出到 stdout 和日志文件"""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def ingest(data_dir: str, batch_upsert_size: int = 10):
    """批量入库主函数"""
    data_dir = os.path.abspath(data_dir)
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])

    # 清空日志
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    _log(f"📂 发现 {len(files)} 个 JSON 文件")
    _log(f"💾 ChromaDB 路径: {CHROMA_PERSIST_DIR}")
    _log("🗑️  清空旧向量库...")
    clear_collection()

    total_chunks = 0
    total_papers = 0
    all_chunks: list[dict] = []
    all_embeddings: list[list[float]] = []

    t_start = time.time()

    for i, fname in enumerate(files):
        filepath = os.path.join(data_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)

        paper_id = obj.get("id", fname.replace(".json", ""))
        chunks = build_chunks(obj)

        if not chunks:
            continue

        total_papers += 1
        total_chunks += len(chunks)

        # 提取文本去 embedding
        texts = [c["document"] for c in chunks]
        vecs = embed_batch(texts)

        all_chunks.extend(chunks)
        all_embeddings.extend(vecs)

        # 每 batch_upsert_size 篇论文写入一次
        if (i + 1) % batch_upsert_size == 0:
            upsert_chunks(all_chunks, all_embeddings)
            elapsed = time.time() - t_start
            _log(f"  ✅ [{i+1}/{len(files)}] {total_chunks:>6} chunks | {elapsed:.0f}s")
            all_chunks.clear()
            all_embeddings.clear()
            time.sleep(0.5)

    # 写入剩余
    if all_chunks:
        upsert_chunks(all_chunks, all_embeddings)
        _log(f"  ✅ [{total_papers}/{len(files)}] {total_chunks} chunks (final)")

    from vector_store import get_collection
    col = get_collection()
    total_count = col.count()
    elapsed = time.time() - t_start
    _log(f"\n{'='*50}")
    _log(f"✅ 入库完成！耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    _log(f"   论文数: {total_papers}")
    _log(f"   总 chunks: {total_count}")
    _log(f"   平均每篇: {total_count / max(total_papers, 1):.1f} chunks")
    _log(f"{'='*50}")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "data")
    ingest(data_dir)

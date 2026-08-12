"""
用 RET RAG 管线为 500 题评测集生成 retrieval_results.json。

用法:
    cd data_evaluation/medical-evidence-rag/part9_evaluation
    source /home/ubuntu/Session-3-202608/RET/venv/bin/activate
    python generate_retrieval_results.py \
        --questions questions_500.json \
        --output retrieval_results.json \
        --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# 引入 RET 管线
RET_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "RET"
sys.path.insert(0, str(RET_ROOT))

from load_metadata import load_metadata_from_dir
from models import Metadata
from retriever import build_bm25_index, hybrid_retrieve


def _metadata_to_eval_format(m: Metadata) -> dict[str, Any]:
    """把 RET Metadata 转成 part9_evaluation 要求的 dict 格式。"""
    return {
        "title": m.title,
        "year": m.year,
        "source_type": m.source_type,
        "id": m.id,
        "text_summary": m.text_summary,
        "text": m.text,
        "image_summary": m.image_summary or [],
        "image_base_64": [],  # 评测不需要图像二进制，避免文件过大
        "evidence_level": m.evidence_level,
        "last_retrieved_at": m.last_retrieved_at,
    }


def generate(
    questions_path: Path,
    output_path: Path,
    top_k: int,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    data_dir = data_dir or (RET_ROOT / "data")
    print(f"⏳ 加载元数据: {data_dir}", flush=True)
    metas = load_metadata_from_dir(str(data_dir))
    print(f"📚 已加载 {len(metas)} 篇论文，预热 BM25...", flush=True)
    build_bm25_index(metas)
    print("✅ BM25 就绪", flush=True)

    with open(questions_path, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset.get("questions", [])
    results: dict[str, list[dict[str, Any]]] = {}
    total = len(questions)

    t0 = time.time()
    for i, q in enumerate(questions, start=1):
        qid = q["id"]
        query = q["question"]
        try:
            raw = hybrid_retrieve(query, metadata_list=metas, top_k=top_k)
            formatted = []
            for item in raw:
                m = item.get("metadata")
                if isinstance(m, Metadata):
                    formatted.append(_metadata_to_eval_format(m))
            results[qid] = formatted
        except Exception as exc:
            print(f"  ⚠️ {qid} 检索失败: {exc}", flush=True)
            results[qid] = []

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            print(
                f"  ✅ [{i}/{total}] 已生成检索结果 | {elapsed:.0f}s",
                flush=True,
            )

    out = {
        "version": "1.0.0",
        "description": f"RET hybrid retrieve (BM25 + keyword + vector), top_k={top_k}",
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n📝 已保存: {output_path}")
    print(f"   总题数: {len(results)}")
    non_empty = sum(1 for v in results.values() if v)
    print(f"   非空检索结果: {non_empty}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为500题生成RAG检索结果")
    parser.add_argument("--questions", type=Path, default=Path("questions_500.json"))
    parser.add_argument("--output", type=Path, default=Path("retrieval_results.json"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--data-dir", type=Path, default=None, help="RET data目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.questions, args.output, args.top_k, args.data_dir)


QUESTION = "大语言模型的推理与训练方法"  # 问题检索
KEYWORDS = "大语言模型 推理 训练"   # 关键词检索，用空格进行分割，后面会进行分词
TOP_K = 3 # 修改K的数值

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import json
import os
import re

import jieba
from rank_bm25 import BM25Okapi


@dataclass
class Metadata:
    title: Optional[str] = None
    year: Optional[str] = None
    source_type: Optional[str] = None
    id: Optional[str] = None
    text: Optional[str] = None
    text_summary: Optional[str] = None
    image_summary: Optional[list[str]] = None
    image_base_64: Optional[list[str]] = None
    evidence_level: Optional[list[str]] = None
    last_retrieved_at: Optional[str] = None


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


def _rank_results(meta_list, scores, reverse=True):
    ranked = sorted(
        [{"metadata": meta_list[i], "score": float(s), "index": i}
         for i, s in enumerate(scores)],
        key=lambda x: x["score"], reverse=reverse,
    )
    for r, item in enumerate(ranked, start=1):
        item["rank"] = r
    return ranked


def bm25_search(metadata_list: list[Metadata], question: str, keywords: str = ""):
    query_tokens = tokenize(keywords or question)
    if not query_tokens:
        return []
    docs = [tokenize(build_document_text(m)) for m in metadata_list]
    scores = BM25Okapi(docs).get_scores(query_tokens)
    return _rank_results(metadata_list, scores)


def keyword_search(metadata_list: list[Metadata], question: str, keywords: str = ""):
    query_tokens = tokenize(keywords or question)
    if not query_tokens:
        return []
    q_set = set(query_tokens)

    scores = []
    for m in metadata_list:
        doc_tokens = set(tokenize(build_document_text(m)))
        title_tokens = set(tokenize(m.title or ""))
        if not doc_tokens and not title_tokens:
            scores.append(0.0)
            continue
        doc_cov = len(q_set & doc_tokens) / len(q_set) if doc_tokens else 0.0
        title_cov = len(q_set & title_tokens) / len(q_set) if title_tokens else 0.0
        scores.append(0.7 * doc_cov + 0.3 * title_cov)

    return _rank_results(metadata_list, scores)


def reciprocal_rank_fusion(*ranked_lists, rrf_k: int = 60):
    fusion = {}
    meta_map = {}

    for lst in ranked_lists:
        for item in lst:
            m = item["metadata"]
            doc_id = m.id or str(item["index"])
            meta_map[doc_id] = m
            fusion[doc_id] = fusion.get(doc_id, 0.0) + 1.0 / (rrf_k + item["rank"])

    sorted_items = sorted(fusion.items(), key=lambda x: x[1], reverse=True)
    if not sorted_items:
        return []

    max_v = sorted_items[0][1]
    return [
        {"metadata": meta_map[doc_id], "score": v / max_v, "raw_rrf_score": v, "rank": r}
        for r, (doc_id, v) in enumerate(sorted_items, start=1)
    ]


def hybrid_retrieve(metadata_list: list[Metadata], question: str, k: int = 100, keywords: str = ""):
    if not metadata_list or not question.strip():
        return []
    fused = reciprocal_rank_fusion(
        bm25_search(metadata_list, question, keywords),
        keyword_search(metadata_list, question, keywords),
    )
    top_k = fused[:k]
    now = datetime.now(timezone.utc).isoformat()
    for item in top_k:
        item["metadata"].last_retrieved_at = now
    return top_k


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


def save_results(results, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    payload = [{
        "rank": item["rank"], "score": item["score"],
        "raw_rrf_score": item["raw_rrf_score"],
        "metadata": {
            k: v for k, v in item["metadata"].__dict__.items()
            if k in ("id", "title", "year", "source_type", "text_summary",
                     "image_summary", "evidence_level", "last_retrieved_at")
        }
    } for item in results]
    path = os.path.join(out_dir, "top3_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"结果已保存至: {path}")


def save_results_txt(results, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    lines.append(f"检索问题: {QUESTION}")
    lines.append(f"Top {TOP_K} 结果\n")
    for item in results:
        m = item["metadata"]
        lines.append(f"Rank {item['rank']}  |  关联度: {item['score']:.4f}")
        lines.append(f"  ID:    {m.id}")
        lines.append(f"  Title: {m.title}  ({m.year})")
        lines.append(f"  Source: {m.source_type}")
        lines.append("")
    path = os.path.join(out_dir, "top3_results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"TXT 结果已保存至: {path}")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    meta_list = load_metadata_from_dir(os.path.join(base, "data"))
    print(f"已加载 {len(meta_list)} 篇论文样例\n")

    results = hybrid_retrieve(meta_list, QUESTION, k=TOP_K, keywords=KEYWORDS)

    print("=" * 70)
    print(f"检索问题: {QUESTION}")
    print("=" * 70)
    for item in results:
        m = item["metadata"]
        print(f"\nRank: {item['rank']}  |  Score: {item['score']:.4f}")
        print(f"  ID:    {m.id}")
        print(f"  Title: {m.title}  ({m.year})")
        print(f"  RRF:   {item['raw_rrf_score']:.6f}")
        print(f"  Retrieved: {m.last_retrieved_at}")

    save_results(results, os.path.join(base, "coarse_sort_result"))
    save_results_txt(results, base)

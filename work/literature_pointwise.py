"""医疗文献元数据打分与筛选。

输入：Python 变量，k1 条 dict（每条对应一个 JSON 记录）
输出：Python 变量，k2 条 dict（按总分降序）

记录中图片相关字段是两个按索引一一对应的列表（下标 i 属于同一张图片）：
    image_summary:  list[str]  图片摘要
    image_base_64:  list[str]  图片详细内容（base64）
"""

from __future__ import annotations

import json
import pickle
from datetime import date
from typing import Any, Callable, Iterable


# 默认权重（可覆盖），总分按权重归一化为 100 分制
DEFAULT_WEIGHTS = {
    "relevance": 50,      # 与查询主题的相关性
    "evidence": 30,       # 证据等级
    "recency": 10,        # 发表年份新近度
    "completeness": 10,   # 文本/图片完整度
}

# 证据等级 -> 分值映射（按证据金字塔，可增改）
DEFAULT_EVIDENCE_MAP = {
    "systematic review": 1.0,
    "meta-analysis": 1.0,
    "rct": 0.9,
    "randomized controlled trial": 0.9,
    "cohort": 0.7,
    "case-control": 0.6,
    "case series": 0.5,
    "case report": 0.4,
    "expert opinion": 0.2,
    "editorial": 0.2,
}

def _get_image_lists(
    rec: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """返回 (image_summary, image_base_64)，两个列表按索引一一对应。"""
    summaries = rec.get("image_summary") or []
    bases = rec.get("image_base_64") or []
    return summaries, bases


def _query_terms(query: str) -> list[str]:
    """把查询拆成匹配词。中文查询按双字片段切分（免分词依赖）。"""
    q = (query or "").strip()
    if not q:
        return []
    terms = q.lower().split()
    if len(terms) == 1 and any("\u4e00" <= ch <= "\u9fff" for ch in q):
        grams = [q[i : i + 2] for i in range(len(q) - 1)]
        return grams or [q]
    return terms


def _text_similarity(
    query: str,
    summary: str | None,
    text: str | None,
    image_summaries: list[str] | None = None,
) -> float:
    """轻量相关性：query 关键词在摘要/全文/图片文字中的命中比例，0~1。

    如果想要真正的语义相似度，把这里替换成 embedding 向量点积即可，
    函数签名和返回范围保持不变。
    """
    terms = _query_terms(query)
    if not terms:
        return 0.0
    parts = [summary or "", text or ""]
    parts.extend(s for s in (image_summaries or []) if s)
    corpus = " ".join(parts).lower()
    hits = sum(1 for t in terms if t in corpus)
    return hits / len(terms)


def _evidence_score(level: str | None, evidence_map: dict[str, float]) -> float:
    if not level:
        return 0.0
    s = level.strip().lower()
    if s in evidence_map:
        return evidence_map[s]
    # 允许 "Randomized Controlled Trial (RCT)" 这类带修饰的写法
    for key, value in evidence_map.items():
        if key in s:
            return value
    return 0.0


def _recency_score(year_raw: str | None, current_year: int) -> float:
    try:
        year = int(str(year_raw).strip())
    except (TypeError, ValueError):
        return 0.0
    span = max(current_year - 2000, 1)
    return max(0.0, min(1.0, (year - 2000) / span))


def _completeness_score(
    text: str | None,
    image_summaries: list[str] | None,
    image_bases: list[str] | None,
) -> float:
    """文本完整度 + 图片完整度（同一索引下 summary 和 base64 都不为空才算完整）。"""
    text = text or ""
    text_score = 1.0 if len(text) >= 500 else len(text) / 500

    summaries = image_summaries or []
    bases = image_bases or []
    n = max(len(summaries), len(bases))
    if n == 0:
        image_score = 0.0
    else:
        complete = sum(
            1
            for i in range(n)
            if (i < len(summaries) and summaries[i])
            and (i < len(bases) and bases[i])
        )
        image_score = complete / n

    return 0.6 * text_score + 0.4 * image_score


def score_record(
    rec: dict[str, Any],
    *,
    query: str | None = None,
    weights: dict[str, float] | None = None,
    evidence_map: dict[str, float] | None = None,
    current_year: int | None = None,
) -> float:
    """对单条记录打分，返回 0~100 的分数。"""
    weights = dict(weights or DEFAULT_WEIGHTS)
    evidence_map = evidence_map or DEFAULT_EVIDENCE_MAP
    current_year = current_year or date.today().year

    img_summaries, img_bases = _get_image_lists(rec)
    parts = {
        "relevance": (
            _text_similarity(
                query or "",
                rec.get("text_summary"),
                rec.get("text"),
                img_summaries,
            )
            if query
            else None
        ),
        "evidence": _evidence_score(rec.get("evidence_level"), evidence_map),
        "recency": _recency_score(rec.get("year"), current_year),
        "completeness": _completeness_score(
            rec.get("text"), img_summaries, img_bases
        ),
    }

    # 没有 query 时去掉相关性维度，其余权重归一化
    active = {k: v for k, v in parts.items() if v is not None}
    total_w = sum(weights[k] for k in active)
    if total_w <= 0:
        return 0.0
    return sum(weights[k] * active[k] for k in active) / total_w * 100


def select_top_k(
    records: Iterable[dict[str, Any]],
    k2: int,
    *,
    query: str | None = None,
    scorer: Callable[[dict[str, Any]], float] | None = None,
    weights: dict[str, float] | None = None,
    evidence_map: dict[str, float] | None = None,
    drop_empty: bool = True,
) -> list[dict[str, Any]]:
    """从 k1 条记录中选出分数最高的 k2 条。

    drop_empty=True 时，text、text_summary 和图片列表全部为空的记录
    视为解析失败，直接排除。

    scorer：可选的自定义打分函数 score(rec) -> 0~100（例如 LLM 打分）；
    不传时使用内置规则打分 score_record，可用 query/weights/evidence_map 调参。
    """
    candidates = list(records)
    if drop_empty:

        def has_content(r: dict[str, Any]) -> bool:
            if r.get("text") or r.get("text_summary"):
                return True
            summaries, bases = _get_image_lists(r)
            return bool(summaries or bases)

        candidates = [r for r in candidates if has_content(r)]

    def score(rec: dict[str, Any]) -> float:
        if scorer is not None:
            return scorer(rec)
        return score_record(
            rec, query=query, weights=weights, evidence_map=evidence_map
        )

    scored = [(rec, score(rec)) for rec in candidates]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [rec for rec, _ in scored[: max(k2, 0)]]


# ---------- 保存/读取工具 ----------

def save_jsonl(records: Iterable[dict[str, Any]], path: str) -> None:
    """每行一个 JSON，适合 k1 很大时逐行追加/读取。"""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(records: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_pickle(records: list[dict[str, Any]], path: str) -> None:
    """原样保存 Python 对象（保留字段顺序、非 JSON 类型），但不可跨语言/不可读。"""
    with open(path, "wb") as f:
        pickle.dump(records, f)


if __name__ == "__main__":
    # 简单自测：3 条记录 -> 取 2 条
    demo = [
        {
            "title": "RCT on new drug",
            "year": "2023",
            "evidence_level": "RCT",
            "text_summary": "A randomized trial of metformin for diabetes.",
            "text": "x" * 1000,
            "image_summary": ["Figure 1 results", "Figure 2 survival"],
            "image_base_64": ["aaa", "bbb"],
        },
        {
            "title": "Old case report",
            "year": "2001",
            "evidence_level": "case report",
            "text_summary": "A single patient case.",
            "text": "x" * 100,
            "image_summary": ["Figure 1"],
            "image_base_64": ["ccc"],
        },
        {
            "title": "Empty record",
            "year": "2010",
            "evidence_level": None,
            "text_summary": None,
            "text": None,
            "image_summary": None,
            "image_base_64": None,
        },
    ]
    picked = select_top_k(demo, 2, query="metformin diabetes")
    for rec in picked:
        print(rec["title"], round(score_record(rec, query="metformin diabetes"), 1))

"""医疗文献两两对比（round-robin）排序。

输入：Python 变量，k1 条 dict（每条对应一个 JSON 记录）
输出：Python 变量，按获胜场次降序排列的 dict 列表

规则：任意两条记录两两对比一次，胜者 +1 分，平局双方都不加分；
最终按总分（获胜场次）从高到低排序，分数相同的保持 k1 原有相对顺序。

默认以文件内独立实现的规则打分（与 literature_pointwise 算法相同，
但不互相依赖、不复用）作为对比依据（分数高者胜），
也可以传入自定义 compare(a, b) 函数替换（例如调用 LLM 当裁判）。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Iterable


# ---------- 内置规则打分（独立实现，与 literature_pointwise 不互相依赖） ----------

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
    """轻量相关性：query 关键词在摘要/全文/图片文字中的命中比例，0~1。"""
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


def _score_record(
    rec: dict[str, Any],
    *,
    query: str | None = None,
    weights: dict[str, float] | None = None,
    evidence_map: dict[str, float] | None = None,
    current_year: int | None = None,
) -> float:
    """对单条记录打分，返回 0~100 的分数（与 pointwise 版本算法一致）。"""
    from datetime import date

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


# ---------- 并行计算辅助 ----------

_WORKER_CTX: dict[str, Any] = {}


def _init_rule_worker(
    records: list[dict[str, Any]],
    query: str | None,
    weights: dict[str, float] | None,
    evidence_map: dict[str, float] | None,
) -> None:
    """进程池初始化：每个工作进程只载入一次全量记录，避免每个任务重复传参。"""
    _WORKER_CTX["records"] = records
    _WORKER_CTX["query"] = query
    _WORKER_CTX["weights"] = weights
    _WORKER_CTX["evidence_map"] = evidence_map


def _judge_rule_chunk(pairs: list[tuple[int, int]]) -> list[int]:
    """进程池任务：对一组 (i, j) 逐对重新打分并返回胜者（i / j / -1）。"""
    recs = _WORKER_CTX["records"]
    query = _WORKER_CTX["query"]
    weights = _WORKER_CTX["weights"]
    evidence_map = _WORKER_CTX["evidence_map"]
    out: list[int] = []
    for i, j in pairs:
        si = _score_record(recs[i], query=query, weights=weights, evidence_map=evidence_map)
        sj = _score_record(recs[j], query=query, weights=weights, evidence_map=evidence_map)
        out.append(i if si > sj else j if si < sj else -1)
    return out


def _resolve_workers(workers: int | None, num_pairs: int) -> int:
    """workers=None 时自动取 CPU 核数；对局太少时退回串行，避免进程启动开销。"""
    if num_pairs < 16:
        return 1
    if workers is None:
        workers = os.cpu_count() or 1
    return max(1, int(workers))


def _chunk_pairs(
    pairs: list[tuple[int, int]], workers: int
) -> list[list[tuple[int, int]]]:
    """把 (i, j) 对列表均分成 workers 份，每份尽量连续。"""
    k = min(workers, len(pairs)) or 1
    size = (len(pairs) + k - 1) // k
    return [pairs[s : s + size] for s in range(0, len(pairs), size)]


def _has_content(rec: dict[str, Any]) -> bool:
    if rec.get("text") or rec.get("text_summary"):
        return True
    return bool(rec.get("image_summary") or rec.get("image_base_64"))


def pairwise_rank(
    records: Iterable[dict[str, Any]],
    k2: int | None = None,
    *,
    compare: Callable[[dict[str, Any], dict[str, Any]], int] | None = None,
    query: str | None = None,
    weights: dict[str, float] | None = None,
    evidence_map: dict[str, float] | None = None,
    drop_empty: bool = True,
    return_counts: bool = False,
    workers: int | None = None,
) -> list[dict[str, Any]] | list[tuple[dict[str, Any], int]]:
    """两两对比并排序。

    参数：
        records:  k1 条记录
        k2:      非 None 时只返回前 k2 条
        compare: 自定义裁判，compare(a, b) 返回 >0 表示 a 胜，<0 表示 b 胜，0 平局
        query/weights/evidence_map: 透传给内置规则打分，仅默认裁判使用
        drop_empty: True 时过滤 text、text_summary、图片列表全空的记录
        return_counts: True 时返回 [(record, 获胜场次), ...]
        workers: 并行度；None=自动（按 CPU 核数），1=串行

    默认规则裁判也是"每对重新打分再比较"（步骤单元是两两打分+比较），
    整体是 O(n^2) 次打分；对局相互独立，可并行：规则裁判用进程池，
    自定义裁判（如 LLM）用线程池。n 较大时建议先粗筛出 top m 再进入两两，
    控制耗时和内存（进程池会把记录复制到每个工作进程一次）。
    """
    candidates = list(records)
    if drop_empty:
        candidates = [r for r in candidates if _has_content(r)]

    n = len(candidates)
    wins = [0] * n

    if compare is None:
        def decide(i: int, j: int) -> int:
            # 步骤单元：先给 a、b 各打一次分，再比较
            si = _score_record(
                candidates[i], query=query, weights=weights, evidence_map=evidence_map
            )
            sj = _score_record(
                candidates[j], query=query, weights=weights, evidence_map=evidence_map
            )
            if si > sj:
                return i
            if si < sj:
                return j
            return -1

    else:

        def decide(i: int, j: int) -> int:
            result = compare(candidates[i], candidates[j])
            if result > 0:
                return i
            if result < 0:
                return j
            return -1

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    workers = _resolve_workers(workers, len(pairs))

    if workers <= 1 or not pairs:
        winners = [decide(i, j) for i, j in pairs]
    elif compare is None:
        # 规则裁判是 CPU 密集计算：进程池并行，记录经 initializer 只拷贝一次/进程
        chunks = _chunk_pairs(pairs, workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_rule_worker,
            initargs=(candidates, query, weights, evidence_map),
        ) as pool:
            results = list(pool.map(_judge_rule_chunk, chunks))
        winners = [w for chunk in results for w in chunk]
    else:
        # 自定义裁判（如 LLM 网络调用）是 I/O 密集：线程池即可，记录共享内存不重复拷贝
        def judge_thread_chunk(chunk: list[tuple[int, int]]) -> list[int]:
            return [decide(i, j) for i, j in chunk]

        chunks = _chunk_pairs(pairs, workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(judge_thread_chunk, chunks))
        winners = [w for chunk in results for w in chunk]

    for (i, j), winner in zip(pairs, winners):
        if winner >= 0:
            wins[winner] += 1

    # 稳定排序：同分的保持 k1 原有相对顺序
    order = sorted(range(n), key=lambda i: wins[i], reverse=True)
    if k2 is not None:
        order = order[: max(k2, 0)]

    if return_counts:
        return [(candidates[i], wins[i]) for i in order]
    return [candidates[i] for i in order]


if __name__ == "__main__":
    # 简单自测：3 条记录两两对比
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
            "title": "Recent editorial",
            "year": "2025",
            "evidence_level": "editorial",
            "text_summary": "Opinion about diabetes management.",
            "text": "x" * 600,
            "image_summary": None,
            "image_base_64": None,
        },
    ]
    ranked = pairwise_rank(demo, query="metformin diabetes", return_counts=True)
    for rec, pts in ranked:
        print(rec["title"], pts)

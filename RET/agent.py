"""
检索 Agent
提供工具化的检索接口，配合 LLM 连调使用。
支持的工具:
  - search_papers: 混合检索（BM25 + Keyword + 向量多路 + RRF）
  - semantic_search: 纯语义向量检索
  - lexical_search: 纯词汇检索（BM25 + Keyword）
  - get_paper: 获取单篇论文详情
  - filter_results: 按 year/source_type 过滤已有结果
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from retriever import (
    Metadata,
    hybrid_retrieve,
    vector_search,
    bm25_search,
    keyword_search,
    reciprocal_rank_fusion,
    load_metadata_from_dir,
)


# ─── 全局状态（懒加载）─────────────────────────────────────────

_metadata_list: list[Metadata] | None = None
_metadata_map: dict[str, Metadata] | None = None


def _ensure_metadata_loaded():
    global _metadata_list, _metadata_map
    if _metadata_list is None:
        base = os.path.dirname(__file__)
        data_dir = os.path.join(base, "data")
        _metadata_list = load_metadata_from_dir(data_dir)
        _metadata_map = {m.id: m for m in _metadata_list if m.id}
        print(f"📚 已加载 {len(_metadata_list)} 篇论文元数据")


# ─── 格式化工具 ────────────────────────────────────────────────

def _format_result(item: dict, index: int, verbose: bool = False) -> str:
    """格式化单条检索结果"""
    lines = []
    lines.append(f"{'─'*60}")
    lines.append(f"🏷️  Rank {index}  |  Score: {item.get('normalized_score', 0):.4f}")
    lines.append(f"📄  Title: {item.get('title', 'N/A')}")
    lines.append(f"📅  Year: {item.get('year', 'N/A')}  |  Source: {item.get('source_type', 'N/A')}")
    lines.append(f"📝  Summary: {item.get('text_summary', 'N/A')[:200]}...")

    if verbose:
        lines.append(f"🆔  Paper ID: {item.get('paper_id', 'N/A')}")
        sources = item.get("sources", [])
        lines.append(f"🔍  Hit sources: {', '.join(sources)}")
        details = item.get("details", {})
        for src, detail in details.items():
            lines.append(f"    {src}: rank={detail.get('rank')}, score={detail.get('score', 0):.4f}")
            if detail.get("best_document"):
                lines.append(f"         best_chunk: {detail['best_document'][:120]}...")

    return "\n".join(lines)


# ─── Agent 工具函数 ───────────────────────────────────────────

def search_papers(
    query: str,
    keywords: str = "",
    top_k: int = 10,
    use_mmr: bool = False,
    year_filter: int | None = None,
    source_type_filter: str | None = None,
    verbose: bool = False,
) -> str:
    """
    🔍 混合检索工具
    综合 BM25 + 关键词 + 向量语义（title/summary/text/image）多路检索，
    通过 RRF 融合排序。

    参数:
        query: 自然语言查询，如 "低密度脂蛋白胆固醇与冠心病的关系"
        keywords: 可选关键词，空格分隔，如 "LDL-C 冠心病 胆固醇"
        top_k: 返回结果数，默认 10
        use_mmr: 是否启用 MMR 多样性重排
        year_filter: 年份下限过滤，如 2015
        source_type_filter: 来源过滤，如 "JAMA"
        verbose: 是否显示各路命中详情
    """
    _ensure_metadata_loaded()

    if not query.strip():
        return "❌ 查询不能为空"

    results = hybrid_retrieve(
        query=query,
        keywords=keywords,
        top_k=top_k,
        metadata_list=_metadata_list,
        use_mmr=use_mmr,
        year_filter=year_filter,
        source_type_filter=source_type_filter,
    )

    if not results:
        return f"❌ 未找到与 \"{query}\" 相关的结果"

    lines = [
        f"🔎 查询: \"{query}\"",
        f"📊 共找到 {len(results)} 条结果",
        "",
    ]

    for i, item in enumerate(results, start=1):
        lines.append(_format_result(item, i, verbose=verbose))
        lines.append("")

    return "\n".join(lines)


def semantic_search(
    query: str,
    top_k: int = 10,
    verbose: bool = False,
) -> str:
    """
    🧠 纯语义向量检索工具
    使用 BGE-M3 分别检索 title/summary/text/image_desc 四路，
    RRF 融合后返回。

    参数:
        query: 自然语言查询
        top_k: 返回结果数
        verbose: 是否显示详情
    """
    _ensure_metadata_loaded()

    if not query.strip():
        return "❌ 查询不能为空"

    vec_results = vector_search(query)
    ranked_lists = [v for v in vec_results.values() if v]
    fused = reciprocal_rank_fusion(*ranked_lists, top_k=top_k)

    meta_by_id = {m.id: m for m in (_metadata_list or []) if m.id}

    lines = [
        f"🧠 语义检索: \"{query}\"",
        f"📊 共找到 {len(fused)} 条结果",
        "",
    ]

    for i, item in enumerate(fused, start=1):
        pid = item["paper_id"]
        m = meta_by_id.get(pid)
        result = {
            **item,
            "title": m.title if m else "N/A",
            "year": m.year if m else "N/A",
            "source_type": m.source_type if m else "N/A",
            "text_summary": m.text_summary if m else "N/A",
        }
        lines.append(_format_result(result, i, verbose=verbose))
        lines.append("")

    return "\n".join(lines)


def lexical_search(
    query: str,
    keywords: str = "",
    top_k: int = 10,
) -> str:
    """
    📖 纯词汇检索工具
    BM25 + Keyword coverage 两路检索，RRF 融合。

    参数:
        query: 查询文本
        keywords: 可选关键词
        top_k: 返回结果数
    """
    _ensure_metadata_loaded()

    if not query.strip():
        return "❌ 查询不能为空"

    bm25 = bm25_search(_metadata_list, query, keywords, top_k=200)
    kw = keyword_search(_metadata_list, query, keywords, top_k=200)
    fused = reciprocal_rank_fusion(bm25, kw, top_k=top_k)

    meta_by_id = {m.id: m for m in (_metadata_list or []) if m.id}

    lines = [
        f"📖 词汇检索: \"{query}\"",
        f"📊 共找到 {len(fused)} 条结果",
        "",
    ]

    for i, item in enumerate(fused, start=1):
        pid = item["paper_id"]
        m = meta_by_id.get(pid)
        result = {
            **item,
            "title": m.title if m else "N/A",
            "year": m.year if m else "N/A",
            "source_type": m.source_type if m else "N/A",
            "text_summary": m.text_summary if m else "N/A",
        }
        lines.append(_format_result(result, i, verbose=False))
        lines.append("")

    return "\n".join(lines)


def get_paper(paper_id: str) -> str:
    """
    📄 获取单篇论文完整详情（含 image_base_64）

    参数:
        paper_id: 论文 ID，如 "10632286"
    """
    _ensure_metadata_loaded()

    m = _metadata_map.get(paper_id) if _metadata_map else None
    if not m:
        return f"❌ 未找到论文 ID: {paper_id}"

    lines = [
        f"{'='*60}",
        f"📄 {m.title}",
        f"{'='*60}",
        f"🆔  ID: {m.id}",
        f"📅  Year: {m.year}",
        f"📚  Source: {m.source_type}",
        f"📝  Summary: {m.text_summary}",
        f"",
        f"📖  Full Text ({len(m.text or '')} chars):",
        f"    {m.text[:3000] if m.text else 'N/A'}...",
        f"",
        f"🖼️  Image Summaries ({len(m.image_summary or [])} items):",
    ]
    for idx, img in enumerate(m.image_summary or []):
        lines.append(f"    [{idx}] {str(img)[:200]}")

    img_count = len(m.image_base_64 or [])
    lines.append(f"")
    lines.append(f"🖼️  Image Base64: {img_count} 张图片已就绪")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


def filter_by_source_type(source_type: str, top_n: int = 20) -> str:
    """
    📚 浏览指定来源的论文

    参数:
        source_type: 期刊名，如 "JAMA", "Circulation", "The Lancet"
        top_n: 返回数量
    """
    _ensure_metadata_loaded()

    matched = []
    for m in (_metadata_list or []):
        if m.source_type and source_type.lower() in m.source_type.lower():
            matched.append(m)

    if not matched:
        available = sorted(set(m.source_type for m in (_metadata_list or []) if m.source_type))
        return f"❌ 未找到来源 \"{source_type}\"\n📚 可用来源 ({len(available)}):\n" + "\n".join(f"  • {s}" for s in available[:30])

    lines = [
        f"📚 来源 \"{source_type}\" 共有 {len(matched)} 篇论文:",
        "",
    ]
    for i, m in enumerate(matched[:top_n], start=1):
        lines.append(f"{i}. [{m.year}] {m.title}")
        lines.append(f"   ID: {m.id}  |  Summary: {(m.text_summary or '')[:100]}...")
        lines.append("")

    return "\n".join(lines)


# ─── Agent 主函数 ─────────────────────────────────────────────

AGENT_TOOLS = {
    "search_papers": {
        "function": search_papers,
        "description": "混合检索：BM25 + 关键词 + 向量语义多路检索，RRF融合。适合通用查询。",
        "parameters": {
            "query": "自然语言查询",
            "keywords": "可选关键词（空格分隔）",
            "top_k": "返回结果数，默认10",
            "use_mmr": "是否MMR多样性重排",
            "year_filter": "年份下限过滤",
            "source_type_filter": "来源过滤",
            "verbose": "是否显示详情",
        },
    },
    "semantic_search": {
        "function": semantic_search,
        "description": "纯语义向量检索：BGE-M3 跨语言语义匹配。适合概念级查询。",
        "parameters": {
            "query": "自然语言查询",
            "top_k": "返回结果数",
            "verbose": "是否显示详情",
        },
    },
    "lexical_search": {
        "function": lexical_search,
        "description": "纯词汇检索：BM25 + 关键词覆盖。适合精确术语匹配。",
        "parameters": {
            "query": "查询文本",
            "keywords": "可选关键词",
            "top_k": "返回结果数",
        },
    },
    "get_paper": {
        "function": get_paper,
        "description": "获取单篇论文完整详情（含全文和图片摘要）。",
        "parameters": {
            "paper_id": "论文ID，如 10632286",
        },
    },
    "filter_by_source_type": {
        "function": filter_by_source_type,
        "description": "浏览指定期刊/来源的所有论文。",
        "parameters": {
            "source_type": "期刊名，如 JAMA",
            "top_n": "返回数量",
        },
    },
}


def run_agent(query: str, tool: str = "search_papers", **kwargs) -> str:
    """
    Agent 入口：根据 tool 名称调用对应工具。

    参数:
        query: 用户的查询
        tool: 工具名称（search_papers / semantic_search / lexical_search / get_paper / filter_by_source_type）
        **kwargs: 传递给工具的其他参数
    """
    if tool not in AGENT_TOOLS:
        available = ", ".join(AGENT_TOOLS.keys())
        return f"❌ 未知工具: {tool}\n可用工具: {available}"

    if tool == "get_paper":
        return AGENT_TOOLS[tool]["function"](paper_id=query, **kwargs)
    elif tool == "filter_by_source_type":
        return AGENT_TOOLS[tool]["function"](source_type=query, **kwargs)
    else:
        return AGENT_TOOLS[tool]["function"](query=query, **kwargs)


def get_tools_schema() -> list[dict]:
    """导出 OpenAI function calling 格式的工具定义，供 LLM 连调使用"""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_papers",
                "description": "混合检索论文库：综合 BM25关键词 + 向量语义多路检索（标题/摘要/全文/图片描述），RRF融合排序。适合大多数查询场景。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "自然语言查询"},
                        "keywords": {"type": "string", "description": "可选关键词（空格分隔）"},
                        "top_k": {"type": "integer", "description": "返回结果数，默认10"},
                        "use_mmr": {"type": "boolean", "description": "是否启用MMR多样性重排"},
                        "year_filter": {"type": "integer", "description": "年份下限过滤，如2015"},
                        "source_type_filter": {"type": "string", "description": "来源过滤，如JAMA"},
                        "verbose": {"type": "boolean", "description": "是否显示各路命中详情"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": "纯语义向量检索：使用 BGE-M3 多语言模型进行跨语言语义匹配（标题/摘要/全文/图片描述）。适合概念级、跨语言查询。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "自然语言查询"},
                        "top_k": {"type": "integer", "description": "返回结果数，默认10"},
                        "verbose": {"type": "boolean", "description": "是否显示详情"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lexical_search",
                "description": "纯词汇检索：BM25 + 关键词覆盖率匹配。适合精确术语、英文缩写、数字范围等精确匹配场景。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "查询文本"},
                        "keywords": {"type": "string", "description": "可选关键词"},
                        "top_k": {"type": "integer", "description": "返回结果数，默认10"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_paper",
                "description": "获取指定论文的完整信息，包括全文、图片摘要、图片base64数据。通常在检索后需要查看某篇论文详情时调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "论文ID，如10632286"},
                    },
                    "required": ["paper_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filter_by_source_type",
                "description": "列出指定期刊/来源的所有论文。用于了解某个期刊收录了哪些论文。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_type": {"type": "string", "description": "期刊名，如JAMA, Circulation"},
                        "top_n": {"type": "integer", "description": "返回数量，默认20"},
                    },
                    "required": ["source_type"],
                },
            },
        },
    ]


# ─── 交互式测试 ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _ensure_metadata_loaded()

    print("=" * 60)
    print("🔬 生物医学论文检索 Agent")
    print("=" * 60)
    print(f"   已加载 {len(_metadata_list)} 篇论文")
    print(f"   可用工具: {', '.join(AGENT_TOOLS.keys())}")
    print()
    print("  示例查询:")
    print("    search_papers: 低密度脂蛋白胆固醇与冠心病的关系")
    print("    semantic_search: LDL-C lowering therapy effectiveness")
    print("    lexical_search: LDL-C 130 mg/dL")
    print("    get_paper: 10632286")
    print("    filter_by_source_type: JAMA")
    print("=" * 60)

    # 默认跑一个测试
    test_query = "他汀类药物对心血管疾病的影响"
    print(f"\n🔍 测试查询: \"{test_query}\"\n")
    result = run_agent(test_query, tool="search_papers", top_k=5, verbose=True)
    print(result)

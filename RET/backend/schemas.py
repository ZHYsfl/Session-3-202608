"""
请求/响应数据模型
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallStep:
    """单次工具调用记录"""
    round: int                     # 第几轮
    action: str                    # "search_local" | "search_pubmed" | "get_paper"
    query: str                     # 搜索查询
    result_count: int              # 返回结果数
    source: str = ""               # 来源标签
    duration_ms: float = 0.0       # 耗时(毫秒)


@dataclass
class EvidenceCard:
    """证据卡片 — 每个引用对应一张"""
    index: int                    # 引用编号 [1], [2]...
    title: str                    # 论文标题
    year: str                     # 年份
    source_type: str              # 期刊/来源
    pmid: str                     # PMID
    doi: str = ""                 # DOI
    link: str = ""                # 可溯源链接
    similarity_score: float = 0.0 # 相似度
    source: str = "local"         # "local" | "external"
    summary_snippet: str = ""     # 摘要片段


@dataclass
class AgentResponse:
    """Agent 完整响应"""
    answer: str                              # 带 [1][2] 引用的回答正文
    evidence_cards: list[EvidenceCard] = field(default_factory=list)
    query: str = ""
    track: str = ""                          # "clinical" | "nutrition"
    model: str = ""
    tokens_used: int = 0
    tool_calls: list[ToolCallStep] = field(default_factory=list)


@dataclass
class SearchResult:
    """单条检索结果"""
    paper_id: str
    title: str
    year: str
    source_type: str
    text_summary: str
    score: float
    chunk_type: str = ""
    image_summary: list[str] | None = None
    image_base_64: list[str] | None = None
    source: str = "local"  # "local" | "external"


@dataclass
class ChatRequest:
    query: str
    track: str = "clinical"  # "clinical" | "nutrition"

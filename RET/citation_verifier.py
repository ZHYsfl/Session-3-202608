"""
引用验证 + 陈述支撑度检测 (赛道三)
=====================================
三层验证：
  L1: 引用存在性 — [1] 是否映射到真实文献
  L2: 陈述支撑度 — 原子陈述能否从检索chunk中推出 (RAGAS faithfulness朴素版)
  L3: 手动标记 — 圈出可核查事实句，划不出支持句就删/改弱/拒答

核心理念：
  不是"假引用"（编号指向不存在的文献），
  而是"陈述支撑度不足"（编号指向真实文献但内容对不上）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from retriever import Metadata


# ─── L1: 引用存在性校验 ────────────────────────────────────────

@dataclass
class CitationCheck:
    """单条引用校验结果"""
    cite_number: int               # 引用编号
    cite_text: str                 # 原文引用标记，如 "[1]" 或 "[ext2]"
    exists_in_map: bool            # 编号是否在 evidence_map 中
    paper_id: str = ""             # 实际映射的 paper_id
    title: str = ""                # 实际文献标题
    has_real_id: bool = False      # 是否有真实 ID (PMID/DOI)
    has_url: bool = False          # 是否有可访问链接
    url: str = ""


@dataclass
class CitationReport:
    """L1 引用验证报告"""
    total_citations: int = 0
    real_citations: int = 0       # 映射到真实文献的
    hallucinated_citations: int = 0  # 编号不存在于 evidence_map
    orphan_citations: int = 0     # 编号存在但文献无真实ID/URL
    existence_rate: float = 0.0   # 引用存在率 = real / total
    checks: list[CitationCheck] = field(default_factory=list)
    hallucinated_numbers: list[int] = field(default_factory=list)


def build_evidence_map(
    local_results: list[dict],
    external_results: list[dict] | None = None,
) -> dict[int, dict]:
    """
    构建 evidence_map: 引用编号 → 文献元数据。
    local_results: [{paper_id, title, year, source_type, score, ...}, ...]
    external_results: [{paper_id, title, link, ...}, ...]
    """
    evidence_map: dict[int, dict] = {}

    for i, r in enumerate(local_results, start=1):
        evidence_map[i] = {
            "paper_id": r.get("paper_id", ""),
            "title": r.get("title", ""),
            "year": r.get("year", ""),
            "source_type": r.get("source_type", ""),
            "source": "local",
            "has_real_id": bool(r.get("paper_id")),
            "has_url": False,
            "url": "",
        }

    if external_results:
        offset = len(local_results)
        for i, r in enumerate(external_results, start=1):
            pid = r.get("paper_id", "").replace("pubmed_", "")
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/" if pid else ""
            evidence_map[offset + i] = {
                "paper_id": pid,
                "title": r.get("title", ""),
                "year": r.get("year", ""),
                "source_type": r.get("source_type", ""),
                "source": "external",
                "has_real_id": bool(pid),
                "has_url": bool(link),
                "url": link,
            }

    return evidence_map


def parse_citations_from_answer(answer: str) -> list[tuple[int, str]]:
    """
    从 LLM 回答中解析所有引用标记。
    返回 [(编号, 原始文本), ...]，按出现顺序。
    支持 [1], [2], [ext1], [ext2] 格式。
    """
    citations: list[tuple[int, str]] = []
    seen: set[int] = set()

    # 匹配 [数字] 和 [ext数字]
    for m in re.finditer(r"\[(?:ext)?(\d+)\]", answer):
        num = int(m.group(1))
        text = m.group(0)
        if num not in seen:
            citations.append((num, text))
            seen.add(num)

    return citations


def verify_citations(
    answer: str,
    evidence_map: dict[int, dict],
) -> CitationReport:
    """
    L1 引用存在性校验。
    检查 LLM 回答中每个引用编号是否在 evidence_map 中存在。
    """
    cited = parse_citations_from_answer(answer)
    report = CitationReport()
    report.total_citations = len(cited)

    for num, text in cited:
        info = evidence_map.get(num)
        if info is None:
            check = CitationCheck(
                cite_number=num, cite_text=text,
                exists_in_map=False,
            )
            report.hallucinated_citations += 1
            report.hallucinated_numbers.append(num)
        else:
            check = CitationCheck(
                cite_number=num, cite_text=text,
                exists_in_map=True,
                paper_id=info["paper_id"],
                title=info["title"],
                has_real_id=info["has_real_id"],
                has_url=info["has_url"],
                url=info["url"],
            )
            if not info["has_real_id"] and not info["has_url"]:
                report.orphan_citations += 1
            else:
                report.real_citations += 1

        report.checks.append(check)

    report.existence_rate = (
        report.real_citations / report.total_citations
        if report.total_citations > 0 else 0.0
    )

    return report


# ─── L2: 陈述支撑度检测 ────────────────────────────────────────

@dataclass
class AtomicClaim:
    """原子陈述"""
    text: str                      # 陈述文本
    sentence_idx: int              # 所在句子序号
    cited_numbers: list[int] = field(default_factory=list)  # 该句引用的编号
    supported: bool = False        # 能否从检索chunk推出
    supporting_chunk: str = ""     # 支撑该陈述的 chunk 文本
    support_score: float = 0.0     # 支撑度 (0-1)
    action: str = ""               # "keep" | "weaken" | "delete" | "abstain"


@dataclass
class FaithfulnessReport:
    """L2 陈述支撑度报告"""
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    weak_support_claims: int = 0   # 有引用但支撑度弱
    faithfulness_score: float = 0.0  # supported / total
    claims: list[AtomicClaim] = field(default_factory=list)


_CITATION_SENTENCE_RE = re.compile(r"\[(?:ext)?\d+\]")


def _split_sentences(text: str) -> list[str]:
    """按句号、分号、换行拆分句子"""
    raw = re.split(r"[。；;.\n]+", text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 5]


def extract_atomic_claims(
    answer: str,
    retrieved_chunks: list[str],
    evidence_map: dict[int, dict],
) -> list[AtomicClaim]:
    """
    将 LLM 回答拆成原子陈述，每句标注引用的编号。
    不包含LLM judge——用简单的词汇重叠做初步支撑度评估。
    """
    sentences = _split_sentences(answer)
    claims: list[AtomicClaim] = []

    for si, sent in enumerate(sentences):
        # 提取该句中的所有引用编号
        cited = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", sent)]

        # 计算支撑度：该句与所有检索chunk的最大词汇重叠率
        best_score = 0.0
        best_chunk = ""
        sent_tokens = set(re.findall(r"[\w一-鿿]+", sent.lower()))
        if sent_tokens:
            for chunk in retrieved_chunks:
                chunk_tokens = set(re.findall(r"[\w一-鿿]+", chunk.lower()))
                if not chunk_tokens:
                    continue
                overlap = len(sent_tokens & chunk_tokens) / len(sent_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_chunk = chunk[:300]

        is_supported = best_score > 0.15  # 阈值：15% 词汇重叠

        # 确定 action
        if is_supported:
            action = "keep"
        elif cited and best_score > 0.05:
            action = "weaken"  # 有引用但支撑不足
        elif best_score > 0.05:
            action = "weaken"
        else:
            action = "delete"  # 无支撑

        claims.append(AtomicClaim(
            text=sent[:500],
            sentence_idx=si,
            cited_numbers=cited,
            supported=is_supported,
            supporting_chunk=best_chunk,
            support_score=best_score,
            action=action,
        ))

    return claims


def evaluate_faithfulness(
    answer: str,
    retrieved_chunks: list[str],
    evidence_map: dict[int, dict],
) -> FaithfulnessReport:
    """
    L2 陈述支撑度评估。
    返回 FaithfulnessReport，标注每句的 action (keep/weaken/delete/abstain)。
    """
    claims = extract_atomic_claims(answer, retrieved_chunks, evidence_map)
    report = FaithfulnessReport()
    report.total_claims = len(claims)
    report.claims = claims

    for c in claims:
        if c.supported:
            report.supported_claims += 1
        elif c.action == "weaken":
            report.weak_support_claims += 1
        else:
            report.unsupported_claims += 1

    report.faithfulness_score = (
        report.supported_claims / report.total_claims
        if report.total_claims > 0 else 0.0
    )

    return report


# ─── L3: 手动标记辅助 ──────────────────────────────────────────

@dataclass
class ManualVerificationSheet:
    """手动验证工作表 — 课堂手动版"""
    answer: str
    claims: list[AtomicClaim] = field(default_factory=list)
    evidence_map: dict[int, dict] = field(default_factory=dict)


def create_manual_sheet(
    answer: str,
    retrieved_chunks: list[str],
    evidence_map: dict[int, dict],
) -> ManualVerificationSheet:
    """
    生成手动验证工作表。
    课堂用法: 圈出可核查事实句 → 在chunk中找支持句 → 划不出就删/改弱/拒答
    """
    claims = extract_atomic_claims(answer, retrieved_chunks, evidence_map)
    return ManualVerificationSheet(
        answer=answer,
        claims=claims,
        evidence_map=evidence_map,
    )


# ─── 综合校验 ──────────────────────────────────────────────────

@dataclass
class FullVerificationReport:
    """完整验证报告"""
    l1_citation: CitationReport = field(default_factory=CitationReport)
    l2_faithfulness: FaithfulnessReport = field(default_factory=FaithfulnessReport)
    answer: str = ""
    query: str = ""
    track: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "📋 引用验证 + 陈述支撑度报告",
            "=" * 60,
            f"查询: {self.query}",
            f"赛道: {self.track}",
            "",
            "── L1: 引用存在性 ──",
            f"  总引用数: {self.l1_citation.total_citations}",
            f"  真实引用: {self.l1_citation.real_citations}",
            f"  虚构引用: {self.l1_citation.hallucinated_citations}",
            f"  孤儿引用(无ID): {self.l1_citation.orphan_citations}",
            f"  引用存在率: {self.l1_citation.existence_rate:.1%}",
        ]
        if self.l1_citation.hallucinated_numbers:
            lines.append(f"  虚构编号: {self.l1_citation.hallucinated_numbers}")

        lines.extend([
            "",
            "── L2: 陈述支撑度 ──",
            f"  原子陈述数: {self.l2_faithfulness.total_claims}",
            f"  有支撑: {self.l2_faithfulness.supported_claims}",
            f"  弱支撑: {self.l2_faithfulness.weak_support_claims}",
            f"  无支撑: {self.l2_faithfulness.unsupported_claims}",
            f"  忠实度分数: {self.l2_faithfulness.faithfulness_score:.1%}",
            "",
            "── 逐句详情 ──",
        ])

        for c in self.l2_faithfulness.claims:
            action_icon = {"keep": "✅", "weaken": "⚠️", "delete": "❌", "abstain": "🚫"}.get(c.action, "❓")
            cited = f" [{', '.join(str(n) for n in c.cited_numbers)}]" if c.cited_numbers else ""
            lines.append(
                f"  {action_icon} [{c.action}]{cited} "
                f"支撑度={c.support_score:.2f} | {c.text[:120]}..."
            )

        lines.append("=" * 60)
        return "\n".join(lines)


def full_verify(
    answer: str,
    query: str,
    track: str,
    local_results: list[dict],
    external_results: list[dict] | None = None,
    retrieved_chunks: list[str] | None = None,
) -> FullVerificationReport:
    """
    完整验证：L1 引用存在性 + L2 陈述支撑度。
    """
    evidence_map = build_evidence_map(local_results, external_results)
    l1 = verify_citations(answer, evidence_map)

    # 收集所有检索chunk文本
    chunks = retrieved_chunks or []
    if not chunks:
        for r in local_results:
            if r.get("text_summary"):
                chunks.append(r["text_summary"])
            if r.get("text"):
                # 截取前2000字符
                chunks.append(r["text"][:2000])

    l2 = evaluate_faithfulness(answer, chunks, evidence_map)

    return FullVerificationReport(
        l1_citation=l1,
        l2_faithfulness=l2,
        answer=answer,
        query=query,
        track=track,
    )

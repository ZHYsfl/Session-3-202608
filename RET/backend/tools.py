"""
外部检索工具
- PubMed / Europe PMC 在线检索
- 返回可溯源结果（含 link）
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

from backend.schemas import SearchResult

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def search_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """
    搜索 PubMed，返回 PMID 列表。
    免费 API，无需 key，但限速 ~3 req/s。
    """
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    })
    url = f"{PUBMED_SEARCH_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        id_list = data.get("esearchresult", {}).get("idlist", [])
        return [{"pmid": pmid} for pmid in id_list]
    except Exception as e:
        print(f"[PubMed Search Error] {e}", flush=True)
        return []


def fetch_pubmed_details(pmids: list[str]) -> list[dict]:
    """
    获取 PubMed 文章详情（标题、期刊、年份、DOI）。
    """
    if not pmids:
        return []

    # 使用 esummary 批量获取
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    })
    url = f"{PUBMED_SUMMARY_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        results = data.get("result", {})
        articles = []
        for pmid in pmids:
            info = results.get(pmid, {})
            if not info:
                continue
            articles.append({
                "pmid": pmid,
                "title": info.get("title", ""),
                "year": info.get("pubdate", "")[:4] if info.get("pubdate") else "",
                "source_type": info.get("source", ""),
                "doi": _extract_doi(info),
                "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return articles
    except Exception as e:
        print(f"[PubMed Fetch Error] {e}", flush=True)
        return []


def _extract_doi(info: dict) -> str:
    """从 esummary 结果中提取 DOI"""
    article_ids = info.get("articleids", [])
    for aid in article_ids:
        if aid.get("idtype") == "doi":
            return aid.get("value", "")
    # fallback: elocationid
    eloc = info.get("elocationid", "")
    if eloc and eloc.startswith("doi:"):
        return eloc[4:]
    return ""


def external_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    外部检索入口：搜索 PubMed 并返回 SearchResult 列表。
    所有外部结果 source="external"，必须提供 link 用于溯源。
    """
    pmid_list = search_pubmed(query, max_results)
    if not pmid_list:
        return []

    # 限速
    time.sleep(0.5)
    details = fetch_pubmed_details([d["pmid"] for d in pmid_list])

    results = []
    for d in details:
        results.append(SearchResult(
            paper_id=f"pubmed_{d['pmid']}",
            title=d.get("title", ""),
            year=d.get("year", ""),
            source_type=d.get("source_type", ""),
            text_summary="",  # PubMed esummary 不含摘要
            score=0.0,
            source="external",
        ))
    return results


def format_external_results_for_prompt(results: list[SearchResult]) -> str:
    """将外部检索结果格式化为 Agent prompt 中的上下文"""
    if not results:
        return "（未检索到外部文献）"

    lines = ["【外部检索结果 — 可溯源文献】", ""]
    for i, r in enumerate(results, start=1):
        lines.append(f"[ext{i}] {r.title}")
        lines.append(f"    来源: {r.source_type} ({r.year})")
        lines.append(f"    PMID: {r.paper_id.replace('pubmed_', '')}")
        lines.append(f"    链接: https://pubmed.ncbi.nlm.nih.gov/{r.paper_id.replace('pubmed_', '')}/")
        lines.append("")
    return "\n".join(lines)

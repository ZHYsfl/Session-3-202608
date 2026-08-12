#!/usr/bin/env python3
"""
Citation Verification Tool
===========================
输入 PMID 或 DOI，核验引用是否真实存在、是否被撤稿、关键字段是否缺失。

特性:
  - PMID 直查 PubMed E-utilities (efetch XML)
  - DOI 通过 PubMed esearch + Europe PMC 双重验证
  - 撤稿检测 (Retracted Publication / Retraction of Publication)
  - 关键字段完整性检查 (title, authors, journal, year, abstract)
  - 指数退避重试机制
  - 同步/异步双模式 API

依赖:
  pip install aiohttp

环境变量:
  PUBMED_API_KEY        PubMed E-utilities API 密钥 (推荐)
  PUBMED_EMAIL           联系邮箱 (必需)

使用:
  python verify_citation.py --pmid 28770321
  python verify_citation.py --doi "10.1007/s00125-017-4318-z"
  python verify_citation.py --pmid 28770321 --pretty

Python API:
  from verify_citation import verify_citation, verify_citation_sync
  result = await verify_citation(pmid="28770321")
  result = verify_citation_sync(doi="10.1007/s00125-017-4318-z")
  print(result.to_dict())
"""

import asyncio
import os
import sys
import json
import re
import datetime
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import quote
import logging

try:
    import aiohttp
except ImportError:
    print("❌ aiohttp 未安装。请执行: pip install aiohttp", file=sys.stderr)
    sys.exit(1)

# =============================================================================
# 日志
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("verify_citation")
logging.getLogger("aiohttp.client").setLevel(logging.WARNING)


# =============================================================================
# 常量
# =============================================================================
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "").strip()
PUBMED_EMAIL = os.environ.get("PUBMED_EMAIL", "researcher@example.com").strip()

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_TOOL = "citation_verifier"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": "CitationVerifier/1.0 (https://github.com/med-lit-tools)",
    "Accept": "application/json, application/xml, text/xml, */*",
}

# 撤稿相关的出版类型关键词
RETRACTION_KEYWORDS = [
    "retracted publication",
    "retraction of publication",
    "retraction notice",
    "expression of concern",
    "partial retraction",
]

# 关键字段列表（用于缺失检查）
CRITICAL_FIELDS = ["title", "authors", "journal", "year", "abstract"]


# =============================================================================
# HTML 清理
# =============================================================================
_HTML_RE = re.compile(r"<[^>]*>")


def strip_html(text: Optional[str]) -> Optional[str]:
    """去除 HTML 标签"""
    if text is None:
        return None
    cleaned = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") \
                  .replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    cleaned = _HTML_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else None


# =============================================================================
# 数据模型
# =============================================================================
@dataclass
class CitationVerificationResult:
    """引用核验结果"""

    # --- 核心核验结果 ---
    exists: bool = False
    is_retracted: bool = False
    retraction_notice: Optional[str] = None

    # --- 文献元数据 ---
    title: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None

    # --- 质量标记 ---
    missing_fields: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)

    # --- 溯源 ---
    source: Optional[str] = None          # "pubmed" | "europe_pmc"
    verified_at: Optional[str] = None

    def __post_init__(self):
        if self.verified_at is None:
            self.verified_at = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """单行摘要，方便 Agent 快速判断"""
        if not self.exists:
            return "❌ 引用不存在"
        parts = []
        if self.is_retracted:
            parts.append("⚠️ 已撤稿")
        else:
            parts.append("✅ 引用有效")
        if self.title:
            parts.append(f"《{self.title[:60]}{'...' if len(self.title) > 60 else ''}》")
        if self.year:
            parts.append(f"({self.year})")
        if self.journal:
            parts.append(f"[{self.journal}]")
        if self.missing_fields:
            parts.append(f"缺失: {', '.join(self.missing_fields)}")
        if self.alerts:
            parts.append(f"提示: {'; '.join(self.alerts)}")
        return " | ".join(parts)


# =============================================================================
# XML 工具
# =============================================================================
def _safe_xml_text(element, tag: str, default: str = "") -> str:
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return default


# =============================================================================
# 引用核验器
# =============================================================================
class CitationVerifier:
    """引用核验器 —— 通过 PMID/DOI 核验文献是否存在"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._session: Optional[aiohttp.ClientSession] = None

        if verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("详细日志模式已开启")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self._session = aiohttp.ClientSession(
                headers=HEADERS,
                timeout=timeout,
                cookie_jar=aiohttp.DummyCookieJar(),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # =========================================================================
    # 重试机制
    # =========================================================================
    async def _retry_get(self, session: aiohttp.ClientSession, url: str,
                         max_retries: int = MAX_RETRIES, **kwargs):
        """带指数退避重试的 HTTP GET"""
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = await session.get(url, **kwargs)
                if resp.status in (429, 502, 503):
                    await resp.release()
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.debug(f"HTTP {resp.status}，{wait}s 后重试 ({attempt+1}/{max_retries})")
                        await asyncio.sleep(wait)
                        continue
                    return resp
                return resp
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.debug(f"请求失败 ({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                    continue
        raise last_error

    # =========================================================================
    # 主入口
    # =========================================================================
    async def verify(self, pmid: Optional[str] = None,
                     doi: Optional[str] = None) -> CitationVerificationResult:
        """
        核验引用。

        参数:
            pmid: PubMed ID (如 "28770321")
            doi: DOI (如 "10.1007/s00125-017-4318-z")

        至少提供一个；两个都提供时 PMID 优先。

        返回: CitationVerificationResult
        """
        # --- 输入规范化 ---
        pmid = self._normalize_pmid(pmid)
        doi = self._normalize_doi(doi)

        if not pmid and not doi:
            return CitationVerificationResult(
                exists=False,
                alerts=["未提供 PMID 或 DOI"],
            )

        # --- PMID 路径 ---
        if pmid:
            logger.info(f"🔍 核验 PMID: {pmid}")
            return await self._verify_by_pmid(pmid)

        # --- DOI 路径 ---
        if doi:
            logger.info(f"🔍 核验 DOI: {doi}")
            return await self._verify_by_doi(doi)

    # =========================================================================
    # PMID 核验
    # =========================================================================
    async def _verify_by_pmid(self, pmid: str) -> CitationVerificationResult:
        """通过 PubMed EFetch 核验 PMID"""
        session = await self._get_session()

        efetch_url = (
            f"{PUBMED_BASE}/efetch.fcgi"
            f"?db=pubmed&id={pmid}"
            f"&rettype=abstract&retmode=xml"
            f"&tool={PUBMED_TOOL}&email={quote(PUBMED_EMAIL)}"
        )
        if PUBMED_API_KEY:
            efetch_url += f"&api_key={PUBMED_API_KEY}"

        try:
            resp = await self._retry_get(session, efetch_url)
            async with resp:
                if resp.status != 200:
                    logger.error(f"PubMed EFetch HTTP {resp.status}")
                    return CitationVerificationResult(
                        exists=False,
                        alerts=[f"PubMed API 返回 HTTP {resp.status}"],
                    )
                xml_text = await resp.text()
        except Exception as e:
            logger.error(f"PubMed EFetch 请求失败: {e}")
            return CitationVerificationResult(
                exists=False,
                alerts=[f"PubMed API 请求失败: {str(e)}"],
            )

        return self._parse_pubmed_xml(xml_text, query_pmid=pmid)

    def _parse_pubmed_xml(self, xml_text: str,
                          query_pmid: str) -> CitationVerificationResult:
        """解析 PubMed EFetch XML 响应"""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"PubMed XML 解析失败: {e}")
            return CitationVerificationResult(
                exists=False,
                alerts=["PubMed 返回数据无法解析"],
            )

        # 检查是否有结果
        article_elem = root.find(".//PubmedArticle")
        if article_elem is None:
            # 无结果 —— 该 PMID 不存在
            logger.warning(f"⚠️ PMID {query_pmid} 在 PubMed 中不存在")
            return CitationVerificationResult(
                exists=False,
                pmid=query_pmid,
                alerts=[f"PMID {query_pmid} 在 PubMed 中未找到"],
                source="pubmed",
            )

        try:
            return self._extract_pubmed_article(article_elem, query_pmid)
        except Exception as e:
            logger.error(f"解析 PubMed 条目出错: {e}")
            return CitationVerificationResult(
                exists=False,
                pmid=query_pmid,
                alerts=[f"解析 PubMed 数据失败: {str(e)}"],
                source="pubmed",
            )

    def _extract_pubmed_article(self, article_elem,
                                 query_pmid: str) -> CitationVerificationResult:
        """从 PubmedArticle XML 元素提取所有字段"""
        # --- PMID ---
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else query_pmid

        article = article_elem.find(".//Article")
        if article is None:
            return CitationVerificationResult(
                exists=True,
                pmid=pmid,
                alerts=["XML 中缺少 Article 节点"],
                source="pubmed",
            )

        # --- 标题 ---
        title_elem = article.find("ArticleTitle")
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else None

        # --- 作者 ---
        authors = None
        author_list = article.find("AuthorList")
        if author_list is not None:
            author_names = []
            for author in author_list.findall("Author"):
                last = _safe_xml_text(author, "LastName")
                fore = _safe_xml_text(author, "ForeName")
                coll = _safe_xml_text(author, "CollectiveName")
                if last and fore:
                    author_names.append(f"{last} {fore}")
                elif last:
                    author_names.append(last)
                elif coll:
                    author_names.append(coll)
            if author_names:
                authors = "; ".join(author_names)

        # --- 期刊 ---
        journal_elem = article.find(".//Journal")
        journal = None
        if journal_elem is not None:
            journal_title = journal_elem.find("Title")
            if journal_title is not None and journal_title.text:
                journal = journal_title.text.strip()
            else:
                iso = journal_elem.find("ISOAbbreviation")
                if iso is not None and iso.text:
                    journal = iso.text.strip()

        # --- 年份 ---
        year = None
        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                year = year_elem.text.strip()
            else:
                md = pub_date.find("MedlineDate")
                if md is not None and md.text:
                    m = re.search(r"(\d{4})", md.text)
                    if m:
                        year = m.group(1)

        # --- 摘要 ---
        abstract = None
        abstract_elem = article.find("Abstract")
        if abstract_elem is not None:
            parts = []
            for at in abstract_elem.findall("AbstractText"):
                label = at.get("Label", "")
                txt = (at.text or "").strip()
                if label and txt:
                    parts.append(f"{label}: {txt}")
                elif txt:
                    parts.append(txt)
            if parts:
                abstract = "\n".join(parts)

        # --- DOI ---
        doi = None
        id_list = article_elem.find(".//PubmedData/ArticleIdList")
        if id_list is not None:
            for aid in id_list.findall("ArticleId"):
                if aid.get("IdType", "") == "doi" and aid.text:
                    doi = aid.text.strip()

        # --- 出版类型 & 撤稿检测 ---
        pub_types = []
        pt_list = article.find("PublicationTypeList")
        if pt_list is not None:
            for pt in pt_list.findall("PublicationType"):
                if pt.text:
                    pub_types.append(pt.text.strip())

        is_retracted, retraction_notice, retraction_alerts = \
            self._check_retraction(pub_types, title, abstract)

        # --- 缺失字段检查 ---
        missing_fields = self._check_missing_fields(
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            abstract=abstract,
        )

        # --- 汇总 alerts ---
        alerts = []
        alerts.extend(retraction_alerts)
        if missing_fields:
            alerts.append(f"缺失字段: {', '.join(missing_fields)}")
        if not abstract:
            alerts.append("无摘要")

        return CitationVerificationResult(
            exists=True,
            is_retracted=is_retracted,
            retraction_notice=retraction_notice,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            abstract=abstract,
            doi=doi,
            pmid=pmid,
            missing_fields=missing_fields,
            alerts=alerts,
            source="pubmed",
        )

    # =========================================================================
    # DOI 核验
    # =========================================================================
    async def _verify_by_doi(self, doi: str) -> CitationVerificationResult:
        """通过 DOI 核验引用 —— 先用 PubMed 查，失败则用 Europe PMC"""
        # 策略 1: PubMed esearch by DOI
        session = await self._get_session()

        esearch_url = (
            f"{PUBMED_BASE}/esearch.fcgi"
            f"?db=pubmed&term={quote(doi)}[doi]&retmax=1&retmode=json"
            f"&tool={PUBMED_TOOL}&email={quote(PUBMED_EMAIL)}"
        )
        if PUBMED_API_KEY:
            esearch_url += f"&api_key={PUBMED_API_KEY}"

        pmid = None
        try:
            resp = await self._retry_get(session, esearch_url)
            async with resp:
                if resp.status == 200:
                    data = await resp.json()
                    id_list = data.get("esearchresult", {}).get("idlist", [])
                    if id_list:
                        pmid = id_list[0]
        except Exception as e:
            logger.debug(f"PubMed ESearch (DOI) 失败: {e}")

        if pmid:
            logger.info(f"✅ DOI → PMID: {pmid}")
            return await self._verify_by_pmid(pmid)

        # 策略 2: Europe PMC
        logger.info("PubMed 无此 DOI，尝试 Europe PMC...")
        return await self._verify_by_doi_europe_pmc(doi)

    async def _verify_by_doi_europe_pmc(self, doi: str) -> CitationVerificationResult:
        """通过 Europe PMC API 核验 DOI"""
        session = await self._get_session()
        url = (
            f"{EUROPEPMC_BASE}/search"
            f"?query=DOI:{quote(doi)}"
            f"&resultType=core&pageSize=1&format=json"
        )

        try:
            resp = await self._retry_get(session, url)
            async with resp:
                if resp.status != 200:
                    logger.error(f"Europe PMC HTTP {resp.status}")
                    return CitationVerificationResult(
                        exists=False,
                        doi=doi,
                        alerts=[f"Europe PMC API 返回 HTTP {resp.status}"],
                    )
                data = await resp.json()
        except Exception as e:
            logger.error(f"Europe PMC 请求失败: {e}")
            return CitationVerificationResult(
                exists=False,
                doi=doi,
                alerts=[f"Europe PMC API 请求失败: {str(e)}"],
            )

        results = data.get("resultList", {}).get("result", [])
        if not results:
            logger.warning(f"⚠️ DOI {doi} 在 PubMed 和 Europe PMC 中均未找到")
            return CitationVerificationResult(
                exists=False,
                doi=doi,
                alerts=[f"DOI {doi} 在 PubMed 和 Europe PMC 中均未找到"],
                source="europe_pmc",
            )

        return self._parse_europe_pmc_result(results[0], doi)

    def _parse_europe_pmc_result(self, item: dict,
                                  query_doi: str) -> CitationVerificationResult:
        """解析 Europe PMC 搜索结果"""
        title = item.get("title")
        year = str(item.get("pubYear")) if item.get("pubYear") else None
        pmid = item.get("pmid")
        doi = item.get("doi") or query_doi
        abstract = strip_html(item.get("abstractText"))
        journal = item.get("journalTitle") or item.get("bookOrReportDetails", {}).get("publisher")

        # 作者
        authors = item.get("authorString")

        # 出版类型 → 撤稿检测
        pub_type_raw = item.get("pubTypeList", "")
        pub_types = []
        if isinstance(pub_type_raw, str) and pub_type_raw:
            pub_types = [pt.strip() for pt in pub_type_raw.split(",") if pt.strip()]
        elif isinstance(pub_type_raw, dict):
            pt_list = pub_type_raw.get("pubType", [])
            if isinstance(pt_list, list):
                pub_types = [str(pt).strip() for pt in pt_list if pt]
            elif isinstance(pt_list, str):
                pub_types = [pt.strip() for pt in pt_list.split(",") if pt.strip()]

        is_retracted, retraction_notice, retraction_alerts = \
            self._check_retraction(pub_types, title, abstract)

        # --- 缺失字段检查 ---
        missing_fields = self._check_missing_fields(
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            abstract=abstract,
        )

        alerts = []
        alerts.extend(retraction_alerts)
        if missing_fields:
            alerts.append(f"缺失字段: {', '.join(missing_fields)}")
        if not abstract:
            alerts.append("无摘要")

        return CitationVerificationResult(
            exists=True,
            is_retracted=is_retracted,
            retraction_notice=retraction_notice,
            title=title,
            authors=authors,
            journal=journal,
            year=year,
            abstract=abstract,
            doi=doi,
            pmid=pmid,
            missing_fields=missing_fields,
            alerts=alerts,
            source="europe_pmc",
        )

    # =========================================================================
    # 撤稿检测
    # =========================================================================
    @staticmethod
    def _check_retraction(pub_types: list[str], title: Optional[str],
                          abstract: Optional[str]) -> tuple[bool, Optional[str], list[str]]:
        """
        检查出版类型是否包含撤稿标记。

        返回: (is_retracted, retraction_notice, alerts)
        """
        is_retracted = False
        retraction_notice = None
        alerts = []

        types_lower = [t.lower().strip() for t in pub_types]

        for kw in RETRACTION_KEYWORDS:
            for pt in types_lower:
                if kw in pt:
                    is_retracted = True
                    if "retraction of publication" in pt:
                        alerts.append("⚠️ 本文已被撤稿 (Retraction Notice)")
                        retraction_notice = f"撤稿声明 — 原因为: {abstract[:200] if abstract else '未提供具体原因'}"
                    elif "retracted publication" in pt:
                        alerts.append("🚫 本文为被撤稿文章 (Retracted Publication)")
                        retraction_notice = "该文章已被标记为 Retracted Publication"
                    elif "expression of concern" in pt:
                        alerts.append("⚠️ 编辑部已对此文发布关注声明 (Expression of Concern)")
                        retraction_notice = "编辑部已发布关注声明，正在审查本文"
                    elif "partial retraction" in pt:
                        alerts.append("⚠️ 本文部分内容已被撤稿 (Partial Retraction)")
                        retraction_notice = "本文部分内容已被撤稿"
                    break

        return is_retracted, retraction_notice, alerts

    # =========================================================================
    # 缺失字段检查
    # =========================================================================
    @staticmethod
    def _check_missing_fields(title: Optional[str], authors: Optional[str],
                               journal: Optional[str], year: Optional[str],
                               abstract: Optional[str]) -> list[str]:
        """检查关键字段中哪些为空"""
        missing = []
        if not title:
            missing.append("title")
        if not authors:
            missing.append("authors")
        if not journal:
            missing.append("journal")
        if not year:
            missing.append("year")
        if not abstract:
            missing.append("abstract")
        return missing

    # =========================================================================
    # 输入规范化
    # =========================================================================
    @staticmethod
    def _normalize_pmid(pmid: Optional[str]) -> Optional[str]:
        """规范化 PMID：去除空白，校验格式"""
        if pmid is None:
            return None
        pmid = pmid.strip()
        if not pmid:
            return None
        # PMID 应为纯数字
        if not pmid.isdigit():
            logger.warning(f"⚠️ PMID 格式异常 (非纯数字): {pmid}")
        return pmid

    @staticmethod
    def _normalize_doi(doi: Optional[str]) -> Optional[str]:
        """规范化 DOI：去除空白，去除多余前缀"""
        if doi is None:
            return None
        doi = doi.strip()
        if not doi:
            return None
        # 去除 https://doi.org/ 前缀
        doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
        return doi


# =============================================================================
# 顶层 API
# =============================================================================

async def verify_citation(pmid: Optional[str] = None,
                          doi: Optional[str] = None,
                          verbose: bool = False) -> CitationVerificationResult:
    """
    异步核验引用 (顶层便捷函数)。

    使用示例:
        result = await verify_citation(pmid="28770321")
        result = await verify_citation(doi="10.1007/s00125-017-4318-z")
    """
    verifier = CitationVerifier(verbose=verbose)
    try:
        return await verifier.verify(pmid=pmid, doi=doi)
    finally:
        await verifier.close()


def verify_citation_sync(pmid: Optional[str] = None,
                         doi: Optional[str] = None,
                         verbose: bool = False) -> CitationVerificationResult:
    """
    同步核验引用 (封装 asyncio.run)。

    使用示例:
        result = verify_citation_sync(pmid="28770321")
        result = verify_citation_sync(doi="10.1007/s00125-017-4318-z")
    """
    return asyncio.run(verify_citation(pmid=pmid, doi=doi, verbose=verbose))


# =============================================================================
# CLI
# =============================================================================
async def main():
    parser = argparse.ArgumentParser(
        description="引用核验工具 — 通过 PMID/DOI 验证文献是否存在、是否撤稿、字段是否完整",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python verify_citation.py --pmid 28770321
  python verify_citation.py --doi "10.1007/s00125-017-4318-z"
  python verify_citation.py --pmid 28770321 --pretty
  python verify_citation.py --pmid 11888888 --doi "10.xxx/yyy"   # PMID 优先

环境变量:
  PUBMED_API_KEY         PubMed E-utilities API 密钥
  PUBMED_EMAIL           联系邮箱 (默认: researcher@example.com)
        """,
    )
    parser.add_argument("--pmid", type=str, default=None, help="PubMed ID (纯数字)")
    parser.add_argument("--doi", type=str, default=None, help="DOI (如 10.1007/xxx)")
    parser.add_argument("--pretty", "-p", action="store_true", help="格式化 JSON 输出")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细调试日志")

    args = parser.parse_args()

    if not args.pmid and not args.doi:
        parser.error("请至少提供 --pmid 或 --doi 中的一个")

    verifier = CitationVerifier(verbose=args.verbose)
    try:
        result = await verifier.verify(pmid=args.pmid, doi=args.doi)
    finally:
        await verifier.close()

    # 输出
    indent = 2 if args.pretty else None
    output = json.dumps(result.to_dict(), ensure_ascii=False, indent=indent)
    print(output)

    # 在 stderr 输出摘要 (人类可读)
    logger.info(result.summary())


if __name__ == "__main__":
    asyncio.run(main())

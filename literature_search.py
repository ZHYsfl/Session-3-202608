#!/usr/bin/env python3
"""
Medical Literature Search Engine
=================================
并发检索 PubMed、ClinicalTrials.gov、Europe PMC、Semantic Scholar 四大数据源，
返回统一的 metadata 对象列表，用于 RAG 知识库构建。

特性:
  - 四源并发搜索 (asyncio)
  - 尽力获取全文 (Europe PMC fullTextXML)
  - 图片下载 + base64 编码 (PMC 页面爬取 + 多 URL 策略竞速)
  - 图片AI内容概括 (SiliconFlow Qwen3-VL 视觉模型)
  - 证据等级自动推断 (Oxford CEBM)
  - DOI/PMID 智能去重 & 多源数据合并
  - 指数退避重试机制
  - Semantic Scholar TLDR 摘要补充 (免费 API, 无需 Key)

依赖:
  pip install aiohttp

环境变量:
  PUBMED_API_KEY        PubMed E-utilities API 密钥 (推荐)
  PUBMED_EMAIL           联系邮箱 (必需)
  SILICONFLOW_API_KEY   硅基流动 API 密钥 (可选，用于图片AI概括)

使用:
  python literature_search.py "metformin diabetes RCT" --max-results 10
  python literature_search.py "COVID-19 vaccine" -n 5 -o results.json --csv
"""

import asyncio
import os
import sys
import json
import csv
import base64
import re
import io
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
logger = logging.getLogger("lit_search")

# 抑制 aiohttp 的 cookie 解析警告 (NCBI 返回了不合规的 cookie 属性)
logging.getLogger("aiohttp.client").setLevel(logging.WARNING)


# =============================================================================
# 常量
# =============================================================================
# --- 必须从环境变量读取的配置 ---
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "").strip()
PUBMED_EMAIL = os.environ.get("PUBMED_EMAIL", "researcher@example.com").strip()
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "").strip()

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_TOOL = "med_literature_search"

CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2"
EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_FIELDS = (
    "title,year,abstract,tldr,authors,journal,"
    "externalIds,publicationTypes,openAccessPdf,publicationDate"
)

# 硅基流动
SILICONFLOW_BASE = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

MAX_CONCURRENT = 5
MAX_IMAGES_PER_PAPER = 10
MAX_RETRIES = 3                       # 指数退避重试次数
FULL_TEXT_TIMEOUT = 15
IMAGE_DOWNLOAD_TIMEOUT = 10
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": f"MedicalLiteratureSearch/1.1 (https://github.com/med-lit-search)",
    "Accept": "application/json, application/xml, text/xml, */*",
}

# PMC 页面爬取时使用更完整的浏览器头
PAGE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

XLINK_NS = "http://www.w3.org/1999/xlink"

# PMC blob CDN URL 正则 (从 PMC 文章页面提取真实图片链接)
# 注意: 页面中存在于 JS/JSON 的 blob URL 不一定可用，必须从 <img src="..."> 中提取
_PMC_IMG_SRC_RE = re.compile(
    r'<img[^>]+src="(https?://cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/'
    r'(?P<hash1>[a-f0-9]+)/(?P<pmcid>\d+)/'
    r'(?P<hash2>[a-f0-9]+)/(?P<filename>[^"]+))"',
    re.IGNORECASE
)

# 备用：从全文页面中匹配所有 blob URL (兜底用)
_PMC_BLOB_FALLBACK_RE = re.compile(
    r'https?://cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/'
    r'(?P<hash1>[a-f0-9]+)/(?P<pmcid>\d+)/'
    r'(?P<hash2>[a-f0-9]+)/(?P<filename>[^"\s<>]+)'
)


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
# 证据等级推断
# =============================================================================
def infer_evidence_level(publication_types: list[str]) -> Optional[str]:
    """Oxford CEBM 标准: 1A > 1B > 2A > 2B > 3A > 3B > 4 > 5"""
    if not publication_types:
        return None

    types_lower = [t.lower().strip() for t in publication_types]
    types_str = " | ".join(types_lower)

    is_meta = any("meta-analysis" in t or "meta analysis" in t for t in types_lower)
    is_sys_review = any("systematic review" in t for t in types_lower)
    is_rct = any("randomized controlled trial" in t or "randomised controlled trial" in t for t in types_lower)

    if is_meta or (is_sys_review and is_rct):
        return "1A"
    if is_sys_review:
        if any("cohort" in t for t in types_lower):
            return "2A"
        if any("case-control" in t or "case control" in t for t in types_lower):
            return "3A"
        return "1A"

    if is_rct:
        return "1B"

    if any(kw in types_str for kw in ["cohort study", "longitudinal study",
                                        "prospective study", "follow-up study"]):
        return "2B"

    if any("case-control" in t or "case control" in t for t in types_lower):
        return "3B"

    if any(kw in types_str for kw in ["case series", "case report"]):
        return "4"

    if any(kw in types_str for kw in ["review", "editorial", "comment", "opinion",
                                       "guideline", "practice guideline", "letter"]):
        return "5"

    if "clinical trial" in types_str:
        return "2B"

    if any(kw in types_str for kw in ["observational study", "cross-sectional",
                                       "cross sectional", "epidemiologic"]):
        return "3B"

    return None


# =============================================================================
# 数据模型
# =============================================================================
@dataclass
class LiteratureMetadata:
    """统一的文献元数据 —— 存入 RAG 数据集的最小单元"""
    title: Optional[str] = None
    year: Optional[str] = None
    source_type: Optional[str] = None          # pubmed | clinicaltrials | europe_pmc
    id: Optional[str] = None                    # PMID / DOI / NCT ID (原始标识符)
    test_summary: Optional[str] = None          # 摘要 (abstract)
    text: Optional[str] = None                  # 全文
    image_summary: Optional[list[str]] = None   # 每张图片的AI内容概括
    image_base_64: Optional[list[str]] = None   # 每张图片的 base64 编码
    evidence_level: Optional[str] = None        # 证据等级
    last_retrieved_at: Optional[str] = None     # 固定为 None

    # 内部追踪（不输出到 JSON）
    _doi: Optional[str] = field(default=None, repr=False)
    _pmid: Optional[str] = field(default=None, repr=False)
    _pmcid: Optional[str] = field(default=None, repr=False)
    _has_full_text: bool = field(default=False, repr=False)

    def __post_init__(self):
        """规范化字段：空字符串统一为 None，填充检索时间戳"""
        if self.last_retrieved_at is None:
            self.last_retrieved_at = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
        if self.test_summary is not None and not self.test_summary.strip():
            self.test_summary = None
        if self.title is not None and not self.title.strip():
            self.title = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ["_doi", "_pmid", "_pmcid", "_has_full_text"]:
            d.pop(key, None)
        return d

    def to_csv_row(self) -> dict:
        """转为 CSV 友好的扁平字典"""
        d = self.to_dict()
        # 列表字段展平为分隔字符串
        if d.get("image_summary"):
            d["image_summary"] = " |===| ".join(d["image_summary"])
        else:
            d["image_summary"] = ""
        if d.get("image_base_64"):
            d["image_base_64"] = " |===| ".join(d["image_base_64"])
        else:
            d["image_base_64"] = ""
        return d


# =============================================================================
# XML / 文本工具
# =============================================================================
def _safe_xml_text(element, tag: str, default: str = "") -> str:
    child = element.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return default


def _extract_all_text(element) -> str:
    """递归提取 XML 元素中所有文本"""
    if element is None:
        return ""
    parts = []
    if element.text:
        parts.append(element.text.strip())
    for child in element:
        parts.append(_extract_all_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


# =============================================================================
# 文献检索引擎
# =============================================================================
class LiteratureSearchEngine:
    """医学文献检索引擎"""

    def __init__(self, max_results: int = 20, fetch_full_text: bool = True,
                 verbose: bool = False):
        self.max_results = max_results
        self.fetch_full_text = fetch_full_text
        self.verbose = verbose
        self._session: Optional[aiohttp.ClientSession] = None
        self._img_session: Optional[aiohttp.ClientSession] = None
        self._page_session: Optional[aiohttp.ClientSession] = None

        if verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("详细日志模式已开启")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            # 使用 DummyCookieJar 避免无效 cookie 属性产生的警告
            self._session = aiohttp.ClientSession(
                headers=HEADERS,
                timeout=timeout,
                cookie_jar=aiohttp.DummyCookieJar(),
            )
        return self._session

    async def _get_image_session(self) -> aiohttp.ClientSession:
        """图片下载专用 Session（独立连接池，缓存复用）"""
        if self._img_session is None or self._img_session.closed:
            timeout = aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT)
            # 图片下载需要 image/* 的 Accept 头，否则 CDN 会拒绝
            img_headers = {
                "User-Agent": PAGE_HEADERS["User-Agent"],
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            }
            self._img_session = aiohttp.ClientSession(
                headers=img_headers,
                timeout=timeout,
                cookie_jar=aiohttp.DummyCookieJar(),
            )
        return self._img_session

    async def _get_page_session(self) -> aiohttp.ClientSession:
        """PMC 页面爬取专用 Session（使用浏览器 User-Agent）"""
        if self._page_session is None or self._page_session.closed:
            timeout = aiohttp.ClientTimeout(total=FULL_TEXT_TIMEOUT)
            self._page_session = aiohttp.ClientSession(
                headers=PAGE_HEADERS,
                timeout=timeout,
                cookie_jar=aiohttp.DummyCookieJar(),
            )
        return self._page_session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        if self._img_session and not self._img_session.closed:
            await self._img_session.close()
        if hasattr(self, '_page_session') and self._page_session and not self._page_session.closed:
            await self._page_session.close()

    # =========================================================================
    # 重试机制
    # =========================================================================
    async def _retry_get(self, session: aiohttp.ClientSession, url: str,
                         max_retries: int = MAX_RETRIES, **kwargs):
        """
        带指数退避重试的 HTTP GET。
        对 429/502/503 和服务端断开错误进行重试。
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = await session.get(url, **kwargs)
                if resp.status in (429, 502, 503):
                    body = await resp.read()
                    resp.release()
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        logger.debug(f"HTTP {resp.status} 于 {url[:80]}，{wait}s 后重试 ({attempt+1}/{max_retries})")
                        await asyncio.sleep(wait)
                        continue
                    # 最后一次重试也失败，构造一个"伪响应"返回
                    resp.status = resp.status  # 保留状态码
                    return resp
                return resp
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.debug(f"请求失败 ({type(e).__name__})，{wait}s 后重试 ({attempt+1}/{max_retries})")
                    await asyncio.sleep(wait)
                    continue
        raise last_error  # type: ignore[misc]

    # =========================================================================
    # 主搜索入口
    # =========================================================================
    async def search(self, query: str) -> list[dict]:
        """并发搜索所有数据源，返回统一的 metadata 字典列表"""
        logger.info(f'🔍 开始检索: "{query}" (最大: {self.max_results})')

        results_pubmed, results_ct, results_epmc, results_ss = await asyncio.gather(
            self._search_pubmed(query),
            self._search_clinicaltrials(query),
            self._search_europe_pmc(query),
            self._search_semantic_scholar(query),
            return_exceptions=True,
        )

        all_results: list[LiteratureMetadata] = []
        for source_name, results in [
            ("PubMed", results_pubmed),
            ("ClinicalTrials.gov", results_ct),
            ("Europe PMC", results_epmc),
            ("Semantic Scholar", results_ss),
        ]:
            if isinstance(results, Exception):
                logger.error(f"❌ {source_name} 搜索失败: {results}")
            else:
                logger.info(f"✅ {source_name}: {len(results)} 条")
                all_results.extend(results)

        all_results = self._deduplicate(all_results)

        if self.fetch_full_text:
            logger.info("📄 开始获取全文与图片...")
            await self._enrich_full_text(all_results)

        logger.info(f"🏁 检索完成: 共 {len(all_results)} 条唯一结果")
        return [r.to_dict() for r in all_results]

    # =========================================================================
    # PubMed
    # =========================================================================
    async def _search_pubmed(self, query: str) -> list[LiteratureMetadata]:
        if not PUBMED_API_KEY:
            logger.warning("⚠️ PUBMED_API_KEY 环境变量未设置，PubMed 可能受到速率限制")

        session = await self._get_session()

        esearch_url = (
            f"{PUBMED_BASE}/esearch.fcgi"
            f"?db=pubmed&term={quote(query)}"
            f"&retmax={self.max_results}&retmode=json&sort=relevance"
            f"&tool={PUBMED_TOOL}&email={quote(PUBMED_EMAIL)}"
        )
        if PUBMED_API_KEY:
            esearch_url += f"&api_key={PUBMED_API_KEY}"

        try:
            resp = await self._retry_get(session, esearch_url)
            async with resp:
                if resp.status != 200:
                    logger.error(f"PubMed ESearch HTTP {resp.status}")
                    return []
                data = await resp.json()
        except Exception as e:
            logger.error(f"PubMed ESearch 失败: {e}")
            return []

        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        ids_str = ",".join(id_list)
        efetch_url = (
            f"{PUBMED_BASE}/efetch.fcgi"
            f"?db=pubmed&id={ids_str}"
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
                    return []
                xml_text = await resp.text()
        except Exception as e:
            logger.error(f"PubMed EFetch 失败: {e}")
            return []

        return self._parse_pubmed_xml(xml_text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[LiteratureMetadata]:
        results = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"PubMed XML 解析失败: {e}")
            return []

        for article_elem in root.iter("PubmedArticle"):
            try:
                pmid_elem = article_elem.find(".//PMID")
                pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None

                article = article_elem.find(".//Article")
                if article is None:
                    continue

                title_elem = article.find("ArticleTitle")
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else None

                # 年份
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
                if year is None:
                    for ad in article.findall(".//ArticleDate"):
                        if ad.get("DateType") == "Electronic":
                            y = ad.find("Year")
                            if y is not None and y.text:
                                year = y.text.strip()
                                break

                # 摘要
                abstract = article.find("Abstract")
                test_summary = None
                if abstract is not None:
                    parts = []
                    for at in abstract.findall("AbstractText"):
                        label = at.get("Label", "")
                        txt = (at.text or "").strip()
                        parts.append(f"{label}: {txt}" if label else txt)
                    test_summary = "\n".join(parts) if parts else None

                # 出版类型
                pub_types = []
                pt_list = article.find("PublicationTypeList")
                if pt_list is not None:
                    for pt in pt_list.findall("PublicationType"):
                        if pt.text:
                            pub_types.append(pt.text.strip())

                # PMCID / DOI
                pmcid = None
                doi = None
                id_list = article_elem.find(".//PubmedData/ArticleIdList")
                if id_list is not None:
                    for aid in id_list.findall("ArticleId"):
                        id_type = aid.get("IdType", "")
                        if id_type == "pmc" and aid.text:
                            pmcid = aid.text.strip()
                        elif id_type == "doi" and aid.text:
                            doi = aid.text.strip()

                evidence_level = infer_evidence_level(pub_types)

                # ID: 优先使用 PMID，其次 DOI
                doc_id = f"pmid:{pmid}" if pmid else (f"doi:{doi}" if doi else None)

                result = LiteratureMetadata(
                    title=title,
                    year=year,
                    source_type="pubmed",
                    id=doc_id,
                    test_summary=test_summary,
                    text=None,
                    image_summary=None,
                    image_base_64=None,
                    evidence_level=evidence_level,
                    last_retrieved_at=None,
                    _doi=doi,
                    _pmid=pmid,
                    _pmcid=pmcid,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"解析 PubMed 条目出错: {e}")
                continue
        return results

    # =========================================================================
    # ClinicalTrials.gov
    # =========================================================================
    async def _search_clinicaltrials(self, query: str) -> list[LiteratureMetadata]:
        session = await self._get_session()
        url = (
            f"{CLINICALTRIALS_BASE}/studies"
            f"?query.term={quote(query)}"
            f"&pageSize={self.max_results}&format=json"
        )
        try:
            resp = await self._retry_get(session, url)
            async with resp:
                if resp.status != 200:
                    logger.error(f"ClinicalTrials.gov HTTP {resp.status}")
                    return []
                data = await resp.json()
        except Exception as e:
            logger.error(f"ClinicalTrials.gov 请求失败: {e}")
            return []
        return self._parse_clinicaltrials_json(data)

    def _parse_clinicaltrials_json(self, data: dict) -> list[LiteratureMetadata]:
        results = []
        for study in data.get("studies", []):
            try:
                ps = study.get("protocolSection", {})
                ident = ps.get("identificationModule", {})
                nct_id = ident.get("nctId")
                brief_title = ident.get("briefTitle") or ident.get("officialTitle")

                status_mod = ps.get("statusModule", {})
                start_date = status_mod.get("startDateStruct", {})
                year = None
                if start_date:
                    y = start_date.get("year") or start_date.get("date")
                    if y:
                        year = str(y)[:4]

                desc = ps.get("descriptionModule", {})
                brief_summary = desc.get("briefSummary")
                detailed_desc = desc.get("detailedDescription")

                design = ps.get("designModule", {})
                study_type = design.get("studyType", "")
                phases = design.get("phases", [])

                summary_parts = []
                if study_type:
                    summary_parts.append(f"Study Type: {study_type}")
                if phases:
                    summary_parts.append(f"Phases: {', '.join(phases)}")
                if brief_summary:
                    summary_parts.append(brief_summary)
                test_summary = "\n".join(summary_parts) if summary_parts else None

                evidence_level = None
                if "INTERVENTIONAL" in study_type.upper():
                    phases_str = str(phases).upper()
                    if "PHASE3" in phases_str or "PHASE4" in phases_str:
                        evidence_level = "1B"
                    elif "PHASE2" in phases_str:
                        evidence_level = "2B"

                doc_id = f"nct:{nct_id}" if nct_id else None

                result = LiteratureMetadata(
                    title=brief_title,
                    year=year,
                    source_type="clinicaltrials",
                    id=doc_id,
                    test_summary=test_summary,
                    text=detailed_desc,
                    image_summary=None,
                    image_base_64=None,
                    evidence_level=evidence_level,
                    last_retrieved_at=None,
                    _pmid=nct_id,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"解析 ClinicalTrials.gov 条目出错: {e}")
                continue
        return results

    # =========================================================================
    # Europe PMC
    # =========================================================================
    async def _search_europe_pmc(self, query: str) -> list[LiteratureMetadata]:
        session = await self._get_session()
        url = (
            f"{EUROPEPMC_BASE}/search"
            f"?query={quote(query)}"
            f"&resultType=core&pageSize={self.max_results}&format=json"
        )
        try:
            resp = await self._retry_get(session, url)
            async with resp:
                if resp.status != 200:
                    logger.error(f"Europe PMC HTTP {resp.status}")
                    return []
                data = await resp.json()
        except Exception as e:
            logger.error(f"Europe PMC 请求失败: {e}")
            return []
        return self._parse_europe_pmc_json(data)

    def _parse_europe_pmc_json(self, data: dict) -> list[LiteratureMetadata]:
        results = []
        for item in data.get("resultList", {}).get("result", []):
            try:
                title = item.get("title")
                year = item.get("pubYear")
                pmid = item.get("pmid")
                pmcid = item.get("pmcid")
                doi = item.get("doi")
                abstract = strip_html(item.get("abstractText"))
                has_full_text = (item.get("inEPMC", "N") == "Y" or item.get("inPMC", "N") == "Y")

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

                evidence_level = infer_evidence_level(pub_types)

                doc_id = f"pmid:{pmid}" if pmid else (f"doi:{doi}" if doi else None)

                result = LiteratureMetadata(
                    title=title,
                    year=str(year) if year else None,
                    source_type="europe_pmc",
                    id=doc_id,
                    test_summary=abstract,
                    text=None,
                    image_summary=None,
                    image_base_64=None,
                    evidence_level=evidence_level,
                    last_retrieved_at=None,
                    _doi=doi,
                    _pmid=pmid,
                    _pmcid=pmcid,
                    _has_full_text=has_full_text,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"解析 Europe PMC 条目出错: {e}")
                continue
        return results

    # =========================================================================
    # Semantic Scholar
    # =========================================================================
    async def _search_semantic_scholar(self, query: str) -> list[LiteratureMetadata]:
        """Semantic Scholar 论文搜索 (免费 API，无需 Key，但有限速: 100 req/5min)"""
        session = await self._get_session()
        url = (
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
            f"?query={quote(query)}"
            f"&limit={self.max_results}"
            f"&fields={SEMANTIC_SCHOLAR_FIELDS}"
        )
        try:
            resp = await self._retry_get(session, url)
            async with resp:
                if resp.status != 200:
                    logger.error(f"Semantic Scholar HTTP {resp.status}")
                    return []
                data = await resp.json()
        except Exception as e:
            logger.error(f"Semantic Scholar 请求失败: {e}")
            return []
        return self._parse_semantic_scholar_json(data)

    def _parse_semantic_scholar_json(self, data: dict) -> list[LiteratureMetadata]:
        results = []
        for paper in data.get("data", []):
            try:
                title = paper.get("title")
                if not title:
                    continue

                external_ids = paper.get("externalIds") or {}
                doi = external_ids.get("DOI")
                pmid = external_ids.get("PubMed")
                paper_id = paper.get("paperId")

                # 年份
                year = paper.get("year")
                year = str(year) if year else None

                # 摘要: 优先 abstract，其次 TLDR
                abstract = strip_html(paper.get("abstract"))
                tldr = paper.get("tldr") or {}
                tldr_text = strip_html(tldr.get("text"))

                # 优先完整 abstract，如果 abstract 为空用 TLDR
                if abstract:
                    test_summary = abstract
                elif tldr_text:
                    test_summary = tldr_text
                else:
                    test_summary = None

                # 出版类型 → 证据等级
                pub_types = paper.get("publicationTypes", []) or []
                evidence_level = infer_evidence_level(pub_types)

                # ID: 优先 PMID > DOI > Semantic Scholar paperId
                if pmid:
                    doc_id = f"pmid:{pmid}"
                elif doi:
                    doc_id = f"doi:{doi}"
                elif paper_id:
                    doc_id = f"ss:{paper_id}"
                else:
                    doc_id = None

                result = LiteratureMetadata(
                    title=title,
                    year=year,
                    source_type="semantic_scholar",
                    id=doc_id,
                    test_summary=test_summary,
                    text=None,
                    image_summary=None,
                    image_base_64=None,
                    evidence_level=evidence_level,
                    last_retrieved_at=None,
                    _doi=doi,
                    _pmid=pmid,
                    _pmcid=None,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"解析 Semantic Scholar 条目出错: {e}")
                continue
        return results

    # =========================================================================
    # 全文获取 & 图片提取
    # =========================================================================
    async def _enrich_full_text(self, results: list[LiteratureMetadata]) -> None:
        """并发为所有有 PMCID 的结果补充全文和图片"""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def enrich_one(result: LiteratureMetadata):
            # 统一条件：只要有 PMCID 就尝试获取全文
            if not result._pmcid:
                return
            async with semaphore:
                try:
                    text, img_summaries, img_b64s = await self._fetch_full_text(result._pmcid)
                    if text:
                        result.text = text
                    if img_summaries:
                        result.image_summary = img_summaries
                    if img_b64s:
                        result.image_base_64 = img_b64s
                except Exception as e:
                    logger.debug(f"全文获取失败 (PMCID={result._pmcid}): {e}")

        await asyncio.gather(*[enrich_one(r) for r in results], return_exceptions=True)

        with_text = sum(1 for r in results if r.text)
        with_images = sum(1 for r in results if r.image_base_64)
        logger.info(f"📄 全文: {with_text}/{len(results)} | 🖼️ 图片: {with_images}/{len(results)}")

    async def _fetch_full_text(
        self, pmcid: str
    ) -> tuple[Optional[str], Optional[list[str]], Optional[list[str]]]:
        """
        从 Europe PMC 获取全文 XML → 提取正文 + 下载图片 + AI概括。
        返回: (body_text, image_summaries, image_base64s)
        """
        if not pmcid:
            return None, None, None

        pmcid_clean = pmcid.replace("PMC", "").strip()
        pmcid_formatted = f"PMC{pmcid_clean}"

        session = await self._get_session()
        url_patterns = [
            f"{EUROPEPMC_BASE}/{pmcid_formatted}/fullTextXML",
            f"{EUROPEPMC_BASE}/PMC/{pmcid_formatted}/fullTextXML",
        ]

        xml_text = None
        for url in url_patterns:
            try:
                timeout = aiohttp.ClientTimeout(total=FULL_TEXT_TIMEOUT)
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        if text and len(text) > 100:
                            xml_text = text
                            break
            except (asyncio.TimeoutError, aiohttp.ClientError, Exception):
                continue

        if not xml_text:
            return None, None, None

        # 改进的 BOM 处理
        xml_text = xml_text.lstrip("﻿")

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return None, None, None

        # --- 提取正文 ---
        body_parts = []
        for body in root.iter("body"):
            for p in body.iter("p"):
                t = _extract_all_text(p)
                if t:
                    body_parts.append(t)

        for abstract in root.iter("abstract"):
            for p in abstract.iter("p"):
                t = _extract_all_text(p)
                if t:
                    body_parts.append(t)

        body_text = "\n\n".join(body_parts) if body_parts else None
        if body_text and len(body_text) > 100000:
            body_text = body_text[:100000] + "\n\n[... 文本已截断 ...]"

        # --- 提取图片信息 ---
        figure_entries: list[dict] = []

        for fig in root.iter("fig"):
            if len(figure_entries) >= MAX_IMAGES_PER_PAPER:
                break

            caption_elem = fig.find("caption")
            caption_text = ""
            if caption_elem is not None:
                caption_text = _extract_all_text(caption_elem)
            label = _safe_xml_text(fig, "label", "")
            if label and caption_text:
                caption_text = f"{label} {caption_text}"
            elif label:
                caption_text = label

            hrefs = []
            for graphic in fig.iter("graphic"):
                href = (
                    graphic.get(f"{{{XLINK_NS}}}href")
                    or graphic.get("href")
                    or graphic.get("xlink:href")
                )
                if href:
                    hrefs.append(href)

            if hrefs:
                figure_entries.append({"caption": caption_text, "hrefs": hrefs})

        if not figure_entries:
            return body_text, None, None

        logger.debug(f"🖼️ {pmcid_formatted}: 发现 {len(figure_entries)} 张图片")

        # --- 收集所有图片文件名 ---
        all_hrefs: set[str] = set()
        for fe in figure_entries:
            for href in fe["hrefs"]:
                fn = href.rsplit("/", 1)[-1] if "/" in href else href
                all_hrefs.add(fn)

        # --- 从 PMC 页面批量下载 blob 图片 (用主 Session, 更快更可靠) ---
        blob_images = await self._fetch_pmc_blob_images(pmcid_formatted, all_hrefs)

        # --- 对 Blob 未覆盖的图片，回退到传统 URL 模式 ---
        img_session = await self._get_image_session()

        async def download_figure(fig_data: dict) -> Optional[dict]:
            for href in fig_data["hrefs"]:
                fn = href.rsplit("/", 1)[-1] if "/" in href else href
                # 优先使用已下载的 blob 图片
                if fn in blob_images:
                    img = blob_images[fn]
                    return {
                        "base64": img["base64"],
                        "media_type": img["media_type"],
                        "caption": fig_data["caption"],
                    }
                # 回退到传统 URL 逐一尝试
                result = await self._download_single_image(
                    img_session, href, pmcid_formatted, blob_url_map=None
                )
                if result:
                    return {
                        "base64": result["base64"],
                        "media_type": result["media_type"],
                        "caption": fig_data["caption"],
                    }
            return None

        downloaded = list(filter(None, await asyncio.gather(
            *[download_figure(f) for f in figure_entries]
        )))

        logger.debug(f"🖼️ {pmcid_formatted}: 下载成功 {len(downloaded)}/{len(figure_entries)}")

        if not downloaded:
            return body_text, None, None

        # --- 并行调用视觉模型生成图片概括 ---
        async def summarize_image(img_data: dict) -> str:
            return await self._vision_describe_image(
                base64_data=img_data["base64"],
                media_type=img_data["media_type"],
                caption_hint=img_data["caption"],
            )

        image_summaries = await asyncio.gather(*[summarize_image(d) for d in downloaded])
        image_base64_list = [d["base64"] for d in downloaded]

        return body_text, image_summaries, image_base64_list

    # =========================================================================
    # PMC 页面爬取 - 直接下载 Blob 图片数据
    # =========================================================================
    async def _fetch_pmc_blob_images(
        self, pmcid: str, filenames: set[str]
    ) -> dict[str, dict]:
        """
        爬取 PMC 文章页面，提取 CDN blob 图片 URL 并直接下载。
        返回: {filename: {base64, media_type}} 映射
        """
        if not filenames:
            return {}

        pmcid_clean = pmcid.replace("PMC", "").strip()
        page_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid_clean}/"
        session = await self._get_session()

        # 1) 获取 PMC 页面 HTML
        try:
            timeout = aiohttp.ClientTimeout(total=FULL_TEXT_TIMEOUT)
            async with session.get(page_url, headers=PAGE_HEADERS, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.debug(f"PMC 页面获取失败 HTTP {resp.status}")
                    return {}
                html = await resp.text()
        except Exception as e:
            logger.debug(f"PMC 页面爬取失败: {e}")
            return {}

        # 2) 从 <img src="..."> 提取 blob URL → filename 映射
        blob_url_map: dict[str, str] = {}
        for m in _PMC_IMG_SRC_RE.finditer(html):
            fn = m.group("filename")
            blob_url_map[fn] = m.group(1)  # 完整 URL

        if not blob_url_map:
            logger.debug(f"📄 {pmcid}: 页面无 blob 图片 URL")
            return {}

        logger.debug(f"📄 {pmcid}: 从页面提取了 {len(blob_url_map)} 个 blob URL")

        # 3) 并发下载匹配的图片 (用主 Session, 30s 超时)
        results: dict[str, dict] = {}

        async def download_one(filename: str) -> Optional[tuple[str, dict]]:
            # 精确匹配或模糊匹配 blob URL
            blob_url = None
            if filename in blob_url_map:
                blob_url = blob_url_map[filename]
            else:
                for bfn, burl in blob_url_map.items():
                    if filename in bfn or bfn in filename:
                        blob_url = burl
                        break

            if not blob_url:
                return None

            try:
                t = aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT)
                async with session.get(blob_url, timeout=t) as resp:
                    if resp.status == 200:
                        ct = resp.headers.get("Content-Type", "")
                        if any(x in ct for x in ["image/", "application/octet-stream"]):
                            data = await resp.read()
                            if len(data) > 100 and self._is_valid_image(data):
                                mt = self._detect_media_type(filename, ct)
                                b64 = base64.b64encode(data).decode("ascii")
                                return (filename, {"base64": b64, "media_type": mt})
            except Exception:
                pass
            return None

        tasks = [download_one(fn) for fn in filenames]
        completed = await asyncio.gather(*tasks)

        for item in completed:
            if item:
                fn, img_data = item
                results[fn] = img_data

        if results:
            logger.debug(f"📄 {pmcid}: blob 下载成功 {len(results)}/{len(filenames)}")
        return results

    # =========================================================================
    # 图片下载
    # =========================================================================
    async def _download_single_image(
        self,
        session: aiohttp.ClientSession,
        href: str,
        pmcid: str,
        blob_url_map: dict[str, str] | None = None,
    ) -> Optional[dict]:
        """
        多 URL 策略并行竞速下载单张图片。
        blob_url_map: 从 PMC 页面提取的文件名→blob URL 映射
        返回: {base64, media_type} 或 None
        """
        pmcid_num = pmcid.replace("PMC", "").strip()
        filename = href.rsplit("/", 1)[-1] if "/" in href else href

        candidates = []

        # 0) Blob CDN URL (如果调用方提供了预获取的映射)
        if blob_url_map:
            if filename in blob_url_map:
                candidates.append(blob_url_map[filename])
            else:
                for fname, burl in blob_url_map.items():
                    if filename in fname or fname in filename:
                        candidates.append(burl)
                        break

        # 1) 绝对 URL (出版商 CDN)
        if href.startswith("http"):
            candidates.append(href)

        # 2) PMC bin (旧版 URL 模式，作为回退)
        candidates.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid_num}/bin/{filename}")

        # 3) Europe PMC 图片 API (旧版，作为回退)
        candidates.append(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextImage"
            f"?imageId={filename}"
        )

        # 4) Frontiers 出版商 CDN 智能推断
        fcdn = self._try_frontiers_cdn(filename)
        if fcdn:
            candidates.append(fcdn)

        # 去重
        seen = set()
        unique_candidates = []
        for u in candidates:
            if u not in seen:
                seen.add(u)
                unique_candidates.append(u)

        timeout = aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT)

        async def try_url(url: str) -> Optional[dict]:
            try:
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        ct = resp.headers.get("Content-Type", "")
                        if any(t in ct for t in ["image/", "application/octet-stream"]):
                            data = await resp.read()
                            if len(data) > 100 and self._is_valid_image(data):
                                mt = self._detect_media_type(filename, ct)
                                return {
                                    "base64": base64.b64encode(data).decode("ascii"),
                                    "media_type": mt,
                                }
            except asyncio.TimeoutError:
                pass
            except aiohttp.ClientError:
                pass
            except Exception:
                pass
            return None

        tasks = [try_url(u) for u in unique_candidates]
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                if result:
                    for t in tasks:
                        t.cancel()
                    return result
            except Exception:
                continue

        logger.debug(f"图片下载全部失败: {filename}")
        return None

    @staticmethod
    def _is_valid_image(data: bytes) -> bool:
        """验证二进制数据是否为真实图片 (通过魔数检查)"""
        if len(data) < 8:
            return False
        # JPEG: FF D8 FF
        if data[:3] == b'\xff\xd8\xff':
            return True
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return True
        # GIF: GIF89a or GIF87a
        if data[:6] in (b'GIF89a', b'GIF87a'):
            return True
        # WebP: RIFF....WEBP
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return True
        # BMP: BM
        if data[:2] == b'BM':
            return True
        return False

    @staticmethod
    def _try_frontiers_cdn(filename: str) -> Optional[str]:
        """
        Frontiers 出版商 CDN URL 构建。
        文件名格式: {journal}-{vol}-{article_id}-{fig_id}.{ext}
        CDN: frontiersin.org/files/Articles/{article_id}/{namebase}-HTML/image_m/{filename}
        """
        m = re.match(r"^([a-zA-Z]+)-(\d+)-(\d+)-([^./\\]+)\.(\w+)$", filename)
        if not m:
            return None
        journal, vol, article_id = m.group(1), m.group(2), m.group(3)
        namebase = f"{journal}-{vol}-{article_id}"
        return (
            f"https://www.frontiersin.org/files/Articles/{article_id}/"
            f"{namebase}-HTML/image_m/{filename}"
        )

    @staticmethod
    def _detect_media_type(filename: str, content_type: str) -> str:
        """根据文件名扩展名或 Content-Type 确定 MIME 类型"""
        if content_type and content_type.startswith("image/"):
            return content_type.split(";")[0].strip()
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mapping = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "bmp": "image/bmp",
            "tiff": "image/tiff", "tif": "image/tiff",
            "svg": "image/svg+xml",
        }
        return mapping.get(ext, "image/jpeg")

    # =========================================================================
    # 视觉模型 - 硅基流动 (SiliconFlow Qwen2.5-VL)
    # =========================================================================
    async def _vision_describe_image(
        self, base64_data: str, media_type: str, caption_hint: str
    ) -> str:
        """
        调用硅基流动 Qwen2.5-VL 视觉模型对图片内容做简要概括。
        失败时回退到 XML caption。
        """
        if not SILICONFLOW_API_KEY:
            logger.debug("SILICONFLOW_API_KEY 未设置，跳过 AI 图片概括")
            if caption_hint:
                return f"[Caption] {caption_hint}"
            return "[无法获取图片描述 - API Key 未配置]"

        prompt = (
            "你是一位医学科研助手。请用2-4句简洁的中文描述这张来自医学研究论文的图片内容。"
            "重点说明: (1)图片类型(图表/流程图/影像等) (2)展示的关键变量或数据 "
            "(3)主要发现或趋势。直接描述内容，不要使用'这张图片展示了'等引导语。"
        )

        try:
            session = await self._get_session()
            headers = {
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": SILICONFLOW_VISION_MODEL,
                "max_tokens": 300,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_data}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            }

            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(
                SILICONFLOW_BASE, json=payload, headers=headers, timeout=timeout
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        if content:
                            return content
                else:
                    logger.debug(f"SiliconFlow Vision HTTP {resp.status}")
        except Exception as e:
            logger.debug(f"SiliconFlow Vision 调用失败: {e}")

        # 回退到 caption
        if caption_hint:
            return f"[Caption] {caption_hint}"
        return "[无法获取图片描述]"

    # =========================================================================
    # 去重
    # =========================================================================
    def _deduplicate(self, results: list[LiteratureMetadata]) -> list[LiteratureMetadata]:
        """基于 DOI 和 PMID 去重，智能合并多源数据以最大化字段完整度"""
        seen_doi: dict[str, int] = {}
        seen_pmid: dict[str, int] = {}
        deduped: list[LiteratureMetadata] = []

        for r in results:
            doi = r._doi
            pmid = r._pmid

            existing_idx: Optional[int] = None
            if doi and doi in seen_doi:
                existing_idx = seen_doi[doi]
            elif pmid and pmid in seen_pmid:
                existing_idx = seen_pmid[pmid]

            if existing_idx is not None:
                # 合并数据：用新结果补全已有记录的缺失字段
                self._merge_record(deduped[existing_idx], r)
                # 如果新条目有 DOI 而旧条目没有，更新 DOI 索引
                if doi and doi not in seen_doi:
                    seen_doi[doi] = existing_idx
                if pmid and pmid not in seen_pmid:
                    seen_pmid[pmid] = existing_idx
                continue

            idx = len(deduped)
            if doi:
                seen_doi[doi] = idx
            if pmid:
                seen_pmid[pmid] = idx
            deduped.append(r)

        if len(results) != len(deduped):
            logger.info(f"🔄 去重+合并: {len(results)} → {len(deduped)}")

        return deduped

    @staticmethod
    def _merge_record(target: LiteratureMetadata, source: LiteratureMetadata) -> None:
        """将 source 中的非空字段补充到 target，不覆盖已有数据"""
        # 摘要：优先保留更长的 (更完整)
        if not target.test_summary and source.test_summary:
            target.test_summary = source.test_summary
        elif target.test_summary and source.test_summary:
            if len(source.test_summary) > len(target.test_summary) * 1.5:
                target.test_summary = source.test_summary
        # 全文
        if not target.text and source.text:
            target.text = source.text
        # 证据等级：从无到有
        if not target.evidence_level and source.evidence_level:
            target.evidence_level = source.evidence_level
        # PMCID (关键：用于后续全文获取！)
        if not target._pmcid and source._pmcid:
            target._pmcid = source._pmcid
        if not target._has_full_text and source._has_full_text:
            target._has_full_text = source._has_full_text


# =============================================================================
# CLI
# =============================================================================
def write_csv(results: list[dict], filepath: str) -> None:
    """将结果输出为 CSV 文件"""
    if not results:
        logger.warning("无结果可输出为 CSV")
        return

    fieldnames = [
        "title", "year", "source_type", "id", "test_summary",
        "text", "image_summary", "image_base_64", "evidence_level",
        "last_retrieved_at",
    ]
    # 限制 base64 列的长度避免 CSV 过大
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            # CSV 中截断过长的 base64，保留完整数据在 JSON
            if row.get("image_base_64"):
                row["image_base_64"] = f"[{len(row['image_base_64'])} images, base64 data truncated in CSV]"
            if row.get("image_summary"):
                if isinstance(row["image_summary"], list):
                    row["image_summary"] = " | ".join(row["image_summary"])
            if row.get("text") and len(row.get("text", "")) > 5000:
                row["text"] = row["text"][:5000] + "... [truncated in CSV]"
            writer.writerow(row)

    logger.info(f"📊 CSV 已保存到: {filepath}")


async def main():
    parser = argparse.ArgumentParser(
        description="医学文献检索引擎 — 并发搜索 PubMed / ClinicalTrials.gov / Europe PMC / Semantic Scholar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python literature_search.py "metformin diabetes RCT" -n 10
  python literature_search.py "COVID-19 vaccine" --no-full-text -o results.json
  python literature_search.py "cancer immunotherapy" -o results.json --csv results.csv --verbose

环境变量:
  PUBMED_API_KEY         PubMed E-utilities API 密钥
  PUBMED_EMAIL           联系邮箱 (默认: researcher@example.com)
  SILICONFLOW_API_KEY    硅基流动 API 密钥 (用于图片AI概括，可选)
        """,
    )
    parser.add_argument("query", type=str, help="搜索查询字符串")
    parser.add_argument("--max-results", "-n", type=int, default=20)
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 JSON 文件")
    parser.add_argument("--csv", type=str, default=None, help="同时输出 CSV 文件路径")
    parser.add_argument("--no-full-text", action="store_true", help="跳过全文/图片获取")
    parser.add_argument("--pretty", "-p", action="store_true", help="格式化 JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细调试日志")

    args = parser.parse_args()

    if not args.query.strip():
        parser.error("查询字符串不能为空")

    if args.verbose and not logger.handlers:
        pass  # level already set in LiteratureSearchEngine.__init__

    engine = LiteratureSearchEngine(
        max_results=args.max_results,
        fetch_full_text=not args.no_full_text,
        verbose=args.verbose,
    )

    try:
        results = await engine.search(args.query)
    finally:
        await engine.close()

    indent = 2 if args.pretty else None
    json_output = json.dumps(results, ensure_ascii=False, indent=indent)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        logger.info(f"💾 JSON 结果已保存到: {args.output}")
    else:
        print(json_output)

    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    asyncio.run(main())

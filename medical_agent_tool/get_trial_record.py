#!/usr/bin/env python3
"""
Clinical Trial Record Lookup Tool
==================================
输入 NCT ID，回查 ClinicalTrials.gov 注册试验，核对是否存在、获取试验状态/分期/干预措施。

特性:
  - NCT ID 直查 ClinicalTrials.gov API v2 单研究端点
  - 自动补全 NCT 前缀 (支持 "01234567" 简写)
  - 返回试验状态、分期、研究类型、干预措施、条件、申办方等
  - 判断是否已有试验结果发布
  - 指数退避重试机制
  - 同步/异步双模式 API

依赖:
  pip install aiohttp

使用:
  python get_trial_record.py NCT04280705
  python get_trial_record.py 04280705              # 自动补 NCT 前缀
  python get_trial_record.py NCT04280705 --pretty

Python API:
  from get_trial_record import get_trial_record, get_trial_record_sync
  result = await get_trial_record("NCT04280705")
  result = get_trial_record_sync("04280705")
  print(result.to_dict())
"""

import asyncio
import os
import sys
import json
import re
import datetime
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional
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
logger = logging.getLogger("trial_record")
logging.getLogger("aiohttp.client").setLevel(logging.WARNING)


# =============================================================================
# 常量
# =============================================================================
CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": "TrialRecordLookup/1.0 (https://github.com/med-lit-tools)",
    "Accept": "application/json",
}

# NCT ID 正则 (支持 NCT01234567 或 01234567)
_NCT_ID_RE = re.compile(r'^(?:NCT)?(\d{8})$', re.IGNORECASE)

# 试验状态分类 (方便 Agent 做判断)
ACTIVE_STATUSES = {
    "NOT_YET_RECRUITING", "RECRUITING", "ENROLLING_BY_INVITATION",
    "ACTIVE_NOT_RECRUITING", "SUSPENDED",
}
COMPLETED_STATUSES = {"COMPLETED", "TERMINATED"}
WITHDRAWN_STATUSES = {"WITHDRAWN", "UNKNOWN"}


# =============================================================================
# 数据模型
# =============================================================================
@dataclass
class TrialRecordResult:
    """临床试验回查结果"""

    # --- 核心核验结果 ---
    exists: bool = False
    nct_id: str = ""

    # --- 试验基本信息 ---
    title: Optional[str] = None              # briefTitle / officialTitle
    status: Optional[str] = None             # 试验状态 (Overall Status)
    phase: Optional[str] = None              # 试验分期 (Phase 1/2/3/4)
    study_type: Optional[str] = None         # Interventional / Observational

    # --- 干预 & 条件 ---
    interventions: list[str] = field(default_factory=list)  # 干预措施名称
    conditions: list[str] = field(default_factory=list)     # 研究疾病/条件

    # --- 组织信息 ---
    sponsor: Optional[str] = None            # 主要申办方

    # --- 时间信息 ---
    start_date: Optional[str] = None         # 开始日期 (YYYY-MM-DD)
    completion_date: Optional[str] = None    # 预计/实际完成日期
    last_update_posted: Optional[str] = None # 最后更新日期

    # --- 规模 & 结果 ---
    enrollment: Optional[int] = None         # 预计招募人数
    has_results: bool = False                # 是否已发布结果

    # --- 链接 & 溯源 ---
    url: Optional[str] = None                # ClinicalTrials.gov 页面
    verified_at: Optional[str] = None        # 核验时间戳

    def __post_init__(self):
        if self.verified_at is None:
            self.verified_at = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

    @property
    def is_active(self) -> bool:
        """是否仍在进行中"""
        return self.status in ACTIVE_STATUSES if self.status else False

    @property
    def is_completed(self) -> bool:
        """是否已完成/终止"""
        return self.status in COMPLETED_STATUSES if self.status else False

    def to_dict(self) -> dict:
        d = asdict(self)
        # 移除 @property 产生的字段
        d.pop("is_active", None)
        d.pop("is_completed", None)
        return d

    def summary(self) -> str:
        """单行摘要，方便 Agent 快速判断"""
        if not self.exists:
            return f"❌ 试验 {self.nct_id} 未注册"
        parts = []
        parts.append(f"✅ {self.nct_id}")
        if self.status:
            parts.append(f"状态: {self.status.replace('_', ' ').title()}")
        if self.phase:
            parts.append(f"分期: {self.phase}")
        if self.study_type:
            parts.append(f"类型: {self.study_type}")
        if self.interventions:
            names = ", ".join(self.interventions[:3])
            if len(self.interventions) > 3:
                names += f" (+{len(self.interventions) - 3})"
            parts.append(f"干预: {names}")
        if self.has_results:
            parts.append("📊 已有结果")
        if self.title:
            short_title = self.title[:80] + "..." if len(self.title) > 80 else self.title
            parts.append(f"《{short_title}》")
        return " | ".join(parts)


# =============================================================================
# 试验回查器
# =============================================================================
class TrialRecordLookup:
    """临床试验回查器 —— 通过 NCT ID 查询注册试验"""

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
    async def lookup(self, nct_id: str) -> TrialRecordResult:
        """
        回查临床试验。

        参数:
            nct_id: NCT 编号，支持 "NCT04280705" 或 "04280705"

        返回: TrialRecordResult
        """
        # --- 输入规范化 ---
        normalized = self._normalize_nct_id(nct_id)
        if not normalized:
            return TrialRecordResult(
                exists=False,
                nct_id=nct_id.strip() if nct_id else "",
                status="NCT ID 格式无效 (需为 8 位数字 + 可选 NCT 前缀)",
            )

        logger.info(f"🔍 回查试验: {normalized}")
        return await self._fetch_study(normalized)

    # =========================================================================
    # API 调用与解析
    # =========================================================================
    async def _fetch_study(self, nct_id: str) -> TrialRecordResult:
        """调用 ClinicalTrials.gov API v2 单研究端点"""
        session = await self._get_session()
        url = f"{CLINICALTRIALS_BASE}/studies/{nct_id}?format=json"

        try:
            resp = await self._retry_get(session, url)
            async with resp:
                if resp.status == 404:
                    logger.warning(f"⚠️ {nct_id} 在 ClinicalTrials.gov 中不存在")
                    return TrialRecordResult(
                        exists=False,
                        nct_id=nct_id,
                        status=f"未找到 — API 返回 404",
                        url=f"https://clinicaltrials.gov/study/{nct_id}",
                    )
                if resp.status != 200:
                    logger.error(f"ClinicalTrials.gov HTTP {resp.status}")
                    return TrialRecordResult(
                        exists=False,
                        nct_id=nct_id,
                        status=f"API 错误 HTTP {resp.status}",
                        url=f"https://clinicaltrials.gov/study/{nct_id}",
                    )
                data = await resp.json()
        except Exception as e:
            logger.error(f"ClinicalTrials.gov 请求失败: {e}")
            return TrialRecordResult(
                exists=False,
                nct_id=nct_id,
                status=f"API 请求失败: {str(e)}",
                url=f"https://clinicaltrials.gov/study/{nct_id}",
            )

        return self._parse_study_response(data, nct_id)

    def _parse_study_response(self, data: dict, nct_id: str) -> TrialRecordResult:
        """
        解析 ClinicalTrials.gov API v2 单研究响应。

        protocolSection 结构:
          - identificationModule  → nctId, briefTitle, officialTitle
          - statusModule          → overallStatus, startDateStruct, completionDateStruct,
                                     lastUpdatePostDateStruct, enrollmentInfo
          - sponsorCollaboratorsModule → leadSponsor
          - designModule          → phases, studyType
          - armsInterventionsModule → armGroups, interventions
          - conditionsModule      → conditions
          - descriptionModule     → briefSummary, detailedDescription
          - resultsSection        → (存在即有结果)
        """
        try:
            ps = data.get("protocolSection", {})

            # --- 标识 ---
            ident = ps.get("identificationModule", {})
            brief_title = ident.get("briefTitle")
            official_title = ident.get("officialTitle")
            title = brief_title or official_title
            actual_nct = ident.get("nctId", nct_id)

            # --- 状态 ---
            status_mod = ps.get("statusModule", {})
            overall_status = status_mod.get("overallStatus")
            # 将状态翻译为中文友好描述
            status_display = self._translate_status(overall_status)

            # 日期
            start_date_struct = status_mod.get("startDateStruct", {})
            if start_date_struct:
                start_date = self._format_date(start_date_struct)
            else:
                start_date = status_mod.get("startDate")

            completion_date_struct = status_mod.get("completionDateStruct", {})
            if completion_date_struct:
                completion_date = self._format_date(completion_date_struct)
            else:
                completion_date = status_mod.get("completionDate")

            last_update_struct = status_mod.get("lastUpdatePostDateStruct", {})
            if last_update_struct:
                last_update = self._format_date(last_update_struct)
            else:
                last_update = status_mod.get("lastUpdatePostDate")

            # 招募人数 (可能在 statusModule 或 designModule 中)
            enrollment_info = (status_mod.get("enrollmentInfo") or
                               ps.get("designModule", {}).get("enrollmentInfo"))
            enrollment = enrollment_info.get("count") if enrollment_info else None

            # --- 申办方 ---
            sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
            lead_sponsor = sponsor_mod.get("leadSponsor", {})
            sponsor_name = lead_sponsor.get("name") if lead_sponsor else None

            # --- 设计 (分期、类型) ---
            design = ps.get("designModule", {})
            phases_list = design.get("phases", []) or []
            phase_str = ", ".join(phases_list) if phases_list else None
            study_type = design.get("studyType")

            # --- 干预措施 ---
            arms_mod = ps.get("armsInterventionsModule", {})
            interventions = self._extract_interventions(arms_mod)

            # --- 条件 ---
            conditions_mod = ps.get("conditionsModule", {})
            conditions = conditions_mod.get("conditions", []) or []

            # --- 是否有结果 ---
            has_results = bool(data.get("resultsSection") or data.get("hasResults"))

            # --- URL ---
            study_url = f"https://clinicaltrials.gov/study/{actual_nct}"

            return TrialRecordResult(
                exists=True,
                nct_id=actual_nct,
                title=title,
                status=status_display,
                phase=phase_str,
                study_type=study_type,
                interventions=interventions,
                conditions=conditions,
                sponsor=sponsor_name,
                start_date=start_date,
                completion_date=completion_date,
                last_update_posted=last_update,
                enrollment=enrollment,
                has_results=has_results,
                url=study_url,
            )

        except Exception as e:
            logger.error(f"解析 ClinicalTrials.gov 响应出错: {e}")
            return TrialRecordResult(
                exists=True,
                nct_id=nct_id,
                status=f"数据解析失败: {str(e)}",
                url=f"https://clinicaltrials.gov/study/{nct_id}",
            )

    # =========================================================================
    # 辅助方法
    # =========================================================================
    @staticmethod
    def _translate_status(status: Optional[str]) -> Optional[str]:
        """将 API 状态码翻译为人类可读描述"""
        if not status:
            return None
        mapping = {
            "NOT_YET_RECRUITING": "尚未招募",
            "RECRUITING": "招募中",
            "ENROLLING_BY_INVITATION": "邀请入组中",
            "ACTIVE_NOT_RECRUITING": "进行中(已停止招募)",
            "COMPLETED": "已完成",
            "SUSPENDED": "暂停中",
            "TERMINATED": "已终止",
            "WITHDRAWN": "已撤回",
            "UNKNOWN": "状态未知",
            "APPROVED_FOR_MARKETING": "已获批上市",
            "AVAILABLE": "可获取",
            "NO_LONGER_AVAILABLE": "不再可获取",
            "TEMPORARILY_NOT_AVAILABLE": "暂时不可获取",
        }
        translated = mapping.get(status, status.replace("_", " ").title())
        # 同时返回原文+翻译，方便 Agent 精确匹配
        return f"{status} ({translated})"

    @staticmethod
    def _extract_interventions(arms_mod: dict) -> list[str]:
        """从 armsInterventionsModule 提取干预措施名称"""
        interventions = []

        # 方式 1: interventions 数组 (v2 API 主要方式)
        for intervention in arms_mod.get("interventions", []) or []:
            name = intervention.get("name")
            i_type = intervention.get("type")
            desc = intervention.get("description")
            if name:
                label = name
                if i_type:
                    label = f"[{i_type}] {label}"
                interventions.append(label)

        # 方式 2: armGroups 中的 interventionNames (旧格式兜底)
        if not interventions:
            for arm in arms_mod.get("armGroups", []) or []:
                for iname in arm.get("interventionNames", []) or []:
                    if iname and iname not in interventions:
                        interventions.append(iname)

        return interventions

    @staticmethod
    def _format_date(date_struct: dict) -> Optional[str]:
        """将日期结构体格式化为 YYYY-MM-DD 字符串。
        支持两种格式:
          - API v2 主要格式: {"date": "2020-02-21", "type": "ACTUAL"}
          - 其他字段格式: {"year": 2020, "month": 2, "day": 21}
        """
        if not date_struct:
            return None
        # 格式 1: 直接 date 字符串 (API v2 标准)
        date_str = date_struct.get("date")
        if date_str and isinstance(date_str, str):
            return date_str
        # 格式 2: 分离的 year/month/day
        year = date_struct.get("year")
        month = date_struct.get("month")
        day = date_struct.get("day")
        parts = []
        if year:
            parts.append(str(year))
        if month:
            parts.append(str(month).zfill(2))
        if day:
            parts.append(str(day).zfill(2))
        return "-".join(parts) if parts else None

    @staticmethod
    def _normalize_nct_id(nct_id: str) -> Optional[str]:
        """规范化 NCT ID：去空白→补前缀→校验格式"""
        if not nct_id:
            return None
        nct_id = nct_id.strip().upper()
        if not nct_id:
            return None

        # 匹配纯数字 (8位)
        m = _NCT_ID_RE.match(nct_id)
        if m:
            return f"NCT{m.group(1)}"

        logger.warning(f"⚠️ NCT ID 格式异常: '{nct_id}' (应为 NCT + 8位数字)")
        return None


# =============================================================================
# 顶层 API
# =============================================================================

async def get_trial_record(nct_id: str,
                           verbose: bool = False) -> TrialRecordResult:
    """
    异步回查临床试验 (顶层便捷函数)。

    使用示例:
        result = await get_trial_record("NCT04280705")
        result = await get_trial_record("04280705")   # 自动补 NCT 前缀
    """
    lookup = TrialRecordLookup(verbose=verbose)
    try:
        return await lookup.lookup(nct_id)
    finally:
        await lookup.close()


def get_trial_record_sync(nct_id: str,
                          verbose: bool = False) -> TrialRecordResult:
    """
    同步回查临床试验 (封装 asyncio.run)。

    使用示例:
        result = get_trial_record_sync("NCT04280705")
        result = get_trial_record_sync("04280705")
    """
    return asyncio.run(get_trial_record(nct_id, verbose=verbose))


# =============================================================================
# CLI
# =============================================================================
async def main():
    parser = argparse.ArgumentParser(
        description="临床试验回查工具 — 通过 NCT ID 查询注册试验的状态、分期、干预",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python get_trial_record.py NCT04280705
  python get_trial_record.py 04280705              # 自动补 NCT 前缀
  python get_trial_record.py NCT04280705 --pretty
        """,
    )
    parser.add_argument("nct_id", type=str, help="NCT 编号 (如 NCT04280705 或 04280705)")
    parser.add_argument("--pretty", "-p", action="store_true", help="格式化 JSON 输出")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细调试日志")

    args = parser.parse_args()

    lookup = TrialRecordLookup(verbose=args.verbose)
    try:
        result = await lookup.lookup(args.nct_id)
    finally:
        await lookup.close()

    # 输出
    indent = 2 if args.pretty else None
    output = json.dumps(result.to_dict(), ensure_ascii=False, indent=indent)
    print(output)

    # 在 stderr 输出摘要 (人类可读)
    logger.info(result.summary())


if __name__ == "__main__":
    asyncio.run(main())

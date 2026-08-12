# 医学文献工具链 — 三轮子总览

三个独立工具，各自无相互依赖，Agent 可按需分流调用。

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent / 上层调度                          │
│                                                             │
│   ┌──────────────────┐  ┌──────────────┐  ┌───────────────┐ │
│   │ literature_search │  │verify_citation│  │get_trial_record│ │
│   │   关键词 → 文献列表  │  │ PMID/DOI→核验  │  │ NCT→试验详情   │ │
│   └────────┬─────────┘  └──────┬───────┘  └───────┬───────┘ │
│            │                   │                   │         │
│   PubMed ─┼─── Europe PMC      │  PubMed           │  CT.gov  │
│   CT.gov ─┘   Semantic Scholar │  Europe PMC       │          │
└─────────────────────────────────────────────────────────────┘
```

---

## 文件总览

| 文件 | 行数 | 一句话功能 |
|------|------|-----------|
| `literature_search.py` | ~1530 | 关键词 + 年份/类型过滤 → 四源并发检索 (数量均分) → 去重合并 |
| `verify_citation.py` | 778 | PMID/DOI → 核验是否存在、是否撤稿、字段完整性 |
| `get_trial_record.py` | 559 | NCT ID → 试验状态 / 分期 / 干预 / 结果状态 |

**三个文件相互独立**，各自有独立的 API 调用封装、重试逻辑、CLI 入口。

---

## 1. literature_search.py — 文献检索引擎

### 一句话
用关键词 + 年份/类型过滤条件，并发搜索四个学术数据源（数量均分），返回去重合并后的文献元数据列表。

### 输入

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | `str` | ✅ | 搜索关键词，如 `"metformin diabetes RCT"` |
| `max_results` | `int` | 否 | **返回总量**，默认 20。**四源均分**：每源分配到 `max(1, max_results//4)` 条 |
| `year` | `int?` | 否 | 发表年份过滤（如 `2023`） |
| `article_type` | `str?` | 否 | 文章类型过滤（如 `rct`, `meta-analysis`, `review`） |
| `fetch_full_text` | `bool` | 否 | 是否获取全文+图片，默认 `True` |
| `verbose` | `bool` | 否 | 详细调试日志 |

**文章类型支持列表：**

| 简称 | 含义 | PubMed/EPMC 查询类型 |
|------|------|---------------------|
| `rct` | 随机对照试验 | Randomized Controlled Trial |
| `meta-analysis` | Meta 分析 | Meta-Analysis |
| `systematic-review` | 系统综述 | Systematic Review |
| `review` | 综述 | Review |
| `clinical-trial` | 临床试验 | Clinical Trial |
| `case-report` | 病例报告 | Case Reports |
| `guideline` | 指南 | Guideline |
| `practice-guideline` | 实践指南 | Practice Guideline |
| `editorial` | 社论 | Editorial |
| `letter` | 信件 | Letter |
| `observational-study` | 观察性研究 | Observational Study |
| `comparative-study` | 比较研究 | Comparative Study |
| `multicenter-study` | 多中心研究 | Multicenter Study |

### 四源过滤策略

不同数据源对年份和文章类型的支持程度不同，采用**服务端 + 客户端两级过滤**：

| 数据源 | 年份过滤 | 文章类型过滤 |
|--------|---------|-------------|
| **PubMed** | `mindate`/`maxdate` 服务端过滤 ✅ | `[Publication Type]` 查询语法 ✅ |
| **Europe PMC** | `FIRST_PDATE:{year}` 查询语法 ✅ | `PUB_TYPE:"{type}"` 查询语法 ✅ |
| **ClinicalTrials.gov** | 客户端侧 `_client_filter()` 过滤 | 客户端侧 `_client_filter()` 过滤 |
| **Semantic Scholar** | `year={y}-{y}` URL 参数 ✅ | 客户端侧 `_client_filter()` 过滤 |

客户端过滤通过 `evidence_level`（证据等级）和摘要关键词双重判断，确保不满足条件的条目被剔除。

### 输出

`list[dict]` — 每个元素为 `LiteratureMetadata` 对象：

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `title` | `str?` | API 原始 | 文献标题 |
| `year` | `str?` | API 原始 | 发表年份 |
| `source_type` | `str` | 常量 | `pubmed` / `clinicaltrials` / `europe_pmc` / `semantic_scholar` |
| `id` | `str?` | API 原始 | `pmid:xxx` / `doi:xxx` / `nct:xxx` / `ss:xxx` |
| `test_summary` | `str?` | API 原始 | 摘要（原样保留，不经 LLM 加工） |
| `text` | `str?` | Europe PMC | 全文（截断至 100,000 字符） |
| `image_summary` | `list[str]?` | AI 生成 | 每张图片的 Qwen3-VL 中文概括 |
| `image_base_64` | `list[str]?` | PMC CDN | 每张图片的 base64 编码 |
| `evidence_level` | `str?` | 规则推断 | Oxford CEBM 证据等级 (1A~5) |
| `last_retrieved_at` | `str` | 系统时钟 | ISO 8601 UTC 时间戳 |

### 功能 & 工作流程

```
用户输入 query + year? + article_type?
    │
    ▼
┌─ _build_query_with_filters() ────────────────────────────┐
│  将年份和文章类型拼接入查询字符串 (PubMed / Europe PMC)      │
│  • PubMed: query + "RCT"[Publication Type]               │
│  • Europe PMC: query + PUB_TYPE:"RCT" + FIRST_PDATE:2023 │
└──────────────────┬──────────────────────────────────────┘
                   ▼
┌─ search(query) ──────────────────────────────────────────┐
│  asyncio.gather 并发调用 (每源 _per_source = max_results/4) │
│  ├─ _search_pubmed()         → mindate/maxdate 年份过滤   │
│  ├─ _search_clinicaltrials() → 客户端侧 _client_filter()  │
│  ├─ _search_europe_pmc()     → PUB_TYPE + FIRST_PDATE    │
│  └─ _search_semantic_scholar()→ year 参数 + 客户端过滤     │
│         │                                                 │
│         ▼                                                 │
│  _deduplicate() ── DOI/PMID 精确去重 + 多源数据合并补全     │
│         │                                                 │
│         ▼                                                 │
│  _enrich_full_text() ── PMC 页面爬取 blob 图片 + AI 概括    │
│         │                                                 │
│         ▼                                                 │
│  返回 list[LiteratureMetadata] → JSON / CSV               │
└───────────────────────────────────────────────────────────┘
```

**关键特性：**
- 四源并发搜索，数量均分（`n=20 → 每源5条`）
- 年份过滤：PubMed/Europe PMC/Semantic Scholar 服务端过滤，CT.gov 客户端过滤
- 文章类型过滤：PubMed/Europe PMC 服务端查询语法，CT.gov/Semantic Scholar 客户端通过证据等级+关键词推断
- DOI/PMID 智能去重，多源数据互补（如 Semantic Scholar 的 TLDR 填补缺失摘要）
- 图片下载采用 5 级兜底策略（Blob CDN → 绝对 URL → PMC bin → Europe PMC API → Frontiers CDN）
- 图片魔数校验 (JPEG/PNG/GIF/WebP/BMP) 防止存入 HTTP 错误页
- 证据等级基于 Oxford CEBM 规则映射，非 LLM 推断
- 指数退避重试 (429/502/503)
- **所有文本数据来自 API 原始响应，不经 LLM 改写**

### API 调用

```python
from literature_search import LiteratureSearchEngine
import asyncio

# 基础搜索 (总量=20, 每源5条)
engine = LiteratureSearchEngine(max_results=20)
results = await engine.search("diabetes metformin")

# 带过滤搜索
engine = LiteratureSearchEngine(
    max_results=40,         # 总量, 每源=10
    year=2023,              # 只看 2023 年
    article_type="rct",     # 只看 RCT
    fetch_full_text=False,  # 跳过全文
)
results = await engine.search("diabetes treatment")
await engine.close()
```

### CLI

```bash
# 基础搜索
python literature_search.py "cancer immunotherapy" -n 20 -o results.json --csv results.csv

# 2024 年的 RCT
python literature_search.py "diabetes" -n 40 --year 2024 --article-type rct

# 2023 年的 Meta-Analysis, 跳过全文
python literature_search.py "heart failure" -n 20 -y 2023 -t meta-analysis --no-full-text

# 2022 年的综述
python literature_search.py "COVID-19 vaccine" -y 2022 -t review -o reviews.json --pretty

### 数据溯源保证

| 可信度 | 字段 |
|--------|------|
| ✅ API 直接提取 | title, year, id, test_summary, text, image_base_64 |
| ⚠️ 规则推断 | evidence_level (基于 API 返回的 publication_types) |
| 🤖 AI 生成 | image_summary (Qwen3-VL-32B，失败回退 XML caption) |

---

## 2. verify_citation.py — 引用核验

### 一句话
输入 PMID 或 DOI，核验这篇文献是否真实存在、是否被撤稿、关键元数据是否完整。

### 输入

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `pmid` | `str?` | 二选一 | PubMed ID（纯数字，如 `"28770321"`） |
| `doi` | `str?` | 二选一 | DOI（如 `"10.1007/s00125-017-4336-x"`） |
| `verbose` | `bool` | 否 | 详细调试日志 |

两个都提供时 **PMID 优先**（直查 PubMed，更快更准确）。

### 输出

`CitationVerificationResult` dataclass：

| 字段 | 类型 | 说明 |
|------|------|------|
| **核心** | | |
| `exists` | `bool` | 引用是否真实存在 |
| `is_retracted` | `bool` | 是否已被撤稿 |
| `retraction_notice` | `str?` | 撤稿声明文本 |
| **元数据** | | |
| `title` | `str?` | 文献标题 |
| `authors` | `str?` | 作者列表（`LastName ForeName; ...`） |
| `journal` | `str?` | 期刊名 |
| `year` | `str?` | 发表年份 |
| `abstract` | `str?` | 摘要文本 |
| `doi` | `str?` | 已确认的 DOI |
| `pmid` | `str?` | 已确认的 PMID |
| **质量标记** | | |
| `missing_fields` | `list[str]` | 缺失的关键字段（title/authors/journal/year/abstract） |
| `alerts` | `list[str]` | 所有警告（撤稿/字段缺失/无摘要/API失败） |
| **溯源** | | |
| `source` | `str` | 数据来源（`pubmed` / `europe_pmc`） |
| `verified_at` | `str` | 核验时间戳 ISO 8601 |

### 功能 & 工作流程

```
输入 PMID                      输入 DOI
    │                              │
    ▼                              ▼
┌─ _verify_by_pmid() ─┐    ┌─ _verify_by_doi() ────────────────┐
│                      │    │  1. PubMed esearch DOI→PMID      │
│  PubMed efetch.fcgi  │    │     ├ 找到 → 走 PMID 路径         │
│  解析 PubmedArticle  │    │     └ 未找到 → 走 Europe PMC      │
│  XML                 │    │  2. Europe PMC search?query=DOI:  │
│                      │    │                                  │
└──────┬───────────────┘    └──────┬───────────────────────────┘
       │                           │
       ▼                           ▼
┌─ 统一输出格式 ────────────────────────────────────────┐
│  ✓ 撤稿检测: PublicationType 含 Retracted/Retraction  │
│  ✓ 字段检查: title/authors/journal/year/abstract     │
│  ✓ 告警汇总: 撤稿/缺失/无摘要                          │
│  → CitationVerificationResult                        │
└──────────────────────────────────────────────────────┘
```

**撤稿检测关键词：**
- `Retracted Publication` — 被撤稿文章
- `Retraction of Publication` — 撤稿声明
- `Expression of Concern` — 编辑部关注声明
- `Partial Retraction` — 部分撤稿

### API 调用

```python
# 异步
from verify_citation import verify_citation
result = await verify_citation(pmid="28770321")

# 同步
from verify_citation import verify_citation_sync
result = verify_citation_sync(doi="10.1007/s00125-017-4336-x")

# 快速判断
print(result.summary())
# ✅ 引用有效 | 《Metformin: clinical use...》| (2017) | [Diabetologia]

print(result.to_dict())
# {'exists': True, 'is_retracted': False, 'title': '...', ...}
```

### CLI

```bash
python verify_citation.py --pmid 28770321
python verify_citation.py --doi "10.1007/s00125-017-4336-x" --pretty
```

---

## 3. get_trial_record.py — 临床试验回查

### 一句话
输入 NCT ID，回查 ClinicalTrials.gov 注册信息，获取试验状态、分期、干预措施等详情。

### 输入

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `nct_id` | `str` | ✅ | NCT 编号，支持 `"NCT04280705"` 或简写 `"04280705"` |
| `verbose` | `bool` | 否 | 详细调试日志 |

自动补全 `NCT` 前缀，自动大写，自动去除空白。

### 输出

`TrialRecordResult` dataclass：

| 字段 | 类型 | 说明 |
|------|------|------|
| **核心** | | |
| `exists` | `bool` | 该试验是否已注册 |
| `nct_id` | `str` | 规范化后的 NCT ID |
| **试验信息** | | |
| `title` | `str?` | 试验标题（briefTitle 或 officialTitle） |
| `status` | `str?` | 试验状态 + 中文翻译（如 `"COMPLETED (已完成)"`） |
| `phase` | `str?` | 试验分期（`PHASE1` / `PHASE2` / `PHASE3` / `PHASE4`） |
| `study_type` | `str?` | 研究类型（`INTERVENTIONAL` / `OBSERVATIONAL`） |
| `interventions` | `list[str]` | 干预措施名称（如 `["[DRUG] Remdesivir", "[OTHER] Placebo"]`） |
| `conditions` | `list[str]` | 研究疾病/条件（如 `["COVID-19"]`） |
| `sponsor` | `str?` | 主要申办方名称 |
| **时间** | | |
| `start_date` | `str?` | 开始日期（`YYYY-MM-DD`） |
| `completion_date` | `str?` | 预计/实际完成日期 |
| `last_update_posted` | `str?` | 最后更新日期 |
| **规模 & 结果** | | |
| `enrollment` | `int?` | 预计/实际招募人数 |
| `has_results` | `bool` | 是否已发布试验结果 |
| **溯源** | | |
| `url` | `str` | ClinicalTrials.gov 页面链接 |
| `verified_at` | `str` | 核验时间戳 ISO 8601 |

**便捷属性：**
- `result.is_active` — 是否仍进行中（Recruiting / Not yet recruiting / Active）
- `result.is_completed` — 是否已完成/终止

### 功能 & 工作流程

```
输入 NCT ID (支持简写)
    │
    ▼
┌─ _normalize_nct_id() ─────────────────────────────────┐
│  "04280705" → "NCT04280705"                           │
│  格式校验: NCT + 8位数字                                │
└──────────────────┬────────────────────────────────────┘
                   ▼
┌─ _fetch_study() ──────────────────────────────────────┐
│  GET clinicaltrials.gov/api/v2/studies/{nctId}        │
│  ├─ HTTP 200 → _parse_study_response()                │
│  └─ HTTP 404 → exists=False                           │
└──────────────────┬────────────────────────────────────┘
                   ▼
┌─ _parse_study_response() ─────────────────────────────┐
│  protocolSection │                                     │
│  ├─ identificationModule     → nctId, title           │
│  ├─ statusModule             → status, dates          │
│  ├─ designModule             → phase, studyType,      │
│  │                             enrollment             │
│  ├─ armsInterventionsModule  → interventions[]         │
│  ├─ conditionsModule         → conditions[]           │
│  └─ sponsorCollaboratorsModule → sponsor              │
│                                                       │
│  resultsSection 存在? → has_results = True            │
│                                                       │
│  → TrialRecordResult                                  │
└───────────────────────────────────────────────────────┘
```

**状态翻译表（内建）：**

| API 值 | 中文 |
|--------|------|
| `RECRUITING` | 招募中 |
| `COMPLETED` | 已完成 |
| `ACTIVE_NOT_RECRUITING` | 进行中(已停止招募) |
| `SUSPENDED` | 暂停中 |
| `TERMINATED` | 已终止 |
| `WITHDRAWN` | 已撤回 |
| `NOT_YET_RECRUITING` | 尚未招募 |

### API 调用

```python
# 异步
from get_trial_record import get_trial_record
result = await get_trial_record("NCT04280705")

# 同步
from get_trial_record import get_trial_record_sync
result = get_trial_record_sync("04280705")  # 简写也行

# 快速判断
print(result.summary())
# ✅ NCT04280705 | 状态: Completed (已完成) | 分期: PHASE3 | 干预: [DRUG] Remdesivir, [OTHER] Placebo | 📊 已有结果 | 《Adaptive COVID-19 Treatment Trial (ACTT)》

# 判断逻辑
if result.exists and not result.has_results:
    print("⚠️ 试验已注册但尚未发布结果")
```

### CLI

```bash
python get_trial_record.py NCT04280705
python get_trial_record.py 04280705 --pretty
```

---

## Agent 调用场景对照表

| 场景 | 使用工具 | 调用方式 |
|------|---------|---------|
| "搜索糖尿病最新RCT研究" | `literature_search` | `search("diabetes", year=2024, article_type="rct")` |
| "找2023年发表的Meta分析，关于心衰的" | `literature_search` | `search("heart failure", year=2023, article_type="meta-analysis")` |
| "查一下这篇文献是不是真的 PMID 28770321" | `verify_citation` | `verify_citation(pmid="28770321")` |
| "这个引用格式的文献还在吗 DOI xxx" | `verify_citation` | `verify_citation(doi="10.xxx/yyy")` |
| "这篇文章被撤稿了吗" | `verify_citation` | `verify_citation(pmid="xxx")` → `is_retracted` |
| "这个临床试验 NCT04280705 结果出来了吗" | `get_trial_record` | `get_trial_record("NCT04280705")` → `has_results` |
| "这个试验用的什么药" | `get_trial_record` | `get_trial_record("xxx")` → `interventions` |
| "论文里引用的这个试验还在进行吗" | `get_trial_record` | `get_trial_record("xxx")` → `is_active` |
| "生成这篇综述需要核验所有引用+补充试验数据" | 三个联合 | 先 search → 逐条 verify_citation + get_trial_record |

---

## 环境变量

三个工具共用同一套环境变量：

| 变量 | 用途 | 影响工具 |
|------|------|---------|
| `PUBMED_API_KEY` | PubMed E-utilities API 密钥 | `literature_search` / `verify_citation` |
| `PUBMED_EMAIL` | 联系邮箱 | `literature_search` / `verify_citation` |
| `SILICONFLOW_API_KEY` | 硅基流动视觉模型 | `literature_search` |

不设置也能运行，但 PubMed 会有速率限制，图片概括回退到 XML caption。

---

## 依赖

```
pip install aiohttp
```

Python 3.10+

---

## 验证清单

```bash
# 1. 文献搜索 — 基础
python literature_search.py "metformin diabetes" -n 8

# 2. 文献搜索 — 年份过滤
python literature_search.py "diabetes" -n 12 --year 2024

# 3. 文献搜索 — 文章类型过滤
python literature_search.py "metformin" -n 12 --article-type rct

# 4. 文献搜索 — 组合过滤
python literature_search.py "diabetes" -n 8 --year 2024 --article-type meta-analysis

# 5. 引用核验 — 真实存在
python verify_citation.py --pmid 28770321 --pretty

# 6. 引用核验 — 不存在
python verify_citation.py --pmid 999999999999 --pretty

# 7. 引用核验 — DOI
python verify_citation.py --doi "10.1007/s00125-017-4336-x" --pretty

# 8. 试验回查 — 真实试验
python get_trial_record.py NCT04280705 --pretty

# 9. 试验回查 — 不存在
python get_trial_record.py NCT99999999 --pretty

# 10. Python 导入测试
python -c "
from literature_search import LiteratureSearchEngine
from verify_citation import verify_citation_sync
from get_trial_record import get_trial_record_sync

# 测试带过滤的引擎构造
engine = LiteratureSearchEngine(max_results=12, year=2023, article_type='rct')
print(f'per_source={engine._per_source}, year={engine.year}, type={engine.article_type}')

result1 = verify_citation_sync(pmid='28770321')
result2 = get_trial_record_sync('NCT04280705')
print(result1.summary())
print(result2.summary())
"
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

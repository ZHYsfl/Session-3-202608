# 医学文献检索引擎 (Medical Literature Search Engine)

## 1. 概述

`literature_search.py` 是一个异步医学文献检索引擎，并发检索 **PubMed**、**ClinicalTrials.gov**、**Europe PMC**、**Semantic Scholar** 四大数据源，返回统一结构的文献元数据（JSON/CSV），用于构建 RAG 知识库。

**核心能力：**
- 四源并发搜索（asyncio）
- 全文获取（Europe PMC fullTextXML）
- 图片下载 + base64 编码（PMC 页面 blob CDN 抓取）
- 图片 AI 内容概括（SiliconFlow Qwen3-VL 视觉模型）
- 证据等级自动推断（Oxford CEBM 标准）
- DOI/PMID 智能去重 + 多源数据合并补全
- 指数退避重试机制
- Semantic Scholar 免费 API（Abstract + TLDR 摘要补充）

---

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install aiohttp
```

### 2.2 配置环境变量

```bash
# 必需：PubMed API Key（https://ncbi.nlm.nih.gov/account/ 注册获取）
export PUBMED_API_KEY="your_api_key_here"

# 可选：PubMed 联系邮箱
export PUBMED_EMAIL="your_email@example.com"

# 可选：硅基流动 API Key（https://siliconflow.cn 注册，用于图片AI描述）
export SILICONFLOW_API_KEY="sk-your_key_here"
```

### 2.3 基本使用

```bash
# 搜索 20 条结果，JSON 输出到 stdout
python literature_search.py "metformin diabetes RCT"

# 限制结果数 + 保存 JSON
python literature_search.py "COVID-19 vaccine efficacy" -n 10 -o results.json

# 同时输出 CSV + 美化 JSON
python literature_search.py "cancer immunotherapy" -n 5 -o results.json --csv results.csv --pretty

# 跳过全文/图片获取（仅元数据）
python literature_search.py "heart failure treatment" --no-full-text

# 详细调试模式
python literature_search.py "stroke prevention" -n 3 -v
```

---

## 3. 架构设计

```
用户输入 (query)
    │
    ▼
┌──────────────────────────────────────────────────┐
│              LiteratureSearchEngine               │
│                                                   │
│  search(query) ─── 主入口                          │
│    │                                              │
│    ├─ _search_pubmed() ──── E-utilities API       │
│    ├─ _search_clinicaltrials() ── CT.gov API v2   │
│    ├─ _search_europe_pmc() ─── Europe PMC REST    │
│    └─ _search_semantic_scholar() ── Semantic Scholar│
│         │                                         │
│         ▼ 并发执行 (asyncio.gather)                │
│    ┌────────────────────┐                         │
│    │  _deduplicate()    │ DOI/PMID 去重 + 合并     │
│    └────────┬───────────┘                         │
│             ▼                                     │
│    ┌──────────────────────────┐                   │
│    │  _enrich_full_text()     │                   │
│    │  ├─ 获取全文 XML          │                   │
│    │  ├─ 提取图片 href         │                   │
│    │  ├─ _fetch_pmc_blob_images() │ PMC 页面抓取   │
│    │  ├─ _download_single_image() │ 兜底 URL 尝试  │
│    │  └─ _vision_describe_image() │ AI 图片概括    │
│    └──────────────────────────┘                   │
│             │                                     │
│             ▼                                     │
│    LiteratureMetadata[] ─── JSON / CSV 输出        │
└──────────────────────────────────────────────────┘
```

### 3.1 Session 管理

脚本维护两个独立的 `aiohttp.ClientSession`：

| Session | 用途 | 超时 | Cookie |
|---------|------|------|--------|
| `_session` | 主 Session（API 调用 + Blob 图片下载） | 30s | DummyCookieJar |
| `_img_session` | 图片下载兜底（回退 URL 模式） | 10s | DummyCookieJar |

**设计原因**：Blob CDN 图片通过主 Session 下载（30s 超时 + 通用 HEADERS），因为图片 Session 的 10s 超时和特殊的 Accept 头会导致 Blob CDN 请求静默超时。

---

## 4. 数据模型

### LiteratureMetadata

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | str? | 文献标题 |
| `year` | str? | 发表年份 |
| `source_type` | str | `pubmed` / `clinicaltrials` / `europe_pmc` / `semantic_scholar` |
| `id` | str? | 标识符 (`pmid:xxx`, `doi:xxx`, `nct:xxx`, `ss:xxx`) |
| `test_summary` | str? | 摘要文本 |
| `text` | str? | 全文文本（截断至 100,000 字符） |
| `image_summary` | list[str]? | 每张图片的 AI 内容概括 |
| `image_base_64` | list[str]? | 每张图片的 base64 编码 |
| `evidence_level` | str? | Oxford CEBM 证据等级 (1A~5) |
| `last_retrieved_at` | str? | 检索时间戳 (ISO 8601 UTC) |

### 4.1 数据溯源 & 可靠性声明

**核心原则：所有文本数据均从权威 API 原始响应中提取，不经 LLM 加工。**

#### 直接检索字段（100% 来自 API 原始数据，零幻觉风险）

| 字段 | 数据来源 | 提取方式 |
|------|---------|---------|
| `title` | PubMed XML → `ArticleTitle` 元素 / CT.gov JSON `briefTitle` / Europe PMC JSON `title` / Semantic Scholar JSON `title` | XML 文本节点 / JSON 字段直接取值 |
| `year` | PubMed `PubDate/Year` / CT.gov `startDateStruct` / Europe PMC `pubYear` / Semantic Scholar `year` | 原始字段，仅做字符串规范化 |
| `source_type` | 硬编码常量 | 标记数据来自哪个 API，不参与检索 |
| `id` | PubMed `PMID` / CT.gov `nctId` / Europe PMC `pmid`/`doi` / Semantic Scholar `externalIds` | 标准学术标识符，全局唯一 |
| `test_summary` | PubMed `Abstract/AbstractText` / CT.gov `briefSummary` / Europe PMC `abstractText` / Semantic Scholar `abstract` 或 `tldr` | 仅做 `strip_html()` 去除 HTML 标签，**文本内容原样保留** |
| `text` | Europe PMC **fullTextXML** API（仅 Open Access 文章） | 递归提取 `<body>` 和 `<abstract>` 下所有 `<p>` 元素文本节点，截断至 100,000 字符 |
| `image_base_64` | NCBI PMC 页面 `<img src="...">` Blob CDN URL → 直接下载二进制数据 | CDN 原始图片字节 → base64 编码；**下载后经魔数校验**（JPEG/PNG/GIF/WebP/BMP），防止将 HTML 错误页误存为图片 |
| `last_retrieved_at` | 系统时钟 | Python `datetime.now(UTC)`，ISO 8601 格式 |

#### 衍生/推断字段（基于真实数据，非原始检索）

| 字段 | 推断方式 | 可信度说明 |
|------|---------|-----------|
| `evidence_level` | 从各 API 返回的 **真实 `publication_types`** 按 Oxford CEBM 标准规则映射 | 输入（出版类型）来自 API 原始数据；映射规则公开可查；**非原始标签，属于最佳推断** |
| `image_summary` | 调用 **Qwen3-VL-32B** 视觉模型对 `image_base_64` 中的真实图片做中文内容概括 | 图片本身真实；描述为 AI 生成，受模型能力限制；失败时自动回退到 XML `<caption>` 原文 |

#### 跨源匹配保证

- **去重依据**：DOI（Digital Object Identifier）和 PMID（PubMed ID），这两个是学术界标准唯一标识符——**同一 DOI = 同一篇论文**，不存在歧义匹配
- **合并策略**：`_merge_record()` 仅补充 **NULL 字段**，绝不覆盖已有真实数据；唯一例外：若新源摘要长度超过旧源 1.5×，替换为更完整版本
- **图片验证**：下载后通过文件头魔数（magic number）校验，JPEG (`\xff\xd8\xff`)、PNG (`\x89PNG`)、GIF (`GIF89a`/`GIF87a`)、WebP (`RIFF....WEBP`)、BMP (`BM`)，确保存入的是真实图片而非 HTTP 错误页面

#### 不会做的事情

- ❌ 不会用 LLM 生成/改写 `test_summary`（摘要）或 `text`（全文）
- ❌ 不会在没有 PMCID 的情况下编造全文数据
- ❌ 不会对 DOI/PMID 进行模糊匹配（仅精确匹配）
- ❌ 不会在图片下载失败时生成占位图片

---

## 5. 核心功能详解

### 5.1 图片下载

图片下载采用**分层策略**：

```
优先级 1: PMC 页面 Blob CDN URL（主 Session，30s 超时）
    ↓ 失败
优先级 2: 原始 href（如果为绝对 URL，如出版商 CDN）
    ↓ 失败
优先级 3: PMC bin URL（旧版模式，作为兜底）
    ↓ 失败
优先级 4: Europe PMC fullTextImage API
    ↓ 失败
优先级 5: Frontiers 出版商 CDN 智能推断
```

**关键实现细节：**
- Blob URL 从 PMC 文章 HTML 的 `<img src="...">` 标签中提取，不使用 JS/JSON 中的 URL（那些通常已失效）
- 使用 `_is_valid_image()` 魔数验证（JPEG/PNG/GIF/WebP/BMP）防止下载到 HTML 错误页

### 5.2 证据等级推断

基于 **Oxford CEBM** 标准（从高到低）：

| 等级 | 研究类型 |
|------|--------|
| 1A | Meta-analysis / Systematic Review of RCTs |
| 1B | Individual RCT / Phase 3-4 Clinical Trial |
| 2A | Systematic Review of Cohort Studies |
| 2B | Cohort Study / Phase 2 Trial / Clinical Trial |
| 3A | Systematic Review of Case-Control Studies |
| 3B | Case-Control / Observational / Cross-Sectional |
| 4 | Case Series / Case Report |
| 5 | Review / Editorial / Guideline / Opinion |

### 5.3 去重与数据合并策略

基于 **DOI** 和 **PMID** 双重去重：
- DOI 匹配优先（更通用）
- PMID 匹配作为补充
- 重复条目保留第一个找到的版本，**智能补全**缺失字段：
  - `test_summary`: 不同源之间互相补充摘要（如 Semantic Scholar 提供 PubMed 缺失的 Abstract）
  - `text`: 全文合并
  - `evidence_level`: 从无到有补全
  - `_pmcid`: 补全 PMCID（使更多条目可触发全文+图片获取）

### 5.4 Semantic Scholar 数据源

**Semantic Scholar** 提供免费的学术论文搜索 API：

| 特性 | 说明 |
|------|------|
| API 端点 | `api.semanticscholar.org/graph/v1/paper/search` |
| 费率限制 | 100 req/5min（无需 API Key） |
| 提供数据 | 标题、摘要、TLDR（AI 一句话总结）、DOI/PMID、出版类型、年份 |
| 优势 | 覆盖预印本 (arXiv)、会议论文；TLDR 填补摘要空白 |

**数据补全逻辑**：当 PubMed / Europe PMC 的结果 `test_summary` 为空时，Semantic Scholar 的 abstract 或 TLDR 会自动合并填充。此外，Semantic Scholar 可能找到 PubMed 未收录的预印本和会议论文，通过 DOI/PMID 去重后可补充缺失数据。

### 5.5 重试机制

所有外部 HTTP 请求使用**指数退避重试**（最多 3 次）：
- 触发条件：HTTP 429（限流）、502/503（服务不可用）、网络异常
- 退避策略：2^attempt 秒（1s → 2s → 4s）

---

## 6. 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `PUBMED_API_KEY` | 推荐 | NCBI E-utilities API Key。不设置仍可使用但受速率限制（3 req/s） |
| `PUBMED_EMAIL` | 否 | 联系邮箱，默认 `researcher@example.com` |
| `SILICONFLOW_API_KEY` | 否 | 硅基流动 API Key。不设置时图片概括回退到 XML caption |

---

## 7. CLI 参数

```
usage: literature_search.py [-h] [--max-results N] [--output FILE] [--csv FILE]
                            [--no-full-text] [--pretty] [--verbose] query

positional arguments:
  query                 搜索查询字符串

optional arguments:
  --max-results, -n N   最大结果数 (默认: 20)
  --output, -o FILE     JSON 输出文件路径
  --csv FILE            CSV 输出文件路径
  --no-full-text        跳过全文/图片获取
  --pretty, -p          格式化 JSON 输出
  --verbose, -v         详细调试日志
```

---

## 8. 输出格式

### JSON

标准 JSON 数组，每个元素为 `LiteratureMetadata` 对象：

```json
[{
  "title": "Metformin: clinical use in type 2 diabetes.",
  "year": "2017",
  "source_type": "pubmed",
  "id": "pmid:28770321",
  "test_summary": "Metformin is one of the most popular...",
  "text": null,
  "image_summary": null,
  "image_base_64": null,
  "evidence_level": "5",
  "last_retrieved_at": "2026-08-11T12:34:56.789012+00:00"
}]
```

### CSV

自动处理以下转换：
- 列表字段：用 ` | ` 分隔合并
- base64 字段：截断标记（保留完整数据在 JSON）
- 文本字段：超过 5000 字符自动截断
- 编码：UTF-8 BOM（Excel 兼容）

---

## 9. Base64 图片数据说明

`image_base_64` 字段存储的是图片的 base64 编码字符串：

```python
import base64

# 还原为图片文件
b64_data = result["image_base_64"][0]
img_bytes = base64.b64decode(b64_data)
with open("output.jpg", "wb") as f:
    f.write(img_bytes)

# 直接用于前端展示
html = f'<img src="data:image/jpeg;base64,{b64_data}">'

# 传给多模态大模型
payload = {
    "type": "image_url",
    "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
}
```

**数据量说明**：base64 编码比原始二进制大约 33%。如原始 100KB 的 JPEG 编码后约 133KB。

---

## 10. 已知限制

| 限制 | 说明 | 影响 |
|------|------|------|
| 仅 Open Access 文章有全文 | 通过 Europe PMC 的 fullTextXML 获取，需文章为 OA | 部分文章无全文/图片 |
| 图片仅从 PMC 获取 | 出版商付费墙后的图片不可下载 | OA 文章图片覆盖率高 |
| PubMed API 速率限制 | 无 API Key 时 3 req/s，有 Key 时 10 req/s | 大量查询时建议配置 Key |
| SilconFlow API 稳定性 | 视觉模型依赖第三方 API | 失败时回退到 XML caption |
| 单次最多 20 条（推荐） | 全文+图片下载耗时随结果数线性增长 | 大量检索建议分批 |

---

## 11. 依赖

```
aiohttp >= 3.8
```

Python 3.10+

---

## 12. 故障排查

### 图片下载为 0

1. 确认检索的文章是 PMC Open Access（`source_type: "pubmed"` 或 `"europe_pmc"`）
2. 设置 `PUBMED_API_KEY` 环境变量
3. 使用 `-v` 查看详细日志

### cookie 警告 `Invalid attribute 'x-enc'`

无害警告，NCBI 服务器返回了不合规的 cookie 属性。已通过 `DummyCookieJar` 抑制（不影响功能）。

### SiliconFlow 返回 403

API Key 过期或无效，图片概括自动回退到 XML caption。

---

## 13. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.2 | 2026-08 | 新增 Semantic Scholar 数据源；增强去重合并逻辑；`last_retrieved_at` 填充真实时间戳 |
| 1.1 | 2026-08 | 重构图片下载（PMC blob CDN），移除硬编码 Key，增加重试/CSV/verbose |
| 1.0 | 2025 | 初始版本 |

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

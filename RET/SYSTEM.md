# 证据智能助手 (Evidence Intelligence) — 系统文档

> 版本: 0.2.0 | 日期: 2026-08-12 | 论文数: 480 | 向量块: 21,433

---

## 一、系统概述

基于 RAG（检索增强生成）的循证医学证据检索与问答系统。核心流程：用户提问 → 本地 ChromaDB 向量检索 + PubMed 外部检索 → DeepSeek-V3 证据整合 → 带引用标记的结构化回答 + 引用验证。

### 三赛道设计

| 赛道 | 名称 | 目标用户 | Prompt 风格 |
|------|------|----------|-------------|
| Track 1 🩺 | 临床证据助手 | 医生/医学生/科研人员 | PICO 框架, 证据层级, 专业审慎 |
| Track 2 🥗 | 健康营养助手 | 普通消费者 | 通俗科普, 不确定性透明, 安全边界 |
| Track 3 📚 | 数据库浏览 | 所有用户 | 直接检索, 论文详情, 统计看板 |

---

## 二、技术架构

### 向量存储
- **嵌入模型**: BGE-M3 (BAAI/bge-m3, 1024维) 通过 SiliconFlow API
- **向量库**: ChromaDB (持久化, cosine 距离, HNSW 索引)
- **分块策略**: 512 tokens, 段落边界分割, 50-token 重叠
- **向量类型**: title (1) + text_summary (1) + text_chunks (N) + image_desc (N)
- **聚合方式**: MaxSim (按 paper_id 取各 chunk 最大相似度)

### 检索架构
- **多路检索**: BM25 词汇 + Keyword 覆盖率 + 向量语义 (title/summary/text/image)
- **融合排序**: Reciprocal Rank Fusion (RRF, k=60)
- **医学概念增强**: 14 类医学术语别名标准化, 实体/方面/约束检测
- **外部检索**: PubMed E-utilities API (免费, 3 req/s 限速)

### Agent 系统
- **LLM**: DeepSeek-V3 (SiliconFlow, free tier)
- **检索策略**: 本地 ChromaDB (top_k=20) + PubMed (max_results=5) 始终同时执行
- **多轮迭代**: LLM 可通过 `[SEARCH: query]` 标记请求补充检索 (最多3轮)
- **热更新**: 外部 PubMed 新论文自动在后台线程完成 embedding + ChromaDB 入库

### 引用验证 (Track 3)
- **L1 引用存在性**: 解析 LLM 回答中的 `[1][2]` 标记 → 对照 evidence_map 检查存在率
- **L2 陈述支撑度**: 拆解原子陈述 → 词汇重叠度 vs 检索 chunk → keep/weaken/delete 判定
- **验证报告**: 返回引用存在率、忠实度分数、逐句详情

---

## 三、项目结构

```
RET/
├── backend/
│   ├── server.py          # FastAPI 服务器 (端口 8000)
│   ├── agents.py          # 双 Agent 逻辑 + 热更新 + 多轮检索
│   ├── schemas.py         # 数据模型 (ChatRequest, AgentResponse, ToolCallStep...)
│   └── tools.py           # PubMed 外部检索工具
├── frontend/
│   └── index.html         # 单页应用 (三标签, 工具调用追踪, 证据卡片)
├── data/                  # 480 篇论文 JSON + 热更新新增论文
├── chroma_db/             # ChromaDB 持久化向量库 (>200MB)
├── config.py              # API Key, 模型参数, 分块配置
├── embedding.py           # BGE-M3 embedding (1.0s 节流 + 429 重试)
├── chunker.py             # 文本分块 + Metadata → Chunk 构建
├── vector_store.py        # ChromaDB CRUD + MaxSim 聚合
├── retriever.py           # BM25/Keyword/Vector 多路检索 + RRF + BM25 缓存
├── retriever_medical.py   # 医学概念感知评分 + 证据层级加权
├── medical_concepts.py    # 医学别名标准化 + 实体/约束检测
├── tokenizer_v2.py        # jieba 中文分词 + stopword 过滤
├── citation_verifier.py   # L1 引用存在性 + L2 陈述支撑度
├── ingest.py              # 批量入库脚本 (480 论文, ~30min)
├── ingest.log             # 入库日志
└── agent.py               # (旧) 工具函数定义 + OpenAI function schema
```

---

## 四、API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/stats` | 数据库统计 (论文数/chunks/年份/期刊) |
| POST | `/api/chat` | 主聊天接口 `{query, track}` → `{answer, evidence_cards, tool_calls, verification}` |
| POST | `/api/search` | 纯检索 (不经过 LLM) |
| GET | `/api/papers/{id}` | 论文详情 |

### Chat 响应示例
```json
{
  "answer": "### 证据概览\n...",
  "evidence_cards": [{"index": 1, "title": "...", "pmid": "...", ...}],
  "tool_calls": [{"round": 1, "action": "search_local", "result_count": 20, ...}],
  "verification": {
    "citation_existence_rate": 1.0,
    "faithfulness_score": 0.53,
    "claims_detail": [{"text": "...", "action": "keep", ...}]
  }
}
```

---

## 五、已知问题 & 优化方向

### 当前瓶颈
- **本地检索延迟 ~30-60s**: BM25/关键词检索即使有缓存仍慢，需 profiler 定位根因
- **LLM 生成 ~30-50s**: DeepSeek-V3 free tier 排队 + 生成

### 已优化
- ✅ BM25/关键词分词结果缓存 (避免每次重分词 480 篇全文)
- ✅ 热更新改为后台线程 (不阻塞用户请求)
- ✅ 去掉热更新后的重复本地检索
- ✅ 元数据启动预加载 (避免首次请求超时)
- ✅ System Prompt 减少不必要多轮检索

### 待优化
- 考虑为 `hybrid_retrieve` 添加分段计时 profiler
- 异步 PubMed abstract 获取 + 批量 embedding
- 可选更快的 LLM 模型 (如 Qwen2.5-7B-Instruct)

---

## 六、启动方式

```bash
cd RET
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000/
```

## 七、数据库热更新

PubMed 检索到本地不存在的论文时，自动触发：
1. 通过 PubMed efetch 获取完整摘要
2. BGE-M3 embedding 生成向量
3. ChromaDB upsert
4. JSON 元数据保存到 `data/pubmed_{pmid}.json`
5. 默认后台线程执行，不阻塞当前请求

首次查询后新论文自动入库，后续查询即可检索到。

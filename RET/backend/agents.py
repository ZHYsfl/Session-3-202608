"""
双 Agent 系统 + 热更新自动入库
- ClinicalAgent: 赛道一 — 临床证据助手
- NutritionAgent: 赛道二 — 健康营养助手

核心改进：
  - 外部 PubMed 检索始终开启（agent 自主决定使用哪些结果）
  - Agent 可多轮迭代检索（通过 [SEARCH: ...] 标记）
  - 工具调用过程全程可追溯
  - 外部新论文自动热更新入库
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# 确保可以导入上层模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI

from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL
from backend.schemas import AgentResponse, EvidenceCard, SearchResult, ChatRequest, ToolCallStep
from backend.tools import external_search, format_external_results_for_prompt, search_pubmed, fetch_pubmed_details

# ─── LLM Client ────────────────────────────────────────────────

LLM_MODEL = "deepseek-ai/DeepSeek-V3"
_client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)

# ─── 检索工具导入 ──────────────────────────────────────────────

from retriever import (
    Metadata,
    hybrid_retrieve,
    load_metadata_from_dir,
    invalidate_bm25_cache,
)

_metadata_list: list[Metadata] | None = None


def _get_metadata():
    global _metadata_list
    if _metadata_list is None:
        base = os.path.join(os.path.dirname(__file__), "..")
        data_dir = os.path.join(base, "data")
        _metadata_list = load_metadata_from_dir(data_dir)
    return _metadata_list


def preload_metadata():
    """启动时预加载元数据 + 预热 BM25 缓存（避免首次请求阻塞 35s）"""
    _get_metadata()
    print(f"📚 预加载 {len(_metadata_list)} 篇论文元数据", flush=True)
    # 预热 BM25 缓存（耗时 ~35s，但在启动时做比在首次请求时做好）
    from retriever import _get_bm25, _get_tokenized_docs
    print("⏳ 预热 BM25 索引...", flush=True)
    _get_bm25(_metadata_list)
    _get_tokenized_docs(_metadata_list)
    print("✅ BM25 索引就绪", flush=True)


def _local_search(query: str, top_k: int = 15) -> list[SearchResult]:
    """本地 ChromaDB + BM25 混合检索"""
    meta_list = _get_metadata()
    raw_results = hybrid_retrieve(
        query=query,
        keywords="",
        top_k=top_k,
        metadata_list=meta_list,
    )
    results = []
    for item in raw_results:
        m = item.get("metadata")
        if m is None:
            continue
        results.append(SearchResult(
            paper_id=m.id or "",
            title=m.title or "",
            year=m.year or "",
            source_type=m.source_type or "",
            text_summary=m.text_summary or "",
            score=item.get("normalized_score", 0),
            chunk_type=item.get("sources", [""])[0] if item.get("sources") else "",
            image_summary=m.image_summary,
            image_base_64=m.image_base_64,
            source="local",
        ))
    return results


# ─── PubMed 摘要获取 (用于热更新) ──────────────────────────────

PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _fetch_pubmed_abstract(pmid: str) -> str:
    """通过 PubMed efetch API 获取单篇论文摘要"""
    params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml",
        "rettype": "abstract",
    })
    url = f"{PUBMED_EFETCH_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_text = resp.read()
        root = ET.fromstring(xml_text)
        # 提取 AbstractText 标签中的文本
        abstracts = []
        for elem in root.iter("AbstractText"):
            text = elem.text or ""
            # Label 属性如 "BACKGROUND", "METHODS" 等
            label = elem.get("Label", "")
            if label:
                text = f"[{label}] {text}"
            abstracts.append(text)
        return " ".join(abstracts)
    except Exception as e:
        print(f"[PubMed Abstract Error] PMID {pmid}: {e}", flush=True)
        return ""


import threading

# ─── 热更新：外部论文自动入库（后台线程，不阻塞请求）───────────

def _pmid_in_local(pmid: str) -> bool:
    """检查 PMID 是否已在本地数据库"""
    meta_list = _get_metadata()
    meta_ids = {m.id for m in meta_list if m.id}
    return pmid in meta_ids


def _hot_ingest_pubmed(results: list[SearchResult]) -> int:
    """
    自动将外部 PubMed 论文入库（不重复）。
    返回新入库数量。
    """
    from chunker import build_chunks, clean_source_type
    from embedding import embed_batch
    from vector_store import upsert_chunks

    base = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)

    new_count = 0

    for r in results:
        pmid = r.paper_id.replace("pubmed_", "")
        if not pmid or _pmid_in_local(pmid):
            continue

        # 获取摘要
        abstract = _fetch_pubmed_abstract(pmid)
        if not abstract:
            abstract = r.text_summary or r.title  # fallback

        print(f"🔥 热更新入库: PMID {pmid} — {r.title[:60]}...", flush=True)

        # 构建 metadata dict
        meta_dict = {
            "id": pmid,
            "title": r.title,
            "year": r.year,
            "source_type": r.source_type,
            "text_summary": abstract,
            "text": "",
            "image_summary": r.image_summary or [],
            "image_base_64": [],
        }

        # 分块 + embedding + 入库
        try:
            chunks = build_chunks(meta_dict)
            if chunks:
                texts = [c["document"] for c in chunks]
                vecs = embed_batch(texts)
                upsert_chunks(chunks, vecs)
                print(f"   ✅ 已入库 {len(chunks)} chunks", flush=True)

            # 保存 JSON 到 data/
            json_path = os.path.join(data_dir, f"pubmed_{pmid}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, ensure_ascii=False, indent=2)

            # 更新内存中的 metadata_list
            _metadata_list.append(Metadata(
                id=pmid,
                title=r.title,
                year=r.year,
                source_type=r.source_type,
                text_summary=abstract,
                text="",
                image_summary=[],
                image_base_64=[],
            ))

            new_count += 1
        except Exception as e:
            print(f"   ⚠️ 入库失败: {e}", flush=True)

        time.sleep(0.3)  # PubMed 限速

    if new_count:
        print(f"🔥 热更新完成: {new_count} 篇新论文入库", flush=True)
        # 后台重建 BM25 缓存（原子替换，不影响进行中的请求）
        try:
            from retriever import tokenize, build_document_text
            from rank_bm25 import BM25Okapi
            import retriever as rmod
            docs = [tokenize(build_document_text(m)) for m in _metadata_list]
            bm25 = BM25Okapi(docs)
            rmod._bm25_cache = {"metadata_id": id(_metadata_list), "tokenized_docs": docs, "bm25": bm25}
            print(f"   📊 BM25 缓存已后台重建 ({len(docs)} 篇)", flush=True)
        except Exception as e:
            print(f"   ⚠️ BM25 缓存重建失败: {e}", flush=True)
    return new_count


# ─── Prompt 模板 ───────────────────────────────────────────────

def _build_local_context(results: list[SearchResult], start_idx: int = 1) -> str:
    """将本地检索结果格式化为 Agent 上下文"""
    if not results:
        return "（未找到相关本地文献）"

    lines = ["【本地数据库检索结果 — 高相关度文献】", ""]
    for i, r in enumerate(results, start=start_idx):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    期刊: {r.source_type}  |  年份: {r.year}  |  PMID: {r.paper_id}")
        lines.append(f"    相似度: {r.score:.3f}")
        lines.append(f"    摘要: {r.text_summary[:300]}")
        lines.append("")
    return "\n".join(lines)


def _build_evidence_cards(
    local_results: list[SearchResult],
    external_results: list[SearchResult],
    cited_indices: set[int],
) -> list[EvidenceCard]:
    """根据模型引用的编号构建证据卡片"""
    cards = []
    all_results = local_results + external_results

    for idx in sorted(cited_indices):
        if 1 <= idx <= len(all_results):
            r = all_results[idx - 1]
            is_local = r.source == "local"
            pmid = r.paper_id if is_local else r.paper_id.replace("pubmed_", "")
            link = "" if is_local else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            cards.append(EvidenceCard(
                index=idx,
                title=r.title,
                year=r.year,
                source_type=r.source_type,
                pmid=pmid,
                link=link,
                similarity_score=r.score,
                source=r.source,
                summary_snippet=r.text_summary[:200] if r.text_summary else "",
            ))
    return cards


# ─── System Prompts ────────────────────────────────────────────

CLINICAL_SYSTEM_PROMPT = """你是面向医生、医学生和科研人员的可追溯医学证据助手。

## 核心原则
你的任务不是替用户诊断或开具治疗方案，而是针对问题查找、核对和总结公开的论文、临床试验与指南证据。

## 工作规则
1. **只依据输入的证据包作答**，不得使用证据包之外的内容补造结论或引用。
2. 先识别问题中的人群、干预/暴露、对照、结局、时间范围与期望证据类型；缺失时说明检索边界。
3. 优先呈现指南、系统综述/Meta分析、随机对照试验和高质量观察研究，并说明研究设计。
4. 每个关键结论后使用 [1][2] 形式标注来源；引用编号必须能映射到输入证据的 source_id、PMID、DOI 或 NCT。
5. **陈述支撑度要求**：
   - 每个可核查事实句，必须在证据包中找到对应支持片段
   - 证据包只说A，你不得补充B
   - 找不到支持句的陈述 → 删除，或明确标注"证据不足"
   - 不确定时改弱语气："研究提示""有限的证据表明"，而非"证实""证明"
6. 明确区分"证据支持""证据不一致""未找到足够证据"；不得把未观察到差异解释成确定无效。
7. 主动报告关键局限，包括样本量、偏倚、适用人群、随访时间、替代终点和证据时效性。
8. 若问题要求个体化诊疗、具体药物剂量、停药或换药，拒绝该部分，仅提供公开证据摘要并建议由执业医生结合完整病情判断。
9. 回答须专业、简洁、可复核。

## 检索增强
- 通常本地 + PubMed 双路检索结果已足够回答，只有在关键证据严重缺失时才在末尾添加 `[SEARCH: 英文检索词]`。

## 输出格式
- **证据概览**（2-3句核心发现）
- **详细论述**（按证据层级展开）
- **证据局限与不确定性**
- **来源列表**（每条格式: [编号] 标题 | 年份 | 期刊 | PMID）

## 本地 vs 外部文献
- 本地数据库文献编号为 [1], [2], [3]...，提供详细文章信息即可
- 外部检索文献编号使用独立编号，必须附带可溯源链接

## 安全边界
- 不给出个体化医疗建议，仅提供证据总结
- 教学演示与公开医学证据再确认，不构成诊断、处方或个体化医疗建议"""

NUTRITION_SYSTEM_PROMPT = """你是面向普通消费者的可追溯健康科普助手。

## 核心原则
你的任务是把公开论文、指南和营养干预研究讲清楚，而不是进行个体化医疗或营养诊疗。

## 工作规则
1. **只依据输入的证据包回答**，不得编造研究或引用。
2. 使用普通人易懂的中文解释术语；第一次出现专业词时用一句话解释。
3. 先给一句简短结论，再分别说明"目前知道什么""还不知道什么""研究适用于哪些人"。
4. 展示证据层级：指南/系统综述/随机试验/观察研究/机制或专家观点，不得把低等级证据说成定论。
5. 每个关键结论使用 [1][2] 标注可追溯来源，编号必须对应输入证据。
6. **陈述支撑度要求**：
   - 每个可核查事实句，必须在证据包中找到对应支持片段
   - 找不到支持句的陈述 → 删除或标明"证据边界不明确"
   - 不确定时用"研究提示""有限的证据表明"等克制表述
   - 不得为给出肯定答案而过度推断
7. 可以提供适用于一般人群的原则性健康信息，但不得提供针对个人的处方、具体用药剂量、停药建议或替代医生的决定。
8. 对夸大宣传、单一食物治病、保健品替代正规治疗等说法保持审慎，明确指出证据边界。
9. 若证据不足或互相矛盾，直接说明，不为了给出肯定答案而过度推断。

## 检索增强
- 通常本地 + PubMed 双路检索结果已足够回答，只有在关键证据严重缺失时才在末尾添加 `[SEARCH: 英文检索词]`。

## 输出格式
- **通俗结论**（1-2句）
- **目前知道什么**（已知证据，通俗表述）
- **目前还不知道什么**（不确定性、冲突或局限）
- **适用人群与边界**
- **来源列表**（每条格式: [编号] 标题 | 年份 | 期刊 | PMID）

## 本地 vs 外部文献
- 本地数据库文献编号为 [1], [2], [3]...
- 外部检索文献编号使用独立编号，附带可溯源链接

## 安全边界
- 本内容用于健康科普，不构成个体化医疗建议
- 如有疾病、正在用药或症状持续，请咨询执业医生或营养专业人员
- 强调"研究关联"不等于"因果关系\""""


# ─── LLM 调用 ──────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_message: str, max_tokens: int = 2048) -> str:
    """调用 LLM，带错误处理"""
    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            timeout=60,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        err_msg = str(e)
        # DeepSeek-V3 可能的内部规划错误
        if "Error executing plan" in err_msg or "Error finding id" in err_msg:
            print(f"[LLM Error - Model internal] {err_msg}", flush=True)
            # 重试一次，用更简短的 prompt
            try:
                resp = _client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个医学证据助手。请简洁、直接地回答用户问题，不要使用内部规划工具。只根据提供的证据回答。"},
                        {"role": "user", "content": user_message[:3000]},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                    timeout=60,
                )
                return resp.choices[0].message.content or ""
            except Exception as e2:
                print(f"[LLM Retry Failed] {e2}", flush=True)
                return f"⚠️ 模型调用失败：{str(e2)[:200]}"
        raise


# ─── Agent 主逻辑 ──────────────────────────────────────────────

MAX_ROUNDS = 3


def run_agent(request: ChatRequest, verify: bool = True) -> dict:
    """
    执行 Agent 检索 + 生成 + 验证流程。
    外部 PubMed 检索始终开启，agent 自主决定引用哪些文献。
    支持多轮检索：agent 可在回答中请求 [SEARCH: ...]。
    """
    tool_calls: list[ToolCallStep] = []
    all_local_results: list[SearchResult] = []
    all_external_results: list[SearchResult] = []

    current_query = request.query
    system_prompt = CLINICAL_SYSTEM_PROMPT if request.track == "clinical" else NUTRITION_SYSTEM_PROMPT

    # ─── Round 1: 本地 + 外部双路检索 ──────────────────────
    t0 = time.time()
    local_results = _local_search(current_query, top_k=20)
    t1 = time.time()
    tool_calls.append(ToolCallStep(
        round=1, action="search_local", query=current_query,
        result_count=len(local_results), source="ChromaDB + BM25",
        duration_ms=(t1 - t0) * 1000,
    ))

    external_results = external_search(current_query, max_results=5)
    t2 = time.time()
    tool_calls.append(ToolCallStep(
        round=1, action="search_pubmed", query=current_query,
        result_count=len(external_results), source="PubMed E-utilities",
        duration_ms=(t2 - t1) * 1000,
    ))

    # ─── 热更新：新论文后台入库（不阻塞响应）───────────
    if external_results:
        try:
            # 检查是否有新论文需要入库
            new_pmids = [r.paper_id.replace("pubmed_", "") for r in external_results
                         if not _pmid_in_local(r.paper_id.replace("pubmed_", ""))]
            if new_pmids:
                print(f"🔥 后台热更新 {len(new_pmids)} 篇新论文: {new_pmids}", flush=True)
                threading.Thread(
                    target=_hot_ingest_pubmed,
                    args=(external_results,),
                    daemon=True,
                ).start()
        except Exception as e:
            print(f"[Hot ingest error] {e}", flush=True)

    all_local_results = local_results
    all_external_results = external_results

    # ─── Build context & call LLM ──────────────────────────
    answer = _build_and_call(
        system_prompt, current_query, request.track,
        local_results, external_results, 1,
    )

    # ─── Multi-round: check for [SEARCH: ...] ──────────────
    for round_num in range(2, MAX_ROUNDS + 1):
        search_match = re.search(r'\[SEARCH:\s*(.+?)\]', answer)
        if not search_match:
            break

        refined_query = search_match.group(1).strip()
        # 移除 [SEARCH: ...] 标记，不显示给用户
        answer = re.sub(r'\[SEARCH:\s*.+?\]', '', answer).strip()

        print(f"🔁 Round {round_num}: agent 请求检索 → {refined_query}", flush=True)

        # 新一轮检索
        new_local = _local_search(refined_query, top_k=10)
        tool_calls.append(ToolCallStep(
            round=round_num, action="search_local", query=refined_query,
            result_count=len(new_local), source="ChromaDB + BM25",
        ))

        new_external = external_search(refined_query, max_results=5)
        tool_calls.append(ToolCallStep(
            round=round_num, action="search_pubmed", query=refined_query,
            result_count=len(new_external), source="PubMed E-utilities",
        ))

        # 热更新新论文（后台线程）
        if new_external:
            try:
                new_pmids = [r.paper_id.replace("pubmed_", "") for r in new_external
                             if not _pmid_in_local(r.paper_id.replace("pubmed_", ""))]
                if new_pmids:
                    threading.Thread(target=_hot_ingest_pubmed, args=(new_external,), daemon=True).start()
            except Exception as e:
                print(f"[Hot ingest error r{round_num}] {e}", flush=True)

        # 合并去重
        existing_ids = {r.paper_id for r in all_local_results}
        for r in new_local:
            if r.paper_id not in existing_ids:
                existing_ids.add(r.paper_id)
                all_local_results.append(r)
        for r in new_external:
            pid = r.paper_id
            if pid not in {e.paper_id for e in all_external_results}:
                all_external_results.append(r)

        # 重新调用 LLM
        answer = _build_and_call(
            system_prompt, request.query, request.track,
            all_local_results, all_external_results, round_num,
            previous_answer=answer,
        )

    # ─── 解析引用 ──────────────────────────────────────────
    cited_local = set()
    cited_ext = set()
    ext_offset = len(all_local_results)

    for m in re.finditer(r"\[(\d+)\]", answer):
        idx = int(m.group(1))
        if 1 <= idx <= len(all_local_results):
            cited_local.add(idx)

    for m in re.finditer(r"\[ext(\d+)\]", answer):
        idx = int(m.group(1))
        if 1 <= idx <= len(all_external_results):
            cited_ext.add(idx)

    all_cited_indices = cited_local | {ext_offset + i for i in cited_ext}
    evidence_cards = _build_evidence_cards(all_local_results, all_external_results, all_cited_indices)

    response = AgentResponse(
        answer=answer,
        evidence_cards=evidence_cards,
        query=request.query,
        track=request.track,
        model=LLM_MODEL,
        tool_calls=tool_calls,
    )

    # ─── 引用验证 ──────────────────────────────────────────
    verification = None
    if verify:
        from citation_verifier import full_verify
        local_dicts = [_search_result_to_dict(r) for r in all_local_results]
        ext_dicts = [_search_result_to_dict(r) for r in all_external_results]
        chunks = [r.text_summary for r in all_local_results if r.text_summary]
        chunks += [r.text_summary for r in all_external_results if r.text_summary]
        verification = full_verify(answer, request.query, request.track, local_dicts, ext_dicts, chunks)

    return {
        "response": response,
        "verification": verification,
    }


def _build_and_call(
    system_prompt: str,
    query: str,
    track: str,
    local_results: list[SearchResult],
    external_results: list[SearchResult],
    round_num: int,
    previous_answer: str = "",
) -> str:
    """构建 prompt 并调用 LLM"""
    local_ctx = _build_local_context(local_results, start_idx=1)
    ext_ctx = format_external_results_for_prompt(external_results)

    if round_num > 1 and previous_answer:
        extra = f"""

## 补充检索结果（第 {round_num} 轮）
以上回答的证据不足，系统已自动补充检索。以下是新增文献，请整合到回答中："""
    else:
        extra = ""

    user_message = f"""## 用户问题
{query}

期望证据类型：{"指南/系统综述/RCT/观察研究" if track == "clinical" else "指南/系统综述/随机试验/观察研究/机制"}

{local_ctx}

{ext_ctx}{extra}

请根据以上文献证据，按照要求格式回答用户的问题。记住：不得编造引用，每个关键结论必须有可追溯的编号来源。"""

    return _call_llm(system_prompt, user_message)


def _search_result_to_dict(r: SearchResult) -> dict:
    return {
        "paper_id": r.paper_id,
        "title": r.title,
        "year": r.year,
        "source_type": r.source_type,
        "text_summary": r.text_summary,
        "score": r.score,
        "source": r.source,
    }

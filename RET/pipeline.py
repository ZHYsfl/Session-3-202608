"""
医学 RAG 端到端 Pipeline（复用 async_tool_calling.py）
========================================================
用法:
    cd RET && source venv/bin/activate
    python pipeline.py "他汀类药物对冠心病二级预防的证据"

流程:
1. 本地 hybrid 检索
2. PubMed 外部检索
3. 将两者作为证据包传给 LLM
4. 返回答案 + 证据卡片
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from async_tool_calling import Agent, Tool, LLMConfig
from backend.agents import _local_search, CLINICAL_SYSTEM_PROMPT
from backend.schemas import SearchResult
from backend.tools import external_search, format_external_results_for_prompt
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL
from retriever import build_bm25_index
from load_metadata import load_metadata_from_dir


LLM_MODEL = os.environ.get("PIPELINE_MODEL", "deepseek-ai/DeepSeek-V3")
LLM_BASE_URL = SILICONFLOW_BASE_URL
LLM_API_KEY = SILICONFLOW_API_KEY


def _guess_config() -> LLMConfig:
    """优先使用 SiliconFlow；可通过环境变量覆盖。"""
    model = os.environ.get("PIPELINE_MODEL", LLM_MODEL)
    base_url = os.environ.get("PIPELINE_BASE_URL", LLM_BASE_URL)
    api_key = os.environ.get("PIPELINE_API_KEY", LLM_API_KEY)
    extra = {}
    if "deepseek" in base_url.lower():
        extra = {"enable_thinking": False}
    return LLMConfig(api_key=api_key, model=model, base_url=base_url, extra_body=extra or None)


class _SearchSession:
    """在工具调用之间保持检索结果，用于生成证据卡片。"""

    def __init__(self):
        self.local_results: list[SearchResult] = []
        self.external_results: list[SearchResult] = []

    def reset(self):
        self.local_results.clear()
        self.external_results.clear()


def _format_local_results(results: list[SearchResult]) -> str:
    if not results:
        return "（未找到相关本地文献）"
    lines = ["【本地数据库检索结果 — 高相关度文献】", ""]
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    期刊: {r.source_type}  |  年份: {r.year}  |  PMID: {r.paper_id}")
        lines.append(f"    相似度: {r.score:.3f}")
        lines.append(f"    摘要: {r.text_summary[:300]}")
        lines.append("")
    return "\n".join(lines)


def _build_evidence_cards(local: list[SearchResult], external: list[SearchResult]) -> list[dict]:
    cards = []
    for i, r in enumerate(local, start=1):
        cards.append({
            "index": i,
            "title": r.title,
            "year": r.year,
            "source_type": r.source_type,
            "pmid": r.paper_id,
            "link": "",
            "similarity_score": r.score,
            "source": "local",
            "summary_snippet": (r.text_summary or "")[:200],
        })
    offset = len(local)
    for i, r in enumerate(external, start=1):
        pmid = r.paper_id.replace("pubmed_", "")
        cards.append({
            "index": offset + i,
            "title": r.title,
            "year": r.year,
            "source_type": r.source_type,
            "pmid": pmid,
            "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "similarity_score": r.score,
            "source": "external",
            "summary_snippet": (r.text_summary or "")[:200],
        })
    return cards


async def run_pipeline(query: str, track: str = "clinical", top_k: int = 10) -> dict[str, Any]:
    """端到端 Pipeline：检索 → LLM 生成 → 证据卡片。"""
    session = _SearchSession()

    def search_local(q: str, k: int = top_k) -> str:
        results = _local_search(q, top_k=k)
        session.local_results = results
        return _format_local_results(results)

    def search_pubmed(q: str, k: int = 5) -> str:
        results = external_search(q, max_results=k)
        session.external_results = results
        return format_external_results_for_prompt(results)

    tools = [
        Tool(
            name="search_local",
            description="检索本地医学文献向量库（ChromaDB + BM25），返回相关文献列表。",
            function=search_local,
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "查询字符串"},
                    "k": {"type": "integer", "description": "返回结果数量", "default": top_k},
                },
                "required": ["q"],
            },
        ),
        Tool(
            name="search_pubmed",
            description="检索 PubMed 外部医学文献，返回可溯源文献列表。",
            function=search_pubmed,
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "英文 PubMed 查询字符串"},
                    "k": {"type": "integer", "description": "返回结果数量", "default": 5},
                },
                "required": ["q"],
            },
        ),
    ]

    config = _guess_config()
    agent = Agent(config, max_tool_retries=1, debug=False)
    for t in tools:
        agent.add_tool(t)

    system_prompt = CLINICAL_SYSTEM_PROMPT if track == "clinical" else "你是健康科普助手。"
    observations = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"## 用户问题\n{query}\n\n请先用 search_local 检索本地文献；如证据不足，再用 search_pubmed 检索外部文献。然后依据证据回答。"},
    ]

    # 限制最多 3 轮工具调用，避免无限循环
    final_observations = observations
    for _round in range(3):
        response = await agent.client.chat.completions.create(
            model=config.model,
            messages=final_observations,
            tools=agent._get_tools(),
            tool_choice="auto",
            extra_body=config.extra_body,
        )
        if response.choices[0].finish_reason != "tool_calls":
            final_observations.append(response.choices[0].message.model_dump())
            break
        final_observations = await agent.chat(final_observations)
    else:
        # 到达最大轮次，强制要求直接回答
        final_observations.append({
            "role": "user",
            "content": "已达到最大检索轮次，请根据已有证据直接给出最终答案。",
        })
        response = await agent.client.chat.completions.create(
            model=config.model,
            messages=final_observations,
            tools=agent._get_tools(),
            tool_choice="none",
            extra_body=config.extra_body,
        )
        final_observations.append(response.choices[0].message.model_dump())

    answer = final_observations[-1].get("content", "")
    evidence_cards = _build_evidence_cards(session.local_results, session.external_results)

    return {
        "query": query,
        "track": track,
        "model": config.model,
        "answer": answer,
        "evidence_cards": evidence_cards,
        "local_count": len(session.local_results),
        "external_count": len(session.external_results),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="医学 RAG 端到端 Pipeline")
    parser.add_argument("query", help="医学问题")
    parser.add_argument("--track", default="clinical", choices=["clinical", "nutrition"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # 确保 BM25 索引已预热（如果还没通过 server 预热）
    from retriever import _bm25_index
    if not _bm25_index.get_paper_ids():
        print("⏳ 预热 BM25 索引...", flush=True)
        metas = load_metadata_from_dir(os.path.join(os.path.dirname(__file__), "data"))
        build_bm25_index(metas)

    result = asyncio.run(run_pipeline(args.query, track=args.track, top_k=args.top_k))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

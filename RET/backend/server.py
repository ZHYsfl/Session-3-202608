"""
FastAPI 后端服务器
提供 REST API 供前端调用
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import asdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.schemas import ChatRequest, AgentResponse, EvidenceCard, SearchResult
from backend.agents import run_agent, _local_search, _get_metadata, preload_metadata


# ─── 启动时预加载 ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时预加载元数据"""
    preload_metadata()
    yield


app = FastAPI(title="证据智能助手 API", version="0.2.0", lifespan=lifespan)

# CORS — 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models for API ──────────────────────────────────

class ChatAPIRequest(BaseModel):
    query: str
    track: str = "clinical"  # "clinical" | "nutrition"


class ToolCallAPI(BaseModel):
    round: int
    action: str
    query: str
    result_count: int
    source: str = ""
    duration_ms: float = 0.0


class EvidenceCardAPI(BaseModel):
    index: int
    title: str
    year: str
    source_type: str
    pmid: str
    doi: str = ""
    link: str = ""
    similarity_score: float = 0.0
    source: str = "local"
    summary_snippet: str = ""


class ChatAPIResponse(BaseModel):
    answer: str
    evidence_cards: list[EvidenceCardAPI]
    query: str
    track: str
    model: str = ""
    tool_calls: list[ToolCallAPI] = []
    verification: dict | None = None


class SearchAPIResponse(BaseModel):
    results: list[dict]
    total: int


# ─── API 路由 ──────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "证据智能助手", "version": "0.2.0"}


@app.post("/api/chat")
async def chat(request: ChatAPIRequest):
    """主聊天接口 — 自动双路检索 + Agent 生成 + 引用验证"""
    try:
        req = ChatRequest(
            query=request.query,
            track=request.track,
        )
        result = run_agent(req, verify=True)

        resp = result["response"]
        verification = result.get("verification")

        # 证据卡片
        cards = [
            EvidenceCardAPI(
                index=c.index,
                title=c.title,
                year=c.year,
                source_type=c.source_type,
                pmid=c.pmid,
                doi=c.doi,
                link=c.link,
                similarity_score=c.similarity_score,
                source=c.source,
                summary_snippet=c.summary_snippet,
            )
            for c in resp.evidence_cards
        ]

        # 工具调用追踪
        tool_calls_api = [
            ToolCallAPI(
                round=t.round,
                action=t.action,
                query=t.query,
                result_count=t.result_count,
                source=t.source,
                duration_ms=t.duration_ms,
            )
            for t in resp.tool_calls
        ]

        # 验证报告
        verif_dict = None
        if verification:
            verif_dict = {
                "citation_existence_rate": verification.l1_citation.existence_rate,
                "total_citations": verification.l1_citation.total_citations,
                "real_citations": verification.l1_citation.real_citations,
                "hallucinated_citations": verification.l1_citation.hallucinated_citations,
                "hallucinated_numbers": verification.l1_citation.hallucinated_numbers,
                "faithfulness_score": verification.l2_faithfulness.faithfulness_score,
                "total_claims": verification.l2_faithfulness.total_claims,
                "supported_claims": verification.l2_faithfulness.supported_claims,
                "weak_support_claims": verification.l2_faithfulness.weak_support_claims,
                "unsupported_claims": verification.l2_faithfulness.unsupported_claims,
                "claims_detail": [
                    {
                        "text": c.text[:200],
                        "action": c.action,
                        "support_score": c.support_score,
                        "cited_numbers": c.cited_numbers,
                        "supported": c.supported,
                    }
                    for c in verification.l2_faithfulness.claims
                ],
            }

        return {
            "answer": resp.answer,
            "evidence_cards": [c.model_dump() for c in cards],
            "query": resp.query,
            "track": resp.track,
            "model": resp.model,
            "tool_calls": [t.model_dump() for t in tool_calls_api],
            "verification": verif_dict,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"检索生成失败: {str(e)[:300]}")


@app.post("/api/search")
async def search(request: ChatAPIRequest):
    """纯检索接口 — 不经过 LLM，直接返回检索结果"""
    try:
        results = _local_search(request.query, top_k=20)
        return SearchAPIResponse(
            results=[asdict(r) for r in results],
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str):
    """获取单篇论文详情"""
    meta_list = _get_metadata()
    for m in meta_list:
        if m.id == paper_id:
            return {
                "id": m.id,
                "title": m.title,
                "year": m.year,
                "source_type": m.source_type,
                "text_summary": m.text_summary,
                "text": (m.text or "")[:5000],
                "image_summary": m.image_summary,
                "image_base_64_count": len(m.image_base_64 or []),
            }
    raise HTTPException(status_code=404, detail="Paper not found")


@app.get("/api/stats")
async def stats():
    """数据库统计信息"""
    meta_list = _get_metadata()
    years = sorted(set(m.year for m in meta_list if m.year))
    source_types = sorted(set(m.source_type for m in meta_list if m.source_type))

    from vector_store import get_collection
    try:
        col = get_collection()
        chunk_count = col.count()
    except Exception:
        chunk_count = 0

    return {
        "total_papers": len(meta_list),
        "total_chunks": chunk_count,
        "year_range": f"{years[0]} - {years[-1]}" if years else "N/A",
        "source_types": source_types[:50],
        "source_type_count": len(source_types),
    }


# ─── 静态文件服务 (前端) ──────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


# ─── 启动入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

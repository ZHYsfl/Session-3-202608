"""LLM 裁判 v2 —— 本地开源模型版（Ollama）。

与 v1（llm_judge.py / llm_judge_v1.py，调用线上 API）的区别：
  - 直接走 Ollama 原生 HTTP 接口（/api/chat），只依赖 Python 标准库
  - 不需要 pip install openai，也不需要 API key

安装模型（约 4.7GB 下载，首次运行前执行一次）：
    ollama pull qwen2.5:7b
显存紧张（如 8GB）就选 7B 量化版；更小机器可用 qwen2.5:3b。

用法：
    from llm_judge_v2 import make_llm_compare
    from literature_pairwise import pairwise_rank

    compare = make_llm_compare(model="qwen2.5:7b", query="metformin for diabetes")
    ranked = pairwise_rank(top_m_records, compare=compare)

先决条件：Ollama 已启动（Windows 安装后一般自动常驻，默认端口 11434）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

# 裁判标准写死在 prompt 里（rubric），LLM 只负责按标准执行
COMPARE_RUBRIC = """你是医学文献评审专家。请严格按以下标准比较两篇医学文献：
1. 与临床问题/检索主题的相关性（最重要）
2. 证据质量与研究设计严谨性（RCT/系统评价 > 队列 > 病例对照 > 个案/专家意见）
3. 时效性（越新越好）
4. 结果的临床实用价值
只输出一个 JSON 对象，不要输出任何其他内容，格式如下：
{"winner": "A" 或 "B" 或 "TIE", "score_a": 0到100, "score_b": 0到100, "reason": "一句话理由"}"""

SCORE_RUBRIC = """你是医学文献评审专家。请对下面这篇医学文献打分（0到100），
考虑：1. 与检索主题的相关性；2. 证据质量与研究设计严谨性；3. 时效性；4. 临床实用价值。
只输出一个 JSON 对象，不要输出任何其他内容，格式如下：
{"score": 0到100, "reason": "一句话理由"}"""


# ---------- Ollama 通信（仅标准库） ----------

def _request_json(base_url: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama 请求失败 HTTP {exc.code}：{body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接 Ollama（{base_url}）。请先启动 Ollama 应用或运行 `ollama serve`：{exc.reason}"
        ) from exc


def _chat_json(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
) -> dict[str, Any]:
    """调 Ollama 原生 /api/chat，format=json 要求模型只输出 JSON。"""
    data = _request_json(
        base_url,
        "/api/chat",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        },
    )
    raw = data.get("message", {}).get("content", "")
    try:
        return _parse_json(raw)
    except Exception as exc:
        raise RuntimeError(f"模型输出无法解析成 JSON，原文前 300 字：{raw[:300]}") from exc


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def ensure_model(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> None:
    """检查模型是否已拉取，未安装时给出明确的安装命令。"""
    data = _request_json(base_url, "/api/tags", None)
    names = {m.get("name") for m in data.get("models", [])}
    if model not in names:
        raise RuntimeError(f"模型 {model} 未安装，请先运行：ollama pull {model}")


# ---------- 记录文本化（与 v1 相同） ----------

def _truncate(s: str | None, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else s[:limit] + "…"


def _record_to_text(rec: dict[str, Any], max_text_chars: int = 2000) -> str:
    """把一条记录压缩成 LLM 能看的文本（base64 图片数据不送，太贵）。"""
    image_summaries = " | ".join(x for x in (rec.get("image_summary") or []) if x)
    return (
        f"标题: {rec.get('title')}\n"
        f"年份: {rec.get('year')}\n"
        f"证据等级: {rec.get('evidence_level')}\n"
        f"摘要: {_truncate(rec.get('text_summary'), 1000)}\n"
        f"正文片段: {_truncate(rec.get('text'), max_text_chars)}\n"
        f"图片摘要: {_truncate(image_summaries, 500)}"
    )


def _record_key(rec: dict[str, Any]) -> str:
    return str(rec.get("id") or rec.get("title") or rec)


# ---------- 裁判工厂 ----------

def make_llm_compare(
    model: str = DEFAULT_MODEL,
    *,
    base_url: str = DEFAULT_BASE_URL,
    query: str | None = None,
    max_text_chars: int = 2000,
    temperature: float = 0.0,
    cache: dict[tuple[str, str], int] | None = None,
) -> Callable[[dict[str, Any], dict[str, Any]], int]:
    """生成 pairwise_rank 可用的 compare 函数：a 胜返回 1，b 胜返回 -1，平局 0。

    query 是统一的检索主题；不传时让模型按证据质量、时效、实用性判断。
    """
    cache = {} if cache is None else cache

    def compare(a: dict[str, Any], b: dict[str, Any]) -> int:
        ka, kb = _record_key(a), _record_key(b)
        if (ka, kb) in cache:
            return cache[(ka, kb)]
        if (kb, ka) in cache:  # 反过来的结果取反，省一半调用
            return -cache[(kb, ka)]

        user = (
            f"检索主题：{query or '（未提供，按证据质量、时效、实用性判断）'}\n\n"
            f"文献 A：\n{_record_to_text(a, max_text_chars)}\n\n"
            f"文献 B：\n{_record_to_text(b, max_text_chars)}"
        )
        data = _chat_json(base_url, model, COMPARE_RUBRIC, user, temperature)

        winner = str(data.get("winner", "TIE")).strip().upper()
        result = 1 if winner == "A" else -1 if winner == "B" else 0
        cache[(ka, kb)] = result
        return result

    return compare


def make_llm_scorer(
    model: str = DEFAULT_MODEL,
    *,
    base_url: str = DEFAULT_BASE_URL,
    query: str | None = None,
    max_text_chars: int = 2000,
    temperature: float = 0.0,
    cache: dict[str, float] | None = None,
) -> Callable[[dict[str, Any]], float]:
    """生成单篇打分函数 score(rec) -> 0~100，可用于：
        ranked = sorted(records, key=llm_score, reverse=True)
    """
    cache = {} if cache is None else cache

    def score(rec: dict[str, Any]) -> float:
        key = _record_key(rec)
        if key in cache:
            return cache[key]
        user = (
            f"检索主题：{query or '（未提供，按证据质量、时效、实用性判断）'}\n\n"
            f"文献：\n{_record_to_text(rec, max_text_chars)}"
        )
        data = _chat_json(base_url, model, SCORE_RUBRIC, user, temperature)
        value = float(data.get("score", 0))
        cache[key] = value
        return value

    return score


if __name__ == "__main__":
    # 端到端自测：需要 Ollama 已启动且模型已拉取
    from literature_pairwise import pairwise_rank

    demo = [
        {
            "id": "r0",
            "title": "RCT on new drug",
            "year": "2023",
            "evidence_level": "RCT",
            "text_summary": "A randomized trial of metformin for diabetes.",
            "text": "x" * 1000,
        },
        {
            "id": "r1",
            "title": "Old case report",
            "year": "2001",
            "evidence_level": "case report",
            "text_summary": "A single patient case.",
            "text": "x" * 100,
        },
        {
            "id": "r2",
            "title": "Recent editorial",
            "year": "2025",
            "evidence_level": "editorial",
            "text_summary": "Opinion about diabetes management.",
            "text": "x" * 600,
        },
    ]
    try:
        ensure_model()
        compare = make_llm_compare(query="metformin for type 2 diabetes")
        ranked = pairwise_rank(demo, compare=compare, return_counts=True)
        for rec, pts in ranked:
            print(rec["id"], pts)
    except RuntimeError as exc:
        print("自测失败：", exc)

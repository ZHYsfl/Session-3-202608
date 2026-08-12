"""把 LLM 接入打分 / 两两对比的示例。

核心思路：LLM 调用被封装成一个普通 Python 函数，签名与现有接口对齐——
  - make_llm_compare() 生成 pairwise_rank 需要的 compare(a, b) -> 1 / -1 / 0
  - make_llm_scorer()  生成单篇打分函数 score(rec) -> 0~100

这样你的排序主流程完全不用改，只是把裁判从“规则算分”换成“LLM 判断”。

想用本地开源模型（Ollama / LM Studio / llama.cpp / vLLM）也很简单：
这些工具都提供 OpenAI 兼容的 HTTP 接口，只需换 base_url 和模型名，
OpenAI SDK 照常使用，例如：

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    compare = make_llm_compare(client, model="qwen2.5:7b-instruct-q4_K_M")

部分本地服务不认 response_format（JSON 模式），传入 json_mode=False 即可。
"""

from __future__ import annotations

import json
from typing import Any, Callable


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


def _get_message_content(response: Any) -> str:
    """兼容 chat.completions 和 responses 两种 SDK 返回结构。"""
    text = getattr(response, "output_text", None)
    if text:
        return text
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return str(response)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _call_llm_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float,
    json_mode: bool = True,
) -> dict[str, Any]:
    """调用 chat.completions 的 JSON 模式并解析结果，解析失败给足报错信息。"""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if json_mode:
        # JSON 模式要求 prompt 里出现 "json" 字样（rubric 里已包含）
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    raw = _get_message_content(response)
    try:
        return _parse_json(raw)
    except Exception as exc:
        raise RuntimeError(f"LLM 返回无法解析成 JSON，原文前 300 字：{raw[:300]}") from exc


def make_llm_compare(
    client: Any,
    model: str = "gpt-4.1-mini",
    *,
    max_text_chars: int = 2000,
    temperature: float = 0.0,
    json_mode: bool = True,
    cache: dict[tuple[str, str], int] | None = None,
) -> Callable[[dict[str, Any], dict[str, Any]], int]:
    """生成 pairwise_rank 可用的 compare 函数：a 胜返回 1，b 胜返回 -1，平局 0。

    用法：
        from openai import OpenAI
        client = OpenAI()                      # 读取 OPENAI_API_KEY 环境变量
        compare = make_llm_compare(client)
        result = pairwise_rank(top_m_records, compare=compare)

    注意 pairwise 是 O(n^2)，LLM 每对要一次调用，建议先用规则筛到 top m 再进来。
    """
    cache = {} if cache is None else cache

    def compare(a: dict[str, Any], b: dict[str, Any]) -> int:
        ka, kb = _record_key(a), _record_key(b)
        if (ka, kb) in cache:
            return cache[(ka, kb)]
        if (kb, ka) in cache:  # 反过来的结果取反，省一半调用
            return -cache[(kb, ka)]

        user = (
            f"检索主题：{COMPARE_QUERY or '（未提供，按证据质量、时效、实用性判断）'}\n\n"
            f"文献 A：\n{_record_to_text(a, max_text_chars)}\n\n"
            f"文献 B：\n{_record_to_text(b, max_text_chars)}"
        )
        data = _call_llm_json(
            client, model, COMPARE_RUBRIC, user, temperature, json_mode=json_mode
        )

        winner = str(data.get("winner", "TIE")).strip().upper()
        result = 1 if winner == "A" else -1 if winner == "B" else 0
        cache[(ka, kb)] = result
        return result

    return compare


# 让 compare 的 prompt 也能带检索主题；默认 None
COMPARE_QUERY: str | None = None


def set_compare_query(query: str | None) -> None:
    """设置两两对比时统一的检索主题（调用 make_llm_compare 之前调用）。"""
    global COMPARE_QUERY
    COMPARE_QUERY = query


def make_llm_scorer(
    client: Any,
    model: str = "gpt-4.1-mini",
    *,
    query: str | None = None,
    max_text_chars: int = 2000,
    temperature: float = 0.0,
    json_mode: bool = True,
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
        data = _call_llm_json(
            client, model, SCORE_RUBRIC, user, temperature, json_mode=json_mode
        )
        value = float(data.get("score", 0))
        cache[key] = value
        return value

    return score


if __name__ == "__main__":
    # 离线自测：用假客户端验证“LLM 裁判”能正确接进 pairwise_rank
    from types import SimpleNamespace

    from literature_pairwise import pairwise_rank

    def _fake_create(**kwargs):
        message = SimpleNamespace(
            content='{"winner": "A", "score_a": 90, "score_b": 10, "reason": "fake"}'
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    _FakeClient = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_fake_create))
    )

    demo = [
        {"id": "r0", "title": "RCT on new drug", "year": "2023",
         "evidence_level": "RCT", "text_summary": "metformin diabetes",
         "text": "x" * 1000},
        {"id": "r1", "title": "Old case report", "year": "2001",
         "evidence_level": "case report", "text_summary": "single patient",
         "text": "x" * 100},
        {"id": "r2", "title": "Recent editorial", "year": "2025",
         "evidence_level": "editorial", "text_summary": "opinion",
         "text": "x" * 600},
    ]
    compare = make_llm_compare(_FakeClient, cache={})
    ranked = pairwise_rank(demo, compare=compare, return_counts=True)
    for rec, pts in ranked:
        print(rec["id"], pts)

    # 真实用法（需要 pip install openai 且设置 OPENAI_API_KEY）：
    # from openai import OpenAI
    # client = OpenAI()
    # set_compare_query("metformin for type 2 diabetes")
    # compare = make_llm_compare(client, model="gpt-4.1-mini")
    # ranked = pairwise_rank(top_m_records, compare=compare)
